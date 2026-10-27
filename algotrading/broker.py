"""From target weights to orders: a broker interface, a local paper broker, an Alpaca
adapter, and a rebalancer with safety limits.

Live trading pipeline (run once a month, after the close of the month's last trading day):

    prices  ->  strategy  ->  target weights  ->  plan_rebalance  ->  safety checks  ->  orders

Rules this module enforces, because automated trading fails in boring ways:
  * dry-run by default: orders are only sent when ``execute=True``
  * gross leverage cap and per-order size cap (a "fat finger" guard)
  * no trading when the market is closed or the account is blocked
  * small drifts are ignored (``min_trade_frac``) to avoid paying costs for noise
  * sells are sent before buys, so cash is freed first
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Order:
    symbol: str
    qty: float          # shares, always positive
    side: str           # "buy" or "sell"

    def notional(self, price: float) -> float:
        return self.qty * price


class Broker(ABC):
    @abstractmethod
    def equity(self) -> float:
        """Net account value (positions + cash, cash may be negative with margin)."""

    @abstractmethod
    def positions(self) -> dict[str, float]:
        """Shares held per symbol."""

    @abstractmethod
    def last_prices(self, symbols: list[str]) -> dict[str, float]:
        ...

    @abstractmethod
    def market_open(self) -> bool:
        ...

    @abstractmethod
    def submit(self, order: Order) -> str:
        """Send one market order; return a broker order id."""


class PaperBroker(Broker):
    """In-memory broker for tests and dry runs: fills market orders instantly at the
    given price, charging ``cost_bps`` per trade. Cash can go negative (margin)."""

    def __init__(self, cash: float, prices: dict[str, float], cost_bps: float = 2.0):
        self.cash, self.prices, self.cost_bps = cash, dict(prices), cost_bps
        self.holdings: dict[str, float] = {}
        self.log: list[Order] = []

    def equity(self) -> float:
        return self.cash + sum(q * self.prices[s] for s, q in self.holdings.items())

    def positions(self) -> dict[str, float]:
        return {s: q for s, q in self.holdings.items() if abs(q) > 1e-9}

    def last_prices(self, symbols: list[str]) -> dict[str, float]:
        return {s: self.prices[s] for s in symbols}

    def market_open(self) -> bool:
        return True

    def submit(self, order: Order) -> str:
        px = self.prices[order.symbol]
        sign = 1 if order.side == "buy" else -1
        self.holdings[order.symbol] = self.holdings.get(order.symbol, 0.0) + sign * order.qty
        self.cash -= sign * order.qty * px + order.qty * px * self.cost_bps / 1e4
        self.log.append(order)
        return f"paper-{len(self.log)}"


class AlpacaBroker(Broker):
    """Alpaca (US stocks/ETFs, commission-free, paper and live accounts) via ``alpaca-py``.

    Keys come from the environment (ALPACA_API_KEY / ALPACA_SECRET_KEY), never from code.
    ``paper=True`` talks to the paper-trading endpoint: fake money, real market data.
    """

    def __init__(self, paper: bool = True, api_key: str | None = None, secret_key: str | None = None):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        key = api_key or os.environ["ALPACA_API_KEY"]
        secret = secret_key or os.environ["ALPACA_SECRET_KEY"]
        self.trading = TradingClient(key, secret, paper=paper)
        self.data = StockHistoricalDataClient(key, secret)

    def equity(self) -> float:
        return float(self.trading.get_account().equity)

    def account_ok(self) -> bool:
        acct = self.trading.get_account()
        status = getattr(acct.status, "value", acct.status)     # enum or plain string
        return not acct.trading_blocked and status == "ACTIVE"

    def positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self.trading.get_all_positions()}

    def last_prices(self, symbols: list[str]) -> dict[str, float]:
        from alpaca.data.requests import StockLatestTradeRequest
        trades = self.data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbols))
        return {s: float(trades[s].price) for s in symbols}

    def daily_closes(self, symbols: list[str], start: str) -> pd.DataFrame:
        """Split- and dividend-adjusted daily closes (a total-return price series)."""
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                               start=pd.Timestamp(start).to_pydatetime(), adjustment=Adjustment.ALL)
        bars = self.data.get_stock_bars(req).df
        closes = bars["close"].unstack(level=0)
        closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
        return closes[symbols]

    def market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    def submit(self, order: Order) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        req = MarketOrderRequest(symbol=order.symbol, qty=round(order.qty, 4),
                                 side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                                 time_in_force=TimeInForce.DAY)
        return str(self.trading.submit_order(req).id)


# --- Planning and safety ---------------------------------------------------------------

@dataclass
class Limits:
    max_gross_leverage: float = 1.5      # sum of |weights| allowed
    max_order_frac: float = 0.75         # no single order larger than this share of equity
    min_trade_frac: float = 0.02         # ignore drifts smaller than this share of equity


def plan_rebalance(target_weights: dict[str, float], equity: float,
                   positions: dict[str, float], prices: dict[str, float],
                   limits: Limits = Limits()) -> list[Order]:
    """Orders that move current holdings to ``target_weights`` (fractions of equity).
    Symbols held but absent from the target are sold. Sells come first."""
    gross = sum(abs(w) for w in target_weights.values())
    if gross > limits.max_gross_leverage + 1e-9:
        raise ValueError(f"target gross leverage {gross:.2f} exceeds limit {limits.max_gross_leverage}")
    if any(w < 0 for w in target_weights.values()):
        raise ValueError("short positions are not supported by this rebalancer")
    if equity <= 0:
        raise ValueError("non-positive equity")
    orders = []
    for sym in sorted(set(target_weights) | set(positions)):
        px = prices[sym]
        target_qty = target_weights.get(sym, 0.0) * equity / px
        delta = target_qty - positions.get(sym, 0.0)
        notional = abs(delta) * px
        full_exit = sym not in target_weights or target_weights[sym] == 0.0
        if notional < limits.min_trade_frac * equity and not (full_exit and delta != 0):
            continue
        if notional > limits.max_order_frac * equity:
            raise ValueError(f"order for {sym} ({notional:,.0f}) exceeds {limits.max_order_frac:.0%} "
                             "of equity: refusing (fat-finger guard)")
        orders.append(Order(sym, abs(delta), "buy" if delta > 0 else "sell"))
    return sorted(orders, key=lambda o: o.side != "sell")


def rebalance(broker: Broker, target_weights: dict[str, float], limits: Limits = Limits(),
              execute: bool = False) -> list[tuple[Order, str]]:
    """Plan and (optionally) send the orders. Returns (order, id or 'DRY-RUN') pairs."""
    if execute and not broker.market_open():
        raise RuntimeError("market is closed: not sending orders")
    if execute and isinstance(broker, AlpacaBroker) and not broker.account_ok():
        raise RuntimeError("account is blocked or inactive")
    symbols = sorted(set(target_weights) | set(broker.positions()))
    orders = plan_rebalance(target_weights, broker.equity(), broker.positions(),
                            broker.last_prices(symbols), limits)
    return [(o, broker.submit(o) if execute else "DRY-RUN") for o in orders]


def frozen_rule_weights(closes: pd.DataFrame, stock: str, bond: str, leverage: float = 1.0,
                        window: int = 10, asof: pd.Timestamp | None = None) -> dict[str, float]:
    """Target weights of the frozen research rule (60/40, each asset held only while its
    month-end close is above its ``window``-month average), from daily adjusted closes.

    Only COMPLETED months are used: if ``asof`` (default: today) is inside a month, that
    partial month is ignored, so the signal can't change mid-month.
    """
    asof = pd.Timestamp(asof or pd.Timestamp.today()).normalize()
    month_ends = closes.groupby(closes.index.to_period("M")).tail(1)
    last = month_ends.index[-1]
    finished = (asof.to_period("M") > last.to_period("M")
                or last.normalize() == last + pd.offsets.BMonthEnd(0))
    if not finished:
        month_ends = month_ends.iloc[:-1]           # current month not finished yet
    sma = month_ends.rolling(window).mean()
    on = (month_ends.iloc[-1] > sma.iloc[-1]).astype(float)
    return {stock: float(0.6 * leverage * on[stock]), bond: float(0.4 * leverage * on[bond])}
