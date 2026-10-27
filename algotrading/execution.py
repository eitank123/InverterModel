"""Realistic execution: trade real instruments (ETFs) with every cost charged.

Costs per trade
  * commission per order (most US brokers: $0; we charge a conservative amount)
  * half the bid/ask spread + slippage, in basis points of the traded amount
  * US regulatory fees on sales: SEC Section 31 fee (per $ sold) and FINRA TAF (per share)
Costs inside the funds (taken out of the ETF's price, like a real fund)
  * expense ratio
  * leveraged ETFs: financing of the borrowed part, (L-1) x (cash rate + swap spread)
  * leveraged ETFs: daily-reset drag ~ L(L-1)/2 x variance (approximated from monthly data)
Cash
  * idle cash earns the cash rate minus a small sweep/T-bill-fund fee

The ledger keeps average-cost basis per symbol, so every sale reports a realized profit/loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Instrument:
    symbol: str
    underlying: str          # "stocks", "bonds" or "cash"
    leverage: float = 1.0
    expense_ratio: float = 0.0
    spread_bps: float = 1.0  # half spread + slippage per trade


# Approximate public figures (expense ratios as published by the issuers; spreads typical).
INSTRUMENTS = {
    "SPY": Instrument("SPY", "stocks", 1.0, 0.000945, 1.0),
    "SSO": Instrument("SSO", "stocks", 2.0, 0.0089, 3.0),
    "UPRO": Instrument("UPRO", "stocks", 3.0, 0.0091, 3.0),
    "IEF": Instrument("IEF", "bonds", 1.0, 0.0015, 2.0),
    "UST": Instrument("UST", "bonds", 2.0, 0.0095, 5.0),       # 2x 7-10y Treasuries (thinly traded)
}


def etf_weights(stocks: float, bonds: float) -> dict[str, float]:
    """Map a stock/bond exposure (e.g. 2.4x stocks, 0.67x bonds) onto unlevered and levered
    ETFs without a margin loan where possible: bonds through IEF (or UST when >1x), stocks
    through SPY + UPRO using exactly the capital left over."""
    w: dict[str, float] = {}
    if bonds > 1.0:
        w["UST"] = bonds / 2
    elif bonds > 0:
        w["IEF"] = bonds
    free = max(1.0 - sum(w.values()), 0.0)
    if stocks <= 0:
        pass
    elif stocks <= free:
        w["SPY"] = stocks
    elif stocks <= 3 * free:
        w["UPRO"] = (stocks - free) / 2
        w["SPY"] = free - w["UPRO"]
    else:
        w["UPRO"] = stocks / 3
    return {k: round(v, 6) for k, v in w.items() if v > 1e-6}


@dataclass
class CostModel:
    commission_per_order: float = 1.00       # $; Alpaca/IBKR Lite charge $0, we stay conservative
    extra_slippage_bps: float = 2.0          # on top of each instrument's spread
    sec_fee_rate: float = 27.80 / 1e6        # Section 31 fee per $ sold (FY2024 rate; changes yearly)
    finra_taf_per_share: float = 0.000166    # FINRA trading activity fee on sales
    finra_taf_max: float = 8.30
    swap_spread: float = 0.005               # leveraged-ETF financing over cash
    cash_fee: float = 0.0014                 # T-bill fund / sweep fee on idle cash


def etf_prices(index_returns: pd.Series, cash_rate: pd.Series, instruments: dict[str, Instrument],
               costs: CostModel, start_price: dict[str, float] | None = None) -> pd.DataFrame:
    """Monthly price paths of ETFs built from their underlying's total return, net of fund costs.

    For a fund with leverage L: r = L*r_u - (L-1)*(cash+swap)/12 - expense/12 - L(L-1)/2 * var_m,
    where var_m is the month's variance proxy (trailing 12-month variance of monthly returns).
    """
    out = {}
    for sym, ins in instruments.items():
        r_u = index_returns[ins.underlying]
        var_m = r_u.rolling(12, min_periods=3).var().bfill()
        L = ins.leverage
        r = (L * r_u - (L - 1) * (cash_rate / 12 + costs.swap_spread / 12)
             - ins.expense_ratio / 12 - 0.5 * L * (L - 1) * var_m)
        p = (1 + r.fillna(0.0)).cumprod()
        out[sym] = p / p.iloc[-1] * (start_price or {}).get(sym, 100.0)
    return pd.DataFrame(out)


@dataclass
class Ledger:
    cash: float
    costs: CostModel = field(default_factory=CostModel)
    shares: dict[str, float] = field(default_factory=dict)
    basis: dict[str, float] = field(default_factory=dict)        # total cost basis per symbol
    trades: list[dict] = field(default_factory=list)
    fees: dict[str, float] = field(default_factory=lambda: {"commission": 0.0, "spread_slippage": 0.0,
                                                             "regulatory": 0.0})

    def value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices[s] for s, q in self.shares.items())

    def trade(self, date, symbol: str, qty: float, price: float, spread_bps: float, reason: str):
        """qty > 0 buys, qty < 0 sells. Costs are paid in cash and recorded on the trade."""
        if abs(qty) < 1e-9:
            return
        c = self.costs
        notional = abs(qty) * price
        commission = c.commission_per_order
        spread = notional * (spread_bps + c.extra_slippage_bps) / 1e4
        regulatory = 0.0
        realized = 0.0
        held = self.shares.get(symbol, 0.0)
        if qty > 0:
            self.basis[symbol] = self.basis.get(symbol, 0.0) + notional + spread + commission
        else:
            sell = min(-qty, held)
            regulatory = notional * c.sec_fee_rate + min(sell * c.finra_taf_per_share, c.finra_taf_max)
            avg = self.basis.get(symbol, 0.0) / held if held else 0.0
            proceeds = notional - spread - commission - regulatory
            realized = proceeds - avg * sell
            self.basis[symbol] = self.basis.get(symbol, 0.0) - avg * sell
        self.cash -= qty * price + spread + commission + regulatory
        self.shares[symbol] = held + qty
        if abs(self.shares[symbol]) < 1e-9:
            self.shares.pop(symbol)
            self.basis.pop(symbol, None)
        self.fees["commission"] += commission
        self.fees["spread_slippage"] += spread
        self.fees["regulatory"] += regulatory
        self.trades.append({"date": pd.Timestamp(date).strftime("%Y-%m-%d"), "symbol": symbol,
                            "side": "buy" if qty > 0 else "sell", "shares": round(abs(qty), 4),
                            "price": round(price, 4), "amount": round(notional, 2),
                            "commission": round(commission, 2), "spread_slippage": round(spread, 2),
                            "regulatory_fees": round(regulatory, 2), "realized_pl": round(realized, 2),
                            "reason": reason})


def simulate(target_fn, prices: pd.DataFrame, cash_rate: pd.Series, start_cash: float = 100_000.0,
             costs: CostModel | None = None, min_trade_frac: float = 0.02,
             instruments: dict[str, Instrument] = INSTRUMENTS) -> dict:
    """Run a monthly strategy on ETF prices with every cost.

    target_fn(date) -> ({symbol: weight}, reason) is called with the decision date; the
    trades execute at the NEXT month's price (signals from monthly-average data, Exp 1).
    Returns equity curve, trades, monthly P/L and the cost breakdown.
    """
    costs = costs or CostModel()
    led = Ledger(start_cash, costs)
    dates = prices.index
    rows, fund_costs = [], 0.0
    pending = target_fn(dates[0])
    for i, d in enumerate(dates):
        px = prices.loc[d].to_dict()
        if i > 0:
            # one month of interest on idle cash (or cost of a negative balance)
            rate = cash_rate.loc[d] / 12
            led.cash *= 1 + (rate - costs.cash_fee / 12 if led.cash > 0 else rate + costs.swap_spread / 12)
            # fund costs embedded in ETF prices this month (reported, not charged twice)
            for s, q in led.shares.items():
                ins = instruments[s]
                fund_costs += q * px[s] * (ins.expense_ratio + (ins.leverage - 1)
                                           * (cash_rate.loc[d] + costs.swap_spread)) / 12
        weights, reason = pending
        equity = led.value(px)
        orders = []
        for s in set(weights) | set(led.shares):
            delta = weights.get(s, 0.0) * equity / px[s] - led.shares.get(s, 0.0)
            full_exit = weights.get(s, 0.0) == 0.0 and led.shares.get(s, 0.0) != 0.0
            if abs(delta) * px[s] >= min_trade_frac * equity or full_exit:
                orders.append((s, delta))
        for s, delta in sorted(orders, key=lambda o: o[1] > 0):        # sells first
            led.trade(d, s, delta, px[s], instruments[s].spread_bps, reason)
        rows.append({"date": d.strftime("%Y-%m-%d"), "equity": round(led.value(px), 2),
                     "cash": round(led.cash, 2),
                     **{f"w_{s}": round(led.shares.get(s, 0.0) * px[s] / led.value(px), 4) for s in prices}})
        pending = target_fn(d)
    eq = pd.DataFrame(rows).set_index("date")
    eq["pl"] = eq["equity"].diff().fillna(eq["equity"].iloc[0] - start_cash)
    return {"equity": eq, "trades": pd.DataFrame(led.trades), "fees": {**led.fees, "fund_costs_embedded": fund_costs},
            "final": float(eq["equity"].iloc[-1])}
