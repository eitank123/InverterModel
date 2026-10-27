"""Forward paper-trading ledger: accounts that start on a real date and only move forward.

Each account holds a strategy, a cash balance (interest earned when positive, borrowing
cost paid when negative) and share holdings. ``step`` is called once per completed month:
accrue interest, mark to market, rebalance to the strategy's target weights, record.
State lives in one JSON file per account so the history survives between runs.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import pandas as pd

from .broker import Limits, PaperBroker, plan_rebalance


@dataclass
class AccountState:
    name: str
    description: str
    symbols: dict[str, str]                     # strategy column -> traded symbol
    start: str
    cash: float
    holdings: dict[str, float] = field(default_factory=dict)
    last_month: str | None = None
    equity_history: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    targets: dict[str, float] = field(default_factory=dict)


class PaperAccount:
    def __init__(self, path: str, state: AccountState):
        self.path, self.state = path, state

    @classmethod
    def load_or_create(cls, path: str, **kwargs) -> "PaperAccount":
        if os.path.exists(path):
            with open(path) as f:
                return cls(path, AccountState(**json.load(f)))
        return cls(path, AccountState(**kwargs))

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(asdict(self.state), f, indent=1, default=float)

    def equity(self, prices: dict[str, float]) -> float:
        return self.state.cash + sum(q * prices[s] for s, q in self.state.holdings.items())

    def step(self, month: pd.Timestamp, prices: dict[str, float], weights: dict[str, float],
             cash_rate: float, borrow_spread: float, cost_bps: float = 2.0,
             max_gross: float = 3.0):
        """Process one completed month: interest, mark-to-market, rebalance, record."""
        st = self.state
        if st.last_month is not None:                       # one month of interest/borrowing
            rate = cash_rate + (borrow_spread if st.cash < 0 else 0.0)
            st.cash *= 1 + rate / 12
        broker = PaperBroker(st.cash, prices, cost_bps)
        broker.holdings = dict(st.holdings)
        target = {st.symbols[k]: float(v) for k, v in weights.items()}
        orders = plan_rebalance(target, broker.equity(), broker.positions(), prices,
                                Limits(max_gross_leverage=max_gross, max_order_frac=10.0))
        for o in orders:
            broker.submit(o)
            st.trades.append({"date": month.strftime("%Y-%m-%d"), "symbol": o.symbol,
                              "side": o.side, "qty": round(o.qty, 4), "price": round(prices[o.symbol], 4)})
        st.cash, st.holdings = broker.cash, broker.positions()
        st.targets = target
        st.last_month = month.strftime("%Y-%m")
        st.equity_history.append({"date": month.strftime("%Y-%m-%d"),
                                  "equity": round(broker.equity(), 2)})
