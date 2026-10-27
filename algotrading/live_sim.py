"""Real-time forward simulation: play money, real market data, every cost.

Each account starts on a real date with $100,000 and only moves forward. Every run reads the
newest daily closes (FRED: S&P 500, 10-year Treasury yield, 3-month T-bill), then for each new
trading day:
  1. idle cash earns the T-bill rate minus a sweep fee (a negative balance pays more);
  2. ETF prices move with the real index (daily-reset leverage, financing and expense ratio,
     exactly as the funds do it);
  3. on the first trading day of a month (and on day one) the strategy's signal is computed
     from completed month-ends only and the account rebalances at that day's close, paying
     commission, spread + slippage and regulatory fees (algotrading.execution.Ledger);
  4. the day's equity is recorded.
State is saved per account, so the record is never rebuilt or backfilled.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import pandas as pd

from .execution import INSTRUMENTS, CostModel, Ledger

def daily_etf_returns(index: pd.DataFrame, cash_rate: pd.Series, costs: CostModel) -> pd.DataFrame:
    """Daily ETF returns from daily index total returns (columns stocks/bonds)."""
    r = index.pct_change().fillna(0.0)
    cr = cash_rate.reindex(index.index).ffill().fillna(0.0)
    out = {}
    for sym, ins in INSTRUMENTS.items():
        L = ins.leverage
        out[sym] = (L * r[ins.underlying] - (L - 1) * (cr + costs.swap_spread) / 252
                    - ins.expense_ratio / 252)
    return pd.DataFrame(out, index=index.index)


class LiveSimAccount:
    def __init__(self, path: str, state: dict):
        self.path, self.st = path, state

    @classmethod
    def load_or_create(cls, path, key, name, description, start_date, cash=100_000.0):
        if os.path.exists(path):
            with open(path) as f:
                return cls(path, json.load(f))
        return cls(path, {"key": key, "name": name, "description": description, "start": start_date,
                          "start_equity": cash, "cash": cash, "holdings": {}, "basis": {},
                          "targets": {}, "exposure": {}, "last_date": None, "last_signal_month": None,
                          "fees": {"commission": 0.0, "spread_slippage": 0.0, "regulatory": 0.0},
                          "fund_costs": 0.0, "equity_history": [], "trades": [], "prices": {}})

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.st, f, indent=1, default=float)

    def advance(self, etf_px: pd.DataFrame, cash_rate: pd.Series, target_fn, costs: CostModel):
        """Process every trading day after ``last_date``. ``target_fn(date)`` returns
        ({symbol: weight}, exposure dict, reason) using data up to that date only."""
        st = self.st
        days = etf_px.index[etf_px.index >= pd.Timestamp(st["start"])]
        if st["last_date"]:
            days = days[days > pd.Timestamp(st["last_date"])]
        led = Ledger(st["cash"], costs, dict(st["holdings"]), dict(st["basis"]), [], dict(st["fees"]))
        new = 0
        for d in days:
            px = etf_px.loc[d].to_dict()
            if st["last_date"]:                                   # one trading day of interest
                rate = float(cash_rate.asof(d)) / 252
                led.cash *= 1 + (rate - costs.cash_fee / 252 if led.cash > 0 else rate + costs.swap_spread / 252)
                for s, q in led.shares.items():                   # costs inside the funds (reported)
                    ins = INSTRUMENTS[s]
                    st["fund_costs"] += q * px[s] * (ins.expense_ratio + (ins.leverage - 1)
                                                     * (float(cash_rate.asof(d)) + costs.swap_spread)) / 252
            month = d.strftime("%Y-%m")
            if st["last_signal_month"] != month:                  # first trading day of a month
                targets, exposure, reason = target_fn(d)
                equity = led.value(px)
                orders = []
                for s in set(targets) | set(led.shares):
                    delta = targets.get(s, 0.0) * equity / px[s] - led.shares.get(s, 0.0)
                    full_exit = targets.get(s, 0.0) == 0.0 and led.shares.get(s, 0.0) != 0.0
                    if abs(delta) * px[s] >= 0.02 * equity or full_exit:
                        orders.append((s, delta))
                for s, delta in sorted(orders, key=lambda o: o[1] > 0):   # sells first
                    led.trade(d, s, delta, px[s], INSTRUMENTS[s].spread_bps, reason)
                st["targets"], st["exposure"], st["last_signal_month"] = targets, exposure, month
            st["equity_history"].append({"date": d.strftime("%Y-%m-%d"), "equity": round(led.value(px), 2)})
            st["last_date"] = d.strftime("%Y-%m-%d")
            st["prices"] = {k: round(v, 4) for k, v in px.items()}
            new += 1
        st["cash"], st["holdings"], st["basis"], st["fees"] = led.cash, led.shares, led.basis, led.fees
        for t in led.trades:                                      # same fields as the paper ledger
            t["qty"], t["price"] = t["shares"], t["price"]
        st["trades"] += led.trades
        return new
