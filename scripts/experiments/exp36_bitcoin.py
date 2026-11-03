"""Experiment 36: does a small Bitcoin sleeve help? (2015-2026, the only history there is)

Liu & Tsyvinski (2021, RFS 'Risks and returns of cryptocurrency'): crypto returns are driven by
crypto momentum and investor attention, not by stock-market factors - in principle a diversifier.
Data: FRED CBBTCUSD (Coinbase daily price). Rules, round and fixed: buy & hold, and a trend filter
(price above its 200-day average, else T-bills; Gayed-Bilello's daily rule from Exp 18). Costs:
0.5% per trade (crypto spreads/fees through a spot ETF or exchange), 0.25%/yr fund fee.
Then the champion (CRSP monthly) with 5% or 10% of the account in the Bitcoin trend sleeve, rebalanced
monthly. Variants counted: 2 rules + 4 combinations = 6. Eleven years is short and dominated by a
few huge bull runs: treat every number here as weak evidence.

    python scripts/experiments/exp36_bitcoin.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import algotrading as at  # noqa: E402
import exp31_combinations as e  # noqa: E402

btc = at.data.load_fred("CBBTCUSD").dropna()
r = btc.pct_change().fillna(0)
tb = (at.data.load_fred("DTB3") / 100).reindex(btc.index).ffill().bfill()
on = (btc > btc.rolling(200).mean()).astype(float).shift(1).fillna(0)
trend = on * r + (1 - on) * tb / 365 - on.diff().abs().fillna(0) * 0.005 - on * 0.0025 / 365
hold = r - 0.0025 / 365


def m(x):
    y = (1 + x).groupby(x.index.to_period("M")).prod() - 1
    y.index = y.index.to_timestamp()
    return y


def stats(x, per=12):
    w = (1 + x).cumprod()
    return {"2016-2020": at.metrics.cagr(x["2016":"2020"], per), "2021-2026": at.metrics.cagr(x["2021":], per),
            "2016-2026": at.metrics.cagr(x["2016":], per), "max DD": float((w["2016":] / w["2016":].cummax() - 1).min())}


bm, tm = m(hold), m(trend)
rows = {"Bitcoin buy & hold": stats(bm), "Bitcoin 200-day trend": stats(tm)}
for lab, base in [("champion", e.portfolio(e.MKT)), ("diversified x2/3", e.portfolio(e.LEGS["diversified (ind mom + stock mom + value)"], scale=2 / 3))]:
    b = base.reindex(tm.index).dropna()
    rows[f"{lab}"] = stats(b)
    for k in (0.05, 0.10):
        rows[f"{lab} + {k:.0%} Bitcoin trend"] = stats((1 - k) * b + k * tm.reindex(b.index))
pd.set_option("display.width", 200)
print(f"Bitcoin {btc.index[0]:%Y-%m-%d} to {btc.index[-1]:%Y-%m-%d}; champion data to {e.DATES[-1]:%Y-%m}\n")
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1%}"))
print(f"\nCorrelation of monthly returns, Bitcoin vs US stocks (2016-2026): {bm['2016':].corr(e.MKT.reindex(bm.index)['2016':]):+.2f}")
