"""Experiment 23: cross-sectional momentum across 13 stock markets (country rotation).

Asness, Moskowitz & Pedersen (2013, 'Value and Momentum Everywhere') find that country equity
indices with the best past 12-month returns (skipping the latest month) keep outperforming.
Rule fixed in advance: each month rank the markets by their 12-1 month return and hold the
top 4 equally (top third), each only while above its 10-month average (Faber 2007 /
Antonacci 2014 absolute filter); otherwise cash. Compared with the equal-weight global trend
sleeve of Exp 21. Same data and costs as Exp 21 (OECD monthly via FRED, +3% dividends,
USD-hedged approximation, 10bp per unit traded, leverage at cash + 0.5%).

    python scripts/experiments/exp23_country_momentum.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

CC = ["US", "DE", "GB", "FR", "IT", "CA", "AU", "NL", "CH", "SE", "ES", "KR", "JP"]
cols = {}
for cc in CC:
    s = at.data.load_fred(f"SPASTT01{cc}M661N")
    s.index = s.index.to_period("M").to_timestamp()
    cols[cc] = s * (1 + 0.03 / 12) ** np.arange(len(s))
px = pd.DataFrame(cols)["1960":]
tb = at.data.load_fred("DTB3") / 100
cash = tb.groupby(tb.index.to_period("M")).mean()
cash.index = cash.index.to_timestamp()
cash = cash.reindex(px.index).ffill()


class CountryRotation(at.PortfolioStrategy):
    """Top ``k`` markets by 12-1 month return, equal weight, each gated by its 10-month average.
    k=None -> all markets (equal-weight global trend, Exp 21)."""

    def __init__(self, lev=1.0, k=4, gate=True):
        self.lev, self.k, self.gate = lev, k, gate

    def weights(self, prices):
        p = prices
        live = p.notna() & p.rolling(13).mean().notna()
        mom = (p.shift(1) / p.shift(12) - 1).where(live)
        on = (p > p.rolling(10).mean()) if self.gate else live
        if self.k:
            rank = mom.rank(axis=1, ascending=False)
            pick = rank <= self.k
            n = pd.Series(self.k, index=p.index)
        else:
            pick = live
            n = live.sum(axis=1)
        w = (pick & on & live).astype(float).div(n.replace(0, np.nan), axis=0).fillna(0.0)
        return w * self.lev


rules = {
    "13 markets equal weight, trend 1x (Exp 21)": CountryRotation(1.0, None),
    "top 4 by momentum, trend-gated 1x": CountryRotation(1.0, 4),
    "top 4 by momentum, no gate 1x": CountryRotation(1.0, 4, gate=False),
    "13 markets equal weight, trend 3x": CountryRotation(3.0, None),
    "top 4 by momentum, trend-gated 3x": CountryRotation(3.0, 4),
    "13 markets buy & hold": CountryRotation(1.0, None, gate=False),
}
eras = {"1961-1990": ("1961", "1990"), "1991-2026": ("1991", "2026"), "1961-2026": ("1961", "2026")}
rows = {}
for name, s in rules.items():
    res = at.run_backtest(px.ffill(), s, cost_bps=10, execution_lag=1, cash_rate=cash, borrow_spread=0.005)
    r = res.returns["1961":]
    row = {k: at.metrics.cagr(r[a:b], 12) for k, (a, b) in eras.items()}
    row["max DD"] = at.metrics.max_drawdown(r)
    row["turnover/yr"] = res.positions.diff().abs().sum(axis=1)["1961":].mean() * 12
    rows[name] = row
pd.set_option("display.width", 200)
out = pd.DataFrame(rows).T
fmt = {c: "{:.1%}".format for c in out.columns}
fmt["turnover/yr"] = "{:.1f}x".format
print(out.to_string(formatters=fmt))

# Robustness: after publication (2013-2026), and the number of markets held (k = 2..6).
print("\nAfter publication (2014-2026) and sensitivity to k (1x, trend-gated):")
rows = {}
for k in [None, 2, 3, 4, 5, 6]:
    res = at.run_backtest(px.ffill(), CountryRotation(1.0, k), cost_bps=10, execution_lag=1,
                          cash_rate=cash, borrow_spread=0.005)
    r = res.returns
    rows["all 13" if k is None else f"top {k}"] = {
        "1961-1990": at.metrics.cagr(r["1961":"1990"], 12), "1991-2013": at.metrics.cagr(r["1991":"2013"], 12),
        "2014-2026 (post-publication)": at.metrics.cagr(r["2014":], 12), "max DD": at.metrics.max_drawdown(r["1961":])}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1%}"))
