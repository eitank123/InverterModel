"""Experiment 21: a GLOBAL trend sleeve: 13 stock markets, each held only above its own
10-month average, equal weight, levered. Does diversification make leverage survivable?

Data: OECD monthly share-price indices (FRED SPASTT01<cc>M661N) for the US and the 12 markets
of Exp 20; +3% dividends; USD-hedged approximation (local price return + US T-bill cash);
markets enter when their data starts; leverage financed at cash + 0.5%; 10bp per unit traded.
Rules are fixed in advance (equal weight, 10-month average, 1x/2x/3x): nothing fitted.
Moskowitz, Ooi & Pedersen (2012) and Hurst et al. (2017) find trend works best across many
markets; Faber (2007) applies the 10-month rule per asset.

    python scripts/experiments/exp21_global_trend.py
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


class GlobalTrend(at.PortfolioStrategy):
    """Equal weight over markets with data (>= 10 months), each held while above its
    10-month average; total exposure ``lev``. ``filter_on=False`` = equal-weight buy & hold."""

    def __init__(self, lev=1.0, filter_on=True, markets=None):
        self.lev, self.filter_on, self.markets = lev, filter_on, markets

    def weights(self, prices):
        p = prices[self.markets] if self.markets else prices
        live = p.notna() & p.rolling(10).mean().notna()
        on = (p > p.rolling(10).mean()) if self.filter_on else live
        n = live.sum(axis=1).replace(0, np.nan)
        w = (on & live).astype(float).div(n, axis=0).fillna(0.0) * self.lev
        return w.reindex(columns=prices.columns, fill_value=0.0)


rules = {
    "US only: buy & hold": GlobalTrend(1.0, False, ["US"]),
    "US only: trend 3x": GlobalTrend(3.0, True, ["US"]),
    "13 markets: buy & hold": GlobalTrend(1.0, False),
    "13 markets: trend 1x": GlobalTrend(1.0, True),
    "13 markets: trend 2x": GlobalTrend(2.0, True),
    "13 markets: trend 3x": GlobalTrend(3.0, True),
}
eras = {"1961-1990": ("1961", "1990"), "1991-2026": ("1991", "2026"), "1961-2026": ("1961", "2026")}
rows = {}
for name, s in rules.items():
    res = at.run_backtest(px.ffill(), s, cost_bps=10, execution_lag=1, cash_rate=cash, borrow_spread=0.005)
    r = res.returns["1961":]
    row = {k: at.metrics.cagr(r[a:b], 12) for k, (a, b) in eras.items()}
    row["vol"] = at.metrics.annual_vol(r, 12)
    row["max DD"] = at.metrics.max_drawdown(r)
    row["worst year"] = ((1 + r).groupby(r.index.year).prod() - 1).min()
    row["avg exposure"] = res.positions["1961":].sum(axis=1).mean()
    rows[name] = row
pd.set_option("display.width", 200)
out = pd.DataFrame(rows).T
fmt = {c: "{:.1%}".format for c in out.columns}
fmt["avg exposure"] = "{:.2f}x".format
print(out.to_string(formatters=fmt))

# Part 2: swap the champion's US trend sleeve for the global one (1961-2026, the years both
# data sets cover). Sleeves are combined at the return level (1/3 each, rebalanced monthly).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loop import CappedValuationLeverage, filtered  # noqa: E402

ctx = at.research.Context.default(0.005)
sleeve = {
    "US trend 3x": ctx.run(at.Levered(filtered(1.0), 3.0)).returns,
    "valuation": ctx.run(CappedValuationLeverage(1.0, 2.0, 0.5, 3.0)).returns,
    "recovery+": ctx.run(at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0)).returns,
}
g3 = at.run_backtest(px.ffill(), GlobalTrend(3.0), cost_bps=10, execution_lag=1, cash_rate=cash,
                     borrow_spread=0.005).returns
g2 = at.run_backtest(px.ffill(), GlobalTrend(2.0), cost_bps=10, execution_lag=1, cash_rate=cash,
                     borrow_spread=0.005).returns
S = pd.DataFrame({**sleeve, "global trend 3x": g3, "global trend 2x": g2})["1961":"2026-09"].dropna()
mixes = {
    "champion (US trend 3x)": S[["US trend 3x", "valuation", "recovery+"]].mean(axis=1),
    "global trend 3x sleeve": S[["global trend 3x", "valuation", "recovery+"]].mean(axis=1),
    "global trend 2x sleeve": S[["global trend 2x", "valuation", "recovery+"]].mean(axis=1),
    "4 sleeves (US + global trend 3x)": S[["US trend 3x", "global trend 3x", "valuation", "recovery+"]].mean(axis=1),
}
rows = {}
for name, r in mixes.items():
    row = {k: at.metrics.cagr(r[a:b], 12) for k, (a, b) in eras.items()}
    row["max DD"] = at.metrics.max_drawdown(r)
    row["worst year"] = ((1 + r).groupby(r.index.year).prod() - 1).min()
    rows[name] = row
print("\nChampion with a global trend sleeve (1961-2026):")
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1%}"))
