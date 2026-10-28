"""Experiment 22: how much leverage? The champion's structure scaled by a single dial, judged
across 13 markets (out of sample) and on US history.

Scale s multiplies every sleeve of the champion (s = 1 is the champion: 3x trend, valuation
0.5-3x, recovery+ 2x/3x with 2x bonds when out). Outside the US the valuation sleeve is
replaced by a neutral 1x trend sleeve (no earnings data), and recovery+ holds cash when out
(no bond data), as in Exp 20. With daily-reset ETFs no sleeve can lose more than it holds, so
monthly losses are capped at -100% per sleeve.

For the owner's objective (expected growth in a market we have not seen) the key column is
the growth of a $1 spread across the 13 markets' histories: median, average of log growth
(Kelly), and the worst.

    python scripts/experiments/exp22_leverage_frontier_13_markets.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from loop import CappedValuationLeverage, filtered  # noqa: E402

SCALES = [1 / 3, 0.5, 2 / 3, 5 / 6, 1.0]
CC = ["US", "DE", "GB", "FR", "IT", "CA", "AU", "NL", "CH", "SE", "ES", "KR", "JP"]
tb = at.data.load_fred("DTB3") / 100
tbm = tb.groupby(tb.index.to_period("M")).mean()
tbm.index = tbm.index.to_timestamp()


def sleeves_abroad():
    return [at.Levered(filtered(1.0), 3.0), filtered(1.0), at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either")]


def sleeves_us():
    return [at.Levered(filtered(1.0), 3.0), CappedValuationLeverage(1.0, 2.0, 0.5, 3.0),
            at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0)]


def mix_returns(sleeve_returns, s):
    """Each sleeve's return at scale s (exposure x s, financing scales too), capped at -100%."""
    return pd.concat([r.clip(lower=-1.0) for r in sleeve_returns], axis=1).mean(axis=1)


rows = []
for s in SCALES:
    growth, dd, ruined = {}, {}, 0
    for cc in CC:
        idx = at.data.load_fred(f"SPASTT01{cc}M661N")
        idx.index = idx.index.to_period("M").to_timestamp()
        tr = idx * (1 + 0.03 / 12) ** np.arange(len(idx))
        px = pd.DataFrame({"stocks": tr, "bonds": tr})
        cash = tbm.reindex(px.index).ffill().fillna(0.015)
        rs = [at.run_backtest(px, at.Levered(sl, s), cost_bps=10, execution_lag=1, cash_rate=cash,
                              borrow_spread=0.005).returns.iloc[12:] for sl in sleeves_abroad()]
        r = mix_returns(rs, s)
        w = (1 + r).cumprod()
        growth[cc] = w.iloc[-1] ** (12 / len(r)) - 1
        dd[cc] = float((w / w.cummax() - 1).min())
        ruined += int(w.min() <= 0.01)
    ctx = at.research.Context.default(0.005)
    rus = mix_returns([ctx.run(at.Levered(sl, s)).returns for sl in sleeves_us()], s)
    eras = [at.metrics.cagr(rus[a:b], 12) for a, b in [("1880", "1929"), ("1930", "1956"), ("1957", "1990"), ("1991", "2026")]]
    g = pd.Series(growth)
    rows.append({"scale": s, "avg stock exposure (US, 1957+)": None,
                 "13 mkts: median growth": g.median(), "13 mkts: mean log growth": float(np.log1p(g).mean()),
                 "13 mkts: worst growth": g.min(), "13 mkts: median max DD": pd.Series(dd).median(),
                 "13 mkts: worst max DD": min(dd.values()), "markets ruined (-99%)": ruined,
                 "US worst era": min(eras), "US 1991-2026": eras[3]})
out = pd.DataFrame(rows).drop(columns="avg stock exposure (US, 1957+)").set_index("scale")
out.index = [f"{s:.2f} x champion" for s in out.index]
pd.set_option("display.width", 220)
fmt = {c: "{:.1%}".format for c in out.columns if c != "markets ruined (-99%)"}
fmt["13 mkts: mean log growth"] = lambda v: f"{np.expm1(v):.1%}"
print(out.to_string(formatters=fmt))
