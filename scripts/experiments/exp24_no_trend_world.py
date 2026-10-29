"""Experiment 24: model risk. What if markets stop trending?

Resample real 1957-2026 monthly (stocks, bonds, cash) returns jointly into synthetic 70-year
histories two ways:
  * i.i.d. (month by month): destroys trends, mean reversion and volatility clustering;
    keeps the return distribution and the stock/bond/cash link. Timing rules should lose here.
  * stationary block bootstrap (mean block 24 months): keeps most of that structure.
The champion's valuation sleeve sees the real CAPE history, which no longer lines up with the
reshuffled prices: in the i.i.d. world valuation carries no information either.
Compared with buy & hold at 1x/2x/3x (margin-financed, losses capped at -100% per month like a
daily-reset fund). 200 paths each; costs and financing as in the loop (10bp, cash + 0.5%).

    python scripts/experiments/exp24_no_trend_world.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from loop import ensemble, filtered, mix  # noqa: E402

ctx = at.research.Context.default(0.005)
hist = pd.DataFrame({"stocks": ctx.px["stocks"].pct_change(), "bonds": ctx.px["bonds"].pct_change(),
                     "cash": ctx.cash / 12})["1957":].dropna()
arr, dates = hist.to_numpy(), hist.index
STRATS = {"S&P 1x": mix(1.0), "S&P 2x": at.Levered(mix(1.0), 2.0), "S&P 3x": at.Levered(mix(1.0), 3.0),
          "trend 3x sleeve": at.Levered(filtered(1.0), 3.0), "champion ensemble": ensemble(3.0)}
N = 200
rng = np.random.default_rng(7)


def run_paths(kind):
    out = {k: [] for k in STRATS}
    for _ in range(N):
        idx = rng.integers(0, len(arr), len(arr)) if kind == "iid" else \
            at.stats.stationary_bootstrap(len(arr), 24, rng)[:len(arr)]
        sim = arr[idx]
        px = pd.DataFrame(np.cumprod(1 + sim[:, :2], axis=0), index=dates, columns=["stocks", "bonds"])
        cash = pd.Series(sim[:, 2] * 12, index=dates)
        for k, s in STRATS.items():
            r = at.run_backtest(px, s, cost_bps=10, execution_lag=1, cash_rate=cash,
                                borrow_spread=0.005).returns.clip(lower=-1.0).iloc[12:]
            w = float((1 + r).prod())
            out[k].append(max(w, 1e-12) ** (12 / len(r)) - 1)
    return pd.DataFrame(out)


rows = []
for kind in ("iid", "block"):
    g = run_paths(kind)
    for k in STRATS:
        rows.append({"world": "no structure (i.i.d.)" if kind == "iid" else "structure kept (blocks)",
                     "strategy": k, "median growth": g[k].median(), "5th pct": g[k].quantile(0.05),
                     "P(beats S&P 1x)": (g[k] > g["S&P 1x"]).mean(),
                     "P(beats S&P 3x)": (g[k] > g["S&P 3x"]).mean()})
res = pd.DataFrame(rows).set_index(["world", "strategy"])
pd.set_option("display.width", 200)
print(res.to_string(float_format=lambda v: f"{v:.1%}"))
