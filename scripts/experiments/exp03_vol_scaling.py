"""Experiment 3: is it the trend signal or the volatility scaling?

Kim, Tse & Wald (2016) argue most of time-series momentum's alpha in Moskowitz, Ooi &
Pedersen (2012) comes from scaling positions by 1/volatility, not from the trend sign.
We run a 2x2 design: {buy & hold, trend} x {unscaled, vol-scaled}, so each effect is
measured separately.

  A. Synthetic: GBM (constant vol) vs GARCH (clustered vol). Neither has a trend.
     Theory: vol scaling helps only if volatility is predictable (Moreira & Muir 2017).
  B. Real: S&P 500 monthly 1871-2026 and DJIA daily 2007-2016.

    python scripts/experiments/exp03_vol_scaling.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.3f}".format)


def grid(trend, vol_lookback, target):
    return {
        "buy_hold": at.BuyAndHold(),
        "buy_hold_volscaled": at.VolTarget(at.BuyAndHold(), target, vol_lookback, 2.0),
        "trend": trend,
        "trend_volscaled": at.VolTarget(trend, target, vol_lookback, 2.0),
    }


def run(prices, strats, **kw):
    out = {}
    for name, s in strats.items():
        r = at.run_backtest(prices, s, name=name, **kw)
        out[name] = {"sharpe": r.stats["sharpe"], "cagr": r.stats["cagr"],
                     "vol": r.stats["ann_vol"], "max_dd": r.stats["max_drawdown"]}
    t = pd.DataFrame(out).T
    sig = t["sharpe"]
    effects = {
        "vol_scaling_effect": ((sig["buy_hold_volscaled"] - sig["buy_hold"])
                               + (sig["trend_volscaled"] - sig["trend"])) / 2,
        "trend_signal_effect": ((sig["trend"] - sig["buy_hold"])
                                + (sig["trend_volscaled"] - sig["buy_hold_volscaled"])) / 2,
    }
    return t, effects


# --- A. Synthetic --------------------------------------------------------------------
print("A. Synthetic, no trends. Mean Sharpe over 20 seeds, 30 years daily, 2bp costs")
trend = at.TrendEnsemble((21, 63, 252), allow_short=False)
for gen_name, gen in [("GBM (constant vol)", at.data.gbm), ("GARCH (clustered vol)", at.data.garch)]:
    tabs, effs = [], []
    for seed in range(20):
        px = gen(n=252 * 30, mu=0.07, seed=seed)["close"]
        t, e = run(px, grid(trend, 21, 0.15), cost_bps=2)
        tabs.append(t["sharpe"])
        effs.append(e)
    print(f"\n-- {gen_name}")
    print(pd.concat(tabs, axis=1).mean(axis=1).to_frame("sharpe").T)
    print({k: round(v, 3) for k, v in pd.DataFrame(effs).mean().items()})

# --- B. Real data -------------------------------------------------------------------
df = at.data.load_shiller_sp500()
cash = (df["long_rate"] - 0.015).clip(lower=0)
print("\nB1. S&P 500 monthly 1871-2026 (lag 1, 10bp, cash = bond yield - 1.5%)")
t, e = run(df["tr"], grid(at.TrendEnsemble((3, 6, 12), allow_short=False), 12, 0.12),
           cost_bps=10, execution_lag=1, cash_rate=cash)
print(t); print({k: round(v, 3) for k, v in e.items()})

dj = at.data.load_djia_daily()["close"]
print("\nB2. DJIA daily 2007-2016 (price only, lag 1, 2bp, cash 0.5%)")
t, e = run(dj, grid(at.TrendEnsemble((21, 63, 252), allow_short=False), 21, 0.12),
           cost_bps=2, execution_lag=1, cash_rate=0.005)
print(t); print({k: round(v, 3) for k, v in e.items()})
print("   (2008 crash is in this sample: one decade, one crisis - treat as an anecdote)")
