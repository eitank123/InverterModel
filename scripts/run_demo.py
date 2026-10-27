"""Demo: run the built-in strategies on three synthetic markets with known structure.

    python scripts/run_demo.py

Expected lessons:
  * Random walk (GBM): nothing beats buy & hold in a meaningful way; costs only hurt.
  * Mean-reverting (OU): the z-score strategy shines, trend strategies bleed.
  * Regime switching: trend following / momentum capture the persistent trends.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402

pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.3f}".format)

MARKETS = {
    "random_walk": at.data.gbm(seed=1),
    "mean_reverting": at.data.ornstein_uhlenbeck(theta=20, sigma=0.25, seed=2),
    "regime_switching": at.data.regime_switching(seed=3),
}
STRATEGIES = {
    "buy_hold": at.BuyAndHold(),
    "sma_20_100": at.SMACrossover(20, 100, allow_short=True),
    "momentum_60": at.Momentum(60),
    "zscore_mr_20": at.ZScoreMeanReversion(20, 1.5, 0.5),
    "momentum_60_vt": at.VolTarget(at.Momentum(60), target_vol=0.15),
}

os.makedirs("results", exist_ok=True)
for name, df in MARKETS.items():
    results = [at.run_backtest(df, s, cost_bps=2.0, name=n) for n, s in STRATEGIES.items()]
    print(f"\n=== {name} ===")
    print(at.compare(results)[["cagr", "ann_vol", "sharpe", "max_drawdown", "ann_turnover"]])
    at.plot(results, path=f"results/{name}.png", title=name)
print("\nPlots saved to results/")
