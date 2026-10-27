"""Demo: how parameter search manufactures fake edges, and how walk-forward exposes them.

    python scripts/overfitting_demo.py

On a pure random walk there is NO edge. Yet the best of ~100 parameter combinations
will look good in-sample. Walk-forward (optimise on the past, trade the unseen future)
shows the truth.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402

prices = at.data.gbm(n=252 * 12, mu=0.0, seed=42)["close"]
grid = {"fast": [5, 10, 15, 20, 30, 40, 50, 60, 80, 100],
        "slow": [50, 75, 100, 125, 150, 175, 200, 250, 300, 350]}
make = lambda fast, slow: at.SMACrossover(fast, slow, allow_short=True)  # noqa: E731

table = at.validation.grid_search(make, grid, prices, cost_bps=2.0)
print(f"Tried {len(table)} parameter sets on a random walk.")
print("Best in-sample:\n", table.head(3)[["fast", "slow", "sharpe", "cagr"]])

wf = at.validation.walk_forward(make, grid, prices, train_bars=756, test_bars=252, cost_bps=2.0)
oos = wf.attrs["oos_returns"]
print("\nWalk-forward windows:\n", wf)
print(f"\nIn-sample Sharpe of chosen params (avg): {wf['is_sharpe'].mean():.2f}")
print(f"Out-of-sample Sharpe (stitched):        {at.metrics.sharpe(oos):.2f}")

null = at.validation.sharpe_null_distribution(at.SMACrossover(20, 100, allow_short=True),
                                              n_paths=100)
print(f"\nNull Sharpe of one fixed SMA(20,100) on 100 random walks: "
      f"mean {null.mean():.2f}, 95th pct {np.percentile(null, 95):.2f}")
