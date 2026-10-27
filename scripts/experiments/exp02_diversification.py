"""Experiment 2: why professional trend followers trade *many* markets with *many*
lookbacks, and why turnover matters. Synthetic data, so the ground truth is known.

  A. Breadth: Sharpe of a trend portfolio vs number of independent markets N.
     Theory ("fundamental law of active management", Grinold 1989):
     SR_portfolio ~ SR_single * sqrt(N) for independent bets.
  B. Parameter risk: single lookback vs an ensemble when we don't know how long trends last.
  C. Costs: how fast does each variant die as trading gets more expensive?

    python scripts/experiments/exp02_diversification.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.2f}".format)
OUT = "results/exp02"
os.makedirs(OUT, exist_ok=True)
SEEDS = range(5)


def trend(lookbacks=(21, 63, 252)):
    """Per-asset: trend signal, scaled to 10% vol per asset."""
    return at.VolTarget(at.TrendEnsemble(lookbacks), target_vol=0.10, lookback=60)


# --- A. Breadth ----------------------------------------------------------------------
print("A. Sharpe vs number of independent markets (mean over seeds, 2 bp costs)")
rows = []
for n in (1, 2, 5, 10, 20):
    srs = []
    for seed in SEEDS:
        px = at.data.panel(at.data.regime_switching, n_assets=n, seed=seed, n=2520)
        port = at.PortfolioVolTarget(at.PerAsset(trend()), target_vol=0.10)
        srs.append(at.run_backtest(px, port, cost_bps=2).stats["sharpe"])
    rows.append({"N": n, "sharpe": np.mean(srs)})
breadth = pd.DataFrame(rows).set_index("N")
breadth["theory_sqrtN"] = breadth.loc[1, "sharpe"] * np.sqrt(breadth.index)
print(breadth)

# --- B. Parameter risk ---------------------------------------------------------------
print("\nB. Which lookback works depends on how persistent trends are (10 assets, Sharpe)")
variants = {"lb_21": (21,), "lb_63": (63,), "lb_252": (252,), "ensemble": (21, 63, 252)}
rows = []
for p_stay in (0.98, 0.99, 0.997, 0.999):
    row = {"p_stay": str(p_stay), "avg_trend_days": round(1 / (1 - p_stay))}
    for name, lbs in variants.items():
        srs = []
        for seed in SEEDS:
            px = at.data.panel(at.data.regime_switching, n_assets=10, seed=seed, n=2520,
                               p_stay=p_stay)
            srs.append(at.run_backtest(px, at.PerAsset(trend(lbs)), cost_bps=2).stats["sharpe"])
        row[name] = np.mean(srs)
    rows.append(row)
param = pd.DataFrame(rows).set_index("p_stay")
print(param)
ranks = param[list(variants)].rank(axis=1, ascending=False)
print("rank of each variant per regime (1 = best):\n", ranks)

# --- C. Costs ------------------------------------------------------------------------
print("\nC. Sharpe vs one-way cost (10 assets, p_stay=0.997)")
pxs = [at.data.panel(at.data.regime_switching, n_assets=10, seed=s, n=2520) for s in SEEDS]
rows = []
for cost in (0, 2, 5, 10, 20, 50):
    row = {"cost_bps": cost}
    for name, lbs in variants.items():
        res = [at.run_backtest(px, at.PerAsset(trend(lbs)), cost_bps=cost) for px in pxs]
        row[name] = np.mean([r.stats["sharpe"] for r in res])
        if cost == 0:
            row[f"turnover_{name}"] = np.mean([r.stats["ann_turnover"] for r in res])
    rows.append(row)
costs = pd.DataFrame(rows).set_index("cost_bps")
print(costs[list(variants)])
print("annual turnover (x capital):", costs.filter(like="turnover").iloc[0].round(1).to_dict())

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(breadth.index, breadth["sharpe"], "o-", label="simulated")
axes[0].plot(breadth.index, breadth["theory_sqrtN"], "--", label="SR1 * sqrt(N)")
axes[0].set_xlabel("number of independent markets"); axes[0].set_ylabel("Sharpe")
axes[0].set_title("Breadth"); axes[0].legend(); axes[0].grid(alpha=.3)
for name in variants:
    axes[1].plot(costs.index, costs[name], "o-", label=name)
axes[1].set_xlabel("one-way cost (bp)"); axes[1].set_ylabel("Sharpe")
axes[1].set_title("Cost sensitivity"); axes[1].legend(); axes[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/breadth_costs.png", dpi=110)
print(f"\nPlot in {OUT}/")
