"""Experiment 44: how often should the champion's three sleeves be reset to equal thirds?

Diversification return (Willenbrock 2011, FAJ; Bouchey, Nemtchinov, Paulsen & Stein 2012,
'Volatility Harvesting'): rebalancing uncorrelated, volatile parts earns extra growth; letting the
winners drift concentrates risk. Round choices only: monthly (the champion), yearly (January), never.
Sleeve returns come from separate backtests (each with its own costs); the comparison uses the same
sleeve returns for every variant, so only the reset rule differs. Variants counted: 2.

    python scripts/experiments/exp44_sleeve_rebalancing.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from loop import CappedValuationLeverage, filtered  # noqa: E402

ERAS = [(1880, 1929), (1930, 1956), (1957, 1990), (1991, 2026)]
ctx = at.research.Context.default(0.005)
sleeves = {"trend": at.Levered(filtered(1.0), 3.0),
           "value": CappedValuationLeverage(1.0, 2.0, 0.5, 3.0),
           "recovery": at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0)}


def rets(res):
    return res.returns if hasattr(res, "returns") else res["returns"]


R = pd.DataFrame({k: rets(ctx.run(s)) for k, s in sleeves.items()}).dropna()
print("sleeve correlations:\n", R.corr().round(2))


def combine(R, reset):
    w = np.full(R.shape[1], 1 / R.shape[1])
    out = []
    for i, (t, row) in enumerate(R.iterrows()):
        if reset == "monthly" or (reset == "yearly" and t.month == 1):
            w = np.full(R.shape[1], 1 / R.shape[1])
        g = w * (1 + row.values)
        out.append(g.sum() / w.sum() - 1)
        w = g / g.sum() if g.sum() > 0 else w
    return pd.Series(out, index=R.index)


def era(r):
    return [(1 + r[str(a):str(b)]).prod() ** (12 / len(r[str(a):str(b)])) - 1 for a, b in ERAS]


def dd(r):
    e = (1 + r).cumprod()
    return (e / e.cummax() - 1).min()


res = {k: combine(R, k) for k in ["monthly", "yearly", "never"]}
rows = []
for k, r in res.items():
    e = era(r)
    rows.append({"reset": k, **{f"{a}-{b}": f"{v:.1%}" for (a, b), v in zip(ERAS, e)},
                 "worst era": f"{min(e):.1%}", "max DD": f"{dd(r):.0%}"})
print(pd.DataFrame(rows).to_string(index=False))
for k in ["yearly", "never"]:
    print(k, "beats monthly everywhere:", at.research.beats_everywhere(res[k], res["monthly"])["passes"])
