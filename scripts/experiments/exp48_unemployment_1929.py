"""Experiment 48: the unemployment gate before 1948 (NBER Macrohistory series 08292).

The new champion (Exp 46) gates its trend sleeve with UNRATE, which starts in 1948. NBER's
Macrohistory database has a monthly, seasonally adjusted unemployment rate for 04/1929-06/1942
(National Industrial Conference Board, m08292a) and 1940-1946 (m08292b). Spliced with FRED UNRATE
(1948+; 1947 is missing and carried forward), the same rule - unemployment above its 12-month
average - can be tested on the Depression, the 1937 crash and WWII. Each source's signal uses only
its own 12 months, so splice levels do not matter. Variants counted: 0 (a check of the champion).

    python scripts/experiments/exp48_unemployment_1929.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
import loop as L  # noqa: E402

NBER = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nber")


def nber(name):
    rows = []
    for line in open(os.path.join(NBER, f"{name}.dat")):
        y, m, v = line.split()[:3]
        if v != ".":
            rows.append((pd.Timestamp(int(y), int(m), 1), float(v)))
    return pd.Series(dict(rows))


def rising(u):
    return u > u.rolling(12).mean()


a, b = nber("m08292a"), nber("m08292b")
gate = pd.concat([rising(a)[:"1939-12"].iloc[11:], rising(b)[b.index[11]:],
                  L.unemployment_rising()["1949":]])
gate = gate[~gate.index.duplicated(keep="last")].sort_index()
gate = gate.reindex(pd.date_range(gate.index[0], gate.index[-1], freq="MS")).ffill().astype(bool)
print(f"gate from {gate.index[0]:%Y-%m}; weak {gate[:'1947'].mean():.0%} of months 1930-1947, "
      f"{gate['1948':].mean():.0%} after")

ctx = at.research.Context.default(0.005)
ERAS = [(1880, 1929), (1930, 1956), (1930, 1947), (1957, 1990), (1991, 2026)]


def rets(x):
    return x.returns if hasattr(x, "returns") else x["returns"]


def era(r):
    return [(1 + r[str(a_):str(b_)]).prod() ** (12 / len(r[str(a_):str(b_)])) - 1 for a_, b_ in ERAS]


def dd(r, a_, b_):
    e = (1 + r[a_:b_]).cumprod()
    return (e / e.cummax() - 1).min()


rows = {"old champion": rets(ctx.run(L.ensemble(3.0))),
        "new champion (gate from 1948)": rets(ctx.run(L.ensemble_unemployment())),
        "new champion (gate from 1930, NBER)": rets(ctx.run(L.ensemble_trend_sleeve(L.MacroGatedTrend(gate))))}
print("eras: " + ", ".join(f"{a_}-{b_}" for a_, b_ in ERAS))
for k, r in rows.items():
    print(f"{k:38s} " + "  ".join(f"{v:6.1%}" for v in era(r)) + f"   max DD 1930-1947 {dd(r, '1930', '1947'):.0%}")
print("NBER-gated beats old champion everywhere:",
      at.research.beats_everywhere(rows["new champion (gate from 1930, NBER)"], rows["old champion"])["passes"])
