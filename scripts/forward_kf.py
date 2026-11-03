"""Delayed forward test of the Exp 31 candidates on CRSP data (Ken French library).

The diversified stock leg (industry momentum + stock momentum + value) has no free real-time price
feed here, so it cannot be a daily play-money account yet. Ken French publishes the month's returns
about a month later. This ledger records, for every NEW month after 2026-09 (the last month the
research saw), the monthly return of each candidate, and never rewrites a recorded month.

    python scripts/forward_kf.py        (run by the loop every round; adds nothing until new data)
"""
import json
import os
import shutil
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import algotrading as at  # noqa: E402

PATH = os.path.join(ROOT, "loop", "forward_kf.json")
LAST_SEEN = "2026-09-01"

if os.path.isdir(at.data.KF_DIR):                       # refresh the cache weekly: new months
    import time
    if time.time() - os.path.getmtime(at.data.KF_DIR) > 7 * 86400:
        shutil.rmtree(at.data.KF_DIR)
import exp31_combinations as e  # noqa: E402

D = e.LEGS["diversified (ind mom + stock mom + value)"]
CANDS = {"current champion (market leg)": e.portfolio(e.MKT),
         "champion x2/3 (market leg)": e.portfolio(e.MKT, scale=0.67),
         "diversified leg x2/3 (max 2x, long only)": e.portfolio(D, scale=0.67),
         "diversified leg + factor sleeve x2/3": e.portfolio(D, extra=e.factor_sleeve() * 0.67, scale=0.67),
         "market buy & hold": e.MKT.reindex(e.DATES)}
state = json.load(open(PATH)) if os.path.exists(PATH) else {"start_after": LAST_SEEN, "months": {}}
new = 0
for name, r in CANDS.items():
    rec = state["months"].setdefault(name, {})
    for d, v in r[r.index > pd.Timestamp(LAST_SEEN)].dropna().items():
        k = d.strftime("%Y-%m")
        if k not in rec:
            rec[k] = round(float(v), 6)
            new += 1
os.makedirs(os.path.dirname(PATH), exist_ok=True)
json.dump(state, open(PATH, "w"), indent=1)
print(f"Ken French data through {e.MKT.index[-1]:%Y-%m}; {new} new monthly records")
for name, rec in state["months"].items():
    tot = 1.0
    for v in rec.values():
        tot *= 1 + v
    print(f"  {name:45s} {len(rec)} months, {tot - 1:+.2%}")
