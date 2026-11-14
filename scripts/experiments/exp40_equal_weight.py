"""Experiment 40: equal-weight instead of cap-weight for the champion's stock part.

Plyakha, Uppal & Vilkov (2012): an equal-weighted S&P portfolio beat the value-weighted one in mean
return, alpha and Sharpe, through size/value tilts and the contrarian effect of rebalancing; the S&P
500 Equal Weight index (fund RSP) is the deployable version. The index's current concentration in a
few mega-caps (Nov 2026 news on Nvidia) is the motivation for checking it now.
Proxies on CRSP 1927-2026 (Ken French, portfolios formed on size, value-weighted within each):
  EW-large  equal weight across the 5 largest size deciles (6-10): roughly the S&P 500 universe with
            equal weight per size bucket, like RSP's tilt toward the smaller half of the index
  EW-all    equal weight across all 10 size deciles (includes small and micro caps)
plus 0.20%/yr extra cost for the monthly rebalancing (RSP's fee 0.20% vs SPY 0.09%).
Each is used (a) on its own at 1x and (b) as the champion's stock leg (Exp 31 method).
Real check: SPY vs RSP total return, 2004-2026 (Yahoo). Variants counted: 4.

    python scripts/experiments/exp40_equal_weight.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import algotrading as at  # noqa: E402
import exp31_combinations as e  # noqa: E402

ME = at.data._kf_block("Portfolios_Formed_on_ME", "Average Value Weight Returns -- Monthly")
dec = ["Lo 10", "2-Dec", "3-Dec", "4-Dec", "5-Dec", "6-Dec", "7-Dec", "8-Dec", "9-Dec", "Hi 10"]
EW_LARGE = ME[dec[5:]].mean(axis=1) - 0.002 / 12
EW_ALL = ME[dec].mean(axis=1) - 0.002 / 12
LEGS = {"cap-weight market (current)": e.MKT, "EW-large (RSP-like)": EW_LARGE, "EW-all deciles": EW_ALL}


def stats(r):
    eras = {"1927-1956": r[:"1956"], "1957-1990": r["1957":"1990"], "1991-2026": r["1991":]}
    row = {k: at.metrics.cagr(v, 12) for k, v in eras.items()}
    row["worst era"] = min(row.values())
    row["1927-2026"] = at.metrics.cagr(r, 12)
    w = (1 + r["1940":]).cumprod()
    row["max DD 1940+"] = float((w / w.cummax() - 1).min())
    return row


rows = {}
for k, leg in LEGS.items():
    rows[f"{k}, 1x"] = stats(leg.reindex(e.DATES))
    rows[f"champion with {k} leg"] = stats(e.portfolio(leg))
pd.set_option("display.width", 220)
print("CRSP 1927-2026, all costs:\n")
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1%}"))

etf = at.data.etf_closes(["SPY", "RSP"], rng="max")
m = etf.groupby(etf.index.to_period("M")).last().pct_change()["2004":]
print("\nReal funds, total return (Yahoo):")
for a, b in [("2004", "2015"), ("2016", "2026"), ("2004", "2026")]:
    print(f"  {a}-{b}: SPY {at.metrics.cagr(m['SPY'][a:b], 12):.1%}  RSP {at.metrics.cagr(m['RSP'][a:b], 12):.1%}")
