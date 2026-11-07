"""Experiment 39: rebalance timing luck. Does splitting the monthly trend decision into 4 weekly
tranches beat deciding on one day a month?

Newfound Research (Hoffstein, 'rebalance timing luck'): a monthly rule that trades on one fixed day
makes results depend on that day's luck; running 4 copies offset by a week and averaging them
('tranching') diversifies it. Exp 34 found the live system's month-end reading whipsawed in April 2025
and April 2026. Test on DAILY CRSP market returns 1927-2026 (Ken French), the champion's trend sleeve:
3x the market while price > its 10-month average (10 observations, one per month, taken on the
tranche's own trading day), cash otherwise; leverage at T-bill + 0.5%, 10bp per unit traded.
  A one decision a month on the last trading day (what the live system does)
  B 4 tranches: trading days 0/5/10/15 before month-end, each 1/4 of the account
  C A with the 2% band (Exp 34)      D B with the 2% band
Variants counted: 4.

    python scripts/experiments/exp39_tranching.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

ff = at.data.load_ff_factors(daily=True)
r = (ff["Mkt-RF"] + ff["RF"])
rf = ff["RF"]
lvl = (1 + r).cumprod()
month = r.index.to_period("M")
pos_from_end = r.groupby(month).cumcount(ascending=False)          # 0 = last trading day


def signal(offset, band):
    """Decision days: the trading day `offset` days before each month-end. Signal: price on that day
    vs the average of the last 10 such observations (with optional hysteresis band). Held until the
    next decision day of the same tranche."""
    dd = lvl[pos_from_end == offset]
    sma = dd.rolling(10).mean()
    state, out = 1.0, []
    for p, s in zip(dd, sma):
        if not np.isnan(s):
            if state and p < s * (1 - band):
                state = 0.0
            elif not state and p > s * (1 + band):
                state = 1.0
        out.append(state)
    sig = pd.Series(out, index=dd.index)
    return sig.reindex(r.index).ffill().shift(1).fillna(1.0)       # trade at the next close


def run(offsets, band, lev=3.0):
    on = sum(signal(o, band) for o in offsets) / len(offsets)
    L = lev * on
    net = L * r - (L - 1).clip(lower=0) * (rf + 0.005 / 252) + (1 - L).clip(lower=0) * rf \
        - L.diff().abs().fillna(0) * 0.001
    return net["1928":]


rows = {}
for lab, offs, band in [("A one day (month-end)", [0], 0.0), ("B 4 weekly tranches", [0, 5, 10, 15], 0.0),
                        ("C one day + 2% band", [0], 0.02), ("D 4 tranches + 2% band", [0, 5, 10, 15], 0.02)]:
    x = run(offs, band)
    row = {e: at.metrics.cagr(x[a:b], 252) for e, (a, b) in
           {"1928-1956": ("1928", "1956"), "1957-1990": ("1957", "1990"), "1991-2026": ("1991", "2026")}.items()}
    row["worst era"] = min(row.values())
    row["2025-26"] = at.metrics.cagr(x["2025":], 252)
    w = (1 + x["1940":]).cumprod()
    row["max DD 1940+"] = float((w / w.cummax() - 1).min())
    rows[lab] = row
# Timing luck itself: the spread of results across the 4 single-day offsets.
singles = {o: run([o], 0.0) for o in (0, 5, 10, 15)}
spread = pd.DataFrame({o: {e: at.metrics.cagr(x[a:b], 252) for e, (a, b) in
                           {"1957-1990": ("1957", "1990"), "1991-2026": ("1991", "2026")}.items()}
                       for o, x in singles.items()}).T
pd.set_option("display.width", 200)
print("Trend sleeve at 3x on daily CRSP returns, all costs:\n")
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1%}"))
print("\nTiming luck - the same rule decided on different single days (days before month-end):\n")
print(spread.to_string(float_format=lambda v: f"{v:.1%}"))
