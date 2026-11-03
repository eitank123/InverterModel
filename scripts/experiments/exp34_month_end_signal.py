"""Experiment 34: the live system reads the trend from MONTH-END closes, the backtests from Shiller's
monthly AVERAGE prices. How much does that difference cost?

Exp 33 found the real-time champion left the market in April 2025 and April 2026 (month-end close
below the 10-month average) and missed both rebounds, while the average-price backtest stayed in.
Here the champion's trend sleeve is rebuilt 1927-2026 on CRSP month-end levels and compared with the
same sleeve on Shiller averages; both earn the same CRSP returns. Fixes tested (published, round):
  * signal on month-end closes, 10-month average (what the live system does)
  * signal on the average of the month's daily closes (= what the backtest assumed, Faber 2007)
  * a 2% band around the 10-month average (Exp 13 / Exp 18 style hysteresis)
Variants counted: 3.

    python scripts/experiments/exp34_month_end_signal.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import algotrading as at  # noqa: E402
import exp31_combinations as e  # noqa: E402

lvl = (1 + e.MKT).cumprod()                                        # CRSP month-end level
d = (1 + e.MKTD).cumprod()
avg = d.groupby(d.index.to_period("M")).mean()                     # average of daily closes
avg.index = avg.index.to_timestamp()


def on_signal(level, band=0.0):
    sma = level.rolling(10).mean()
    out, state = [], True
    for p, s in zip(level, sma):
        if pd.isna(s):
            out.append(1.0)
            continue
        if state and p < s * (1 - band):
            state = False
        elif not state and p > s * (1 + band):
            state = True
        out.append(float(state))
    return pd.Series(out, index=level.index)


shiller_on = (e.SLEEVES["trend"]["stocks"] > 0).astype(float)
SIG = {"Shiller monthly averages (backtests so far)": shiller_on,
       "month-end closes (live system)": on_signal(lvl),
       "average of daily closes": on_signal(avg),
       "month-end closes with a 2% band": on_signal(lvl, 0.02)}
rows = {}
for k, on in SIG.items():
    W = pd.DataFrame({"stocks": 3.0 * on, "bonds": 0.0}).reindex(e.DATES).fillna(0)
    Wl = W.shift(1).fillna(0)
    r = (Wl["stocks"] * e.MKT.reindex(e.DATES) + (1 - Wl["stocks"]) * e.RF.reindex(e.DATES)
         - (Wl["stocks"] - 1).clip(lower=0) * 0.005 / 12 - Wl["stocks"].diff().abs().fillna(0) * 0.001)
    row = {"1927-1956": at.metrics.cagr(r[:"1956"], 12), "1957-1990": at.metrics.cagr(r["1957":"1990"], 12),
           "1991-2026": at.metrics.cagr(r["1991":], 12)}
    row["worst era"] = min(row.values())
    row["switches/yr"] = on.reindex(e.DATES).diff().abs().sum() / (len(e.DATES) / 12)
    row["2025-26"] = at.metrics.cagr(r["2025":], 12)
    rows[k] = row
pd.set_option("display.width", 200)
print("Trend sleeve at 3x (S&P above its 10-month average), CRSP returns, all costs:\n")
t = pd.DataFrame(rows).T
print(t.to_string(formatters={c: ("{:.2f}".format if c == "switches/yr" else "{:.1%}".format) for c in t.columns}))

# The full champion with its signals read from month-end closes (as live), without and with the band.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loop import ensemble  # noqa: E402

bonds = e.idx["bonds"].copy()
bonds.index = bonds.index.to_period("M").to_timestamp()
idx_me = pd.DataFrame({"stocks": lvl, "bonds": bonds.reindex(lvl.index).ffill()}).dropna()
rows = {}
for lab, strat in [("champion, Shiller averages (backtests so far)", None),
                   ("champion, month-end closes (live)", ensemble(3.0)),
                   ("champion, month-end closes + 2% band", ensemble(3.0, band=0.02))]:
    if strat is None:
        r = e.portfolio(e.MKT)
    else:
        W = strat.weights(idx_me).shift(1).reindex(e.DATES).fillna(0)
        g = W["stocks"] + W["bonds"]
        r = (W["stocks"] * e.MKT.reindex(e.DATES) + W["bonds"] * e.RB.reindex(e.DATES) + (1 - g) * e.RF.reindex(e.DATES)
             - (g - 1).clip(lower=0) * 0.005 / 12 - W.diff().abs().sum(axis=1).fillna(0) * 0.001)
    row = {"1927-1956": at.metrics.cagr(r[:"1956"], 12), "1957-1990": at.metrics.cagr(r["1957":"1990"], 12),
           "1991-2026": at.metrics.cagr(r["1991":], 12)}
    row["worst era"] = min(row.values())
    row["2025-26"] = at.metrics.cagr(r["2025":], 12)
    w = (1 + r["1940":]).cumprod()
    row["max DD 1940+"] = float((w / w.cummax() - 1).min())
    rows[lab] = row
print("\nFull champion, CRSP returns, all costs:\n")
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1%}"))
