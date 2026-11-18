"""Experiment 47: does the new champion's unemployment gate survive real-time data?

The gate (Exp 46) used today's revised UNRATE. Here it uses only what was published at the time:
the Philadelphia Fed Real-Time Data Set, quarterly vintages of the monthly unemployment rate
(RUC, vintages 1965Q4 onward, each dated the middle of its quarter). At each month-end the latest
vintage then available is used; its last observation is compared with the average of its last 12.
That is staler than live use (a new figure arrives every month), so it is a conservative test.
Before the first vintage the revised series is used. Variants counted: 0 (a check, not a candidate).

    python scripts/experiments/exp47_realtime_unemployment.py <path to rucqvmd.xlsx>
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
import loop as L  # noqa: E402

URL = ("https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/real-time-data/"
       "data-files/xlsx/rucqvmd.xlsx")
src = sys.argv[1] if len(sys.argv) > 1 else URL
raw = pd.read_excel(src)
raw.index = pd.to_datetime(raw["DATE"].str.replace(":", "-") + "-01")
raw = raw.drop(columns="DATE")


def vintage_date(col):                                  # RUC65Q4 -> 1965-11-15
    yy, q = int(col[3:5]), int(col[6])
    year = 1900 + yy if yy >= 47 else 2000 + yy
    return pd.Timestamp(year, 3 * q - 1, 15)


vint = {vintage_date(c): raw[c].dropna() for c in raw.columns}
dates = sorted(vint)
months = pd.date_range("1948-01-01", "2026-10-01", freq="MS")
revised = L.unemployment_rising()
rt = {}
for m in months:
    month_end = m + pd.offsets.MonthEnd(0)
    avail = [d for d in dates if d <= month_end]
    if not avail:
        rt[m] = revised.get(m, np.nan)
        continue
    u = vint[avail[-1]]
    rt[m] = bool(u.iloc[-1] > u.iloc[-12:].mean())
rt = pd.Series(rt).astype(bool)
# MacroGatedTrend lags its input one month (publication delay). The real-time series is already
# what was known at each month-end, so shift it back one month to keep the timing identical.
rt_input = rt.shift(-1).ffill().astype(bool)

ctx = at.research.Context.default(0.005)
ERAS = [(1880, 1929), (1930, 1956), (1957, 1990), (1991, 2026)]


def rets(x):
    return x.returns if hasattr(x, "returns") else x["returns"]


def era(r):
    return [(1 + r[str(a):str(b)]).prod() ** (12 / len(r[str(a):str(b)])) - 1 for a, b in ERAS]


old = rets(ctx.run(L.ensemble(3.0)))
new = rets(ctx.run(L.ensemble_unemployment()))
real = rets(ctx.run(L.ensemble_trend_sleeve(L.MacroGatedTrend(rt_input))))
agree = (rt["1966":] == revised.shift(1).reindex(rt["1966":].index).fillna(True)).mean()
print(f"gate agreement real-time vs revised (1966+, revised lagged a month): {agree:.0%}")
for name, r in [("old champion", old), ("new champion (revised data)", new), ("new champion (real-time data)", real)]:
    e = era(r)
    print(f"{name:32s} " + "  ".join(f"{v:5.1%}" for v in e) + f"   worst {min(e):.1%}   1966-2026 "
          f"{(1 + r['1966':]).prod() ** (12 / len(r['1966':])) - 1:.1%}")
print("real-time beats old champion everywhere:", at.research.beats_everywhere(real, old)["passes"])
