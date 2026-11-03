"""Experiment 38: the Exp 37 round-4 pick (2/3 champion + 1/3 satellite at 2x) on REAL ETF prices.

The satellite was built from index data (local-currency foreign stocks, corporate bonds from yields),
which Exp 35 showed flatters it. Here the satellite is rebuilt from real, dividend-adjusted ETFs
(SPY, EFA, IEF, LQD, GLD; Yahoo), US-dollar, margin at T-bill + 1.5%, 2006-2026:
  risk parity + trend 10% vol, time-series momentum 10% vol each, gold (GLD) 10-month trend;
  equal thirds, levered 2x. The champion part is the live-style champion (month-end, 2% band, CRSP).

    python scripts/experiments/exp38_real_check.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import algotrading as at  # noqa: E402
import exp31_combinations as e  # noqa: E402
from exp37_meta import champion_live  # noqa: E402

etf = at.data.etf_closes(["SPY", "EFA", "IEF", "LQD", "GLD"], rng="max")
mo = etf.groupby(etf.index.to_period("M")).last()
mo.index = mo.index.to_timestamp()
R = mo.pct_change()["2005":]
rf = e.RF.reindex(R.index).ffill()
lv = (1 + R.fillna(0)).cumprod()
on = lv > lv.rolling(10).mean()
vol = R.rolling(12).std() * np.sqrt(12)


def fin(W):
    W = W.shift(1).fillna(0)
    g = W.sum(axis=1)
    return (W * R.fillna(0)).sum(axis=1) + (1 - g) * rf - (g - 1).clip(lower=0) * 0.015 / 12 \
        - W.diff().abs().sum(axis=1).fillna(0) * 0.001


base = (1 / vol).div((1 / vol).sum(axis=1), axis=0)
pv = pd.Series([np.sqrt(np.nan_to_num(base.iloc[i].to_numpy()) @ np.nan_to_num(R.iloc[max(0, i - 11):i + 1].cov().to_numpy())
                        @ np.nan_to_num(base.iloc[i].to_numpy()) * 12) for i in range(len(R))], index=R.index)
rp = fin(base.mul((0.10 / pv).clip(upper=3.0), axis=0).where(on, 0.0).fillna(0))
ex12 = (1 + R).rolling(12).apply(np.prod, raw=True).sub((1 + rf).rolling(12).apply(np.prod, raw=True), axis=0)
w3 = ((ex12 > 0).astype(float) * (0.10 / vol) / 5).fillna(0)
ts = fin(w3.div((w3.sum(axis=1) / 3).clip(lower=1), axis=0))
gold = fin(pd.DataFrame({"GLD": on["GLD"].astype(float)}).reindex(columns=R.columns, fill_value=0.0))
sat = 2 * (rp + ts + gold) / 3 - (rf + 0.015 / 12)
ch = champion_live().reindex(R.index)
P = pd.DataFrame({"champion (live-style)": ch, "satellite 2x (real ETFs)": sat,
                  "2/3 champion + 1/3 satellite": 2 / 3 * ch + 1 / 3 * sat, "SPY": R["SPY"]})["2006":"2026-09"].dropna()
rows = {}
for k, x in P.items():
    w = (1 + x).cumprod()
    rows[k] = {"2006-2015": at.metrics.cagr(x[:"2015"], 12), "2016-2026": at.metrics.cagr(x["2016":], 12),
               "2006-2026": at.metrics.cagr(x, 12), "max DD": float((w / w.cummax() - 1).min()),
               "2022": float((1 + x["2022"]).prod() - 1)}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1%}"))
print(f"\ncorrelation champion vs satellite: {P.iloc[:, 0].corr(P.iloc[:, 1]):+.2f}")
