"""Experiment 52: the new champion's unemployment gate on 12 other markets (out of sample).

Setup as Exp 20 (OECD monthly share-price indices from FRED, +3%/yr dividends, US T-bill cash,
10bp costs, cash +0.5% financing). Gate: each country's own OECD unemployment rate
(FRED LRHUTTTT<cc>M156S harmonised, else LRUNTTTT; Switzerland has none) above its 12-month average, lagged a month;
where the series has no data yet the trend exit is always obeyed. Same rule, no re-fitting.
Champion-like mix = 3x trend sleeve + 1x trend-filtered stand-in for valuation + recovery+.
Variants counted: 0 (a check of the champion).

    python scripts/experiments/exp52_twelve_markets_unemployment.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from loop import MacroGatedTrend, _monthly, filtered  # noqa: E402

COUNTRIES = {"DE": "Germany", "GB": "UK", "FR": "France", "IT": "Italy", "CA": "Canada",
             "AU": "Australia", "NL": "Netherlands", "CH": "Switzerland", "SE": "Sweden",
             "ES": "Spain", "KR": "Korea", "JP": "Japan"}
tb = at.data.load_fred("DTB3") / 100
tbm = tb.groupby(tb.index.to_period("M")).mean()
tbm.index = tbm.index.to_timestamp()
rec = at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either")
rows = []
for cc, name in COUNTRIES.items():
    s = at.data.load_fred(f"SPASTT01{cc}M661N")
    s.index = s.index.to_period("M").to_timestamp()
    tr = s * (1 + 0.03 / 12) ** np.arange(len(s))
    px = pd.DataFrame({"stocks": tr, "bonds": tr})
    cash = tbm.reindex(px.index).ffill().fillna(0.015)
    u = pd.Series(dtype=float)
    for series in (f"LRHUTTTT{cc}M156S", f"LRUNTTTT{cc}M156S"):       # harmonised first
        try:
            u = _monthly(series)
            break
        except Exception:
            continue
    weak = (u > u.rolling(12).mean()) if len(u) else pd.Series(True, index=px.index)
    start = max(px.index[12], u.index[12]) if len(u) > 12 else px.index[12]   # test only where the gate exists
    strats = {"trend 3x old": at.Levered(filtered(1.0), 3.0), "trend 3x gated": MacroGatedTrend(weak),
              "mix old": at.Blend({at.Levered(filtered(1.0), 3.0): 1 / 3, filtered(1.0): 1 / 3, rec: 1 / 3}),
              "mix gated": at.Blend({MacroGatedTrend(weak): 1 / 3, filtered(1.0): 1 / 3, rec: 1 / 3})}
    row = {"country": name, "from": f"{start:%Y}"}
    for k, st in strats.items():
        r = at.run_backtest(px, st, cost_bps=10, execution_lag=1, cash_rate=cash, borrow_spread=0.005).returns[start:]
        row[f"{k} growth"] = at.metrics.cagr(r, 12)
        row[f"{k} DD"] = at.metrics.max_drawdown(r)
    rows.append(row)
T = pd.DataFrame(rows).set_index("country")
pd.set_option("display.width", 250)
print(T.to_string(float_format=lambda v: f"{v:.1%}"))
g = [c for c in T if c.endswith("growth")]
d = [c for c in T if c.endswith("DD")]
print("\nmedian growth:", T[g].median().map("{:.1%}".format).to_dict())
print("worst growth:", T[g].min().map("{:.1%}".format).to_dict())
print("worst DD:", T[d].min().map("{:.0%}".format).to_dict())
print("countries with DD below -90%:", (T[d] < -0.90).sum().to_dict())
T = T.drop(index=[c for c in T.index if c == "Switzerland"])        # no unemployment data: gate inactive
print("gated mix beats old mix in", int((T["mix gated growth"] > T["mix old growth"]).sum()), "of", len(T))
