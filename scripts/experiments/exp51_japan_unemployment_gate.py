"""Experiment 51: the new champion's unemployment gate on a market it was never fitted to (Japan).

Champion candidates must report their sleeves on Japan 1950-2026 (loop/README.md, since checkpoint
10). Setup as Exp 19 (Nikkei 225 month-ends, +1.5%/yr dividends, US T-bill cash, 10bp costs, cash
+0.5% financing). Gate: Japan's own unemployment rate (OECD, FRED LRUNTTTTJPM156S, 1955+) above its
12-month average, lagged a month; before 1955 the trend exit is always obeyed. Same rule and
parameters as the US champion. The valuation sleeve is stood in by 1x trend-filtered (no Japanese
earnings), as in Exp 19. Variants counted: 0 (a check of the champion).

    python scripts/experiments/exp51_japan_unemployment_gate.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from loop import MacroGatedTrend, filtered, _monthly  # noqa: E402

nk = at.data.load_fred("NIKKEI225")
tb = at.data.load_fred("DTB3") / 100
me = nk.groupby(nk.index.to_period("M")).last()
me.index = me.index.to_timestamp()
tr = me * (1 + 0.015 / 12) ** pd.Series(range(len(me)), index=me.index)
px = pd.DataFrame({"stocks": tr, "bonds": tr})
cash = tb.groupby(tb.index.to_period("M")).last()
cash.index = cash.index.to_timestamp()
cash = cash.reindex(px.index).ffill().fillna(0.015)

u = _monthly("LRUNTTTTJPM156S")
weak = u > u.rolling(12).mean()
print(f"Japan unemployment {u.index[0]:%Y-%m} to {u.index[-1]:%Y-%m}; rising in {weak.mean():.0%} of months")

rec = at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either")
rules = {
    "trend 3x (old champion's sleeve)": at.Levered(filtered(1.0), 3.0),
    "trend 3x gated by unemployment (new)": MacroGatedTrend(weak),
    "ensemble-like, old": at.Blend({at.Levered(filtered(1.0), 3.0): 1 / 3, filtered(1.0): 1 / 3, rec: 1 / 3}),
    "ensemble-like, new": at.Blend({MacroGatedTrend(weak): 1 / 3, filtered(1.0): 1 / 3, rec: 1 / 3}),
}
eras = {"1956-1989 (boom)": ("1956", "1989"), "1990-2012 (bust)": ("1990", "2012"),
        "2013-2026": ("2013", "2026"), "1956-2026": ("1956", "2026")}
rows = {}
for name, s in rules.items():
    r = at.run_backtest(px, s, cost_bps=10, execution_lag=1, cash_rate=cash, borrow_spread=0.005).returns
    row = {k: at.metrics.cagr(r[a:b], 12) for k, (a, b) in eras.items()}
    row["max DD"] = at.metrics.max_drawdown(r["1956":])
    row["worst year"] = ((1 + r["1956":]).groupby(r["1956":].index.year).prod() - 1).min()
    rows[name] = row
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1%}"))
