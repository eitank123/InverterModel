"""Experiment 19: does the trend rule work on a market it was never fitted to? (Japan, 1950-2026)

Every rule in the loop was designed on US data. The Nikkei 225 (FRED, daily since 1949) is a
true out-of-sample market: a boom to 1989, then a 20-year bust (-80%). Monthly month-end
closes, dividends added at 1.5%/yr, cash = US 3-month T-bill (1.5% before 1954; a USD-hedged
investor earns the local excess return plus USD cash, so this is an approximation).
Same rules and parameters as the US champion's sleeves, nothing re-fitted:
  buy & hold 1x / 3x; 10-month average 1x / 2x / 3x; recovery leverage (2x, 3x after >15%
  falls, 'either' signal); valuation is not testable (no Japanese earnings data).
Costs 10bp per unit traded, leverage financed at cash + 0.5%.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from loop import filtered, mix  # noqa: E402

nk = at.data.load_fred("NIKKEI225")
tb = at.data.load_fred("DTB3") / 100
me = nk.groupby(nk.index.to_period("M")).last()
me.index = me.index.to_timestamp()
tr = me * (1 + 0.015 / 12) ** pd.Series(range(len(me)), index=me.index)   # + dividends
px = pd.DataFrame({"stocks": tr, "bonds": tr})          # no Japanese bond series: bonds unused
cash = tb.groupby(tb.index.to_period("M")).last()
cash.index = cash.index.to_timestamp()
cash = cash.reindex(px.index).ffill().fillna(0.015)

rules = {
    "buy & hold 1x": mix(1.0),
    "buy & hold 3x": at.Levered(mix(1.0), 3.0),
    "10-month average 1x": filtered(1.0),
    "10-month average 2x": at.Levered(filtered(1.0), 2.0),
    "10-month average 3x": at.Levered(filtered(1.0), 3.0),
    "recovery+ (cash when out)": at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either"),
}
eras = {"1950-1989 (boom)": ("1950", "1989"), "1990-2012 (bust)": ("1990", "2012"),
        "2013-2026": ("2013", "2026"), "1950-2026": ("1950", "2026")}
rows = {}
for name, s in rules.items():
    r = at.run_backtest(px, s, cost_bps=10, execution_lag=1, cash_rate=cash, borrow_spread=0.005).returns
    row = {k: at.metrics.cagr(r[a:b], 12) for k, (a, b) in eras.items()}
    row["max DD"] = at.metrics.max_drawdown(r["1950":])
    row["worst year"] = ((1 + r["1950":]).groupby(r["1950":].index.year).prod() - 1).min()
    rows[name] = row
res = pd.DataFrame(rows).T
pd.set_option("display.width", 200)
print(f"Nikkei 225 month-ends {me.index[0]:%Y-%m} to {me.index[-1]:%Y-%m}\n")
print(res.to_string(float_format=lambda v: f"{v:.1%}"))

# Part 2: the champion's structure in Japan. No Japanese earnings, so the valuation sleeve is
# replaced by its neutral setting (1x, trend-filtered): an approximation, stated as such.
print("\nEnsemble-like mixes (valuation sleeve stood in by 1x trend-filtered):")
val_standin = filtered(1.0)
rec = at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either")
mixes = {f"trend {t:g}x + 1x stand-in + recovery+ (L {lev:g})":
         at.Blend({at.Levered(filtered(1.0), t): 1 / 3, val_standin: 1 / 3,
                   at.RecoveryLeverage(lev, lev * 1.5, 0.15, signal="either"): 1 / 3})
         for t, lev in [(3.0, 2.0), (2.0, 2.0), (2.0, 1.0), (1.0, 1.0)]}
rows = {}
for name, s in mixes.items():
    r = at.run_backtest(px, s, cost_bps=10, execution_lag=1, cash_rate=cash, borrow_spread=0.005).returns
    row = {k: at.metrics.cagr(r[a:b], 12) for k, (a, b) in eras.items()}
    row["max DD"] = at.metrics.max_drawdown(r["1950":])
    rows[name] = row
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1%}"))
