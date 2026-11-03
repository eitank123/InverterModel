"""Experiment 26: turn-of-the-month leverage overlay, tested on data the loop has not used.

Lakonishok & Smidt (1988) and McConnell & Xu (2008, 'Equity returns at the turn of the month')
find that US stock returns concentrate in the last trading day of a month plus the first three
trading days of the next one; McConnell & Xu report it across 1926-2005 and 30+ countries.
Exp 17 looked at it on Dow data 2007-2016. Here the window is fixed in advance at the published
-1..+3 and tested out of sample on FRED's S&P 500 (2016-2026, total return with a 1.2% dividend
proxy). Two pre-set rules (2 variants):
  * 1x outside the window, 3x inside (pure overlay on buy & hold);
  * 2x outside, 3x inside (overlay on the champion's current 2x stance).
Leverage financed at the T-bill + 0.5%; 5bp per side on every unit of exposure changed (two
switches a month). Compared with constant 1x/2x/3x and with 'inside the window only' returns.

    python scripts/experiments/exp26_turn_of_month.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

px = at.data.fred_daily_proxies()["stocks"]
r = px.pct_change().dropna()
tb = (at.data.load_fred("DTB3") / 100).reindex(r.index).ffill().bfill()
m = r.index.to_period("M")
k = r.groupby(m).cumcount()                                   # 0 = first trading day
n = r.groupby(m).transform("size")
tom = (k <= 2) | (k == n - 1)                                 # last day + first 3 days

print(f"S&P 500 days {r.index[0]:%Y-%m-%d} to {r.index[-1]:%Y-%m-%d}: window {tom.mean():.0%} of days")
for lab, mask in [("window (-1..+3)", tom), ("other days", ~tom)]:
    x = r[mask]
    print(f"  {lab:16s} mean {x.mean()*1e4:5.1f}bp/day, annualised {x.mean()*252:6.1%}, "
          f"t={x.mean()/x.std()*len(x)**0.5:4.1f}")
print(f"  window share of total log return: {(r[tom].pipe(lambda s: (1+s).prod())):.2f}x growth "
      f"vs {(1+r[~tom]).prod():.2f}x on the other days")


def run(lev):
    """lev: daily leverage series (set the day before, i.e. known in advance: the calendar)."""
    gross = lev * r - (lev - 1).clip(lower=0) * (tb + 0.005) / 252
    turn = lev.diff().abs().fillna(lev.iloc[0])
    return gross - turn * 0.0005


rules = {"1x constant": pd.Series(1.0, r.index), "2x constant": pd.Series(2.0, r.index),
         "3x constant": pd.Series(3.0, r.index),
         "1.38x constant (same avg)": pd.Series(1.38, r.index),
         "2.19x constant (same avg)": pd.Series(2.19, r.index),
         "1x, 3x in window": tom.map({True: 3.0, False: 1.0}),
         "2x, 3x in window": tom.map({True: 3.0, False: 2.0})}
rows = {}
for name, lev in rules.items():
    x = run(lev.astype(float))
    rows[name] = {"2016-2021": at.metrics.cagr(x[:"2021"], 252), "2022-2026": at.metrics.cagr(x["2022":], 252),
                  "2016-2026": at.metrics.cagr(x, 252), "max DD": at.metrics.max_drawdown(x),
                  "avg leverage": lev.mean()}
out = pd.DataFrame(rows).T
print(out.to_string(formatters={c: "{:.1%}".format for c in out.columns[:-1]} | {"avg leverage": "{:.2f}x".format}))
