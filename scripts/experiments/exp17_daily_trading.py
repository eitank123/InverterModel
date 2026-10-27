"""Experiment 17: does daily trading beat the monthly approach once costs are counted?

Data: Dow Jones Industrial Average daily, 2007-2016 (price index, no dividends: the only
daily data reachable offline). Published short-horizon rules, no parameters fitted here:
  * buy & hold
  * 200-day average filter (daily version of the monthly trend rule)
  * 1-day reversal: hold the day after a down day (short-term reversal, Lehmann 1990)
  * 20-day momentum: hold while the 20-day return is positive
  * RSI(2) (Connors & Alvarez 2009): buy when RSI(2) < 10 above the 200-day average,
    sell when the close rises above the 5-day average
  * turn of the month: last trading day + first 3 days of each month (Lakonishok & Smidt 1988)
  * overnight only (close -> next open) vs intraday only (open -> close)
    (Lou, Polk & Skouras 2019, "A tug of war: overnight versus intraday expected returns").
    Caveat: an index's published "open" is computed partly from stale prices, so this split is
    unreliable on index data; the published effect is measured on ETF/stock opens.

Signals use the close of day t and are traded at the close of day t+1 (one day of delay),
except the overnight/intraday split, which trades every open and close by construction.
Costs are per side, as a share of the amount traded: 1bp (best case for SPY), 5bp (typical
retail with slippage). Halves 2007-2011 and 2012-2016 are shown separately.

    python scripts/experiments/exp17_daily_trading.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

d = at.data.load_djia_daily()
c, o = d["close"], d["open"]
r_cc = c.pct_change().fillna(0.0)
r_on = (o / c.shift(1) - 1).fillna(0.0)            # close -> next open
r_id = (c / o - 1).fillna(0.0)                     # open -> close


def rsi(s, n):
    ch = s.diff()
    up = ch.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-ch.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def connors(c):
    sma200, sma5, r2 = c.rolling(200).mean(), c.rolling(5).mean(), rsi(c, 2)
    pos, held = [], 0.0
    for i in range(len(c)):
        if held == 0 and r2.iloc[i] < 10 and c.iloc[i] > sma200.iloc[i]:
            held = 1.0
        elif held == 1 and c.iloc[i] > sma5.iloc[i]:
            held = 0.0
        pos.append(held)
    return pd.Series(pos, index=c.index)


month = c.index.to_period("M")
day_in_month = c.groupby(month).cumcount()
days_left = c.groupby(month).cumcount(ascending=False)
signals = {                                         # position decided at close t
    "buy & hold": pd.Series(1.0, index=c.index),
    "200-day average": (c > c.rolling(200).mean()).astype(float),
    "1-day reversal": (r_cc < 0).astype(float),
    "20-day momentum": (c.pct_change(20) > 0).astype(float),
    "RSI(2) Connors": connors(c),
}
# Turn of month: hold from the close BEFORE the last trading day through day 3 of the next month.
tom_hold_day = (days_left == 0) | (day_in_month <= 2)          # days whose return we want
signals["turn of the month"] = tom_hold_day.shift(-1, fill_value=False).astype(float)


def run(pos_decided, cost_bps):
    """pos decided at close t, executed at close t+1, earns r_cc from t+1 to t+2."""
    pos = pos_decided.shift(1).fillna(0.0)       # one day of delay
    gross = pos.shift(1).fillna(0.0) * r_cc
    trades = pos.diff().abs().fillna(pos.abs())
    return gross - trades * cost_bps / 1e4, trades


def run_split(which, cost_bps):
    """Overnight-only or intraday-only: two trades per day by construction."""
    r = r_on if which == "overnight" else r_id
    return r - 2 * cost_bps / 1e4, pd.Series(2.0, index=r.index)


def stats(r, trades):
    yrs = len(r) / 252
    return {"CAGR": (1 + r).prod() ** (1 / yrs) - 1, "Sharpe": r.mean() / r.std() * np.sqrt(252),
            "max DD": at.metrics.max_drawdown(r), "trades/yr": trades.sum() / yrs / 2,
            "2007-11": (1 + r[:"2011"]).prod() ** (252 / len(r[:"2011"])) - 1,
            "2012-16": (1 + r["2012":]).prod() ** (252 / len(r["2012":])) - 1}


rows = []
for cost in (0.0, 1.0, 5.0):
    for name, sig in signals.items():
        r, t = run(sig, cost)
        rows.append({"rule": name, "cost bp": cost, **stats(r, t)})
    for which in ("overnight", "intraday"):
        r, t = run_split(which, cost)
        rows.append({"rule": f"{which} only", "cost bp": cost, **stats(r, t)})
res = pd.DataFrame(rows)

# Break-even cost: bp per side at which the rule's CAGR falls to buy & hold's.
bh = (1 + r_cc).prod() ** (252 / len(r_cc)) - 1
be = {}
for name, sig in {**{k: v for k, v in signals.items() if k != "buy & hold"},
                  "overnight only": None, "intraday only": None}.items():
    lo, hi = 0.0, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2
        r = run(sig, mid)[0] if sig is not None else run_split(name.split()[0], mid)[0]
        g = (1 + r).prod() ** (252 / len(r)) - 1
        lo, hi = (mid, hi) if g > bh else (lo, mid)
    be[name] = lo if lo > 0 else np.nan

pd.set_option("display.width", 200)
fmt = {"CAGR": "{:.1%}", "Sharpe": "{:.2f}", "max DD": "{:.0%}", "trades/yr": "{:.0f}",
       "2007-11": "{:.1%}", "2012-16": "{:.1%}"}
for cost in (0.0, 1.0, 5.0):
    t = res[res["cost bp"] == cost].drop(columns="cost bp").set_index("rule")
    print(f"\n=== cost {cost:g} bp per side ===")
    print(t.to_string(formatters={k: v.format for k, v in fmt.items()}))
print(f"\nBuy & hold CAGR {bh:.1%}. Break-even cost (bp per side) to match buy & hold "
      "(blank = loses even with free trading):")
for k, v in be.items():
    print(f"  {k:20s} {'' if np.isnan(v) else f'{v:.1f}'}")
