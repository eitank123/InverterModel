"""Experiment 18: does a DAILY exit improve the 3x monthly trend sleeve?

The champion's trend sleeve checks the S&P against its 10-month average once a month, so a
fast crash (Oct 2008, Mar 2020) is held at 3x until month-end. Gayed & Bilello (2016,
"Leverage for the Long Run") apply a DAILY 200-day average to leveraged S&P exposure instead.

Data: daily 2007-2026, DJIA 2007-2016 (data/ file) chained to the FRED S&P 500 2016-2026
(price indices; dividends added at 1.9%/yr), T-bills from FRED. One era only, so this is an
experiment, not a champion candidate: the worst-era rule needs 1880+ data that is monthly.

All rules hold 3x daily-reset exposure (financing at T-bill + 0.5%, 0.91% expense ratio) or
T-bills, signals at the close of day t, traded at the close of day t+1, 5bp per side.
  A. monthly 10-month average (the current sleeve)
  B. daily 200-day average (Gayed-Bilello)
  C. daily 200-day average with a 2% band (enter above +2%, exit below -2%; Exp 13)
  D. monthly signal + daily emergency exit below the 200-day average -2%; re-entry only at
     a month-end with the monthly signal on
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

dj = at.data.load_djia_daily()["close"]
sp = at.data.load_fred("SP500")
tb = at.data.load_fred("DTB3") / 100
overlap = sp.index[0]                                    # chain daily returns: DJIA, then S&P
ret = pd.concat([dj.pct_change()[dj.index < overlap], sp.pct_change()[sp.index >= overlap]]).fillna(0.0)
px = (1 + ret).cumprod() * 100
r = px.pct_change().fillna(0.0) + 0.019 / 252            # add dividends
cash = tb.reindex(px.index).ffill().bfill() / 252
L, cost = 3.0, 5e-4
r3 = L * r - (L - 1) * (cash + 0.005 / 252) - 0.0091 / 252

me = px.groupby(px.index.to_period("M")).transform("last")
is_me = px.index.to_series().groupby(px.index.to_period("M")).transform("max") == px.index.to_series()
me_close = px[is_me]
sma10m = me_close.rolling(10).mean()
monthly_on = (me_close > sma10m).reindex(px.index).ffill().fillna(False)
sma200 = px.rolling(200).mean()
daily_on = px > sma200


def banded(px, sma, band=0.02):
    on, out = False, []
    for p, s in zip(px, sma):
        if np.isnan(s):
            out.append(False)
            continue
        if not on and p > s * (1 + band):
            on = True
        elif on and p < s * (1 - band):
            on = False
        out.append(on)
    return pd.Series(out, index=px.index)


def emergency(px, sma, monthly_on, is_me, band=0.02):
    on, out = bool(monthly_on.iloc[0]), []
    for d in px.index:
        if on and not np.isnan(sma[d]) and px[d] < sma[d] * (1 - band):
            on = False
        if is_me[d]:
            on = bool(monthly_on[d])
        out.append(on)
    return pd.Series(out, index=px.index)


rules = {
    "S&P buy & hold (1x)": None,
    "3x buy & hold (UPRO-like)": pd.Series(True, index=px.index),
    "A. monthly 10-month average 3x": monthly_on,
    "B. daily 200-day average 3x": daily_on,
    "C. daily 200-day ±2% band 3x": banded(px, sma200),
    "D. monthly + daily emergency exit 3x": emergency(px, sma200, monthly_on, is_me),
}


def run(sig):
    if sig is None:
        return r, 0
    pos = sig.astype(float).shift(1).fillna(0.0)             # traded at the next close
    held = pos.shift(1).fillna(0.0)
    switches = pos.diff().abs().fillna(0.0)
    out = held * r3 + (1 - held) * cash - switches * cost
    return out, switches.sum()


def stats(x, n):
    yrs = len(x) / 252
    eq = (1 + x).cumprod()
    yr = (1 + x).groupby(x.index.year).prod() - 1
    return {"CAGR": eq.iloc[-1] ** (1 / yrs) - 1, "max DD": (eq / eq.cummax() - 1).min(),
            "worst day": x.min(), "switches/yr": n / yrs, "2008": yr.get(2008), "2020": yr.get(2020),
            "2022": yr.get(2022), "2007-16": (1 + x[:"2016"]).prod() ** (252 / len(x[:"2016"])) - 1,
            "2017-26": (1 + x["2017":]).prod() ** (252 / len(x["2017":])) - 1}


rows = {k: stats(*run(v)) for k, v in rules.items()}
res = pd.DataFrame(rows).T
pd.set_option("display.width", 200)
fmt = {c: "{:.1%}".format for c in res.columns if c != "switches/yr"}
fmt["switches/yr"] = "{:.1f}".format
print(f"Daily {px.index[0]:%Y-%m-%d} to {px.index[-1]:%Y-%m-%d} ({len(px)} days)\n")
print(res.to_string(formatters=fmt))
