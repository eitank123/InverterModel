"""Experiment 58: a daily crash guard for the champion's 3x trend sleeve (weekly-report idea, W47).

Week 47 showed the weakness of a month-end rule at 3x: the S&P fell 8.6% in five days and the monthly
3x accounts lost 24% with no chance to act, while the daily 200-day account (Gayed & Bilello 2016,
'Leverage for the Long Run') stepped aside. Test on DAILY CRSP market returns 1928-2026 (Ken French),
3x while 'on', cash otherwise, leverage at T-bill + 0.5%, 10bp per unit traded, trades at the next close:
  A  monthly: price at month-end > its 10-month average (the old champion's sleeve)
  A' monthly gated by unemployment (the new champion's sleeve; UNRATE 1948+, NBER 1930-47, Exp 48)
  B  A AND a daily guard: out below 200-day average x0.98, back above x1.02   (Gayed-Bilello with band)
  B' A' AND the same daily guard
  C  the daily rule alone (the live 'Daily trend 3x' account)
Variants counted: 3 (B, B', C).

    python scripts/experiments/exp58_daily_crash_guard.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
import loop as L  # noqa: E402

ff = at.data.load_ff_factors(daily=True)
r, rf = ff["Mkt-RF"] + ff["RF"], ff["RF"]
lvl = (1 + r).cumprod()
month = r.index.to_period("M")
is_me = r.groupby(month).cumcount(ascending=False) == 0

# monthly 10-month signal, decided on the last trading day, traded next close
me = lvl[is_me]
m_on = (me > me.rolling(10).mean()).astype(float)
# unemployment gate (rising = obey the exit), lagged one month for publication; NBER 1930-47 + UNRATE
sys.argv = sys.argv[:1]
from exp48_unemployment_1929 import gate as unemp_gate  # noqa: E402
weak = unemp_gate.shift(1).reindex(pd.PeriodIndex(me.index, freq="M").to_timestamp()).fillna(True).astype(bool).values
gm_on = pd.Series(np.where(~weak, 1.0, m_on.values), index=me.index)


def daily(sig_monthly):
    return sig_monthly.reindex(r.index).ffill().shift(1).fillna(1.0)


sma200 = lvl.rolling(200).mean()
state, g = 1.0, []
for p, s in zip(lvl, sma200):
    if not np.isnan(s):
        if state and p < s * 0.98:
            state = 0.0
        elif not state and p > s * 1.02:
            state = 1.0
    g.append(state)
guard = pd.Series(g, index=r.index).shift(1).fillna(1.0)


def run(on, lev=3.0):
    Lx = lev * on
    net = Lx * r - (Lx - 1).clip(lower=0) * (rf + 0.005 / 252) + (1 - Lx).clip(lower=0) * rf \
        - Lx.diff().abs().fillna(0) * 0.001
    return net["1930":]


V = {"A monthly (old sleeve)": daily(m_on), "A' monthly, unemployment-gated (new sleeve)": daily(gm_on),
     "B monthly + daily guard": daily(m_on) * guard, "B' gated + daily guard": daily(gm_on) * guard,
     "C daily 200-day rule alone": guard}
ERAS = {"1930-1956": ("1930", "1956"), "1957-1990": ("1957", "1990"), "1991-2026": ("1991", "2026")}
rows = {}
for k, on in V.items():
    x = run(on)
    row = {e: at.metrics.cagr(x[a:b], 252) for e, (a, b) in ERAS.items()}
    row["worst era"] = min(row.values())
    w = (1 + x["1940":]).cumprod()
    row["max DD 1940+"] = float((w / w.cummax() - 1).min())
    row["worst year"] = float(((1 + x).groupby(x.index.year).prod() - 1).min())
    row["1987"] = float((1 + x["1987"]).prod() - 1)
    row["trades/yr"] = float(on.diff().abs().sum() / (len(on) / 252))
    rows[k] = row
pd.set_option("display.width", 220)
print("3x trend sleeve on daily CRSP returns 1930-2026, all costs:\n")
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1%}" if abs(v) < 5 else f"{v:.1f}"))


# --- Part 2: the whole champion on daily data ----------------------------------------------------
# The other two sleeves (valuation leverage, recovery+) keep their monthly Shiller decisions: the
# weight decided for month m (execution lag 1) is held every day of month m+1, applied to daily CRSP
# returns as a daily-reset leveraged position. Recovery+'s Treasuries when out are held as cash (no
# daily bond series): a conservative simplification shared by all variants.
ctx = at.research.Context.default(0.005)
def monthly_to_daily(w):
    w = w.copy()
    w.index = pd.PeriodIndex(w.index, freq="M") + 1
    return pd.Series(w.reindex(month).values, index=r.index).ffill().fillna(0.0)


val = monthly_to_daily(L.CappedValuationLeverage(1.0, 2.0, 0.5, 3.0).weights(ctx.px)["stocks"])
rec = monthly_to_daily(at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0).weights(ctx.px)["stocks"])


def run_exposure(E_, fin=0.005):
    net = E_ * r - (E_ - 1).clip(lower=0) * (rf + fin / 252) + (1 - E_).clip(lower=0) * rf \
        - E_.diff().abs().fillna(0) * 0.001
    return net["1930":]


ens = {"old champion (monthly trend)": 3 * daily(m_on), "new champion (gated monthly)": 3 * daily(gm_on),
       "champion with daily trend sleeve": 3 * guard, "champion with gated monthly + daily guard": 3 * daily(gm_on) * guard,
       "50/50 hedge (old + new monthly)": 1.5 * daily(m_on) + 1.5 * daily(gm_on)}
rows = {}
for fin in (0.005, 0.015):
    for k, t in ens.items():
        x = run_exposure((t + val + rec) / 3, fin)
        row = {e: at.metrics.cagr(x[a:b], 252) for e, (a, b) in ERAS.items()}
        row["worst era"] = min(row.values())
        w = (1 + x["1940":]).cumprod()
        row["max DD 1940+"] = float((w / w.cummax() - 1).min())
        row["2022"] = float((1 + x["2022"]).prod() - 1)
        rows[(f"+{fin:.1%}", k)] = row
print("\nWhole ensemble on daily CRSP 1930-2026 (other sleeves monthly), all costs:\n")
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.1%}"))


# --- Part 3: out of sample, Japan (Nikkei 225 daily, FRED, 1950-2026) ------------------------------
# Same rules, nothing re-fitted. Dividends +1.5%/yr, cash = US T-bill (as Exp 19), 3x daily-reset.
nk = at.data.load_fred("NIKKEI225")
nr = nk.pct_change().dropna() + 0.015 / 252
tb = (at.data.load_fred("DTB3") / 100 / 252).reindex(nr.index).ffill().fillna(0.015 / 252)
nl = (1 + nr).cumprod()
nme = nl[nr.groupby(nr.index.to_period("M")).cumcount(ascending=False) == 0]
n_m = (nme > nme.rolling(10).mean()).astype(float).reindex(nr.index).ffill().shift(1).fillna(1.0)
s2 = nl.rolling(200).mean()
st_, g2 = 1.0, []
for p, s in zip(nl, s2):
    if not np.isnan(s):
        if st_ and p < s * 0.98:
            st_ = 0.0
        elif not st_ and p > s * 1.02:
            st_ = 1.0
    g2.append(st_)
n_d = pd.Series(g2, index=nr.index).shift(1).fillna(1.0)
print("\nJapan, 3x trend sleeve (Nikkei daily, all costs):")
for k, on in {"monthly 10-month rule": n_m, "daily 200-day rule (+-2%)": n_d}.items():
    Lx = 3 * on
    x = (Lx * nr - (Lx - 1).clip(lower=0) * (tb + 0.005 / 252) + (1 - Lx).clip(lower=0) * tb
         - Lx.diff().abs().fillna(0) * 0.001)["1951":]
    w = (1 + x).cumprod()
    print(f"  {k:28s} 1951-1989 {at.metrics.cagr(x[:'1989'], 252):6.1%}  1990-2012 {at.metrics.cagr(x['1990':'2012'], 252):6.1%}"
          f"  2013-2026 {at.metrics.cagr(x['2013':], 252):6.1%}  all {at.metrics.cagr(x, 252):6.1%}  max DD {float((w / w.cummax() - 1).min()):.0%}")


# --- Part 4 (Exp 59): fixed 50/50 trend sleeves, daily rule + a monthly rule -----------------------
print("\nExp 59 - trend sleeve split 50/50 between the daily rule and a monthly rule (whole ensemble):")
mix = {"half daily, half old monthly": 1.5 * guard + 1.5 * daily(m_on),
       "half daily, half gated monthly": 1.5 * guard + 1.5 * daily(gm_on)}
for fin in (0.005, 0.015):
    for k, t in {**{"new champion (gated monthly)": 3 * daily(gm_on), "champion with daily trend sleeve": 3 * guard}, **mix}.items():
        x = run_exposure((t + val + rec) / 3, fin)
        v = [at.metrics.cagr(x[a:b], 252) for a, b in ERAS.values()]
        w = (1 + x["1940":]).cumprod()
        print(f"  +{fin:.1%} {k:36s} " + " ".join(f"{q:6.1%}" for q in v) + f"  worst {min(v):6.1%}  DD1940+ "
              f"{float((w / w.cummax() - 1).min()):.0%}  2022 {float((1 + x['2022']).prod() - 1):.0%}")
