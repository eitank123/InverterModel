"""Experiment 7: does valuation predict returns, and can a trader use it in real time?

Math (Campbell & Shiller 1988): log return identity
    r_{t+1} ~ k + rho * pd_{t+1} - pd_t + dg_{t+1}
iterated forward says today's price/dividend (or price/earnings) ratio must forecast
future returns, future dividend growth, or both. Empirically it forecasts *returns*,
weakly at 1 year and strongly at 10 years (Campbell & Shiller 1998, CAPE).

But: Goyal & Welch (2008) showed almost no predictor beats the historical mean
*out of sample*. Campbell & Thompson (2008) showed simple restrictions (no negative
forecasts) help. And overlapping 10-year returns make naive t-stats meaningless:
Newey-West standard errors are required.

    python scripts/experiments/exp07_valuation.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", "{:.3f}".format)
OUT = "results/exp07"
os.makedirs(OUT, exist_ok=True)

df = at.data.load_shiller_sp500()
PUB_LAG = 6          # months: earnings are published with a delay; be conservative
cape = df["cape"].shift(PUB_LAG)
infl10 = (df["cpi"] / df["cpi"].shift(120)) ** (1 / 10) - 1
real_bond = df["long_rate"] - infl10
ecy = (1 / cape - real_bond.shift(PUB_LAG))          # Shiller's excess CAPE yield
log_real = np.log(df["tr"] / df["cpi"])

# --- A. In-sample predictive regressions ---------------------------------------------
print("A. Future real return (annualised) on log(CAPE), 1881-2023, Newey-West t-stats")
rows = []
for h in (1, 12, 60, 120):
    fut = (log_real.shift(-h) - log_real) * 12 / h
    for name, x in [("log_cape", np.log(cape)), ("ecy", ecy)]:
        res = at.stats.ols_newey_west(fut, x.rename(name).to_frame(), lags=max(h, 1))
        rows.append({"horizon_m": h, "predictor": name, "coef": res.loc[name, "coef"],
                     "t_newey_west": res.loc[name, "t_nw"], "R2": res.attrs["r2"],
                     "n": res.attrs["n"]})
print(pd.DataFrame(rows).set_index(["horizon_m", "predictor"]))

# --- B. Out-of-sample R^2 (Goyal-Welch test) -----------------------------------------
print("\nB. Out-of-sample R^2 vs the expanding historical mean (from 1911; >0 = useful)")
rows = []
for h in (12, 60, 120):
    fut = (log_real.shift(-h) - log_real) * 12 / h
    for name, x in [("log_cape", np.log(cape)), ("ecy", ecy)]:
        for nonneg in (False, True):
            r2, _ = at.stats.oos_r2(fut, x, horizon=h, min_train=360, nonneg=nonneg)
            rows.append({"horizon_m": h, "predictor": name,
                         "campbell_thompson": nonneg, "oos_R2": r2})
print(pd.DataFrame(rows).pivot_table(index=["horizon_m", "predictor"],
                                     columns="campbell_thompson", values="oos_R2"))
print("   (1-month OOS R^2 of even 0.5% is economically large; 10-year R^2 overlaps heavily)")

# What does the model say about the next 10 years? Use the latest data point.
fut120 = (log_real.shift(-120) - log_real) * 12 / 120
fit = at.stats.ols_newey_west(fut120, np.log(cape).rename("lc").to_frame(), lags=120)
for c in (30.8, 41.0):
    pred = fit.loc["const", "coef"] + fit.loc["lc", "coef"] * np.log(c)
    print(f"   CAPE {c:4.1f} -> regression-implied real return next 10y: {pred:+.1%}/yr")

# --- C. A real-time valuation strategy ------------------------------------------------
print("\nC. Valuation-tilted stock/bond allocation (walk-forward, lag 1, 10bp)")
px = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
cash = (df["long_rate"] - 0.015).clip(lower=0)
pct = ecy.expanding(min_periods=240).rank(pct=True).ffill()   # percentile vs own past only


core = at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4})
sma = at.PriceAboveSMA(10)
STRATS = {
    "60_40": core,
    "60_40_sma10": at.TrendFilter(core, sma),
    "value_tilt": at.ValueTilt(pct),
    "value_tilt_sma10": at.TrendFilter(at.ValueTilt(pct), sma),
}
start = pct.first_valid_index()
KW = dict(cost_bps=10, execution_lag=1, cash_rate=cash[start:])
res = {n: at.run_backtest(px[start:], s, name=n, **KW) for n, s in STRATS.items()}
t = at.compare(list(res.values()))
print(f"   sample {start.date()} - {px.index[-1].date()} (CAPE frozen after 2023-09)")
print(t[["cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "ann_turnover"]])

# Asness-Moskowitz-Pedersen: value and momentum active returns should be negatively correlated
base = res["60_40"].returns
val_active = res["value_tilt"].returns - base
mom_active = res["60_40_sma10"].returns - base
print(f"\n   corr(value active, trend active) = {val_active.corr(mom_active):+.2f}  "
      "(AMP 2013: about -0.5 across asset classes)")
yr = lambda r: r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)  # noqa: E731
print(f"   annual: corr = {yr(val_active).corr(yr(mom_active)):+.2f}")
print("   current stock weight of value_tilt:",
      f"{STRATS['value_tilt'].weights(px).iloc[-1]['stocks']:.0%}")
at.plot(list(res.values()), f"{OUT}/value_trend.png", "Value tilt and trend filter, 1901-2026")
