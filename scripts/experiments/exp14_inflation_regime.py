"""Experiment 14: does an inflation-regime signal add value beyond the trend filter?

Motivation (Exp 4): the stock-bond correlation turns positive when inflation drives
markets (1968-99, 2021-26), so bonds stop hedging. Idea: when trailing 12-month CPI
inflation is high (above a threshold) AND rising, cut the bond sleeve to cash.
CPI is published with a ~1 month delay, so the signal is lagged 2 months.

Discipline: thresholds chosen on TRAIN 1957-1990, judged on TEST 1991-2026.
Data: bonds reliable from 1953 (Exp 11), so 1957+ only.

    python scripts/experiments/exp14_inflation_regime.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", "{:.3f}".format)

df = at.data.load_shiller_sp500()
px = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
cash = (df["long_rate"] - 0.015).clip(lower=0)
KW = dict(cost_bps=10, execution_lag=1)
infl = (df["cpi"] / df["cpi"].shift(12) - 1).shift(2)        # published with a delay
rising = infl > infl.shift(6)


class InflationGuard(at.PortfolioStrategy):
    """Zero the bond weight when inflation > threshold (and rising, if required)."""

    def __init__(self, base, threshold=0.04, need_rising=True):
        self.base, self.threshold, self.need_rising = base, threshold, need_rising

    def weights(self, prices):
        w = self.base.weights(prices).copy()
        bad = (infl > self.threshold) & (rising if self.need_rising else True)
        w.loc[bad.reindex(w.index).fillna(False).to_numpy(), "bonds"] = 0.0
        return w


core = at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4})
filt = at.TrendFilter(core, at.PriceAboveSMA(10))
variants = {"60/40": core, "60/40+SMA10 [frozen]": filt}
for th in (0.03, 0.04, 0.05, 0.06):
    for rise in (True, False):
        variants[f"guard {th:.0%}{' rising' if rise else ''}"] = InflationGuard(filt, th, rise)

rows = []
rets = {}
for n, s in variants.items():
    r = at.run_backtest(px, s, cash_rate=cash, **KW).returns
    rets[n] = r
    for per, (a, b) in {"train": ("1957", "1990"), "test": ("1991", "2026")}.items():
        x = r[a:b]
        ex = x - cash.reindex(x.index) / 12
        rows.append({"variant": n, "period": per, "excess_sharpe": at.metrics.sharpe(ex),
                     "cagr": at.metrics.cagr(x), "max_dd": at.metrics.max_drawdown(x)})
t = pd.DataFrame(rows).pivot(index="variant", columns="period")
print(t)
train_sr = t[("excess_sharpe", "train")].drop(["60/40", "60/40+SMA10 [frozen]"])
best = train_sr.idxmax()
print(f"\nSelected on train: {best}; test excess Sharpe {t.loc[best, ('excess_sharpe', 'test')]:.3f} "
      f"vs frozen {t.loc['60/40+SMA10 [frozen]', ('excess_sharpe', 'test')]:.3f}")
th = float(best.split()[1].strip("%")) / 100
active = (infl > th) & (rising if "rising" in best else True)
print(f"Selected guard active in {active['1957':'1990'].mean():.0%} of train months, "
      f"{active['1991':'2026'].mean():.0%} of test months")
R = pd.DataFrame(rets)["1991":]
print("2022 return:", {k: f"{(1 + R[k]['2022']).prod() - 1:+.1%}" for k in ("60/40", "60/40+SMA10 [frozen]", best)})
