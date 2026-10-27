"""Experiment 10: can we reduce the trend filter's whipsaw without overfitting?

Weak spot found in Exp 9: in fast V-shaped crashes (1987, 2020, 2025) the SMA-10 filter
sells near the bottom and buys back late. Ideas from the literature:
  * average several windows (parameter diversification, Exp 2)
  * partial de-risking instead of all-or-nothing
  * rotate the stock sleeve into bonds instead of cash ("dual momentum", Antonacci 2014)
  * risk-parity base weights (inverse volatility)
  * combine with the valuation tilt (Exp 7)

Discipline: choose using TRAIN = 1901-1960 only. Then report TEST = 1961-2026 for every
variant, so we can see whether the train ranking survives (it usually doesn't fully).

    python scripts/experiments/exp10_improve_filter.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", "{:.3f}".format)

df = at.data.load_shiller_sp500()
px = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
cash = (df["long_rate"] - 0.015).clip(lower=0)
KW = dict(cost_bps=10, execution_lag=1, borrow_spread=0.01)
infl10 = (df["cpi"] / df["cpi"].shift(120)) ** (1 / 10) - 1
ecy = 1 / df["cape"].shift(6) - (df["long_rate"] - infl10).shift(6)
pct = ecy.expanding(min_periods=240).rank(pct=True).ffill()


class Partial(at.Strategy):
    """Hold ``floor`` when the filter says out, 1 when in."""

    def __init__(self, filt, floor=0.5):
        self.filt, self.floor = filt, floor

    def positions(self, prices):
        return self.floor + (1 - self.floor) * self.filt.positions(prices)


class RotateToBonds(at.PortfolioStrategy):
    """60/40 with SMA filters, but the stock sleeve moves into bonds (if bonds are
    trending up) instead of cash when stocks trend down."""

    def __init__(self, window=10, stock_w=0.6):
        self.window, self.stock_w = window, stock_w

    def weights(self, prices):
        f = at.PriceAboveSMA(self.window)
        s_on, b_on = f.positions(prices["stocks"]), f.positions(prices["bonds"])
        ws = self.stock_w * s_on
        wb = (1 - self.stock_w) * b_on + self.stock_w * (1 - s_on) * b_on
        return pd.DataFrame({"stocks": ws, "bonds": wb}, index=prices.index)


core = at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4})
sma10 = at.PriceAboveSMA(10)
sma_avg = at.AverageSignal([at.PriceAboveSMA(w) for w in (6, 8, 10, 12)])
VARIANTS = {
    "60/40 (benchmark)": core,
    "A sma10 [frozen]": at.TrendFilter(core, sma10),
    "B sma avg 6-12": at.TrendFilter(core, sma_avg),
    "C tsmom 3/6/12": at.TrendFilter(core, at.TrendEnsemble((3, 6, 12), allow_short=False)),
    "D sma10 partial 50%": at.TrendFilter(core, Partial(sma10, 0.5)),
    "E rotate to bonds": RotateToBonds(),
    "F inv-vol + sma10": at.TrendFilter(at.InverseVol(36), sma10),
    "G value + sma10": at.TrendFilter(at.ValueTilt(pct), sma10),
    "H value + sma avg": at.TrendFilter(at.ValueTilt(pct), sma_avg),
    "I inv-vol + sma avg, 8% vol": at.PortfolioVolTarget(
        at.TrendFilter(at.InverseVol(36), sma_avg), 0.08, 36, 2.0),
}
PERIODS = {"train 1901-1960": ("1901", "1960"), "test 1961-2026": ("1961", "2026")}
rows = []
rets = {}
for name, s in VARIANTS.items():
    full = at.run_backtest(px["1891":], s, cash_rate=cash["1891":], **KW).returns
    rets[name] = full
    for pname, (a, b) in PERIODS.items():
        r = full[a:b]
        rows.append({"variant": name, "period": pname, "sharpe": at.metrics.sharpe(r),
                     "cagr": at.metrics.cagr(r), "max_dd": at.metrics.max_drawdown(r),
                     "calmar": at.metrics.calmar(r)})
t = pd.DataFrame(rows).pivot(index="variant", columns="period")
print(t)

tr = t[("sharpe", "train 1901-1960")]
te = t[("sharpe", "test 1961-2026")]
print(f"\nRank correlation of Sharpe, train vs test: {tr.rank().corr(te.rank()):+.2f}")
best = tr.drop("60/40 (benchmark)").idxmax()
print(f"Selected on train: {best}  -> test Sharpe {te[best]:.3f} vs frozen A {te['A sma10 [frozen]']:.3f}"
      f" vs 60/40 {te['60/40 (benchmark)']:.3f}")
n_trials = len(VARIANTS) - 1
test_rets = pd.DataFrame(rets)["1961":]
dsr = at.metrics.deflated_sharpe(test_rets[best], n_trials, test_rets.apply(at.metrics.sharpe).std())
print(f"Deflated Sharpe of the selection on test ({n_trials} trials): {dsr:.2f} "
      "(vs zero; the harder question is vs 60/40, see the bootstrap below)")
pd.set_option("display.float_format", "{:.3f}".format)
sh = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
boot = at.stats.bootstrap_stat(test_rets, lambda d: d.apply(sh), n_boot=500, mean_block=24)
boot = pd.DataFrame(list(boot), columns=test_rets.columns)
print("P(variant's test Sharpe > frozen A), block bootstrap:")
print((boot.sub(boot["A sma10 [frozen]"], axis=0) > 0).mean().drop("A sma10 [frozen]").round(2).to_string())

print("\nWhipsaw check: returns in fast V-shaped crashes and rebounds")
episodes = {"1987 crash+rebound": ("1987-08", "1988-12"), "2020 COVID": ("2020-01", "2020-12"),
            "2025": ("2025-01", "2025-12"), "2008-09 slow bear": ("2007-11", "2009-12"),
            "2022 inflation": ("2022-01", "2022-12")}
print(pd.DataFrame({k: (test_rets[a:b] + 1).prod() - 1 for k, (a, b) in episodes.items()})
      .map(lambda v: f"{v:+.1%}"))
