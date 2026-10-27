"""Experiment 11: re-check every stock/bond conclusion on the era with real bond data.

Data flaw found in Exp 10: before 1953 Shiller's long rate is interpolated from annual
data. Monthly yield changes are smooth (autocorrelation 0.85, vs 0.3 after 1953), so the
rebuilt bond returns have ~0.5% volatility and a raw Sharpe of 5-9: impossible for real
bonds. Every stock/bond result that includes pre-1953 data is flattered. Monthly 10-year
Treasury yields exist from April 1953, so we evaluate 1957-2026 (allowing indicator warm-up).

    python scripts/experiments/exp11_clean_bond_era.py
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
START = "1957"

core = at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4})
sma10 = at.PriceAboveSMA(10)
sma_avg = at.AverageSignal([at.PriceAboveSMA(w) for w in (6, 8, 10, 12)])
C = {
    "S&P500": at.PerAsset(at.BuyAndHold(), {"stocks": 1.0}),
    "bonds": at.PerAsset(at.BuyAndHold(), {"bonds": 1.0}),
    "60/40": core,
    "60/40+SMA10 [frozen]": at.TrendFilter(core, sma10),
    "60/40+SMA avg": at.TrendFilter(core, sma_avg),
    "S&P500+SMA10": at.PerAsset(sma10, {"stocks": 1.0}),
    "value+SMA10": at.TrendFilter(at.ValueTilt(pct), sma10),
    "inv-vol+SMA10": at.TrendFilter(at.InverseVol(36), sma10),
    "60/40+SMA10 x1.5": at.Levered(at.TrendFilter(core, sma10), 1.5),
    "tsmom 2-asset volscaled": at.PortfolioVolTarget(
        at.PerAsset(at.VolTarget(at.TrendEnsemble((1, 3, 12)), 0.10, 12, 3.0)), 0.10, 24, 3.0),
}
R = pd.DataFrame({n: at.run_backtest(px, s, cash_rate=cash, **KW).returns for n, s in C.items()})[START:]
rf = cash.reindex(R.index) / 12
stats = pd.DataFrame({
    "cagr": R.apply(at.metrics.cagr),
    "vol": R.apply(at.metrics.annual_vol),
    "sharpe_raw": R.apply(at.metrics.sharpe),
    "sharpe_excess": R.sub(rf, axis=0).apply(at.metrics.sharpe),
    "max_dd": R.apply(at.metrics.max_drawdown),
})
print(f"A. {START}-2026, real monthly bond data, lag 1, 10bp, cash = bond yield - 1.5%")
print(stats)

sh = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
ex = R.sub(rf, axis=0)
boot = pd.DataFrame(list(at.stats.bootstrap_stat(ex, lambda d: d.apply(sh), n_boot=1000,
                                                 mean_block=24)), columns=R.columns)
print("\nB. Excess-Sharpe bootstrap (block 24m): 95% CI and P(beats 60/40)")
print(pd.DataFrame({"lo": boot.quantile(.025), "hi": boot.quantile(.975),
                    "P>60/40": (boot.sub(boot["60/40"], axis=0) > 0).mean()}))

print("\nC. Kelly fraction (excess), clean era:")
for n in ("S&P500", "60/40", "60/40+SMA10 [frozen]"):
    print(f"   {n:22s} f* = {at.stats.kelly_fraction(ex[n]):.1f}")

print("\nD. Comparison with the earlier full-history numbers (1901-2026 Sharpe raw):")
R_full = pd.DataFrame({n: at.run_backtest(px, C[n], cash_rate=cash, **KW).returns
                       for n in ("60/40", "60/40+SMA10 [frozen]", "bonds")})["1901":]
print("   ", R_full.apply(at.metrics.sharpe).round(2).to_dict())

# --- E. Is the drawdown reduction robust? Max drawdown is ONE observation per history.
print("\nE. Bootstrap of max drawdown and of the worst 12-month return (block 24m)")


def worst12(d):
    return (1 + d).rolling(12).apply(np.prod, raw=True).min() - 1


pairs = [("60/40", "60/40+SMA10 [frozen]"), ("S&P500", "S&P500+SMA10")]
cols = [c for p in pairs for c in p]
dd = pd.DataFrame(list(at.stats.bootstrap_stat(
    R[cols], lambda d: d.apply(at.metrics.max_drawdown), n_boot=500, mean_block=24)), columns=cols)
w12 = pd.DataFrame(list(at.stats.bootstrap_stat(
    R[cols], lambda d: d.apply(worst12), n_boot=500, mean_block=24)), columns=cols)
for a, b in pairs:
    print(f"   {b} vs {a}: median maxDD {dd[b].median():.0%} vs {dd[a].median():.0%}, "
          f"P(smaller maxDD) = {(dd[b] > dd[a]).mean():.0%}; "
          f"median worst 12m {w12[b].median():.0%} vs {w12[a].median():.0%}, "
          f"P(better) = {(w12[b] > w12[a]).mean():.0%}")
