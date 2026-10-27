"""Experiment 12: is the trend filter's drawdown reduction real, or just less exposure?

A filter that is out of the market 25% of the time has lower volatility, and lower
volatility alone shrinks drawdowns. The right control is a CONSTANT-exposure portfolio
with the same volatility. If the filter's drawdown is smaller than the vol-matched
control's, the filter is timing crashes (the effect we want). Otherwise it is just holding less.

  A. Real data 1957-2026: filter vs vol-matched constant exposure.
  B. Synthetic markets (truth known): GBM (no structure), GARCH (vol clustering),
     regime switching (trends). Theory: the filter's edge over the control should
     appear only where returns are serially dependent.

    python scripts/experiments/exp12_insurance_or_exposure.py
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
START = "1957"


def compare(prices, base, filt, cash_rate, start=None):
    """Return stats for base, base+filter, and base scaled to the filter's volatility."""
    f = at.run_backtest(prices, at.TrendFilter(base, filt), cash_rate=cash_rate, **KW).returns
    b = at.run_backtest(prices, base, cash_rate=cash_rate, **KW).returns
    if start:
        f, b = f[start:], b[start:]
    k = f.std() / b.std()
    ctrl = at.run_backtest(prices, at.Levered(base, k), cash_rate=cash_rate, **KW).returns
    if start:
        ctrl = ctrl[start:]
    out = {}
    for name, r in [("base", b), ("filtered", f), (f"constant {k:.2f}x", ctrl)]:
        out[name] = {"cagr": at.metrics.cagr(r), "vol": at.metrics.annual_vol(r),
                     "max_dd": at.metrics.max_drawdown(r), "calmar": at.metrics.calmar(r)}
    return pd.DataFrame(out).T, f, ctrl


print(f"A. Real data {START}-2026 (monthly, lag 1, 10bp, cash = bond yield - 1.5%)")
for label, base in [("S&P 500", at.PerAsset(at.BuyAndHold(), {"stocks": 1.0})),
                    ("60/40", at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4}))]:
    t, f, ctrl = compare(px, base, at.PriceAboveSMA(10), cash, START)
    print(f"\n-- {label}")
    print(t)
    both = pd.DataFrame({"f": f, "c": ctrl})
    dd = pd.DataFrame(list(at.stats.bootstrap_stat(
        both, lambda d: d.apply(at.metrics.max_drawdown), n_boot=500, mean_block=24)),
        columns=["f", "c"])
    print(f"   bootstrap P(filter has smaller max DD than vol-matched control) = "
          f"{(dd['f'] > dd['c']).mean():.0%}; P(higher CAGR) = "
          f"{(pd.DataFrame(list(at.stats.bootstrap_stat(both, lambda d: d.mean(), 500, 24)), columns=['f', 'c']).pipe(lambda x: (x.f > x.c).mean())):.0%}")

print("\nB. Synthetic markets, 40 years monthly, 100 seeds each (no costs, cash 0)")
gens = {
    "GBM (no structure)": lambda s: at.data.gbm(n=480 * 21, mu=0.08, sigma=0.16, seed=s),
    "GARCH (vol clusters)": lambda s: at.data.garch(n=480 * 21, mu=0.08, seed=s),
    "regime switching (trends)": lambda s: at.data.regime_switching(
        n=480 * 21, mus=(0.15, -0.15), sigmas=(0.12, 0.25), p_stay=0.997, seed=s),
}
rows = []
for gname, g in gens.items():
    for seed in range(100):
        p = g(seed)["close"]
        if len(p) > 1000:                       # daily generators -> month-end prices
            p = p.resample("MS").last()
        prices = pd.DataFrame({"stocks": p})
        base = at.PerAsset(at.BuyAndHold(), {"stocks": 1.0})
        t, _, _ = compare(prices, base, at.PriceAboveSMA(10), 0.0)
        ctrl_name = t.index[2]
        rows.append({"market": gname,
                     "dd_filtered": t.loc["filtered", "max_dd"], "dd_control": t.loc[ctrl_name, "max_dd"],
                     "cagr_filtered": t.loc["filtered", "cagr"], "cagr_control": t.loc[ctrl_name, "cagr"]})
syn = pd.DataFrame(rows)
syn["filter_better_dd"] = syn["dd_filtered"] > syn["dd_control"]
syn["filter_better_cagr"] = syn["cagr_filtered"] > syn["cagr_control"]
print(syn.groupby("market").mean(numeric_only=True))
