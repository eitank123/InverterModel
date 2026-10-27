"""Experiment 4: stocks + bonds, 1871-2026. Does trend following improve a 60/40?

The 60/40 portfolio (60% stocks, 40% government bonds) is the default for long-term
investors. Its weakness: when stocks and bonds fall together (1970s inflation, 2022).
Hypothesis (Hurst, Ooi & Pedersen): trend following on both assets helps exactly then.

Bonds: a 10-year constant-maturity bond built from Shiller's long-rate series.

    python scripts/experiments/exp04_stocks_bonds.py
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
OUT = "results/exp04"
os.makedirs(OUT, exist_ok=True)

df = at.data.load_shiller_sp500()
px = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
cash = (df["long_rate"] - 0.015).clip(lower=0)
KW = dict(cost_bps=10, execution_lag=1, cash_rate=cash)

r = px.pct_change().dropna()
print("Asset stats (monthly, 1871-2026):")
for c in px:
    print(f"  {c:7s} CAGR {at.metrics.cagr(r[c]):.1%}  vol {at.metrics.annual_vol(r[c]):.1%}  "
          f"Sharpe {at.metrics.sharpe(r[c]):.2f}  maxDD {at.metrics.max_drawdown(r[c]):.0%}")

print("\nStock-bond correlation by era (it is NOT a constant):")
eras = [("1871", "1928"), ("1929", "1949"), ("1950", "1967"), ("1968", "1999"), ("2000", "2020"),
        ("2021", "2026")]
for a, b in eras:
    print(f"  {a}-{b}: {r['stocks'][a:b].corr(r['bonds'][a:b]):+.2f}")

tsmom = at.TrendEnsemble((1, 3, 12), allow_short=True)
core = at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4})
trend_unscaled = at.PerAsset(tsmom)
trend_scaled = at.PortfolioVolTarget(
    at.PerAsset(at.VolTarget(tsmom, 0.10, 12, 3.0)), target_vol=0.10, lookback=24, max_leverage=3.0)
STRATS = {
    "stocks_100": at.PerAsset(at.BuyAndHold(), {"stocks": 1.0}),
    "bonds_100": at.PerAsset(at.BuyAndHold(), {"bonds": 1.0}),
    "60_40": core,
    "60_40_sma10_filter": at.PerAsset(at.PriceAboveSMA(10), {"stocks": 0.6, "bonds": 0.4}),
    "tsmom_unscaled": trend_unscaled,
    "tsmom_volscaled": trend_scaled,
    "80%60_40+20%tsmom": at.Blend({core: 0.8, trend_scaled: 0.2}),
    "60_40+30%tsmom_overlay": at.Blend({core: 1.0, trend_scaled: 0.3}),
}
res = [at.run_backtest(px, s, name=n, **KW) for n, s in STRATS.items()]
t = at.compare(res)
print("\nFull period (lag 1, 10bp, cash = bond yield - 1.5%)")
print(t[["cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "exposure", "ann_turnover"]])
at.plot([x for x in res if x.name in ("stocks_100", "60_40", "tsmom_volscaled",
                                       "60_40+30%tsmom_overlay")],
        f"{OUT}/portfolios.png", "Stocks, bonds and trend, 1871-2026")

# Risk contribution: how much of each portfolio's risk comes from stocks?
print("\nShare of portfolio variance from stocks (avg):")
for x in res:
    if x.name in ("60_40", "tsmom_unscaled", "tsmom_volscaled"):
        w = x.positions
        contrib = (w["stocks"] * r["stocks"]).reindex(w.index).fillna(0)
        tot = x.gross_returns
        print(f"  {x.name:16s} {contrib.cov(tot) / tot.var():.0%}")

print("\nCrises (total return over the window):")
crises = {"1929-32 crash": ("1929-09", "1932-06"), "1937-42": ("1937-03", "1942-04"),
          "1966-82 stagflation": ("1966-01", "1981-09"), "1973-74": ("1973-01", "1974-09"),
          "2000-02 dot-com": ("2000-09", "2002-09"), "2008 GFC": ("2007-11", "2009-02"),
          "2022 inflation": ("2022-01", "2022-10")}
ct = pd.DataFrame({k: {x.name: at.metrics.total_return(x.returns[a:b]) for x in res}
                   for k, (a, b) in crises.items()})
print(ct.T[["stocks_100", "60_40", "60_40_sma10_filter", "tsmom_volscaled",
            "60_40+30%tsmom_overlay"]])

# Diagnosis of the 1966-82 loss: the bond trend whipsawed while levered ~1.2x, and the
# high turnover was expensive. Futures cost ~1-2bp, so check cost sensitivity too.
for bp in (2, 10):
    x = at.run_backtest(px, trend_scaled, **{**KW, "cost_bps": bp})
    print(f"tsmom_volscaled at {bp:2d}bp: Sharpe {x.stats['sharpe']:.2f}, "
          f"1966-81 return {at.metrics.total_return(x.returns['1966':'1981-09']):+.0%}")
bond_trend = at.run_backtest(px["bonds"]["1966":"1981"], tsmom, execution_lag=1)
print(f"bond-only trend Sharpe 1966-81: {bond_trend.stats['sharpe']:.2f}")

print("\nCorrelation of tsmom_volscaled with stocks (monthly): "
      f"{res[5].returns.corr(r['stocks']):+.2f}")
print(f"Plot in {OUT}/")
