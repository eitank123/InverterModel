"""Experiment 9: how robust are the candidates, what did they earn each year, and what
should we realistically expect over the next 10 years?

Candidates are FROZEN here (no more tuning on this data):
  S&P 500, 60/40, 60/40 + SMA-10 filter, value tilt + SMA-10, 60/40 + SMA-10 at 1.5x.

  A. Annual profits: calendar years 2000-2026 YTD, decades since 1900, trailing periods.
  B. Stationary block bootstrap (Politis & Romano 1994): Sharpe CIs and P(better than 60/40).
  C. Parameter plateau: SMA window x stock weight.
  D. Forward 10 years: block-bootstrap simulation, historical vs valuation-adjusted means.

    python scripts/experiments/exp09_robustness_and_profits.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 80)
pd.set_option("display.float_format", "{:.1%}".format)
OUT = "results/exp09"
os.makedirs(OUT, exist_ok=True)

df = at.data.load_shiller_sp500()
px = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
cash = (df["long_rate"] - 0.015).clip(lower=0)
KW = dict(cost_bps=10, execution_lag=1, borrow_spread=0.01)

infl10 = (df["cpi"] / df["cpi"].shift(120)) ** (1 / 10) - 1
ecy = 1 / df["cape"].shift(6) - (df["long_rate"] - infl10).shift(6)
pct = ecy.expanding(min_periods=240).rank(pct=True).ffill()


def make(stock_w=0.6, window=10):
    return at.TrendFilter(at.PerAsset(at.BuyAndHold(), {"stocks": stock_w, "bonds": 1 - stock_w}),
                    at.PriceAboveSMA(window))


CANDIDATES = {
    "S&P500": at.PerAsset(at.BuyAndHold(), {"stocks": 1.0}),
    "60/40": at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4}),
    "60/40+SMA10": make(),
    "value+SMA10": at.TrendFilter(at.ValueTilt(pct), at.PriceAboveSMA(10)),
    "60/40+SMA10 x1.5": at.Levered(make(), 1.5),
}
res = {n: at.run_backtest(px, s, cash_rate=cash, name=n, **KW) for n, s in CANDIDATES.items()}
R = pd.DataFrame({n: r.returns for n, r in res.items()})["1901":]

# --- A. Annual profits ---------------------------------------------------------------
yearly = (1 + R).groupby(R.index.year).prod() - 1
print("A1. Calendar-year returns 2000-2026 (2026 = Jan-Sep)")
print(yearly.loc[2000:])
print("\nA2. Average annual return (CAGR) by decade")
dec = R.groupby(R.index.year // 10 * 10).apply(lambda x: (1 + x).prod() ** (12 / len(x)) - 1)
dec.index = [f"{d}s" for d in dec.index]
print(dec)
print("\nA3. Trailing annualised returns to", R.index[-1].date())
trail = {}
for yrs in (1, 3, 5, 10, 20, 50, 100):
    sub = R.iloc[-12 * yrs:]
    trail[f"{yrs}y"] = (1 + sub).prod() ** (1 / yrs) - 1
print(pd.DataFrame(trail).T)
print("\n   Worst calendar year:", {n: f"{yearly[n].min():.0%} ({yearly[n].idxmin()})" for n in R})
print("   Losing years (%):", {n: f"{(yearly[n] < 0).mean():.0%}" for n in R})

# --- B. Bootstrap ----------------------------------------------------------------------
pd.set_option("display.float_format", "{:.3f}".format)
print("\nB. Stationary block bootstrap (mean block 24 months, 1000 resamples), 1901-2026")
sharpe = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
boot = at.stats.bootstrap_stat(R, lambda d: d.apply(sharpe), n_boot=1000, mean_block=24)
boot = pd.DataFrame(list(boot), columns=R.columns)
summary = pd.DataFrame({"sharpe": R.apply(sharpe), "ci_2.5%": boot.quantile(.025),
                        "ci_97.5%": boot.quantile(.975),
                        "P(beats 60/40)": (boot.sub(boot["60/40"], axis=0) > 0).mean()})
print(summary)

# --- C. Parameter plateau -----------------------------------------------------------
print("\nC. Sharpe (and max drawdown) of stock_weight/40-bond + SMA filter, by window")
grid_sr, grid_dd = {}, {}
for w in (0.4, 0.5, 0.6, 0.7, 0.8):
    for win in (4, 6, 8, 10, 12, 14, 16):
        r = at.run_backtest(px["1901":], make(w, win), cash_rate=cash["1901":], **KW)
        grid_sr[(w, win)] = r.stats["sharpe"]
        grid_dd[(w, win)] = r.stats["max_drawdown"]
print(pd.Series(grid_sr).unstack())
print(pd.Series(grid_dd).unstack())

# --- D. Forward-looking 10-year simulation --------------------------------------------
print("\nD. Next 10 years: 1000 block-bootstrap paths of history (stocks, bonds, cash jointly)")
hist = pd.DataFrame({"stocks": px["stocks"].pct_change(), "bonds": px["bonds"].pct_change(),
                     "cash": cash / 12})["1901":].dropna()
mu_hist = hist.mean() * 12
# Valuation-adjusted: CAPE ~41 -> earnings yield ~2.4% real; + ~2.5% inflation = ~5% nominal.
# Bonds: expected return ~ today's 10y yield. Cash ~ today's short rate proxy.
target = pd.Series({"stocks": 0.05, "bonds": df["long_rate"].iloc[-1],
                    "cash": max(df["long_rate"].iloc[-1] - 0.015, 0)})
print(f"   historical means: {mu_hist.round(3).to_dict()}; valuation-adjusted: {target.round(3).to_dict()}")
rng = np.random.default_rng(7)
warm = px.iloc[-24:]
H = 120
rows = []
for scen in ("historical", "valuation-adjusted"):
    adj = hist.copy()
    if scen != "historical":
        adj = adj + (target - mu_hist) / 12
    arr = adj.to_numpy()
    for i in range(1000):
        idx = at.stats.stationary_bootstrap(len(arr), 24, rng)[:H]
        sim = arr[idx]
        dates = pd.date_range(warm.index[-1] + pd.offsets.MonthBegin(), periods=H, freq="MS")
        sim_px = pd.DataFrame(np.cumprod(1 + sim[:, :2], axis=0) * warm.iloc[-1].to_numpy(),
                              index=dates, columns=["stocks", "bonds"])
        path = pd.concat([warm, sim_px])
        c = pd.Series(np.r_[np.full(len(warm), sim[0, 2]), sim[:, 2]] * 12, index=path.index)
        for n, s in CANDIDATES.items():
            r = at.run_backtest(path, s, cash_rate=c, **KW).returns.iloc[-H:]
            rows.append({"scenario": scen, "strategy": n, "cagr": (1 + r).prod() ** 0.1 - 1,
                         "max_dd": at.metrics.max_drawdown(r)})
sim = pd.DataFrame(rows)
pd.set_option("display.float_format", "{:.1%}".format)
q = sim.groupby(["scenario", "strategy"]).agg(
    cagr_median=("cagr", "median"), cagr_5pct=("cagr", lambda x: x.quantile(.05)),
    cagr_95pct=("cagr", lambda x: x.quantile(.95)), p_loss_10y=("cagr", lambda x: (x < 0).mean()),
    maxdd_median=("max_dd", "median"), p_dd_over_30=("max_dd", lambda x: (x < -0.3).mean()))
print(q)
sim.to_csv(f"{OUT}/forward_sims.csv", index=False)
