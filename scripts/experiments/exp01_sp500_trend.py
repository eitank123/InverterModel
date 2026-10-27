"""Experiment 1: does trend-following timing work on 155 years of the S&P 500?

Hypothesis (Faber 2007; Hurst, Ooi & Pedersen 2017): holding stocks only while they
trend up, and cash otherwise, gives similar returns to buy & hold with much smaller
drawdowns, i.e. a higher Sharpe.

    python scripts/experiments/exp01_sp500_trend.py
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
OUT = "results/exp01"
os.makedirs(OUT, exist_ok=True)

df = at.data.load_shiller_sp500()
tr = df["tr"]                       # total return index: we earn dividends when invested
rate = df["long_rate"]              # only rate series available; an *upper bound* for cash
COST = 10                           # bp per switch; history was costlier, ETFs today ~1-2bp

STRATS = {
    "buy_hold": at.BuyAndHold(),
    "faber_sma10": at.PriceAboveSMA(10),
    "mom12_long": at.Momentum(12, allow_short=False),
    "ensemble_1_3_12": at.TrendEnsemble((1, 3, 12), allow_short=False),
}


def table(lag: int, cash) -> pd.DataFrame:
    res = [at.run_backtest(tr, s, cost_bps=COST, execution_lag=lag, cash_rate=cash, name=n)
           for n, s in STRATS.items()]
    t = at.compare(res)
    # Sharpe in excess of cash (the textbook definition) using the same cash rate.
    rf = cash if isinstance(cash, pd.Series) else pd.Series(cash, index=tr.index)
    t["excess_sharpe"] = [at.metrics.sharpe(r.returns, risk_free=rf) for r in res]
    return t, res


print("=" * 100, "\nA. Full period", tr.index[0].date(), "->", tr.index[-1].date())
cols = ["cagr", "ann_vol", "sharpe", "excess_sharpe", "max_drawdown", "exposure", "ann_turnover"]
for lag in (0, 1):
    for cash_name, cash in [("cash=0%", 0.0), ("cash=bond yield", rate)]:
        t, res = table(lag, cash)
        print(f"\n-- execution_lag={lag}, {cash_name}")
        print(t[cols])
        if lag == 0 and cash_name != "cash=0%":
            at.plot(res, f"{OUT}/full_period.png", "S&P 500 total return, 1871-2026 (log)")
            main_results = res

# --- B. Robustness: is there a plateau of good parameters, or one lucky spike? ---------
print("\n" + "=" * 100, "\nB. Parameter robustness (lag=1, cash=bond yield, excess Sharpe)")
rows = []
for w in range(2, 25):
    for kind, strat in [("sma", at.PriceAboveSMA(w)), ("mom", at.Momentum(w, allow_short=False))]:
        r = at.run_backtest(tr, strat, cost_bps=COST, execution_lag=1, cash_rate=rate)
        rows.append({"kind": kind, "window": w,
                     "excess_sharpe": at.metrics.sharpe(r.returns, risk_free=rate),
                     "sharpe": r.stats["sharpe"], "max_dd": r.stats["max_drawdown"]})
rob = pd.DataFrame(rows)
bh = at.run_backtest(tr, at.BuyAndHold(), cash_rate=rate)
bh_ex = at.metrics.sharpe(bh.returns, risk_free=rate)
print(rob.pivot(index="window", columns="kind", values="excess_sharpe").T)
print(f"buy & hold excess Sharpe: {bh_ex:.3f}")
print("share of variants beating buy & hold:",
      f"{(rob['excess_sharpe'] > bh_ex).mean():.0%}")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(figsize=(9, 4))
for kind, g in rob.groupby("kind"):
    ax.plot(g["window"], g["excess_sharpe"], marker="o", label=kind)
ax.axhline(bh_ex, color="k", ls="--", label="buy & hold")
ax.set_xlabel("lookback (months)"); ax.set_ylabel("excess Sharpe"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Robustness: trend rules on S&P 500 1871-2026 (lag=1)")
fig.tight_layout(); fig.savefig(f"{OUT}/robustness.png", dpi=110)

# --- C. Deflated Sharpe: correct for having tried 46 variants -------------------------
best = rob.sort_values("sharpe").iloc[-1]
strat = (at.PriceAboveSMA if best["kind"] == "sma" else
         lambda w: at.Momentum(w, allow_short=False))(int(best["window"]))
best_ret = at.run_backtest(tr, strat, cost_bps=COST, execution_lag=1, cash_rate=rate).returns
n_trials = len(rob) + len(STRATS)
dsr = at.metrics.deflated_sharpe(best_ret, n_trials, rob["sharpe"].std())
print("\n" + "=" * 100)
print(f"C. Best variant {best['kind']}{int(best['window'])}: Sharpe {best['sharpe']:.2f}; "
      f"expected best-of-{n_trials} by luck "
      f"{at.metrics.expected_max_sharpe(n_trials, rob['sharpe'].std()):.2f} above the mean; "
      f"PSR(>0) {at.metrics.probabilistic_sharpe(best_ret):.3f}, DSR {dsr:.3f}")
# A harder test: is timing *better than buy & hold*, not just better than zero?
# Scale timing to buy & hold's volatility (a constant), then test the return difference.
scaled = best_ret * (bh.returns.std() / best_ret.std())
print(f"   PSR that vol-matched timing beats buy & hold: "
      f"{at.metrics.probabilistic_sharpe(scaled - bh.returns):.3f}")
# The answer hinges on what cash earned. Bills usually yield ~1-2% less than 10y bonds.
print("   pre-specified rules (not searched), vol-matched vs buy & hold, by cash assumption:")
for cash_name, cash in [("0%", 0.0), ("bond-1.5%", (rate - 0.015).clip(lower=0)),
                        ("bond yield", rate)]:
    out = []
    for n in ("faber_sma10", "ensemble_1_3_12"):
        r = at.run_backtest(tr, STRATS[n], cost_bps=COST, execution_lag=1, cash_rate=cash).returns
        sc = r * (bh.returns.std() / r.std())
        out.append(f"{n} {at.metrics.probabilistic_sharpe(sc - bh.returns):.3f}")
    print(f"     cash={cash_name:10s}  PSR: " + ",  ".join(out))

# --- D. Sub-periods: does it work in every era, incl. after publication (2007)? -------
print("\n" + "=" * 100, "\nD. Sub-periods (lag=1, cash=bond yield)")
periods = [("1871", "1928"), ("1929", "1949"), ("1950", "1999"), ("2000", "2026"),
           ("2007", "2026")]
rows = []
for a, b in periods:
    for n in ("buy_hold", "faber_sma10", "ensemble_1_3_12"):
        r = at.run_backtest(tr[a:b], STRATS[n], cost_bps=COST, execution_lag=1,
                            cash_rate=rate[a:b])
        rows.append({"period": f"{a}-{b}", "strategy": n, "cagr": r.stats["cagr"],
                     "sharpe": r.stats["sharpe"], "max_dd": r.stats["max_drawdown"]})
print(pd.DataFrame(rows).pivot(index="period", columns="strategy"))

# --- E. Crises -----------------------------------------------------------------------
print("\n" + "=" * 100, "\nE. Crisis drawdowns (lag=0, cash=bond yield)")
crises = {"1929 crash": ("1929-08", "1932-07"), "1973-74": ("1973-01", "1974-12"),
          "dot-com": ("2000-08", "2002-10"), "GFC": ("2007-10", "2009-03"),
          "COVID": ("2020-01", "2020-04"), "2022": ("2022-01", "2022-10"),
          "1987 crash": ("1987-08", "1987-12")}
rows = {}
for name, (a, b) in crises.items():
    rows[name] = {r.name: at.metrics.total_return(r.returns[a:b]) for r in main_results}
print(pd.DataFrame(rows).T)

# --- F. The averaging artifact, measured on synthetic data where the truth is known -----
print("\n" + "=" * 100, "\nF. Monthly *averaged* prices vs month-end prices (random walk, no edge)")
ac = df["close"][:"2023-06"].pct_change().autocorr()
print(f"   lag-1 autocorrelation of Shiller monthly returns 1871-2023: {ac:.3f}")
rows = []
for seed in range(20):
    daily = at.data.gbm(n=252 * 60, mu=0.07, sigma=0.16, seed=seed)["close"]
    monthly_end = daily.resample("MS").last()
    monthly_avg = daily.resample("MS").mean()
    for kind, px in [("month_end", monthly_end), ("month_avg", monthly_avg)]:
        for lag in (0, 1):
            r = at.run_backtest(px, STRATS["ensemble_1_3_12"], cost_bps=0, execution_lag=lag)
            rows.append({"prices": kind, "lag": lag, "autocorr": px.pct_change().autocorr(),
                         "sharpe_minus_bh": r.stats["sharpe"] - at.run_backtest(
                             px, at.BuyAndHold(), cost_bps=0).stats["sharpe"]})
print(pd.DataFrame(rows).groupby(["prices", "lag"]).mean())
print(f"\nPlots in {OUT}/")
