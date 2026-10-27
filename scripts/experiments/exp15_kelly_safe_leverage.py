"""Experiment 15: use Kelly leverage only where it is "safe": calm assets, calm times,
uptrends. Can it beat the 60/40 + SMA10 x1.5 baseline without a bigger drawdown?

Math: multi-asset Kelly w = lambda * Sigma^-1 mu (lambda = Kelly fraction). Because
Sigma^-1 scales with 1/variance, Kelly naturally levers the calmer asset (bonds) and
calm periods. Setting mu = 0 when an asset is below its 10-month SMA switches leverage off
in downtrends. The open question is how to estimate mu:
  K1  literature priors: equity premium 5%/yr, term premium 1.5%/yr (Dimson-Marsh-Staunton)
  K2  valuation: stocks mu = excess CAPE yield (Shiller), bonds 1.5%
  K3  ridge regression on macro features (walk-forward)
  K4  small neural network ensemble (walk-forward)
All with the trend gate. Plus risk-parity baselines.

Protocol: lambda and the gross cap are chosen on TRAIN 1957-1990 by the rule
"highest CAGR with max drawdown no worse than -25%"; TEST 1991-2026 is reported once.
Leverage costs cash + 1%/yr; 10bp per trade; signals traded one month late.

    python scripts/experiments/exp15_kelly_safe_leverage.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", "{:.3f}".format)
OUT = "results/exp15"
os.makedirs(OUT, exist_ok=True)

df = at.data.load_shiller_sp500()
px = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
cash = (df["long_rate"] - 0.015).clip(lower=0)
KW = dict(cost_bps=10, execution_lag=1, borrow_spread=0.01, cash_rate=cash)
TRAIN, TEST, FULL = ("1957", "1990"), ("1991", "2026"), ("1957", "2026")
DD_LIMIT = -0.25

trend = pd.DataFrame({c: at.PriceAboveSMA(10).positions(px[c]) for c in px})
excess = px.pct_change().sub(cash / 12, axis=0)

# --- mu estimators (annual excess return forecasts, point-in-time) --------------------
mu = {}
prior = pd.DataFrame({"stocks": 0.05, "bonds": 0.015}, index=px.index)
mu["K1 priors"] = prior * trend
feats = at.ml.macro_features(df, px)
ecy = feats["ecy"].clip(0, 0.10)
mu["K2 valuation"] = pd.DataFrame({"stocks": ecy, "bonds": 0.015}, index=px.index) * trend

from sklearn.linear_model import Ridge  # noqa: E402

ml_preds = {}
for name, factory in [("ridge", lambda: Ridge(alpha=30.0)), ("nn", lambda: at.ml.EnsembleMLP())]:
    cache = f"{OUT}/ml_preds_{name}.csv"
    if os.path.exists(cache):
        ml_preds[name] = pd.read_csv(cache, index_col=0, parse_dates=True)
        continue
    cols = {}
    for asset in px:
        y = excess[asset].shift(-2)                     # return earned by a weight set at t
        cols[asset] = at.ml.walk_forward(factory, feats.drop(columns=["log_cape"]), y,
                                         start="1965", horizon=2, min_train=120,
                                         refit_every=12, train_from="1954") * 12
    ml_preds[name] = pd.DataFrame(cols)
    ml_preds[name].to_csv(cache)
mu["K3 ridge"] = ml_preds["ridge"] * trend
mu["K4 neural net"] = ml_preds["nn"] * trend

# Diagnostic: do the ML forecasts beat the expanding historical mean (Goyal-Welch)?
print("A. Out-of-sample R^2 of next-month excess-return forecasts vs expanding mean (1965-2026)")
for name, P in ml_preds.items():
    for asset in px:
        y = excess[asset].shift(-2)
        d = pd.DataFrame({"y": y, "f": P[asset] / 12}).dropna()
        bench = y.expanding(120).mean().shift(2).reindex(d.index)
        r2 = 1 - ((d.y - d.f) ** 2).sum() / ((d.y - bench) ** 2).sum()
        hit = (np.sign(d.f) == np.sign(d.y)).mean()
        print(f"   {name:6s} {asset:7s} OOS R2 {r2:+.2%}   sign hit-rate {hit:.1%}")

# --- Candidate strategies -------------------------------------------------------------
core = at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4})
filt = at.TrendFilter(core, at.PriceAboveSMA(10))


def risk_parity(target):
    return at.PortfolioVolTarget(at.TrendFilter(at.InverseVol(36), at.PriceAboveSMA(10)),
                                 target, 36, 3.0)


def evaluate(strategy):
    r = at.run_backtest(px, strategy, **KW)
    out = {}
    for pname, (a, b) in {"train": TRAIN, "test": TEST, "full": FULL}.items():
        x = r.returns[a:b]
        ex = x - cash.reindex(x.index) / 12
        out[pname] = {"cagr": at.metrics.cagr(x), "max_dd": at.metrics.max_drawdown(x),
                      "excess_sharpe": at.metrics.sharpe(ex),
                      "avg_gross": r.positions[a:b].sum(axis=1).mean()}
    return out, r


# K5: literature priors, but Sigma floored at long-run vols (stocks 15%, bonds 7%).
FLOOR = {"stocks": 0.15, "bonds": 0.07}
floors = {name: None for name in mu}
mu["K5 priors+vol floor"] = mu["K1 priors"]
floors["K5 priors+vol floor"] = FLOOR

# Select lambda / cap on TRAIN only.
print("\nB. Selecting Kelly fraction and cap on TRAIN 1957-1990 (max DD must be >= "
      f"{DD_LIMIT:.0%})")
selected = {}
for name, m in mu.items():
    best = None
    for frac in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0):
        for cap in (1.0, 1.5, 2.0, 3.0):
            o, _ = evaluate(at.KellyPortfolio(m, frac, 36, cap, floors[name]))
            ok = o["train"]["max_dd"] >= DD_LIMIT
            if ok and (best is None or o["train"]["cagr"] > best[2]):
                best = (frac, cap, o["train"]["cagr"])
    if best is None:
        print(f"   {name:15s} -> no setting meets the drawdown limit on train: rejected")
        continue
    selected[name] = best
    print(f"   {name:15s} -> fraction {best[0]}, cap {best[1]} (train CAGR {best[2]:.1%})")
best_rp = None
for tv in (0.05, 0.06, 0.08, 0.10, 0.12):
    o, _ = evaluate(risk_parity(tv))
    if o["train"]["max_dd"] >= DD_LIMIT and (best_rp is None or o["train"]["cagr"] > best_rp[1]):
        best_rp = (tv, o["train"]["cagr"])
print(f"   risk parity     -> vol target {best_rp[0]:.0%} (train CAGR {best_rp[1]:.1%})")
best_lev = None
for lev in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
    o, _ = evaluate(at.Levered(filt, lev))
    if o["train"]["max_dd"] >= DD_LIMIT and (best_lev is None or o["train"]["cagr"] > best_lev[1]):
        best_lev = (lev, o["train"]["cagr"])
print(f"   60/40+SMA10 xL  -> leverage {best_lev[0]} (train CAGR {best_lev[1]:.1%})")
# "Lever only when calm AND trending up": vol-target the trend-filtered 60/40.
best_vt = None
for tv in (0.06, 0.08, 0.10, 0.12, 0.14):
    for cap in (1.5, 2.0, 3.0):
        o, _ = evaluate(at.PortfolioVolTarget(filt, tv, 12, cap))
        if o["train"]["max_dd"] >= DD_LIMIT and (best_vt is None or o["train"]["cagr"] > best_vt[2]):
            best_vt = (tv, cap, o["train"]["cagr"])
print(f"   60/40+SMA10 vol-target -> {best_vt[0]:.0%} vol, cap {best_vt[1]} (train CAGR {best_vt[2]:.1%})")

FINAL = {
    "60/40": core,
    "60/40+SMA10": filt,
    "60/40+SMA10 x1.5 (baseline)": at.Levered(filt, 1.5),
    f"60/40+SMA10 x{best_lev[0]} (train-picked)": at.Levered(filt, best_lev[0]),
    f"risk parity+SMA {best_rp[0]:.0%} vol": risk_parity(best_rp[0]),
    f"60/40+SMA10 vol-target {best_vt[0]:.0%} cap {best_vt[1]}":
        at.PortfolioVolTarget(filt, best_vt[0], 12, best_vt[1]),
}
for name, (frac, cap, _) in selected.items():
    FINAL[f"{name} f={frac} cap={cap}"] = at.KellyPortfolio(mu[name], frac, 36, cap, floors[name])

rows, rets = [], {}
for name, s in FINAL.items():
    o, r = evaluate(s)
    rets[name] = r.returns
    for p, v in o.items():
        rows.append({"strategy": name, "period": p, **v})
t = pd.DataFrame(rows).pivot(index="strategy", columns="period")
print("\nC. Results (train = selection period, TEST = untouched)")
print(t[[("cagr", "train"), ("cagr", "test"), ("cagr", "full"), ("max_dd", "test"),
         ("max_dd", "full"), ("excess_sharpe", "test"), ("excess_sharpe", "full"),
         ("avg_gross", "full")]])

R = pd.DataFrame(rets)["1991":]
base = "60/40+SMA10 x1.5 (baseline)"
sh = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
ex = R.sub(cash.reindex(R.index) / 12, axis=0)
boot = pd.DataFrame(list(at.stats.bootstrap_stat(ex, lambda d: d.apply(sh), 500, 24)),
                    columns=R.columns)
cg = pd.DataFrame(list(at.stats.bootstrap_stat(R, lambda d: d.apply(at.metrics.cagr, ppy=12), 500, 24)),
                  columns=R.columns)
print("\nD. TEST 1991-2026 bootstrap vs the x1.5 baseline")
print(pd.DataFrame({"P(higher excess Sharpe)": (boot.sub(boot[base], axis=0) > 0).mean(),
                    "P(higher CAGR)": (cg.sub(cg[base], axis=0) > 0).mean()}).drop(base))

print("\nE. Calendar-year returns, last 6 years (test period)")
yr = (1 + R).groupby(R.index.year).prod() - 1
print(yr.loc[2021:].T.map(lambda v: f"{v:+.1%}"))
at.plot([at.run_backtest(px["1957":], s, **{**KW, "cash_rate": cash["1957":]}, name=n)
         for n, s in FINAL.items() if "60/40" in n or "K1" in n or "K4" in n],
        f"{OUT}/kelly_strategies.png", "Kelly-guided leverage, 1957-2026")

# --- F. Is "lever when calm" better than plain leverage at the same risk? -------------
print("\nF. Vol-targeting vs a constant leverage with the same volatility (1957-2026)")
vt_s = at.PortfolioVolTarget(filt, best_vt[0], 12, best_vt[1])
vt = at.run_backtest(px, vt_s, **KW).returns["1957":]
plain = at.run_backtest(px, filt, **KW).returns["1957":]
k = vt.std() / plain.std()
ctrl = at.run_backtest(px, at.Levered(filt, k), **KW).returns["1957":]
both = pd.DataFrame({"vt": vt, "ctrl": ctrl})
cg2 = pd.DataFrame(list(at.stats.bootstrap_stat(both, lambda d: d.apply(at.metrics.cagr, ppy=12), 500, 24)),
                   columns=both.columns)
dd2 = pd.DataFrame(list(at.stats.bootstrap_stat(both, lambda d: d.apply(at.metrics.max_drawdown), 500, 24)),
                   columns=both.columns)
print(f"   vol-target: CAGR {at.metrics.cagr(vt):.1%}, DD {at.metrics.max_drawdown(vt):.0%} | "
      f"constant x{k:.2f}: CAGR {at.metrics.cagr(ctrl):.1%}, DD {at.metrics.max_drawdown(ctrl):.0%}")
print(f"   P(vol-target higher CAGR) {(cg2.vt > cg2.ctrl).mean():.0%}, "
      f"P(smaller DD) {(dd2.vt > dd2.ctrl).mean():.0%}  -> no reliable edge from timing leverage")

# --- G. What does borrowing cost do? --------------------------------------------------
print("\nG. Borrow spread over cash (futures ~0.2-0.5%, IBKR ~1.5%, typical retail margin ~3-6%)")
rows = []
for spread in (0.0, 0.005, 0.015, 0.03, 0.05):
    for name, s in [("60/40+SMA10 x1.5", at.Levered(filt, 1.5)),
                    ("60/40+SMA10 x2.0", at.Levered(filt, 2.0)),
                    (f"vol-target {best_vt[0]:.0%}", vt_s)]:
        r = at.run_backtest(px, s, **{**KW, "borrow_spread": spread}).returns["1957":]
        rows.append({"spread": f"{spread:.1%}", "strategy": name, "cagr": at.metrics.cagr(r)})
print(pd.DataFrame(rows).pivot(index="strategy", columns="spread", values="cagr")
      .map(lambda v: f"{v:.1%}"))
print(f"   unlevered 60/40+SMA10: {at.metrics.cagr(plain):.1%}; 60/40: "
      f"{at.metrics.cagr(at.run_backtest(px, core, **KW).returns['1957':]):.1%}")
