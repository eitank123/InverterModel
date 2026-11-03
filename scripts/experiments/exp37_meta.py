"""Experiment 37: one algorithm over everything that has worked - a meta-allocator across
independent strategy sleeves, with a research loop of rounds (test -> infer -> improve -> retest).

Sleeves (monthly returns, 1972-2026, every cost; leverage beyond 1x financed at T-bill + 1.5% unless
the sleeve holds leveraged ETFs):
  champion_live   the champion as it trades live (month-end closes) with the 2% band (Exp 34)
  diversified     industry momentum + stock momentum + value legs, champion signals, x2/3 (Exp 31)
  risk_parity     5 asset classes, inverse-vol, trend-filtered, 10% vol (Exp 35, R5)
  tsmom           time-series momentum per asset class, 10% vol each (Exp 35, R3)
  factor          market-neutral momentum (vol-scaled) + value long/short, 1x notional (Exp 31, S2)
  gold_trend      gold above its 10-month average, else T-bills
Combination methods:
  EW     fixed equal weight (DeMiguel-Garlappi-Uppal 2009: 1/N is hard to beat out of sample)
  IV     inverse volatility (trailing 36 months)
  KELLY  growth-optimal weights from expanding-window means/covariance, half-Kelly, long-only, sum 1
  MOM    sleeve momentum: equal weight over sleeves with positive 12-month excess return (Gupta-Kelly
         2019 'factor momentum'), else the best of cash/EW
  NN     walk-forward neural network (5 small MLPs) predicting each sleeve's next-month return from its
         own past returns/vol/drawdown and the macro state (CAPE, 10y yield and its 12-month change,
         market trend); weights proportional to max(prediction, 0) / variance, half blended with EW
         (Zhang-Zohren-Roberts 2020 'Deep learning for portfolio optimization' motivates the attempt)
Loop: decisions use TRAIN 1972-1999 only; TEST 2000-2026 is reported for every round, never used.
  Round 1  all sleeves, all five combination methods
  Round 2  leave-one-out on TRAIN: drop any sleeve whose removal raises the EW combo's train
           Sharpe ratio; re-run
  Round 3  lever the best-on-train combo to the champion's TRAIN volatility (one scale factor fixed on
           TRAIN), so growth is compared at equal risk
Results are appended to loop/meta_rounds.json.

    python scripts/experiments/exp37_meta.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import algotrading as at  # noqa: E402
import exp31_combinations as e  # noqa: E402
import exp35_multi_asset as m  # noqa: E402
from algotrading.ml import EnsembleMLP  # noqa: E402
from loop import ensemble  # noqa: E402

START, TRAIN_END = "1972-01-01", "1999-12-01"
RF = e.RF


# ---------------------------------------------------------------- sleeves -----------------------
def champion_live():
    lvl = (1 + e.MKT).cumprod()
    bonds = e.idx["bonds"].copy()
    bonds.index = bonds.index.to_period("M").to_timestamp()
    idx_me = pd.DataFrame({"stocks": lvl, "bonds": bonds.reindex(lvl.index).ffill()}).dropna()
    W = ensemble(3.0, band=0.02).weights(idx_me).shift(1).reindex(e.DATES).fillna(0)
    g = W["stocks"] + W["bonds"]
    return (W["stocks"] * e.MKT.reindex(e.DATES) + W["bonds"] * e.RB.reindex(e.DATES) + (1 - g) * RF.reindex(e.DATES)
            - (g - 1).clip(lower=0) * 0.005 / 12 - W.diff().abs().sum(axis=1).fillna(0) * 0.001)


def financed(W, rets):
    W = W.shift(1).reindex(rets.index).fillna(0)
    g = W.sum(axis=1)
    rf = RF.reindex(rets.index).fillna(0)
    return ((W * rets.fillna(0)).sum(axis=1) + (1 - g) * rf - (g - 1).clip(lower=0) * 0.015 / 12
            - W.diff().abs().sum(axis=1).fillna(0) * 0.001)


def gold_trend():
    g = m.R["Gold"]
    lv = (1 + g.fillna(0)).cumprod()
    on = (lv > lv.rolling(10).mean()).astype(float).shift(1).fillna(0)
    rf = RF.reindex(g.index).fillna(0)
    return on * g.fillna(0) + (1 - on) * rf - on.diff().abs().fillna(0) * 0.001


def sleeves():
    fs = e.factor_sleeve()
    S = pd.DataFrame({
        "champion_live": champion_live(),
        "diversified": e.portfolio(e.LEGS["diversified (ind mom + stock mom + value)"], scale=2 / 3),
        "risk_parity": financed(m.W5, m.R),
        "tsmom": financed(m.W3, m.R),
        "factor": 2 * (fs + 0.002) - 0.002 + RF,
        "gold_trend": gold_trend(),
    })
    return S[START:"2026-09"].dropna()


# ---------------------------------------------------------------- combiners ---------------------
def combine(S, W):
    """W: weights decided at the end of month t-1 (index = month t). Monthly rebalanced, with 10bp
    per unit of sleeve weight changed (sleeves' own trading costs are already inside S)."""
    W = W.reindex(S.index).ffill().fillna(1 / S.shape[1])
    r = (W * S).sum(axis=1) - W.diff().abs().sum(axis=1).fillna(0) * 0.001
    return r


def w_ew(S):
    return pd.DataFrame(1 / S.shape[1], index=S.index, columns=S.columns)


def w_iv(S):
    v = S.rolling(36, min_periods=12).std().shift(1)
    iv = 1 / v
    return iv.div(iv.sum(axis=1), axis=0).fillna(1 / S.shape[1])


def w_kelly(S):
    rf = RF.reindex(S.index).fillna(0)
    X = S.sub(rf, axis=0)
    out = []
    for i in range(len(S)):
        if i < 36:
            out.append(np.repeat(1 / S.shape[1], S.shape[1]))
            continue
        h = X.iloc[:i]
        mu, cov = h.mean().to_numpy(), h.cov().to_numpy()
        cov = 0.5 * cov + 0.5 * np.diag(np.diag(cov))                 # shrink correlations
        w = np.clip(0.5 * np.linalg.solve(cov, mu), 0, None)
        out.append(w / w.sum() if w.sum() > 0 else np.repeat(1 / S.shape[1], S.shape[1]))
    return pd.DataFrame(out, index=S.index, columns=S.columns)


def w_mom(S):
    rf = RF.reindex(S.index).fillna(0)
    ex = (1 + S).rolling(12).apply(np.prod, raw=True).sub((1 + rf).rolling(12).apply(np.prod, raw=True), axis=0).shift(1)
    pos = (ex > 0).astype(float)
    n = pos.sum(axis=1)
    return pos.div(n.replace(0, np.nan), axis=0).fillna(1 / S.shape[1])


def macro():
    df = e.df
    cape = df["cape"] if "cape" in df else df["price"] / df["earnings"].rolling(120).mean()
    y = df["long_rate"]
    mk = pd.DataFrame({"cape": np.log(cape), "y10": y, "dy12": y.diff(12)})
    mk.index = mk.index.to_period("M").to_timestamp()
    lvl = (1 + e.MKT).cumprod()
    mk["mkt_trend"] = (lvl > lvl.rolling(10).mean()).astype(float).reindex(mk.index)
    return mk


def w_nn(S):
    rf = RF.reindex(S.index).fillna(0)
    mk = macro().reindex(S.index).ffill()
    feats, names = {}, list(S.columns)
    for c in names:
        x = S[c]
        lv = (1 + x).cumprod()
        feats[c] = pd.DataFrame({"r1": x, "r3": x.rolling(3).sum(), "r6": x.rolling(6).sum(), "r12": x.rolling(12).sum(),
                                 "vol12": x.rolling(12).std(), "dd": lv / lv.cummax() - 1,
                                 "id": float(names.index(c))}).join(mk)
    target = S.shift(-1)                                              # next month's return
    preds = pd.DataFrame(np.nan, index=S.index, columns=names)
    for y0 in range(1980, 2027, 5):
        lo, hi = pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y0 + 5}-01-01")
        tr = S.index[(S.index < lo - pd.DateOffset(months=1))]
        Xtr = np.vstack([feats[c].loc[tr].to_numpy() for c in names])
        ytr = np.concatenate([target[c].loc[tr].to_numpy() for c in names])
        ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
        if ok.sum() < 200:
            continue
        mu, sd = Xtr[ok].mean(0), Xtr[ok].std(0) + 1e-12
        model = EnsembleMLP(hidden=(8,), alpha=1.0).fit((Xtr[ok] - mu) / sd, ytr[ok])
        te = S.index[(S.index >= lo) & (S.index < hi)]
        for c in names:
            X = feats[c].loc[te].to_numpy()
            good = np.isfinite(X).all(1)
            p = np.full(len(te), np.nan)
            if good.any():
                p[good] = model.predict((X[good] - mu) / sd)
            preds.loc[te, c] = p
    var = S.rolling(36, min_periods=12).var()
    raw = (preds.sub(rf, axis=0).clip(lower=0) / var).fillna(0)
    w = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0)
    w = w.fillna(1 / S.shape[1])
    return (0.5 * w + 0.5 / S.shape[1]).shift(1)                      # decided with data to t-1


def w_hist_mean(S):
    """Same mapping as NN but with expanding historical means as the forecast (control)."""
    rf = RF.reindex(S.index).fillna(0)
    mu = S.expanding(36).mean()
    var = S.rolling(36, min_periods=12).var()
    raw = (mu.sub(rf, axis=0).clip(lower=0) / var).fillna(0)
    w = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(1 / S.shape[1])
    return (0.5 * w + 0.5 / S.shape[1]).shift(1)


# ---------------------------------------------------------------- evaluation --------------------
def stats(r):
    rf = RF.reindex(r.index).fillna(0)
    out = {}
    for lab, (a, b) in {"TRAIN 1972-99": ("1972", "1999"), "test 2000-12": ("2000", "2012"),
                        "test 2013-26": ("2013", "2026"), "TEST 2000-26": ("2000", "2026")}.items():
        out[lab] = at.metrics.cagr(r[a:b], 12)
    ex = (r - rf)
    out["train Sharpe"] = float(ex[:TRAIN_END].mean() / ex[:TRAIN_END].std() * np.sqrt(12))
    out["test Sharpe"] = float(ex["2000":].mean() / ex["2000":].std() * np.sqrt(12))
    w = (1 + r["2000":]).cumprod()
    out["TEST max DD"] = float((w / w.cummax() - 1).min())
    out["2025-26"] = at.metrics.cagr(r["2025":], 12)
    return out


def table(d):
    t = pd.DataFrame({k: stats(v) for k, v in d.items()}).T
    return t.to_string(formatters={c: ("{:.2f}".format if "Sharpe" in c else "{:.1%}".format) for c in t.columns})


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    S = sleeves()
    log = {"date": pd.Timestamp.today().strftime("%Y-%m-%d"), "rounds": []}
    print(f"Sleeves {S.index[0]:%Y-%m} to {S.index[-1]:%Y-%m}; correlations (TRAIN):\n{S[:TRAIN_END].corr().round(2).to_string()}\n")
    print("Each sleeve alone:\n" + table({c: S[c] for c in S.columns}))

    def run_round(S, label):
        combos = {"EW": combine(S, w_ew(S)), "IV": combine(S, w_iv(S)), "KELLY": combine(S, w_kelly(S)),
                  "MOM": combine(S, w_mom(S)), "HIST-MEAN (control)": combine(S, w_hist_mean(S)),
                  "NN": combine(S, w_nn(S))}
        print(f"\n{label} - sleeves: {', '.join(S.columns)}\n" + table(combos))
        log["rounds"].append({"label": label, "sleeves": list(S.columns),
                              "results": {k: stats(v) for k, v in combos.items()}})
        return combos

    c1 = run_round(S, "Round 1: all sleeves, five combiners")

    # Round 2: leave-one-out on TRAIN only (EW combo Sharpe).
    rf = RF.reindex(S.index).fillna(0)

    def sharpe_train(r):
        x = (r - rf)[:TRAIN_END]
        return float(x.mean() / x.std() * np.sqrt(12))

    base = sharpe_train(combine(S, w_ew(S)))
    drop = [c for c in S.columns if sharpe_train(combine(S.drop(columns=c), w_ew(S.drop(columns=c)))) > base]
    print(f"\nRound 2 inference (TRAIN only): EW train Sharpe {base:.2f}; dropping {drop or 'nothing'}")
    S2 = S.drop(columns=drop) if drop else S
    c2 = run_round(S2, f"Round 2: leave-one-out on train (dropped {drop or 'none'})")

    # Round 3: pick the combiner with the best TRAIN Sharpe in round 2; lever it to the champion's
    # TRAIN volatility with one fixed factor (financed at T-bill + 1.5%).
    best = max(c2, key=lambda k: sharpe_train(c2[k]))
    r = c2[best]
    k = float(S["champion_live"][:TRAIN_END].std() / r[:TRAIN_END].std())
    levered = k * r - (k - 1) * (rf + 0.015 / 12)
    print(f"\nRound 3 inference (TRAIN only): best combiner {best}; scale x{k:.2f} to match the champion's train volatility")
    d3 = {f"{best} x{k:.2f} (vol-matched)": levered, f"{best} x1": r, "champion_live alone": S["champion_live"],
          "diversified alone": S["diversified"]}
    print(table(d3))
    log["rounds"].append({"label": f"Round 3: {best} levered x{k:.2f}", "results": {kk: stats(v) for kk, v in d3.items()}})
    # Round 4 (inference from rounds 1-3): Sharpe is the wrong yardstick for the owner's goal (it
    # dropped the growth engines), the NN was no better than its historical-mean control, and the
    # factor sleeve decayed after publication. So: keep the stock sleeves as the core, add the
    # diversifiers as a satellite levered 2x (a margin account can hold it), fixed round weights, and
    # choose on TRAIN by the WORST of two train halves (1972-85, 1986-99) - the champion rule.
    div3 = S[["risk_parity", "tsmom", "gold_trend"]].mean(axis=1)
    sat = 2 * div3 - (rf + 0.015 / 12)
    cands = {"champion_live alone": S["champion_live"], "diversified alone": S["diversified"],
             "50% champion + 50% diversified": 0.5 * S["champion_live"] + 0.5 * S["diversified"],
             "50% champion + 50% satellite(2x)": 0.5 * S["champion_live"] + 0.5 * sat,
             "2/3 champion + 1/3 satellite(2x)": 2 / 3 * S["champion_live"] + 1 / 3 * sat,
             "50% diversified + 50% satellite(2x)": 0.5 * S["diversified"] + 0.5 * sat,
             "1/3 champion + 1/3 diversified + 1/3 satellite(2x)": (S["champion_live"] + S["diversified"] + sat) / 3,
             "EW all six x1.5": 1.5 * c1["EW"] - 0.5 * (rf + 0.015 / 12)}

    def worst_half(r):
        return min(at.metrics.cagr(r["1972":"1985"], 12), at.metrics.cagr(r["1986":"1999"], 12))

    pick = max(cands, key=lambda k: worst_half(cands[k]))
    print("\nRound 4: core + satellite, chosen by the worst train half\n" + table(cands))
    print("  train worst half: " + ", ".join(f"{k} {worst_half(v):.1%}" for k, v in cands.items()))
    print(f"  -> chosen on TRAIN: {pick}")
    log["rounds"].append({"label": "Round 4: core + satellite", "pick": pick,
                          "results": {kk: stats(v) for kk, v in cands.items()}})
    # Round 5 (owner: 'give each algorithm an amount of money chosen by a NN or other models'):
    # the money split between the core (champion) and the satellite decided every month by each
    # combiner, against the fixed 2/3-1/3 picked in round 4.
    two = pd.DataFrame({"champion_live": S["champion_live"], "satellite": sat}).dropna()
    fixed = pd.DataFrame({"champion_live": 2 / 3, "satellite": 1 / 3}, index=two.index)
    c5 = {"fixed 2/3 + 1/3 (round 4)": combine(two, fixed), "EW 50/50": combine(two, w_ew(two)),
          "IV": combine(two, w_iv(two)), "KELLY": combine(two, w_kelly(two)), "MOM": combine(two, w_mom(two)),
          "HIST-MEAN (control)": combine(two, w_hist_mean(two)), "NN": combine(two, w_nn(two))}
    print("\nRound 5: the money split between champion and satellite chosen every month\n" + table(c5))
    nnw = w_nn(two)["champion_live"]["2000":]
    print(f"  NN's share for the champion 2000-2026: mean {nnw.mean():.0%}, range {nnw.min():.0%}-{nnw.max():.0%}")
    log["rounds"].append({"label": "Round 5: dynamic core/satellite split", "results": {kk: stats(v) for kk, v in c5.items()}})
    path = os.path.join(ROOT, "loop", "meta_rounds.json")
    hist = json.load(open(path)) if os.path.exists(path) else []
    hist.append(log)
    json.dump(hist, open(path, "w"), indent=1, default=float)
