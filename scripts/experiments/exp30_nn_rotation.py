"""Experiment 30: a neural network that picks industries every day / week / month, after costs and
a 25% capital-gains tax, against plain industry momentum; and both as the champion's stock leg.

Data: Ken French 12 US industry portfolios, DAILY value-weighted total returns 1926-2026 (no
intraday/hourly data exists for 100 years; hourly is not testable here). Cash earns the T-bill.

Network: the loop's EnsembleMLP (5 small sklearn MLPs, 8 hidden units, strong L2), retrained every
5 years on all earlier data (walk-forward, expanding window; the first model is trained on
1927-1956, so results start in 1957). Inputs per industry and day, all known at that close:
log returns over the past 1, 5, 21, 63, 126 and 252 days (raw and minus the average industry),
63-day volatility, and two market inputs (market above its 200-day average; market 21-day return).
Target: the industry's return over the next holding period minus the average industry.
Rule (both NN and momentum): hold the top 3 of 12 equally; trade a holding only when it is more than
2% of the account away from target (Exp 25 band); 10bp per side.

Tax: 25% on realized gains (Israel's rate), average-cost basis per industry, losses carried
forward, paid at each year-end out of the account; whatever is left is taxed when sold at the end.
Momentum = 12-1 month return (Moskowitz-Grinblatt 1999), same top 3, same frequencies.
Variants counted: 3 frequencies x {NN, momentum} + 2 champion combinations = 8.

    python scripts/experiments/exp30_nn_rotation.py        (about 10-20 minutes)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from algotrading.ml import EnsembleMLP  # noqa: E402

R = at.data.load_ff_industries(12, daily=True).fillna(0.0)
FF = at.data.load_ff_factors(daily=True)
RF = FF["RF"].reindex(R.index).fillna(0.0)
MKT = (FF["Mkt-RF"] + FF["RF"]).reindex(R.index).fillna(0.0)
COST, BAND, TAX, K = 0.001, 0.02, 0.25, 3
FREQS = {"daily": 1, "weekly": 5, "monthly": 21}
START = "1957-01-01"
LR = np.log1p(R)


def features():
    f = {}
    for h in (1, 5, 21, 63, 126, 252):
        x = LR.rolling(h).sum()
        f[f"r{h}"] = x
        f[f"x{h}"] = x.sub(x.mean(axis=1), axis=0)
    f["vol63"] = LR.rolling(63).std() * np.sqrt(252)
    lvl = (1 + MKT).cumprod()
    on = (lvl > lvl.rolling(200).mean()).astype(float)
    m21 = np.log1p(MKT).rolling(21).sum()
    f["mkt_on"] = pd.DataFrame({c: on for c in R.columns})
    f["mkt21"] = pd.DataFrame({c: m21 for c in R.columns})
    return pd.concat(f, axis=1)            # columns (feature, industry)


def target(h):
    fwd = LR[::-1].rolling(h).sum()[::-1].shift(-1)    # days t+1..t+h
    return fwd.sub(fwd.mean(axis=1), axis=0)


F = features()
NAMES = sorted(set(F.columns.get_level_values(0)), key=list(F.columns.get_level_values(0)).index)


def panel(dates):
    X = np.stack([F[n].loc[dates].to_numpy() for n in NAMES], axis=-1)   # (T, 12, nfeat)
    return X


def nn_scores(h):
    """Walk-forward NN predictions (T x 12) on decision dates every h days from START."""
    y = target(h)
    dates = R.index[(R.index >= START)][::h]
    out = pd.DataFrame(np.nan, index=dates, columns=R.columns)
    for y0 in range(1957, 2027, 5):
        lo, hi = pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y0 + 5}-01-01")
        tr_dates = R.index[(R.index >= "1928-01-01") & (R.index < lo - pd.Timedelta(days=2 * h))][::max(h, 5)]
        Xtr = panel(tr_dates).reshape(-1, len(NAMES))
        ytr = y.loc[tr_dates].to_numpy().reshape(-1)
        ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
        Xtr, ytr = Xtr[ok], ytr[ok]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        model = EnsembleMLP(hidden=(8,), alpha=1.0).fit((Xtr - mu) / sd, ytr)
        d = dates[(dates >= lo) & (dates < hi)]
        if len(d) == 0:
            continue
        Xte = panel(d)
        pred = model.predict(((Xte.reshape(-1, len(NAMES)) - mu) / sd)).reshape(len(d), -1)
        out.loc[d] = pred
        print(f"    trained to {y0 - 1} on {len(ytr):,} rows; scored {len(d)} dates", flush=True)
    return out


def mom_scores(h):
    dates = R.index[(R.index >= START)][::h]
    m = LR.rolling(231).sum().shift(21)                  # 12-1 months in trading days
    return m.loc[dates]


def picks(scores):
    rank = scores.rank(axis=1, ascending=False)
    return (rank <= K).astype(float) / K


def simulate(w_dec, tax=TAX):
    """Daily simulation with band, costs and year-end tax. ``w_dec``: target weights on decision
    dates (decided at that close, traded at the next close). Returns daily net returns (after tax
    paid along the way) and the after-tax liquidation value."""
    tgt = w_dec.shift(1).reindex(R.index[R.index >= w_dec.index[0]]).ffill().fillna(0.0)
    days = tgt.index
    rets, rf = R.loc[days].to_numpy(), RF.loc[days].to_numpy()
    T = tgt.to_numpy()
    newdec = pd.Series(w_dec.shift(1).index, index=w_dec.index).reindex(days).notna().to_numpy()
    n = R.shape[1]
    hold, basis, cash = np.zeros(n), np.zeros(n), 1.0
    realized, carry, tax_paid = 0.0, 0.0, 0.0
    eq_prev, out = 1.0, np.empty(len(days))
    years = days.year
    for i in range(len(days)):
        hold *= 1 + rets[i]
        cash *= 1 + rf[i]
        eq = hold.sum() + cash
        if newdec[i]:
            want = T[i] * eq
            diff = want - hold
            trade = (np.abs(diff) > BAND * eq) | ((T[i] == 0) & (hold > 0))
            for j in np.where(trade & (diff < 0))[0]:       # sells first
                q = -diff[j]
                frac = q / hold[j] if hold[j] > 0 else 0.0
                realized += q - basis[j] * frac
                basis[j] *= 1 - frac
                hold[j] -= q
                cash += q * (1 - COST)
            for j in np.where(trade & (diff > 0))[0]:
                q = diff[j]
                hold[j] += q
                basis[j] += q
                cash -= q * (1 + COST)
        if tax and (i == len(days) - 1 or years[i + 1] != years[i]):   # year-end tax
            net = realized + carry
            if net > 0:
                t = tax * net
                cash -= t
                tax_paid += t
                carry = 0.0
            else:
                carry = net
            realized = 0.0
        eq = hold.sum() + cash
        out[i] = eq / eq_prev - 1
        eq_prev = eq
    gains = hold.sum() - basis.sum() + carry
    liq = eq - (tax * gains if tax and gains > 0 else 0.0)
    return pd.Series(out, index=days), liq, tax_paid


def summarize(r, liq=None):
    row = {"1957-1990": at.metrics.cagr(r["1957":"1990"], 252), "1991-2026": at.metrics.cagr(r["1991":], 252),
           "2000-2026": at.metrics.cagr(r["2000":], 252)}
    row["worst era"] = min(row["1957-1990"], row["1991-2026"])
    row["1957-2026"] = at.metrics.cagr(r, 252)
    yrs = len(r) / 252
    row["after tax, sold at end"] = liq ** (1 / yrs) - 1 if liq else np.nan
    row["max DD"] = at.metrics.max_drawdown(r)
    return row


if __name__ == "__main__":
    rows, monthly_legs = {}, {}
    ew = pd.DataFrame(1 / 12, index=R.index[R.index >= START][::21], columns=R.columns)
    for lab, w in [("12 industries equal weight (monthly)", ew)]:
        r_pre, _, _ = simulate(w, tax=0)
        r_tax, liq, _ = simulate(w)
        rows[lab] = summarize(r_pre, liq)
    m = MKT[START:]
    rows["market (buy & hold, no tax until sold)"] = summarize(m, (1 + m).prod() * (1 - TAX) + TAX)
    for fname, h in FREQS.items():
        for kind, fn in [("momentum", mom_scores), ("neural net", nn_scores)]:
            print(f"{kind} {fname}...", flush=True)
            w = picks(fn(h))
            r_pre, _, _ = simulate(w, tax=0)
            r_tax, liq, paid = simulate(w)
            lab = f"{kind} top 3, {fname}"
            rows[lab] = summarize(r_pre, liq)
            dec = w.diff().abs().sum(axis=1) / 2
            rows[lab]["turnover/yr"] = dec.sum() / (len(w) * h / 252)
            if fname == "monthly":
                monthly_legs[kind] = r_pre
    out = pd.DataFrame(rows).T
    pd.set_option("display.width", 250)
    print("\nPre-tax CAGR by era (all trading costs), plus after-tax CAGR if sold at the end (25%):\n")
    print(out.to_string(float_format=lambda v: f"{v:.1%}"))

    # Champion with the stock leg in the monthly picks (same construction as Exp 27, pre-tax).
    from loop import ensemble
    df = at.data.load_shiller_sp500()
    idx = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
    W = ensemble(3.0).weights(idx).shift(1)
    W.index = W.index.to_period("M").to_timestamp()
    rb = idx["bonds"].pct_change()
    rb.index = rb.index.to_period("M").to_timestamp()
    mrf = at.data.load_ff_factors()["RF"]
    mm = at.data.load_ff_factors()["Mkt-RF"] + mrf
    legs = {"S&P (current champion)": mm}
    for kind, r in monthly_legs.items():
        x = (1 + r).groupby(r.index.to_period("M")).prod() - 1
        x.index = x.index.to_timestamp()
        legs[f"{kind} top 3"] = x
    common = W.index.intersection(legs["neural net top 3"].index)
    print("\nChampion's exposures with different stock legs (monthly, pre-tax, 1957-2026):")
    for lab, rs in legs.items():
        w = W.loc[common]
        g = w.sum(axis=1)
        r = (w["stocks"] * rs.reindex(common) + w["bonds"] * rb.reindex(common) + (1 - g) * mrf.reindex(common)
             - (g - 1).clip(lower=0) * 0.005 / 12 - w.diff().abs().sum(axis=1).fillna(0) * 0.001)
        e1, e2 = at.metrics.cagr(r[:"1990"], 12), at.metrics.cagr(r["1991":], 12)
        print(f"  {lab:28s} 1957-1990 {e1:.1%}  1991-2026 {e2:.1%}  worst {min(e1, e2):.1%}  "
              f"max DD {at.metrics.max_drawdown(r):.1%}")
