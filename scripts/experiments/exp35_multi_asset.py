"""Experiment 35: widen the investment areas - stocks abroad, corporate bonds, gold - and combine
them with the published multi-asset rules; then combine the result with the champion family.

Owner: 'widen your investments areas, test everything, combine with what we use'. Monthly data:
  US stocks   CRSP value-weighted total return (Ken French)
  Intl stocks equal-weight 12 developed markets (OECD share prices via FRED, +3%/yr dividends,
              local-currency returns = roughly currency-hedged; as in Exp 20)
  Treasuries  10-year, from Shiller yields (constant-maturity approximation)
  Corp bonds  Moody's Baa yield (FRED BAA), 8-year duration approximation, -0.5%/yr for defaults
  Gold        London price (datahub gold-prices), minus 0.4%/yr (GLD's fee); free market from 1971
Eras (three ~equal, gold trades freely): 1972-1990, 1991-2007, 2008-2026.
Published rules, round parameters, 10bp per unit traded, leverage at T-bill + 0.5%:
  R1 equal weight, buy and hold (rebalanced monthly)
  R2 Faber (2007) GTAA: equal weight, each asset held only above its 10-month average, else T-bills
  R3 time-series momentum (Moskowitz-Ooi-Pedersen 2012; Hurst-Ooi-Pedersen 2017): each asset held
     while its 12-month return beats T-bills, sized to 10% volatility (trailing 12 months), total <= 3x
  R4 risk parity (Asness-Frazzini-Pedersen 2012): inverse-volatility weights scaled to 10%
     portfolio volatility, <= 3x; R5 = R4 with each asset's trend filter
  C  combinations, fixed 50/50: the champion (S&P leg) and the diversified x2/3 account (CRSP legs,
     Exp 31) each paired with R3 or R5.
Variants counted: 5 rules + 4 combinations = 9.

    python scripts/experiments/exp35_multi_asset.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import algotrading as at  # noqa: E402
import exp31_combinations as e  # noqa: E402

GOLD_URL = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
GOLD_PATH = os.path.join(os.path.dirname(at.data.SHILLER_PATH), "gold_monthly.csv")


def gold():
    if not os.path.exists(GOLD_PATH):
        import urllib.request
        urllib.request.urlretrieve(GOLD_URL, GOLD_PATH)
    g = pd.read_csv(GOLD_PATH)
    g.index = pd.to_datetime(g["Date"])
    return g["Price"].pct_change() - 0.004 / 12


def intl():
    cols = {}
    for cc in ["DE", "GB", "FR", "IT", "CA", "AU", "NL", "CH", "SE", "ES", "JP", "KR"]:
        s = at.data.load_fred(f"SPASTT01{cc}M661N")
        s.index = s.index.to_period("M").to_timestamp()
        cols[cc] = s.pct_change() + 0.03 / 12
    return pd.DataFrame(cols).mean(axis=1, skipna=True)


def corp():
    y = at.data.load_fred("BAA") / 100
    y.index = y.index.to_period("M").to_timestamp()
    return y.shift(1) / 12 - 8.0 * y.diff() - 0.005 / 12


R = pd.DataFrame({"US stocks": e.MKT, "Intl stocks": intl(), "Treasuries": e.RB, "Corp bonds": corp(),
                  "Gold": gold()})
R = R["1960":].dropna(how="all")
RF = e.RF.reindex(R.index).fillna(0)
START = "1972-01-01"
ERAS = {"1972-1990": ("1972", "1990"), "1991-2007": ("1991", "2007"), "2008-2026": ("2008", "2026")}


def run(W, rets=R):
    W = W.shift(1).reindex(rets.index).fillna(0)                # decided at t-1
    g = W.sum(axis=1)
    r = (W * rets.fillna(0)).sum(axis=1) + (1 - g) * RF - (g - 1).clip(lower=0) * 0.005 / 12 \
        - W.diff().abs().sum(axis=1).fillna(0) * 0.001
    return r[START:]


lvl = (1 + R.fillna(0)).cumprod()
on = (lvl > lvl.rolling(10).mean()) & R.notna()
vol = R.rolling(12).std() * np.sqrt(12)
live = R.notna() & R.shift(12).notna()
n = live.sum(axis=1).replace(0, np.nan)
ex12 = ((1 + R).rolling(12).apply(np.prod, raw=True)).sub((1 + RF).rolling(12).apply(np.prod, raw=True), axis=0)

W1 = live.astype(float).div(n, axis=0)
W2 = (live & on).astype(float).div(n, axis=0)
w3 = ((ex12 > 0) & live).astype(float) * (0.10 / vol) / n.values[:, None]
W3 = w3.div((w3.sum(axis=1) / 3).clip(lower=1), axis=0)                # cap 3x
iv = (1 / vol).where(live)
base = iv.div(iv.sum(axis=1), axis=0)
cov_vol = pd.Series([np.sqrt(np.nan_to_num(base.iloc[i].to_numpy()) @ np.nan_to_num(R.iloc[max(0, i - 11):i + 1].cov().to_numpy()) @ np.nan_to_num(base.iloc[i].to_numpy()) * 12)
                     for i in range(len(R))], index=R.index)
W4 = base.mul((0.10 / cov_vol).clip(upper=3.0), axis=0).fillna(0)
W5 = W4.where(on, 0.0)

RULES = {"R1 equal weight (5 assets)": run(W1), "R2 Faber GTAA (trend filter)": run(W2),
         "R3 time-series momentum, 10% vol each": run(W3), "R4 risk parity 10% vol": run(W4),
         "R5 risk parity + trend filter": run(W5)}


def stats(r):
    row = {k: at.metrics.cagr(r[a:b], 12) for k, (a, b) in ERAS.items()}
    row["worst era"] = min(row.values())
    row["1972-2026"] = at.metrics.cagr(r, 12)
    w = (1 + r).cumprod()
    row["max DD"] = float((w / w.cummax() - 1).min())
    row["worst 12m"] = float(((1 + r).rolling(12).apply(np.prod, raw=True) - 1).min())
    return row


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    fmt = lambda v: f"{v:.1%}"  # noqa: E731
    rows = {k: stats(R[k][START:].fillna(0)) for k in R.columns}
    print("Single assets, buy & hold:\n")
    print(pd.DataFrame(rows).T.to_string(float_format=fmt))
    print(f"\nCorrelations 1972-2026:\n{R[START:].corr().round(2).to_string()}")
    rows = {k: stats(v) for k, v in RULES.items()}
    print("\nMulti-asset rules:\n")
    print(pd.DataFrame(rows).T.to_string(float_format=fmt))
    print(f"  avg gross exposure: R3 {W3[START:].sum(axis=1).mean():.2f}x, R4 {W4[START:].sum(axis=1).mean():.2f}x, "
          f"R5 {W5[START:].sum(axis=1).mean():.2f}x")

    D = e.LEGS["diversified (ind mom + stock mom + value)"]
    champ = e.portfolio(e.MKT).reindex(R.index)[START:]
    div23 = e.portfolio(D, scale=2 / 3).reindex(R.index)[START:]
    rows = {"champion (S&P leg)": stats(champ), "diversified x2/3": stats(div23)}
    for a, ra in [("champion", champ), ("diversified x2/3", div23)]:
        for b in ("R3 time-series momentum, 10% vol each", "R5 risk parity + trend filter"):
            rows[f"50/50 {a} + {b[:2]}"] = stats(0.5 * ra + 0.5 * RULES[b])
    print("\nCombinations with the champion family (fixed 50/50, rebalanced monthly):\n")
    print(pd.DataFrame(rows).T.to_string(float_format=fmt))
    print(f"\nCorrelation of monthly returns: champion vs R3 {champ.corr(RULES['R3 time-series momentum, 10% vol each']):+.2f}, "
          f"champion vs R5 {champ.corr(RULES['R5 risk parity + trend filter']):+.2f}")

    # R5 at a 15% volatility target (round; AQR/Hurst-Ooi-Pedersen use 10-15%) and its combinations.
    W5b = base.mul((0.15 / cov_vol).clip(upper=3.0), axis=0).fillna(0).where(on, 0.0)
    r5b = run(W5b)
    rows = {"R5 at 15% vol": stats(r5b), "50/50 champion + R5 at 15% vol": stats(0.5 * champ + 0.5 * r5b),
            "50/50 diversified x2/3 + R5 at 15% vol": stats(0.5 * div23 + 0.5 * r5b)}
    print("\nHigher risk budget (variants +3):\n")
    print(pd.DataFrame(rows).T.to_string(float_format=fmt))
    print(f"  avg gross exposure R5 at 15%: {W5b[START:].sum(axis=1).mean():.2f}x")

    # Real ETFs (Yahoo, dividend-adjusted, US-dollar, unhedged): SPY, EFA, IEF, LQD, GLD, 2004-2026.
    etf = at.data.etf_closes(["SPY", "EFA", "IEF", "LQD", "GLD"], rng="max")
    m = etf.groupby(etf.index.to_period("M")).last()
    m.index = m.index.to_timestamp()
    Rr = m.pct_change()["2005":]
    Rr.columns = R.columns
    rf = e.RF.reindex(Rr.index).ffill()
    lv = (1 + Rr.fillna(0)).cumprod()
    on_r = lv > lv.rolling(10).mean()
    vol_r = Rr.rolling(12).std() * np.sqrt(12)
    base_r = (1 / vol_r).div((1 / vol_r).sum(axis=1), axis=0)
    pv = pd.Series([np.sqrt(np.nan_to_num(base_r.iloc[i].to_numpy()) @ np.nan_to_num(Rr.iloc[max(0, i - 11):i + 1].cov().to_numpy())
                            @ np.nan_to_num(base_r.iloc[i].to_numpy()) * 12) for i in range(len(Rr))], index=Rr.index)
    out = {}
    for tv in (0.10, 0.15):
        W = base_r.mul((tv / pv).clip(upper=3.0), axis=0).where(on_r, 0.0).fillna(0).shift(1).fillna(0)
        g = W.sum(axis=1)
        r = ((W * Rr.fillna(0)).sum(axis=1) + (1 - g) * rf - (g - 1).clip(lower=0) * 0.015 / 12
             - W.diff().abs().sum(axis=1).fillna(0) * 0.001)["2006":]
        w = (1 + r).cumprod()
        out[f"R5 on real ETFs, {tv:.0%} vol, margin T-bill+1.5%"] = {
            "2006-2015": at.metrics.cagr(r[:"2015"], 12), "2016-2026": at.metrics.cagr(r["2016":], 12),
            "2006-2026": at.metrics.cagr(r, 12), "max DD": float((w / w.cummax() - 1).min()), "avg gross": g["2006":].mean()}
    spy = Rr["US stocks"]["2006":]
    w = (1 + spy).cumprod()
    out["SPY buy & hold"] = {"2006-2015": at.metrics.cagr(spy[:"2015"], 12), "2016-2026": at.metrics.cagr(spy["2016":], 12),
                             "2006-2026": at.metrics.cagr(spy, 12), "max DD": float((w / w.cummax() - 1).min()), "avg gross": 1.0}
    print("\nReal-ETF check (variants +2):\n")
    t = pd.DataFrame(out).T
    print(t.to_string(formatters={c: ("{:.2f}x".format if c == "avg gross" else "{:.1%}".format) for c in t.columns}))

    # The combination that matters, with the real-ETF R5 (2006-2026; champion from CRSP months).
    for tv in (0.10, 0.15):
        W = base_r.mul((tv / pv).clip(upper=3.0), axis=0).where(on_r, 0.0).fillna(0).shift(1).fillna(0)
        g = W.sum(axis=1)
        r5r = ((W * Rr.fillna(0)).sum(axis=1) + (1 - g) * rf - (g - 1).clip(lower=0) * 0.015 / 12
               - W.diff().abs().sum(axis=1).fillna(0) * 0.001)["2006":"2026-09"]
        ch = e.portfolio(e.MKT).reindex(r5r.index)
        dv = e.portfolio(D, scale=2 / 3).reindex(r5r.index)
        for lab, x in [("champion alone", ch), ("diversified x2/3 alone", dv),
                       (f"50/50 champion + real R5 {tv:.0%}", 0.5 * ch + 0.5 * r5r),
                       (f"50/50 diversified x2/3 + real R5 {tv:.0%}", 0.5 * dv + 0.5 * r5r)]:
            if tv == 0.15 and "alone" in lab:
                continue
            w = (1 + x).cumprod()
            print(f"  {lab:45s} 2006-2015 {at.metrics.cagr(x[:'2015'], 12):6.1%}  2016-2026 {at.metrics.cagr(x['2016':], 12):6.1%}  "
                  f"all {at.metrics.cagr(x, 12):6.1%}  max DD {float((w / w.cummax() - 1).min()):6.1%}")
