"""Experiment 31: more growth AND more safety? Combining independent published effects with the
champion, 1927-2026 (CRSP data from Ken French; the champion's signals from the Shiller data).

The owner's goal: the largest mean growth, but not everything in one place, and safety up where it
costs little growth. Every building block is a published effect with round parameters; every
combination is a FIXED equal weight (no fitted weights). Monthly, signals from completed months,
traded the next month, 10bp per unit traded, leverage at the T-bill + 0.5%.

Stock legs (what the champion's 'stocks' exposure buys):
  L0 market (CRSP value-weighted) - the current champion
  L1 industry momentum: top 3 of 12 industries by 12-1 month return (Moskowitz-Grinblatt 1999)
  L2 stock momentum: top 30% of stocks by 12-2 month return (Jegadeesh-Titman 1993), value-weighted
  L3 value: top 30% of stocks by book-to-market (Fama-French 1992), value-weighted
  L4 diversified: equal thirds of L1, L2, L3 (three effects in one leg)
Extra sleeves / overlays (each tested on top of L4):
  S1 short side: the trend sleeve goes -1x (short the market) instead of to cash when the market is
     below its 10-month average (time-series momentum, Moskowitz-Ooi-Pedersen 2012)
  S2 factor sleeve: + 0.5x of a market-neutral long/short momentum + value factor mix (Asness,
     Moskowitz & Pedersen 2013: value and momentum hedge each other), momentum part
     volatility-scaled to 12% (Barroso & Santa-Clara 2015); 20bp a month of extra costs
  S3 volatility-managed overlay: scale total exposure by (median variance / last month's variance),
     clipped to 0.5-1.5 (Moreira & Muir 2017)
  S4 short-term reversal factor (Lehmann 1990; de Groot et al. 2012), as +0.5x, after 80bp a month
     of costs (two legs, ~100% monthly turnover, 20bp per side)
Safety metrics: worst era, max drawdown (all, and from 1940 to leave out the Great Depression), worst 12
months, share of months more than 30% below the peak. Variants counted: 5 legs + 5 sleeves + 14 scalings = 24.

    python scripts/experiments/exp31_combinations.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from loop import CappedValuationLeverage, filtered  # noqa: E402

kf = at.data._kf_block
FF = at.data.load_ff_factors()
RF, MKT = FF["RF"], FF["Mkt-RF"] + FF["RF"]
IND = at.data.load_ff_industries(12)
PRIOR = kf("10_Portfolios_Prior_12_2", "Value Weight Returns -- Monthly")
BM = kf("Portfolios_Formed_on_BE-ME", "Value Weight Returns -- Monthly")
MOM = kf("F-F_Momentum_Factor", "")["Mom"]
STREV = kf("F-F_ST_Reversal_Factor", "")["ST_Rev"]
MOMD = kf("F-F_Momentum_Factor_daily", "")["Mom"]
MKTD = (lambda f: f["Mkt-RF"] + f["RF"])(at.data.load_ff_factors(daily=True))
HML = FF["HML"]

# --- stock legs -------------------------------------------------------------------------------
mom12 = (1 + IND).rolling(11).apply(np.prod, raw=True).shift(1) - 1
rank = mom12.where(IND.notna() & mom12.notna()).rank(axis=1, ascending=False)
top = ((rank <= 3).astype(float) / 3).shift(1)                  # decided at t-1, held in t
L1 = (top * IND.fillna(0)).sum(axis=1) - top.diff().abs().sum(axis=1).fillna(0) * 0.001
L2 = PRIOR[["PRIOR 8", "PRIOR 9", "Hi PRIOR"]].mean(axis=1) - 0.0015    # ~ momentum fund turnover
L3 = BM["Hi 30"] - 0.0003
LEGS = {"market": MKT, "industry momentum": L1, "stock momentum": L2, "value": L3,
        "diversified (ind mom + stock mom + value)": (L1 + L2 + L3) / 3}

# --- the champion's three sleeves, signals from the Shiller data -------------------------------
df = at.data.load_shiller_sp500()
idx = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})


def mi(x):
    x = x.copy()
    x.index = x.index.to_period("M").to_timestamp()
    return x


SLEEVES = {"trend": mi(at.Levered(filtered(1.0), 3.0).weights(idx)),
           "value": mi(CappedValuationLeverage(1.0, 2.0, 0.5, 3.0).weights(idx)),
           "recovery": mi(at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0).weights(idx))}
RB = mi(idx["bonds"].pct_change())
DATES = MKT.index[(MKT.index >= "1927-07-01")]
DATES = DATES[DATES.isin(SLEEVES["trend"].index) & DATES.isin(L1.dropna().index)]


def portfolio(leg, sleeves=("trend", "value", "recovery"), short=False, extra=None, volman=False,
              scale=1.0):
    W = sum(SLEEVES[s] for s in sleeves) / len(sleeves) * scale
    if short:                                            # trend sleeve short when out
        on = (SLEEVES["trend"]["stocks"] > 0).astype(float)
        W = W.copy()
        W["stocks"] = W["stocks"] - (1 - on) * 1.0 / len(sleeves) * scale
        W["bonds"] = W["bonds"].where(on > 0, W["bonds"] - SLEEVES["trend"]["bonds"] / len(sleeves) * scale)
    W = W.shift(1).reindex(DATES)                        # decided at t-1
    if volman:
        var = (MKTD ** 2).groupby(MKTD.index.to_period("M")).sum()
        var.index = var.index.to_timestamp()
        mult = (var.expanding(120).median() / var).clip(0.5, 1.5).shift(1).reindex(DATES).fillna(1.0)
        W = W.mul(mult, axis=0)
    g = W["stocks"].clip(lower=0) + W["bonds"].clip(lower=0)
    short_amt = (-W["stocks"]).clip(lower=0)
    r = (W["stocks"] * leg.reindex(DATES) + W["bonds"] * RB.reindex(DATES) + (1 - W.sum(axis=1)) * RF.reindex(DATES)
         - (g - 1).clip(lower=0) * 0.005 / 12 - short_amt * 0.005 / 12
         - W.diff().abs().sum(axis=1).fillna(0) * 0.001)
    if extra is not None:
        r = r + extra.reindex(DATES).fillna(0)
    return r


def factor_sleeve():
    vol = MOMD.rolling(126).std() * np.sqrt(252)
    v = vol.groupby(MOMD.index.to_period("M")).last()
    v.index = v.index.to_timestamp()
    mom_scaled = MOM * (0.12 / v.shift(1)).clip(upper=2.0)
    return 0.5 * (0.5 * mom_scaled + 0.5 * HML) - 0.002


def stats(r):
    eras = {"1927-1956": r[:"1956"], "1957-1990": r["1957":"1990"], "1991-2026": r["1991":]}
    row = {k: at.metrics.cagr(v, 12) for k, v in eras.items()}
    row["worst era"] = min(row.values())
    row["1927-2026"] = at.metrics.cagr(r, 12)
    w = (1 + r).cumprod()
    row["max DD"] = float((w / w.cummax() - 1).min())
    w2 = (1 + r["1940":]).cumprod()
    row["max DD 1940+"] = float((w2 / w2.cummax() - 1).min())
    row["worst 12m"] = float(((1 + r).rolling(12).apply(np.prod, raw=True) - 1).min())
    row["months >30% down"] = float((w / w.cummax() - 1 < -0.30).mean())
    return row


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    fmt = lambda v: f"{v:.1%}"  # noqa: E731
    rows = {"market buy & hold": stats(MKT.reindex(DATES))}
    for name, leg in LEGS.items():
        rows[f"champion, {name} leg"] = stats(portfolio(leg))
    print("A. The champion's exposures with different stock legs (1927-2026, monthly, all costs):\n")
    print(pd.DataFrame(rows).T.to_string(float_format=fmt))

    D = LEGS["diversified (ind mom + stock mom + value)"]
    fs = factor_sleeve()
    rev = 0.5 * (STREV - 0.008)
    rows = {"champion, diversified leg (base)": stats(portfolio(D)),
            "+ S1 short side in the trend sleeve": stats(portfolio(D, short=True)),
            "+ S2 factor sleeve (mom + value L/S)": stats(portfolio(D, extra=fs)),
            "+ S3 volatility-managed overlay": stats(portfolio(D, volman=True)),
            "+ S4 short-term reversal (after costs)": stats(portfolio(D, extra=rev)),
            "+ S2 + S3": stats(portfolio(D, extra=fs, volman=True))}
    print("\nB. Extra sleeves on top of the diversified-leg champion:\n")
    print(pd.DataFrame(rows).T.to_string(float_format=fmt))

    print("\nC. Growth vs safety: scale all exposure up or down (Kelly: half the bet keeps about 3/4 of the growth):\n")
    rows = {}
    for s in (0.5, 0.67, 0.75, 1.0, 1.25):
        rows[f"current champion (market leg) x{s:.2f}"] = stats(portfolio(MKT, scale=s))
    for s in (0.5, 0.67, 0.75, 1.0):
        rows[f"diversified leg, long only x{s:.2f}"] = stats(portfolio(D, scale=s))
    for s in (0.5, 0.67, 0.75, 1.0, 1.25):
        rows[f"diversified leg + factor sleeve x{s:.2f}"] = stats(portfolio(D, extra=fs * s, scale=s))
    print(pd.DataFrame(rows).T.to_string(float_format=fmt))
    print(f"\nFactor sleeve alone: {at.metrics.cagr(fs['1927':] + RF['1927':], 12):.1%}/yr incl. cash, "
          f"corr with market {fs.corr(MKT):+.2f}; reversal after costs {at.metrics.cagr(rev['1927':] + RF['1927':], 12):.1%}/yr; "
          f"reversal gross mean {STREV.mean() * 12:.1%}/yr")

    # Selection bias: had the loop picked the best of these variants each January using only
    # past data (and held it a year), what would it have earned? (research.walk_forward_selection)
    D = LEGS["diversified (ind mom + stock mom + value)"]
    variants = {f"leg {k}": portfolio(v) for k, v in LEGS.items()}
    variants.update({"S1": portfolio(D, short=True), "S2": portfolio(D, extra=fs), "S3": portfolio(D, volman=True),
                     "S4": portfolio(D, extra=rev), "S2+S3": portfolio(D, extra=fs, volman=True)})
    for s in (0.5, 0.67, 0.75, 1.0):
        variants[f"div x{s}"] = portfolio(D, scale=s)
        variants[f"div+S2 x{s}"] = portfolio(D, extra=fs * s, scale=s)
        variants[f"champ x{s}"] = portfolio(MKT, scale=s)
    wf = at.research.walk_forward_selection(pd.DataFrame(variants).dropna(), start=1957)
    print(f"\nWalk-forward selection over {len(variants)} variants, 1957-2026: {at.metrics.cagr(wf, 12):.1%}/yr "
          f"(1957-1990 {at.metrics.cagr(wf[:'1990'], 12):.1%}, 1991-2026 {at.metrics.cagr(wf['1991':], 12):.1%})")
