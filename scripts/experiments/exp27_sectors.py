"""Experiment 27: sectors and themes. Should we hold semiconductors / tech instead of the index,
or rotate into the strongest industries?

Data: Ken French's 49 value-weighted US industry portfolios, 1926-2026 (total returns, CRSP),
and the market (Mkt-RF + RF). Costs 10bp per unit traded; leverage financed at the T-bill + 0.5%;
signals use completed months only and trade one month later (execution lag 1).

A. Hold one hot industry from the start (Chips = semiconductors, Softw, Hardw): the question is
   whether today's favourite was knowable in advance. Each era is shown separately.
B. Industry momentum (Moskowitz & Grinblatt 1999, JF): each month hold the 5 industries with the
   best 12-1 month return, equal weight (round, fixed in advance; ~top tenth of 49).
C. B gated by the market trend (S&P-style 10-month average on the market), at 1x and 3x:
   the champion's trend sleeve with the stock leg replaced by the momentum industries.
D. Reference: market and the market trend sleeve at 1x and 3x.
E. Robustness: 10/12/17/30/48/49 industry sets, top 3-5, 1x / trend-gated 1x / 2x (18 more).
F. The champion's own exposures with the stock leg in the top 3 of 12 industries (all of it, or
   only the first 1x with the rest in the S&P, which is what sector ETFs + UPRO can deliver): 2.
Eras: 1927-1956, 1957-1990, 1991-2026 (no 1880-1926 data here). Variants counted: 26.

    python scripts/experiments/exp27_sectors.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

ind = at.data.load_ff_industries(49)
ff = at.data.load_ff_factors()
rf = ff["RF"]
mkt = ff["Mkt-RF"] + rf
ERAS = {"1927-1956": ("1927", "1956"), "1957-1990": ("1957", "1990"), "1991-2026": ("1991", "2026")}
COST, SPREAD = 0.001, 0.005


def run(w_risky, rets, lev_cap=None):
    """w_risky: weights decided at month-end t (DataFrame aligned to rets columns); traded
    from month t+1. Returns monthly net returns."""
    w = w_risky.shift(1).fillna(0.0)
    gross = w.sum(axis=1)
    r = (w * rets.fillna(0.0)).sum(axis=1) + (1 - gross) * rf
    r -= (gross - 1).clip(lower=0) * SPREAD / 12                 # financing spread above T-bill
    r -= w.diff().abs().sum(axis=1).fillna(0.0) * COST
    return r


def stats(r):
    row = {k: at.metrics.cagr(r[a:b], 12) for k, (a, b) in ERAS.items()}
    row["worst era"] = min(row.values())
    row["1927-2026"] = at.metrics.cagr(r["1927":], 12)
    row["max DD"] = at.metrics.max_drawdown(r["1927":])
    return row


mk = pd.DataFrame({"mkt": mkt})
mkt_level = (1 + mkt).cumprod()
trend_on = (mkt_level > mkt_level.rolling(10).mean()).astype(float)
rows = {}
rows["market 1x"] = stats(run(pd.DataFrame({"mkt": 1.0}, index=mkt.index), mk))
rows["market trend 3x (champion's trend sleeve)"] = stats(run(pd.DataFrame({"mkt": 3.0 * trend_on}), mk))
for name in ["Chips", "Softw", "Hardw"]:
    x = ind[[name]]
    w = x.notna().astype(float)
    rows[f"{name} buy & hold"] = stats(run(w, x))

mom = (1 + ind).rolling(11).apply(np.prod, raw=True).shift(1) - 1   # months t-11..t-1 (12-1)
live = ind.notna() & mom.notna()
rank = mom.where(live).rank(axis=1, ascending=False)
top5 = (rank <= 5).astype(float) / 5
rows["industry momentum top 5, 1x"] = stats(run(top5, ind))
rows["industry momentum top 5, trend-gated 1x"] = stats(run(top5.mul(trend_on, axis=0), ind))
rows["industry momentum top 5, trend-gated 3x"] = stats(run(3 * top5.mul(trend_on, axis=0), ind))
out = pd.DataFrame(rows).T
pd.set_option("display.width", 220)
print("US industries (Ken French 49), all costs, leverage at T-bill + 0.5%:\n")
print(out.to_string(float_format=lambda v: f"{v:.1%}"))

# What a semiconductor fund costs in risk: volatility and worst 12 months.
print("\nRisk of single industries vs the market (1991-2026):")
for name, r in [("market", mkt), ("Chips", ind["Chips"]), ("Softw", ind["Softw"]), ("Hardw", ind["Hardw"])]:
    x = r["1991":]
    print(f"  {name:7s} vol {x.std() * 12 ** 0.5:5.1%}, worst 12 months "
          f"{((1 + x).rolling(12).apply(np.prod, raw=True) - 1).min():6.1%}, max DD {at.metrics.max_drawdown(x):6.1%}")

# Post-publication check for momentum (Moskowitz & Grinblatt published 1999).
r = run(top5, ind)
print(f"\nIndustry momentum 1x after publication (2000-2026): {at.metrics.cagr(r['2000':], 12):.1%} "
      f"vs market {at.metrics.cagr(mkt['2000':], 12):.1%}")


# --- Robustness across industry sets (top ~tenth-quarter, fixed) and the champion with a sector stock leg ---
print("\nRobustness across Ken French industry sets, and the champion with a sector-momentum stock leg:")
ff=at.data.load_ff_factors(); rf=ff["RF"]; mkt=ff["Mkt-RF"]+rf
lvl=(1+mkt).cumprod(); on=(lvl>lvl.rolling(10).mean()).astype(float)
def run(w,rets):
    w=w.shift(1).fillna(0); g=w.sum(axis=1)
    r=(w*rets.fillna(0)).sum(axis=1)+(1-g)*rf-(g-1).clip(lower=0)*0.005/12-w.diff().abs().sum(axis=1).fillna(0)*0.001
    return r
E={"27-56":("1927","1956"),"57-90":("1957","1990"),"91-26":("1991","2026"),"00-26":("2000","2026")}
rows={}
for n,k in [(10,3),(12,3),(17,4),(30,5),(48,5),(49,5)]:
    ind=at.data.load_ff_industries(n)
    mom=(1+ind).rolling(11).apply(np.prod,raw=True).shift(1)-1
    rank=mom.where(ind.notna()&mom.notna()).rank(axis=1,ascending=False)
    top=(rank<=k).astype(float)/k
    for lab,w in [("1x",top),("gated 1x",top.mul(on,axis=0)),("gated 2x",2*top.mul(on,axis=0))]:
        r=run(w,ind)
        row={e:at.metrics.cagr(r[a:b],12) for e,(a,b) in E.items()}
        row["worst"]=min(row[e] for e in ["27-56","57-90","91-26"]); row["maxDD"]=at.metrics.max_drawdown(r["1927":])
        rows[f"{n} ind top{k} {lab}"]=row
    ew=ind.notna().astype(float).div(ind.notna().sum(axis=1),axis=0)
    r=run(ew,ind); rows[f"{n} ind equal-wt 1x"]={e:at.metrics.cagr(r[a:b],12) for e,(a,b) in E.items()}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v:f"{v:.1%}"))

# Champion's own exposure, stock leg = market vs 12-industry top-3 momentum
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loop import ensemble
df=at.data.load_shiller_sp500()
idx=pd.DataFrame({"stocks":df["tr"],"bonds":at.data.bond_total_return(df["long_rate"])})
W=ensemble(3.0).weights(idx).shift(1)            # traded next month
W.index=W.index.to_period("M").to_timestamp()
rb=idx["bonds"].pct_change(); rb.index=rb.index.to_period("M").to_timestamp()
ind=at.data.load_ff_industries(12)
mom=(1+ind).rolling(11).apply(np.prod,raw=True).shift(1)-1
rank=mom.where(ind.notna()&mom.notna()).rank(axis=1,ascending=False); top=((rank<=3).astype(float)/3).shift(1)
rmom=(top*ind.fillna(0)).sum(axis=1)-top.diff().abs().sum(axis=1)*0.001
common=W.index.intersection(rmom.index)[13:]
def champ(rs):
    w=W.loc[common]; g=w.sum(axis=1)
    return w["stocks"]*rs.loc[common]+w["bonds"]*rb.loc[common]+(1-g)*rf.loc[common]-(g-1).clip(lower=0)*0.005/12-w.diff().abs().sum(axis=1).fillna(0)*0.001
for lab,rs in [("champion, S&P stock leg",mkt),("champion, sector-momentum stock leg",rmom)]:
    r=champ(rs); row={e:at.metrics.cagr(r[a:b],12) for e,(a,b) in E.items()}
    print(lab, {k:f"{v:.1%}" for k,v in row.items()}, "worst", f"{min(row[e] for e in ['27-56','57-90','91-26']):.1%}", "DD", f"{at.metrics.max_drawdown(r):.1%}")
# Hybrid: first 1x of stocks in the top-3 sectors (unlevered sector ETFs), the rest via S&P (UPRO)
w=W.loc[common]; s=w["stocks"]; core=s.clip(upper=1.0)
rs=(core*rmom.loc[common]+(s-core)*mkt.loc[common]).div(s.replace(0,np.nan)).fillna(0)
r=champ(rs.reindex(mkt.index).fillna(0)); row={e:at.metrics.cagr(r[a:b],12) for e,(a,b) in E.items()}
print("hybrid (1x sectors + S&P leverage)", {k:f"{v:.1%}" for k,v in row.items()}, "worst", f"{min(row[e] for e in ['27-56','57-90','91-26']):.1%}", "DD", f"{at.metrics.max_drawdown(r):.1%}")
