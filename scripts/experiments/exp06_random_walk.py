"""Experiment 6: is the stock market a random walk? At which horizons does it trend,
and at which does it mean-revert?

Math: if log prices follow a random walk, k-period return variance grows linearly in k,
so the variance ratio VR(q) = Var(r_t + ... + r_{t-q+1}) / (q Var(r_t)) = 1.
VR(q) - 1 ~ 2 sum_{k<q} (1 - k/q) rho_k, so VR > 1 means positive autocorrelation
(momentum), and VR < 1 means negative autocorrelation (mean reversion).

Literature: Lo & MacKinlay (1988) reject the random walk for weekly US returns;
Poterba & Summers (1988) and Fama & French (1988) find mean reversion at 3-5 years;
Moskowitz, Ooi & Pedersen (2012) find momentum at 1-12 months.

Trap: Shiller's prices are monthly *averages*, which by itself creates VR > 1 at short q
(Working 1960). So the null distribution is simulated from averaged random walks.

    python scripts/experiments/exp06_random_walk.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.3f}".format)
OUT = "results/exp06"
os.makedirs(OUT, exist_ok=True)

df = at.data.load_shiller_sp500()
stocks = np.log(df["tr"]).diff().dropna()
real = np.log(df["tr"] / df["cpi"]).diff().dropna()
bonds = np.log(at.data.bond_total_return(df["long_rate"])).diff().dropna()
QS = [2, 3, 6, 12, 24, 36, 60, 120]


def averaged_walk(n, sigma_m, rng, averaged=True, days=21):
    """n monthly log returns of a daily random walk, observed as monthly averages."""
    lp = np.cumsum(rng.standard_normal((n + 1) * days) * sigma_m / np.sqrt(days))
    m = lp.reshape(n + 1, days)
    lvl = np.log(np.exp(m).mean(axis=1)) if averaged else m[:, -1]
    return np.diff(lvl)


def null_band(n, sigma_m, qs, averaged=True, sims=300, seed=0):
    """VR quantiles for a random walk of n months, observed as monthly averages."""
    rng = np.random.default_rng(seed)
    out = {q: [] for q in qs}
    for _ in range(sims):
        r = averaged_walk(n, sigma_m, rng, averaged)
        for q in qs:
            out[q].append(at.stats.variance_ratio(r, q)[0])
    return {q: np.percentile(v, [5, 50, 95]) for q, v in out.items()}


def vr_table(r, label, averaged=True):
    band = null_band(len(r), r.std(), QS, averaged)
    rows = []
    for q in QS:
        vr, z = at.stats.variance_ratio(r, q)
        lo, med, hi = band[q]
        verdict = "TREND" if vr > hi else ("MEAN-REV" if vr < lo else "random walk")
        rows.append({"q_months": q, "VR": vr, "z_LoMac": z, "null_5%": lo, "null_50%": med,
                     "null_95%": hi, "verdict": verdict})
    t = pd.DataFrame(rows).set_index("q_months")
    print(f"\n-- {label} (n={len(r)} months)")
    print(t)
    return t


print("A. Variance ratios vs a simulated averaged-averaged-random-walk null (90% band)")
t_nom = vr_table(stocks, "S&P 500 total return, nominal, 1871-2026")
t_real = vr_table(real[:"2023-06"], "S&P 500 total return, real (CPI-deflated), 1871-2023")
vr_table(stocks[:"1945"], "S&P 500 1871-1945")
vr_table(stocks["1946":], "S&P 500 1946-2026")
vr_table(bonds, "10y bond total return 1871-2026")
# Bond returns = carry (yield/12, which moves slowly for decades) + duration * yield change.
# A persistent carry makes VR > 1 without any exploitable price trend, so test the
# price component alone as well.
carry = np.log1p(df["long_rate"].shift(1) / 12).reindex(bonds.index)
vr_table((bonds - carry).dropna(), "10y bond return minus carry (price component only)")

print("\nB. Annual (non-overlapping) return autocorrelations, real S&P 500 1872-2022")
yearly = real[:"2022"].groupby(real[:"2022"].index.year).sum()
for k in range(1, 6):
    print(f"   lag {k} year(s): {yearly.autocorr(k):+.3f}  (±{2 / np.sqrt(len(yearly)):.3f} ~ 95%)")

print("\nC. Hurst exponent (R/S), monthly log returns")
rng = np.random.default_rng(1)
h_null = [at.stats.hurst_exponent(averaged_walk(len(stocks), 0.04, rng)) for _ in range(200)]
print(f"   S&P 500: {at.stats.hurst_exponent(stocks):.3f}; "
      f"bonds: {at.stats.hurst_exponent(bonds):.3f}; averaged-random-walk null "
      f"{np.mean(h_null):.3f} [{np.percentile(h_null, 5):.3f}, {np.percentile(h_null, 95):.3f}]")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.fill_between(t_nom.index, t_nom["null_5%"], t_nom["null_95%"], alpha=.25, color="gray",
                label="random walk (monthly-averaged) 90% band")
ax.plot(t_nom.index, t_nom["VR"], "o-", label="S&P 500 nominal")
ax.plot(t_real.index, t_real["VR"], "s-", label="S&P 500 real")
ax.axhline(1, color="k", lw=.8)
ax.set_xscale("log"); ax.set_xticks(QS); ax.set_xticklabels(QS)
ax.set_xlabel("horizon q (months)"); ax.set_ylabel("variance ratio")
ax.set_title("Variance ratios, 1871-2026: >1 trend, <1 mean reversion"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/variance_ratios.png", dpi=110)
print(f"\nPlot in {OUT}/")
