"""Experiment 8: how much to bet? Kelly sizing, leverage and estimation error.

Math. With excess return mean mu and variance sigma^2 per period, the expected log growth
at leverage f is (continuous-time approximation)
        g(f) = r + f mu - f^2 sigma^2 / 2,
maximised at the Kelly fraction f* = mu / sigma^2 with g* = r + mu^2 / (2 sigma^2) =
r + SR^2 / 2. Beyond 2 f* growth is below cash. For geometric Brownian motion at a
fraction lambda of Kelly, the probability that wealth EVER falls to a fraction x of its
starting value is x^(2/lambda - 1): at full Kelly there is a 50% chance of halving.
(Kelly 1956; Thorp 1969; MacLean, Thorp & Ziemba 2011.)

Reality checks: returns have fat tails and crashes (gaps), mu is estimated with huge
error, and borrowing costs more than cash earns.

    python scripts/experiments/exp08_kelly_leverage.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.3f}".format)

df = at.data.load_shiller_sp500()
px = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
cash = (df["long_rate"] - 0.015).clip(lower=0)
KW = dict(cost_bps=10, execution_lag=1, cash_rate=cash, borrow_spread=0.01)


BASES = {
    "stocks": at.PerAsset(at.BuyAndHold(), {"stocks": 1.0}),
    "60_40": at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4}),
    "60_40_sma10": at.TrendFilter(at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4}),
                            at.PriceAboveSMA(10)),
}

# --- A. Theory: Kelly fraction from full-sample moments -------------------------------
print("A. Kelly fraction from full-sample moments (excess of cash, 1871-2026)")
rows = []
for name, s in BASES.items():
    r = at.run_backtest(px, s, **KW).returns
    ex = r - cash.reindex(r.index) / 12
    f = at.stats.kelly_fraction(ex)
    sr = ex.mean() / ex.std() * np.sqrt(12)
    rows.append({"base": name, "excess_mu": ex.mean() * 12, "sigma": ex.std() * np.sqrt(12),
                 "sharpe": sr, "kelly_f": f, "g*_minus_cash": sr**2 / 2})
print(pd.DataFrame(rows).set_index("base"))

# --- B. What actually happens at each leverage ----------------------------------------
print("\nB. Realised results by leverage (borrow at cash + 1%, monthly rebalanced)")
rows = []
for name, s in BASES.items():
    for f in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        r = at.run_backtest(px, at.Levered(s, f), **KW).returns
        eq = (1 + r).cumprod()
        ruined = (eq <= 0).any() or (r <= -1).any()
        rows.append({"base": name, "leverage": f,
                     "cagr": -1.0 if ruined else at.metrics.cagr(r),
                     "max_dd": -1.0 if ruined else at.metrics.max_drawdown(r),
                     "worst_month": r.min()})
lev = pd.DataFrame(rows)
print(lev.pivot(index="leverage", columns="base", values=["cagr", "max_dd"]))
print("   (cagr -1.000 = wiped out)")

# --- C. Estimation error: Kelly estimated in real time -------------------------------
print("\nC. Kelly fraction estimated on trailing 30-year windows (stocks)")
r_st = at.run_backtest(px, BASES["stocks"], **KW).returns
ex_st = r_st - cash.reindex(r_st.index) / 12
roll = (ex_st.rolling(360).mean() / ex_st.rolling(360).var()).dropna()
dec = roll.groupby(roll.index.year // 10 * 10).last()
print("   by decade end:", ", ".join(f"{d}s {v:.1f}" for d, v in dec.items()))
print(f"   range {roll.min():.1f} .. {roll.max():.1f}: the 'optimal' bet size swings by a factor "
      f"of {roll.max() / max(roll.min(), 0.1):.0f}")


class RollingKelly(at.PortfolioStrategy):
    """Leverage = fraction * trailing Kelly estimate, capped; uses only past data."""

    def __init__(self, base, frac=0.5, window=360, cap=2.0):
        self.base, self.frac, self.window, self.cap = base, frac, window, cap

    def weights(self, prices):
        w = self.base.weights(prices)
        r = (w.shift(1) * prices.pct_change()).sum(axis=1)
        c = cash.reindex(prices.index).ffill() / 12
        ex = r - c
        k = (ex.rolling(self.window).mean() / ex.rolling(self.window).var())
        lev = (self.frac * k).clip(0, self.cap).fillna(1.0)
        return w.mul(lev, axis=0)


rows = []
for name in ("stocks", "60_40_sma10"):
    for label, s in [("fixed 1x", at.Levered(BASES[name], 1.0)),
                     ("half-Kelly rolling, cap 2", RollingKelly(BASES[name], 0.5)),
                     ("full-Kelly rolling, cap 4", RollingKelly(BASES[name], 1.0, cap=4.0))]:
        r = at.run_backtest(px["1901":], s, **{**KW, "cash_rate": cash["1901":]}).returns
        rows.append({"base": name, "sizing": label, "cagr": at.metrics.cagr(r),
                     "vol": at.metrics.annual_vol(r), "max_dd": at.metrics.max_drawdown(r),
                     "sharpe": at.metrics.sharpe(r)})
print(pd.DataFrame(rows).set_index(["base", "sizing"]))

# --- D. Drawdown probabilities: theory vs history --------------------------------------
print("\nD. P(wealth ever falls to x of start) for a GBM bet at lambda * Kelly: x^(2/lambda - 1)")
for lam in (1.0, 0.5, 0.25):
    print(f"   lambda={lam:4.2f}: halve {0.5 ** (2 / lam - 1):.1%}, lose 80% {0.2 ** (2 / lam - 1):.2%}")
