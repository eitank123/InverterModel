"""Experiment 29: does news tone predict the S&P 500 over longer horizons (a week to 3 months)?

Same data as Exp 28: GDELT daily tone of English 'stock market' articles, 2017-2026, against the
S&P 500 (FRED, total return proxy). Horizons fixed in advance: 1, 5, 10, 21 and 63 trading days.
Overlapping windows inflate t-statistics, so each correlation is also measured on NON-overlapping
windows (every h-th day). Tetlock (2007) predicts the sign: gloomy news -> lower returns the next
day, then a reversal (gloom -> HIGHER returns over the following week).

Then one pre-set contrarian rule (the reversal direction from the paper, not from this data):
2x S&P, raised to 3x for 21 trading days after the 5-day tone falls 1 standard deviation below
its 250-day mean. Checked against 200 random time shifts of the tone series. Variants: 6.

    python scripts/experiments/exp29_news_horizons.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

tone = pd.read_csv(at.news.GDELT_TONE_PATH, index_col=0, parse_dates=True)["tone"]
sp = at.data.fred_daily_proxies()["stocks"]
r = sp.pct_change().dropna()
tb = (at.data.load_fred("DTB3") / 100).reindex(r.index).ffill().bfill()
t = tone.reindex(tone.index.union(r.index)).ffill().reindex(r.index)
z = at.news.tone_zscore(t)
lr = np.log1p(r)

print(f"News tone {tone.index[0]:%Y-%m-%d} to {tone.index[-1]:%Y-%m-%d}\n")
print(f"{'horizon':>9s} {'signal':>18s} {'corr (all days)':>16s} {'corr (non-overlap)':>19s} {'t (non-overlap)':>16s} {'n':>5s}")
for h in (1, 5, 10, 21, 63):
    fwd = lr[::-1].rolling(h).sum()[::-1].shift(-1)          # log return over days t+1..t+h
    for lab, sig in [("tone level", t), ("tone z (5d vs 1y)", z)]:
        x = pd.concat([sig, fwd], axis=1).dropna()
        c_all = x.corr().iloc[0, 1]
        xn = x.iloc[::h]
        c = xn.corr().iloc[0, 1]
        print(f"{h:>6d} d  {lab:>18s} {c_all:>+16.3f} {c:>+19.3f} {c * (len(xn) - 2) ** 0.5 / (1 - c * c) ** 0.5:>+16.1f} {len(xn):>5d}")

# Gloomy-news days: average forward returns after z < -1 vs all days
print("\nAverage S&P return after gloomy news (z < -1) vs all days:")
for h in (5, 21, 63):
    fwd = lr[::-1].rolling(h).sum()[::-1].shift(-1)
    g, a = fwd[z < -1].mean(), fwd[z.notna()].mean()
    print(f"  next {h:>2d} days: after gloom {np.expm1(g):+.2%}, all days {np.expm1(a):+.2%}  "
          f"({(z < -1).sum()} gloomy days, heavily overlapping)")


def contrarian(zs):
    trig = (zs < -1).astype(float)
    boost = trig.rolling(21, min_periods=1).max()            # 21 days after any trigger
    lev = (2.0 + boost).shift(1).fillna(2.0)
    net = lev * r - (lev - 1) * (tb + 0.005) / 252 - lev.diff().abs().fillna(0) * 0.0005
    return net["2018":], lev["2018":]


g = lambda x: at.metrics.cagr(x, 252)  # noqa: E731
base = (2 * r - (tb + 0.005) / 252)["2018":]
net, lev = contrarian(z)
print(f"\nPlain 2x: {g(base):.1%}/yr, max DD {at.metrics.max_drawdown(base):.1%}")
print(f"Contrarian (3x for a month after gloom): {g(net):.1%}/yr, max DD {at.metrics.max_drawdown(net):.1%}, "
      f"avg {lev.mean():.2f}x")
const = (lev.mean() * r - (lev.mean() - 1) * (tb + 0.005) / 252)["2018":]
print(f"Constant leverage at the same average ({lev.mean():.2f}x): {g(const):.1%}/yr")
rng = np.random.default_rng(29)
null = np.array([g(contrarian(pd.Series(np.roll(z.to_numpy(), int(rng.integers(60, len(z) - 60))),
                                        index=z.index))[0]) for _ in range(200)])
print(f"Null (tone shifted at random): median {np.median(null):.1%}, 95th pct {np.percentile(null, 95):.1%}, "
      f"p = {(null >= g(net)).mean():.2f}")
