"""Experiment 5: can news headlines tell us when to own stocks?

Data: top-25 r/worldnews headlines per day (2008-08 to 2016-07, ~50k headlines) and the
daily Dow Jones index. Scorer: Loughran-McDonald finance dictionary.

Hypotheses (decide before looking):
  H1 (popular claim): today's headline sentiment predicts tomorrow's market direction.
  H2 (Tetlock 2007): pessimistic news predicts low returns next day, then a reversal.
  H3: bad news predicts *volatility* (risk), even if not direction.

The trap we test explicitly: many tutorials on this dataset predict the SAME day's
move from the SAME day's headlines, which is lookahead (headlines react to the market).

    python scripts/experiments/exp05_news_sentiment.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.3f}".format)
OUT = "results/exp05"
os.makedirs(OUT, exist_ok=True)

# --- Data ----------------------------------------------------------------------------
cache = f"{OUT}/daily_sentiment.csv"
heads = at.news.load_reddit_headlines()
if os.path.exists(cache):
    daily = pd.read_csv(cache, index_col=0, parse_dates=True)
else:
    heads["score"] = at.news.LexiconScorer().score(heads["title"].tolist())
    g = heads.groupby("date")["score"]
    daily = pd.DataFrame({"sentiment": g.mean(), "neg_share": g.apply(lambda s: (s < 0).mean())})
    daily.to_csv(cache)
sent = daily["sentiment"]

dj = at.data.load_djia_daily()
close, opens = dj["close"], dj["open"]
ret = close.pct_change()
print(f"{len(heads)} headlines, {len(daily)} days; {(heads.title.str.len() > 0).mean():.0%} non-empty")

# Sanity check: does the dataset's Label equal "DJIA closed up or flat that same day"?
label = heads.groupby("date")["label"].first()
up = (ret >= 0).astype(int).reindex(label.index)
print(f"Label == same-day up/flat: {(label == up).mean():.1%} agreement "
      "-> the Kaggle target is the SAME day's move")

df = pd.DataFrame({"sent": sent, "r0": ret.reindex(sent.index)}).dropna()
fwd = pd.DataFrame({f"r{k}": ret.shift(-k) for k in range(1, 6)}).reindex(df.index)
df = df.join(fwd)
df["sent_z"] = (df["sent"] - df["sent"].rolling(250, min_periods=60).mean()) / \
    df["sent"].rolling(250, min_periods=60).std()

# --- A. Same day vs next day ---------------------------------------------------------
print("\nA. Correlation of daily sentiment with returns (t = news date)")
n = len(df)
se = 1 / np.sqrt(n)
for col, lab in [("r0", "same day t   (lookahead!)"), ("r1", "next day t+1"), ("r2", "day t+2"),
                 ("r3", "day t+3"), ("r4", "day t+4"), ("r5", "day t+5")]:
    c = df["sent_z"].corr(df[col])
    print(f"   {lab:26s} corr {c:+.3f}   (|corr| > {2 * se:.3f} is ~significant)")


def acc(target):
    sub = df[["sent_z", target]].dropna()
    return ((sub["sent_z"] > 0) == (sub[target] >= 0)).mean()


print(f"   direction 'accuracy': same day {acc('r0'):.1%}, next day {acc('r1'):.1%} "
      f"(always-up baseline {(df['r1'] >= 0).mean():.1%})")

# --- B. Risk instead of direction -----------------------------------------------------
print("\nB. Does bad news predict volatility? corr(sentiment_t, |return_{t+1}|)")
absr = df["r1"].abs()
print(f"   corr {df['sent_z'].corr(absr):+.3f}; "
      f"after the most negative 20% of days |r| = {absr[df['sent_z'] < df['sent_z'].quantile(.2)].mean():.2%}, "
      f"after the most positive 20% = {absr[df['sent_z'] > df['sent_z'].quantile(.8)].mean():.2%}")
# Fair comparison: does sentiment add anything beyond yesterday's own volatility?
lag_vol = ret.abs().rolling(20).mean().reindex(df.index)
X = pd.DataFrame({"const": 1.0, "lag_vol": lag_vol, "sent_z": df["sent_z"]})
ok = X.notna().all(axis=1) & absr.notna()
beta, *_ = np.linalg.lstsq(X[ok].to_numpy(), absr[ok].to_numpy(), rcond=None)
resid = absr[ok] - X[ok] @ beta
cov = np.linalg.inv(X[ok].T @ X[ok]) * resid.var()
print(f"   regression |r_t+1| ~ recent vol + sentiment: t-stat(sentiment) = "
      f"{beta[2] / np.sqrt(cov.iloc[2, 2] if hasattr(cov, 'iloc') else cov[2, 2]):+.2f}")

# --- C. Trading it -------------------------------------------------------------------
print("\nC. Sentiment timing strategies on the DJIA (2 bp, cash 0.5%)")
start, end = sent.index[0], sent.index[-1]
px_close = close[start:end]
px_open = opens[start:end]
bh = at.run_backtest(px_close, at.BuyAndHold(), cost_bps=2, cash_rate=0.005)
rows = []
variants = [(sm, lo) for sm in (1, 5, 20) for lo in (0.0, -1.0)]
for sm, lo in variants:
    strat = at.news.SentimentStrategy(sent, smooth=sm, low=lo)
    for timing, px, lag in [("close_t (leaky)", px_close, 0), ("close_t+1", px_close, 1),
                            ("open_t+1", px_open, 1)]:
        r = at.run_backtest(px, strat, cost_bps=2, execution_lag=lag, cash_rate=0.005)
        rows.append({"smooth": sm, "when_negative": "short" if lo < 0 else "cash",
                     "timing": timing, "sharpe": r.stats["sharpe"], "cagr": r.stats["cagr"],
                     "max_dd": r.stats["max_drawdown"], "turnover": r.stats["ann_turnover"],
                     "_ret": r.returns})
tab = pd.DataFrame(rows)
print(tab.pivot_table(index=["smooth", "when_negative"], columns="timing", values="sharpe"))
print(f"   buy & hold Sharpe {bh.stats['sharpe']:.3f}, CAGR {bh.stats['cagr']:.1%}, "
      f"maxDD {bh.stats['max_drawdown']:.0%}")

# --- D. Is the best honest variant better than luck? ----------------------------------
honest = tab[tab["timing"] == "open_t+1"].sort_values("sharpe", ascending=False)
best = honest.iloc[0]
dsr = at.metrics.deflated_sharpe(best["_ret"], n_trials=len(tab), trial_sharpes_std=tab["sharpe"].std())
print(f"\nD. Best honest variant: smooth={best['smooth']}, {best['when_negative']}: "
      f"Sharpe {best['sharpe']:.2f}; DSR (vs {len(tab)} trials) = {dsr:.2f}")
# Permutation test: rotate the sentiment series by a random offset. This keeps its own
# autocorrelation but breaks any link to prices. How often does luck do as well?
rng = np.random.default_rng(0)
null = []
strat_best = at.news.SentimentStrategy(sent, smooth=int(best["smooth"]),
                                       low=-1.0 if best["when_negative"] == "short" else 0.0)
for _ in range(300):
    k = int(rng.integers(250, len(sent) - 250))
    rotated = pd.Series(np.roll(sent.to_numpy(), k), index=sent.index)
    strat_best.sentiment = rotated
    null.append(at.run_backtest(px_open, strat_best, cost_bps=2, execution_lag=1,
                                cash_rate=0.005).stats["sharpe"])
null = np.array(null)
print(f"   permutation p-value: {(null >= best['sharpe']).mean():.2f} "
      f"(null Sharpe mean {null.mean():.2f}, 95th pct {np.percentile(null, 95):.2f})")
