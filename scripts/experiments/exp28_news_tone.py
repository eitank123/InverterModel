"""Experiment 28: does the tone of the day's news predict the S&P 500? (GDELT 2017-2026)

GDELT 2.0 scores the tone of every online news article worldwide; its DOC API returns the daily
average tone of articles matching a query, back to 2017, with no look-ahead (the tone of day t
is known at the end of day t). Query fixed in advance: English articles mentioning "stock market".
Tetlock (2007, JF): media pessimism predicts lower returns the next day, then a reversal.

Tests, all trading at the close of the NEXT day (one day of delay), 5bp per side:
  1. correlation of tone(t) with the S&P return on day t+1 and on days t+1..t+5;
  2. overlay on 2x S&P: cut to 1x while the 5-day average tone is more than 1 standard deviation
     below its trailing 250-day mean (round, fixed in advance), else 2x;
  3. the same rule with the tone series shifted by a random number of days (200 draws): the null
     distribution. Exp 5 showed that any rule that sits out part of a crash can look good by luck.
Variants counted: 2 (the correlation check and the overlay).

    python scripts/experiments/exp28_news_tone.py
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "news", "gdelt_tone_stock_market.csv")
URL = ("https://api.gdeltproject.org/api/v2/doc/doc?query=%22stock%20market%22%20sourcelang:english"
       "&mode=timelinetone&format=json&startdatetime=20170101000000&enddatetime={end}000000")


def load_tone():
    if not os.path.exists(PATH):
        end = pd.Timestamp.today().strftime("%Y%m%d")
        for _ in range(4):                       # GDELT rate-limits; retry politely
            try:
                req = urllib.request.Request(URL.format(end=end), headers={"User-Agent": "algotrading-research/0.1"})
                d = json.loads(urllib.request.urlopen(req, timeout=60).read())
                break
            except Exception:
                time.sleep(15)
        s = pd.DataFrame(d["timeline"][0]["data"])
        s["date"] = pd.to_datetime(s["date"].str[:8])
        s.set_index("date")["value"].rename("tone").to_csv(PATH)
    return pd.read_csv(PATH, index_col=0, parse_dates=True)["tone"]


tone = load_tone()
sp = at.data.fred_daily_proxies()["stocks"]
r = sp.pct_change().dropna()
tb = (at.data.load_fred("DTB3") / 100).reindex(r.index).ffill().bfill()
t = tone.reindex(tone.index.union(r.index)).ffill().reindex(r.index)   # weekend news -> Monday
print(f"Tone {tone.index[0]:%Y-%m-%d} to {tone.index[-1]:%Y-%m-%d}; S&P days {len(r)}")

fwd1, fwd5 = r.shift(-1), r[::-1].rolling(5).sum()[::-1].shift(-1)
for lab, f in [("next day", fwd1), ("next 5 days", fwd5)]:
    x = pd.concat([t, f], axis=1).dropna()
    c = x.corr().iloc[0, 1]
    print(f"  corr(tone today, S&P {lab}) = {c:+.3f}  (t ~ {c * len(x) ** 0.5:+.1f}, overlapping for 5 days)")
x = pd.concat([t.diff(), r], axis=1).dropna()
print(f"  same-day: corr(tone change, S&P return) = {x.corr().iloc[0, 1]:+.3f}  (news describes the day)")


def overlay(tone_series):
    s5 = tone_series.rolling(5).mean()
    z = (s5 - s5.rolling(250).mean()) / s5.rolling(250).std()
    lev = pd.Series(np.where(z < -1, 1.0, 2.0), index=r.index).where(z.notna(), 2.0)
    lev = lev.shift(1).fillna(2.0)                     # known at close t, held from t+1
    net = lev * r - (lev - 1) * (tb + 0.005) / 252 - lev.diff().abs().fillna(0) * 0.0005
    return net["2018":], lev["2018":]


base = 2 * r - (tb + 0.005) / 252
g = lambda x: at.metrics.cagr(x, 252)  # noqa: E731
net, lev = overlay(t)
print(f"\n2x S&P constant: {g(base['2018':]):.1%}/yr, max DD {at.metrics.max_drawdown(base['2018':]):.1%}")
print(f"News overlay (1x when tone z<-1): {g(net):.1%}/yr, max DD {at.metrics.max_drawdown(net):.1%}, "
      f"at 1x {(lev == 1).mean():.0%} of days, avg {lev.mean():.2f}x")
rng = np.random.default_rng(28)
null = []
for _ in range(200):
    k = int(rng.integers(60, len(t) - 60))
    shifted = pd.Series(np.roll(t.to_numpy(), k), index=t.index)
    null.append(g(overlay(shifted)[0]))
null = np.array(null)
print(f"Null (tone shifted at random): median {np.median(null):.1%}, 95th pct {np.percentile(null, 95):.1%}; "
      f"p = {(null >= g(net)).mean():.2f}")
