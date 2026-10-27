# Lesson 3: News, information and why "read the news, then trade" is hard

## 1. How fast is news priced in?
When the Fed announces a rate decision at 14:00:00 New York time, the first trades react within
**microseconds**. Firms co-locate servers at the exchange and parse machine-readable news feeds
(Reuters, Bloomberg, Dow Jones Newswires) with pre-built rules. By the time a human reads the
headline on a website, the "obvious" move has already happened.

What remains for slower traders:
* **Drift after the reaction.** Prices sometimes under-react to complex or ambiguous news
  (earnings surprises: "post-earnings-announcement drift"; slowly diffusing news in small,
  little-followed stocks). This is where academic news-sentiment results live.
* **Risk timing.** Scheduled events (FOMC, CPI, jobs report, earnings) are known in advance, and
  volatility around them is predictably higher. You can't predict the direction, but you can size
  positions before the event.
* **Interpretation that machines find hard.** This is where LLMs might help. Lopez-Lira & Tang
  report that LLM-scored *firm-level* headlines predicted next-day returns, mostly in small caps
  and for bad news, and that the effect is shrinking as more funds use LLMs.

## 2. Choosing news sources for a systematic strategy
| Criterion | Why it matters |
|---|---|
| **Machine-readable** (RSS, API) | A program has to consume it |
| **Exact timestamps** | Without them you can't avoid lookahead |
| **Primary over commentary** | A Fed statement or SEC filing is the event itself; an article about it is a delayed, opinionated copy |
| **Historical archive** | You can only backtest what you can replay as it was on the day |
| **Relevance to what you trade** | World-politics headlines are not market news (Exp 5 proved it) |

Sources chosen for this project (`algotrading/news.py: SOURCES`):
1. **Federal Reserve press releases**: rate decisions move every asset. Primary source, archived.
2. **SEC EDGAR 8-K filings**: companies must report material events within 4 business days.
   Exact timestamps, free, full archive since 1994. The best free firm-level news source.
3. **BLS releases (CPI, jobs)**: the scheduled macro surprises.
4. **GDELT**: free global news index since 2015 with tone scores. Backtestable.
5. **MarketWatch / CNBC / Yahoo Finance RSS**: fast financial headlines, but no free archives,
   so they are useful for live monitoring only.
* *Paid, professional:* Reuters / Bloomberg / Dow Jones Newswires (millisecond timestamps, full
  archives), RavenPack (pre-scored news).

(None of these were reachable from the sandbox, so the URLs in `SOURCES` are unverified. The
network policy has to allow them.)

## 3. The four ways news backtests lie
1. **Same-day leakage.** A headline dated "Tuesday" may be published after Tuesday's close.
   The popular Kaggle "Reddit news → DJIA" dataset labels each day with *that same day's*
   market move (we verified: 99.8% match). Models "predicting" it are reading the answer.
2. **Train/test shuffling.** Random K-fold on time series lets the model train on the future.
   Always split by time (walk-forward).
3. **LLM lookahead.** An LLM trained on the internet up to 2024 has read what happened after every
   2015 headline. Its "sentiment" on old news is contaminated. Only headlines published after
   the model's training cutoff are a fair test.
4. **Wrong benchmark.** A strategy in the market half the time beat buy & hold over 2008–2016
   because it randomly missed part of the crash. Compare against the same strategy fed
   *shuffled* news (a permutation test). That is the test Exp 5's "edge" failed (p = 0.57).

## 4. How we will use news, then
Not as "headline is positive → buy". Instead:
* **Firm-level events** (8-K filings, earnings) for individual stocks, scored by an LLM
  (`news.ClaudeScorer`), evaluated only on post-cutoff data with next-session timing.
* **Event-risk calendar**: reduce exposure before scheduled macro events if tests show
  volatility rises without compensation.
* **Context for humans**: a daily digest of what moved and why. Useful for understanding, not
  automatically tradeable.
