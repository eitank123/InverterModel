# Research log

Each entry records: **hypothesis → setup → result → lesson → next step**. Numbers are
reproducible with the listed script. Write the hypothesis *before* running anything.

---

## Background reading (2026-10-27)

| Source | Key claim | What we test |
|---|---|---|
| Faber (2007), *A Quantitative Approach to Tactical Asset Allocation*, J. Wealth Mgmt; 10-yr update in JPM (2018) | Hold an asset only when price > its 10-month SMA, otherwise cash. Similar return to buy & hold, much smaller drawdowns. The out-of-sample update reports Sharpe improving 0.45 → 0.80, but underperforming stocks in 6 of the 8 years after 2009. | Exp 1 |
| Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, JFE | Past 12-month return sign predicts next month's, in 58 futures across all asset classes. Positions are volatility-scaled. | Exp 1, 2 |
| Kim, Tse & Wald (2016), critique of TSMOM | Much of TSMOM's alpha comes from **volatility scaling**, not from the trend signal. Without scaling, TSMOM ≈ buy & hold. | Next: Exp 3 |
| Hurst, Ooi & Pedersen (2017), *A Century of Evidence on Trend-Following*, AQR / JPM | Equal mix of 1-, 3- and 12-month TSMOM, 67 markets, 1880–2016. Positive in every decade, low correlation to stocks, and positive in 8 of the 10 worst 60/40 drawdowns. | Exp 1, 2 |
| Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, JPM | After N trials the best Sharpe is inflated by ≈ σ_SR · E[max of N normals]. Correct for it before believing a result. | Implemented in `metrics.deflated_sharpe` |

Note: the publishers' sites were not reachable from the sandbox. These summaries come
from abstracts and secondary sources, so verify the details in the papers.

---

## Experiment 1: Trend timing on 155 years of the S&P 500
`python scripts/experiments/exp01_sp500_trend.py` · data: Shiller monthly, 1871-01 → 2026-09 · cost 10 bp/switch

**Hypothesis.** Faber's SMA-10 rule and the 1/3/12-month trend ensemble (long or cash)
give a higher Sharpe and a much smaller drawdown than buy & hold, in every era.

**Result 1: the headline numbers depend heavily on two "boring" assumptions.**

| | Buy & hold | Faber SMA10 | Ensemble 1/3/12 |
|---|---|---|---|
| Sharpe, trade same month, cash = bond yield | 0.71 | 1.11 | **1.26** |
| Sharpe, trade *next* month, cash = bond yield | 0.71 | 0.94 | 0.95 |
| Sharpe, trade next month, cash = 0% | 0.71 | 0.80 | 0.67 |
| Max drawdown (next month, bond yield) | −82% | −51% | −44% |
| CAGR (next month, bond yield) | 9.4% | 9.1% | 7.4% |

**Result 2: a data artifact found and measured.** Shiller's prices are *monthly averages*
of daily closes. Averaging a random walk creates lag-1 autocorrelation of ≈0.25 (Working, 1960).
We measured **0.275** in the real data. On simulated random walks with no edge, the same
ensemble strategy *beat* buy & hold by +0.20 Sharpe on averaged prices traded at lag 0, but
*lost* to it by −0.10 to −0.15 on month-end prices or with a one-month lag. **The impressive 1.26 is
mostly artifact.** Short lookbacks (1 month) are hit hardest. From now on we use `execution_lag=1` on this data.

**Result 3: the effect is a plateau, not a spike.** We swept lookbacks of 2–24 months for SMA and momentum
(46 variants, lag 1, Sharpe in excess of cash). 70% of the variants beat buy & hold (0.39). There is a broad
plateau at 6–12 months (≈0.46–0.50). Very short lookbacks (2–4 months) do worse than buy & hold. A broad plateau is what a real
effect looks like; a single sharp peak is what overfitting looks like.

**Result 4: is it *significantly* better than buy & hold?** We scaled each pre-specified
rule to buy & hold's volatility, then computed the Probabilistic Sharpe Ratio of the return difference:

| Cash assumption | Faber SMA10 | Ensemble |
|---|---|---|
| 0% | 0.91 | 0.28 |
| bond yield − 1.5% (≈ T-bills) | 0.996 | 0.98 |
| bond yield (upper bound) | 0.999 | 0.999 |

Whether trend timing "beats" the market depends on what your cash earns while you are out.
That is a real-world fact, not a bug. During the 1970s, cash earned 6–9% a year.

**Result 5: every era, including after publication.** Faber's Sharpe beat buy & hold in all
five sub-periods: 1871–1928 (1.01 vs 0.78), 1929–49 (0.35 vs 0.26), 1950–99 (1.29 vs 1.15),
2000–26 (1.10 vs 0.71), and post-publication 2007–26 (1.05 vs 0.87). However, its CAGR was *lower* in
the strong bull markets of 1950–99 and 2007–26. **You buy insurance, and insurance has a premium.**

**Result 6: crisis behaviour.** In slow bear markets (1929–32: −79% → −25%; 1973–74: −39% → −1%;
GFC: −48% → −5%) the rule exits early. In fast crashes it can't react on monthly data
(COVID 2020: −12% vs −16% for the rule, which sold near the bottom and bought back late; this is called *whipsaw*).

**Lessons.**
1. Know how your data was constructed. Averaging, survivorship, and dividend handling can each
   create or destroy a "strategy".
2. Report results under conservative assumptions (lag, costs, cash) and treat the gap
   from the optimistic ones as a warning.
3. Trend timing on one market is mainly a *risk-reduction* tool: drawdowns are cut in half,
   with a modest risk-adjusted improvement, and it underperforms in bull markets.

---

## Experiment 2: Breadth, parameter risk and costs (synthetic)
`python scripts/experiments/exp02_diversification.py` · regime-switching markets · 10 years daily · 5 seeds

**Hypothesis.** (a) Sharpe grows like √N with independent markets. (b) An ensemble of lookbacks
is more robust than any single lookback when trend length is unknown. (c) Fast signals die
first as costs rise.

**A: Breadth.** Per-asset trend signals are each scaled to 10% vol, and the portfolio is scaled to 10% vol.

| N markets | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Sharpe | 1.08 | 1.79 | 2.35 | 3.44 | 4.57 |
| SR₁·√N | 1.08 | 1.53 | 2.42 | 3.43 | 4.85 |

√N holds well. This is the **fundamental law of active management** and the single best
reason to trade many uncorrelated markets. (These Sharpes are unrealistically high because the
simulated trends are strong and markets fully independent. Real markets are correlated, and
real diversified trend programmes achieve ≈0.5–1.0.)

**B: Parameter risk.** Sharpe by lookback, depending on average trend duration:

| avg trend length | 50 d | 100 d | 333 d | 1000 d |
|---|---|---|---|---|
| 1-month lookback | 1.14 | 1.49 | 2.20 | 2.48 |
| 3-month | 1.10 | 1.73 | **3.48** | 3.81 |
| 12-month | 0.46 | 1.07 | 2.76 | **4.28** |
| Ensemble | **1.22** | **1.82** | 3.44 | 4.17 |

Each single lookback wins in one regime and loses badly in another. The ensemble ranked 1st,
1st, 2nd and 2nd. **When you don't know the right parameter, diversify across parameters.**

**C: Costs.** Annual turnover: 1-month lookback 26× capital, 3-month 11×, 12-month 5×, ensemble 12×.

| one-way cost | 0 bp | 5 bp | 20 bp | 50 bp |
|---|---|---|---|---|
| 1-month | 2.36 | 1.96 | 0.76 | **−1.44** |
| 3-month | 3.55 | 3.37 | 2.82 | 1.74 |
| 12-month | 2.79 | 2.71 | 2.47 | **1.98** |
| ensemble | 3.54 | 3.29 | 2.53 | 1.05 |

Cost drag ≈ turnover × cost. The fast signal goes from best to negative. **The more
expensive your market (small caps, crypto, retail), the slower your signals must be.**

---

## Background reading, round 2 (2026-10-27)

| Source | Key claim | What we test |
|---|---|---|
| Kim, Tse & Wald (2016), critique of MOP | TSMOM's alpha is mostly from volatility scaling. Without it, TSMOM ≈ buy & hold. | Exp 3 |
| Moreira & Muir (2017), *Volatility-Managed Portfolios*, JF | Cutting exposure when vol is high raises Sharpe, because vol is predictable and returns don't rise with it. | Exp 3 (GARCH) |
| Tetlock (2007), *Giving Content to Investor Sentiment*, JF | Pessimism in a WSJ market column predicts lower returns next day, then a reversal within a week. | Exp 5 |
| Lopez-Lira & Tang (2023–25), *Can ChatGPT Forecast Stock Price Movements?* | LLM scores of firm-specific headlines predict the next-day reaction, mostly for small caps and negative news. Returns are fading as adoption grows. | `news.ClaudeScorer` (not yet run: needs an API key) |
| Faber (2018) 10-year update | The SMA-10 out-of-sample Sharpe held up, but the rule lagged stocks in 6 of 8 years after 2009. | Exp 1, Exp 4 |

---

## Experiment 3: Trend signal or volatility scaling?
`python scripts/experiments/exp03_vol_scaling.py` · 2×2 design: {buy & hold, trend} × {unscaled, vol-scaled}

**Hypothesis.** Following Kim, Tse & Wald, vol scaling explains most of the trend strategy's Sharpe.

| Market | Effect of vol scaling | Effect of trend signal |
|---|---|---|
| Synthetic GBM, constant vol (20 seeds) | −0.02 | −0.12 |
| Synthetic GARCH, clustered vol (20 seeds) | **+0.03** | −0.15 |
| S&P 500 monthly 1871–2026 | −0.08 | **+0.16** |
| DJIA daily 2007–2016 | −0.03 | −0.05 |

(Effects are average Sharpe differences over the two levels of the other factor.)

**Result.**
- On synthetic data without trends, the trend signal *costs* 0.12–0.15 Sharpe. Sitting in cash during dips of an upward-drifting asset is pure cost when there is no trend.
- Vol scaling helps only when volatility clusters (GARCH), and only a little.
- On 155 years of the S&P 500 the **trend signal**, not the scaling, is what helped. Monthly vol estimates from 12 points are noisy and lag by a month.
- On one decade of daily DJIA data neither factor mattered. One decade with one crash is an anecdote, not evidence.

**Lesson.** The Kim–Tse–Wald effect is mostly about *cross-asset* risk balancing across 58
futures with very different volatilities, not about timing one market. Our single-asset test
cannot confirm or refute it. That needs many assets (Exp 4 is a first, small step).

---

## Experiment 4: Stocks + bonds, 60/40 and trend (1871–2026)
`python scripts/experiments/exp04_stocks_bonds.py` · bonds = 10-year constant-maturity bond built from Shiller's yields · lag 1, 10 bp, cash = bond yield − 1.5%

**Data check.** Rebuilt 10-year bond returns: 4.5%/yr, 4.3% vol, worst drawdown −23%.
2022: −15% (the real 10-year Treasury lost ≈15–16%). 1941–81: 2.8%/yr while inflation ran
~4.7%, the great bond bear market. *A bug was caught here:* the first version counted the
coupon twice (9.3%/yr). Sanity-checking against known history caught it.

**Stock–bond correlation is not a constant:** +0.03 (1871–1928), +0.09 (1929–49),
−0.10 (1950–67), **+0.28 (1968–99)**, **−0.33 (2000–20)**, **+0.33 (2021–26)**. Bonds hedged
stocks for only about 20 of the last 155 years. When inflation drives markets, both fall together.

| Portfolio | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| 100% stocks | 9.4% | 14.0% | 0.71 | −82% |
| 60/40 | 7.7% | 8.6% | 0.90 | −61% |
| **60/40, each asset with SMA-10 filter** | 7.0% | 6.1% | **1.14** | **−32%** |
| TSMOM 2 assets, unscaled | 4.5% | 5.9% | 0.78 | −34% |
| TSMOM 2 assets, vol-scaled (levered ≈2.3×) | 6.5% | 9.1% | 0.74 | −60% |
| 80% 60/40 + 20% TSMOM | 7.6% | 7.5% | 1.02 | −52% |

Crises (total return): 1929–32: 60/40 −60% vs filtered −17%. 2008: −25% vs +6%. 2022: −19% vs −3%.

**The surprise: vol-scaled TSMOM lost 51% in 1966–82.** Diagnosis: it held ~1.2× gross
bond exposure, the bond trend whipsawed (bond-only trend Sharpe −0.51 over 1966–81), and
turnover of 9×/yr cost 27% over 15 years. At futures-like 2 bp costs it still lost 39%.
Published trend-following made money in the 1970s mainly from **commodities and currencies**,
which we don't have. With two assets, there is no breadth (Exp 2).

**Lessons.**
1. A 60/40 is not "safe". It lost 61% in 1929–32, and bonds stop hedging when inflation is the problem.
2. The simplest overlay, the SMA-10 filter per asset, was the best risk-adjusted result in the
   whole project so far: Sharpe 0.90 → 1.14 and drawdown halved, at a cost of 0.7%/yr CAGR.
3. Vol-scaling + leverage on a tiny universe amplifies whipsaw. Don't lever a strategy
   whose edge comes from breadth until you have the breadth.

---

## Experiment 5: News sentiment (r/worldnews headlines vs DJIA, 2008–2016)
`python scripts/experiments/exp05_news_sentiment.py` · ~50k headlines · Loughran-McDonald scorer

**Data check.** The dataset's `Label` matches "DJIA closed up that *same* day" 99.8% of the time.
Tutorials that predict it from the same day's headlines are using information from after the
prediction moment (lookahead).

| Test | Result | Verdict |
|---|---|---|
| corr(sentiment_t, return same day) | −0.001 | nothing, even with lookahead |
| corr(sentiment_t, return t+1 … t+5) | +0.010, +0.019, +0.004, −0.007, −0.026 | all within noise (±0.045) |
| Next-day direction "accuracy" | 50.4% | worse than "always up" (53.6%) |
| Bad news → higher volatility next day? | t-stat −0.18 after controlling for recent vol | no |
| Best honest strategy (5-day smoothing, cash when negative) | Sharpe 0.43 vs buy & hold 0.37 | looks like a win… |
| …vs the same strategy with randomly time-shifted news | null mean 0.47, p = 0.57 | **…it is not.** |

**The key lesson of the whole project so far.** The "sentiment" strategy beat buy & hold on Sharpe,
and a naive researcher would stop there. But a strategy that is randomly in the market half the
time *also* beats buy & hold over 2008–2016, because it sits out half of the 2008 crash by chance.
**The right benchmark is the strategy's own null distribution, not buy & hold.**

**Why it failed.**
1. **Wrong source.** World-politics headlines voted on by Reddit users are not market news.
   Same-day correlation of zero shows these headlines don't even *describe* the market.
2. **Wrong granularity.** The published LLM results (Lopez-Lira & Tang) are firm-level:
   a headline about company X predicting X's stock, mostly small caps. Aggregate world news predicting
   an index is far harder, because the index is the most efficiently priced asset in the world.
3. **Speed.** Anything in a headline about the S&P 500 is priced in within seconds.

**Next for news.** (a) Firm-level: SEC 8-K filings and per-ticker headlines, which need network
access. (b) Macro events: FOMC/CPI release days as a *risk* calendar, not a direction signal.
(c) `ClaudeScorer` on headlines *after* the model's training cutoff, which needs an API key.

---

## Open questions / next experiments
* **More breadth:** commodities and currencies are where 1970s trend profits came from.
  Shiller has no such data. We need network access (Yahoo/Stooq) or GitHub-hosted datasets.
* **Out-of-sample discipline:** freeze the SMA-10 60/40 rule now, and re-test it in 6–12 months on unseen data.
* **Firm-level news + LLM scoring** (see Exp 5).
* **Event-driven engine** for paper trading (Roadmap phase 5).
