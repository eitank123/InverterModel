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
> ⚠ Pre-1953 bond data is interpolated and too smooth. Sharpe ratios here are flattered; see Exp 11 for the clean re-check.

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

## Background reading, round 3: the math (2026-10-27)

| Source | Claim | Used in |
|---|---|---|
| Lo & MacKinlay (1988), *Stock Market Prices Do Not Follow Random Walks* | Variance-ratio test with a heteroskedasticity-robust z | `stats.variance_ratio`, Exp 6 |
| Working (1960) | Averaging a random walk induces ρ₁ = 0.25 | Exp 1, Exp 6 null |
| Poterba & Summers (1988); Fama & French (1988) | Mean reversion in stock prices at 3–5 year horizons | Exp 6 |
| Campbell & Shiller (1988, 1998) | Valuation ratios forecast long-horizon returns | Exp 7 |
| Goyal & Welch (2008); Goyal, Welch & Zafirov (update) | Most predictors fail out of sample, and most "have lost their empirical support" | `stats.oos_r2`, Exp 7 |
| Campbell & Thompson (2008) | Sign restrictions improve out-of-sample forecasts | Exp 7 |
| Kelly (1956); Thorp; MacLean, Thorp & Ziemba (2011) | Growth-optimal betting f* = μ/σ² and its drawdown risk | Exp 8 |
| Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere* | Value and momentum are negatively correlated (≈ −0.5), so combining them helps | Exp 7 |
| Politis & Romano (1994) | Stationary block bootstrap | `stats.bootstrap_stat`, Exp 9–11 |
| Industry check: SG Trend Index | ≈ +2.6% in 2024 and ≈ +2.4% in 2025 (secondary sources), versus S&P 500 ≈ +18% in 2025 | Context for "annual profits" |

---

## Experiment 6: Is the market a random walk?
`exp06_random_walk.py` · variance ratios at 2–120 months vs a simulated *averaged* random-walk null

* The textbook test says "not a random walk" (z = +7 at 2 months), but that is **entirely the
  averaging artifact**. Against the right null, stocks are a random walk at nearly every horizon.
* 12 months: VR 1.66 vs 95% null 1.65, a borderline **trend**. This matches TSMOM's 12-month horizon.
* 10 years: VR 0.64 before 1945 (**mean reversion**, as Poterba & Summers found). None after 1945.
* Annual autocorrelations of real returns: lag 2 = −0.22 (just significant), others noise.
* Bonds "trend" at every horizon, but subtracting carry (yield/12) removes it all except at 2 months.
  The predictable part of bonds is their **yield**, which is known in advance.
* Hurst exponent: S&P 0.64, inside the averaged-random-walk band [0.565, 0.642].

**Lesson:** the edge, if any, is tiny and lives at about 12 months (trend) and about 10 years (valuation).
Everything in between is noise.

---

## Experiment 7: Does valuation predict returns?
`exp07_valuation.py` · CAPE and excess CAPE yield (ECY), published with a 6-month lag

| Horizon | log CAPE: coef (NW t) | in-sample R² | OOS R² (plain / Campbell-Thompson) |
|---|---|---|---|
| 1 month | −0.09 (−2.7) | 0.6% | n/a |
| 12 months | −0.09 (−2.9) | 4% | −1.3% / −0.5% (ECY: +2.0% / +3.3%) |
| 5 years | −0.06 (−2.8) | 12% | −21% / −9% |
| 10 years | −0.06 (−4.6) | 24% | +2% / +6% (ECY: −28% / +16%) |

* In-sample evidence is strong. Out-of-sample it is unstable, the Goyal–Welch pattern exactly.
* Regression-implied 10-year real return at CAPE 30.8 (last in data, 2023): +2.3%/yr. **At CAPE ≈ 41
  (October 2026): +0.6%/yr real.**
* A real-time valuation tilt (30–90% stocks by ECY percentile) raised CAGR 8.0% → 8.9%. Combined with
  SMA-10 it gave the best Calmar ratio (0.29, max drawdown −27%). Today it would hold **48% stocks**.
* Value and trend active returns: ρ = −0.05, not the −0.5 that AMP report (they used cross-sectional
  value and momentum across many assets, not timing on one market).

---

## Experiment 8: Kelly sizing and leverage
`exp08_kelly_leverage.py` · borrowing at cash + 1%

* Full-sample Kelly: stocks 3.6×, 60/40 6.4×, 60/40+SMA10 10.7×. At 4× on stocks, 1929 wipes you out.
  60/40+SMA10 still compounds at 6×, but with a −99.5% drawdown. Kelly ignores fat tails, and no
  human sits through that.
* The trailing 30-year Kelly estimate ranged 1.0–9.4. A rolling half-Kelly reduced Sharpe in every case.
* **60/40+SMA10 at 1.5×: 8.6% CAGR, −46% max DD**, about stock-like returns with half the drawdown.
  (Exp 11 later shows that leverage adds nothing to *excess* Sharpe; it is just a dial.)

---

## Experiment 9: Annual profits, robustness, and the next 10 years
`exp09_robustness_and_profits.py`

**Annual profits (simulated; monthly data; 2026 = Jan–Sep):**

| | 2022 | 2023 | 2024 | 2025 | 2026 YTD | last 12m | 10y p.a. | 100y p.a. | worst year |
|---|---|---|---|---|---|---|---|---|---|
| S&P 500 | −15.0% | +21.7% | +30.3% | +15.8% | +13.2% | +18.3% | 15.5% | 10.5% | −42% (1931) |
| 60/40 | −14.8% | +12.7% | +19.3% | +11.2% | +9.2% | +12.6% | 9.6% | 8.5% | −26% (1931) |
| 60/40+SMA10 | −2.8% | +8.8% | +19.1% | +3.6% | +9.2% | +12.6% | 8.0% | 7.6% | −10% (1937) |
| 60/40+SMA10 ×1.5 | −5.0% | +11.9% | +27.6% | +3.6% | +12.4% | +17.2% | 11.0% | 9.3% | −16% (1937) |

The filter won big in 2000–02 (+4%/yr vs −20%/yr for stocks) and in 2008 (+7.6% vs −39%), and lagged in fast
rebounds (2020, 2025).

**Parameter plateau:** Sharpe is flat across SMA windows of 6–16 months (8 is best, 10 is fine), so the result is not a lucky parameter.

**Next 10 years (1000 block-bootstrap paths):**

| | historical means: median CAGR | valuation-adjusted: median CAGR | P(10-year loss), valuation-adjusted |
|---|---|---|---|
| S&P 500 | 10.7% | 4.2% | 23% |
| 60/40 | 8.3% | 4.4% | 12% |
| 60/40+SMA10 | 7.1% | 4.3% | 2.7% |
| value+SMA10 | 6.4% | 4.2% | 1.4% |

At today's valuations, all candidates are expected to earn about 4–5%/yr nominal. **The strategies
differ in risk, not in expected return.**

---

## Experiment 10: Ten ways to reduce whipsaw (train 1901–1960, test 1961–2026)
`exp10_improve_filter.py`

* Train/test rank correlation of Sharpe: +0.32. Train rankings barely predict test rankings.
* The train winner (inverse-vol + SMA, train Sharpe **3.04**) was an artifact of fake-smooth early bond data (next entry).
* No variant beat the frozen SMA-10 rule with confidence (best P = 0.76). Partial de-risking cut whipsaw losses in
  2020/2025 but gave up most of the 2008 protection. **You cannot tell a V-crash from a slow bear in advance.**
* Decision: **keep SMA-10 frozen.**

---

## Experiment 11: Data flaw found, conclusions re-checked on 1957–2026
`exp11_clean_bond_era.py`

**Flaw.** Before 1953 Shiller's long rate is interpolated from annual data: monthly yield changes have
autocorrelation 0.85 (0.3 after 1953). Rebuilt bond returns then have 0.5% volatility and a raw Sharpe of 5–9,
which is impossible for real bonds. Every stock/bond Sharpe that included pre-1953 data was flattered.

**Re-check, 1957–2026, Sharpe in excess of cash:**

| | CAGR | excess Sharpe (95% CI) | P(beats 60/40) | max DD |
|---|---|---|---|---|
| S&P 500 | 10.7% | 0.56 (0.26–0.85) | 25% | −49% |
| 60/40 | 8.9% | 0.59 (0.30–0.90) | n/a | −27% |
| **60/40+SMA10** | 8.0% | 0.63 (0.36–0.89) | **63%** | **−15%** |
| S&P 500+SMA10 | 9.9% | 0.63 (0.39–0.87) | 63% | −25% |
| value+SMA10 | 8.0% | 0.64 (0.39–0.90) | 68% | −16% |
| 2-asset vol-scaled TSMOM | 4.1% | 0.04 | 0% | −63% |

**Correction of earlier claims.** "P(60/40+SMA10 beats 60/40) = 99.5%" (Exp 9) is withdrawn. It came from
raw Sharpe (cash interest counted as skill) plus fake-smooth pre-1953 bonds. On clean data the risk-adjusted
improvement is small and **not significant**.

**What survives, robustly:** the filter halves drawdowns. Bootstrap P(smaller max DD) = 100% for both 60/40
(−15% vs −27%) and the S&P 500 (−25% vs −49%). The worst 12-month return also improves in 99% of resamples
(−12% vs −24%). **The trend filter is crash insurance with a small premium, not alpha.**

---

## Experiment 12: Is the drawdown reduction just lower exposure?
`exp12_insurance_or_exposure.py` · control = the same base portfolio at a **constant** exposure with the filter's volatility

A filter that is out of the market 25% of the time is less volatile, and less volatility alone shrinks drawdowns.
The fair comparison is a vol-matched constant-exposure portfolio.

| 1957–2026 | CAGR | vol | max DD | P(filter DD smaller) | P(filter CAGR higher) |
|---|---|---|---|---|---|
| S&P 500 + SMA10 | 9.9% | 9.0% | −25% | 92% | 84% |
| S&P 500 at constant 0.74× | 9.2% | 9.0% | −38% | | |
| 60/40 + SMA10 | 8.0% | 5.9% | −15% | 90% | 69% |
| 60/40 at constant 0.74× | 7.7% | 5.9% | −20% | | |

Synthetic check (100 × 40-year markets, truth known):

| Market | Filter DD better than control | Filter CAGR better |
|---|---|---|
| GBM, no serial structure | 40% | 15% |
| GARCH, vol clusters only | 29% | 13% |
| Regime switching (trends) | **80%** | **82%** |

**Lesson.** Where returns have no trend structure, the filter is *worse* than simply holding less. It
helps only when trends exist. On real data it beat the control about 90% of the time, which is indirect
evidence that real markets have exploitable slow trends (mostly slow bear markets), consistent with the
12-month variance ratio of Exp 6. It is not proof (p ≈ 0.1), but the pieces agree.

---

## Experiment 13: How to implement the filter in practice
`exp13_implementation.py` · daily trading

| DJIA 2007–2016 | CAGR | Sharpe | max DD | switches/yr |
|---|---|---|---|---|
| buy & hold | 4.7% | 0.34 | −54% | n/a |
| monthly SMA10, month-end close | 4.4% | 0.46 | −16% | 1.5 |
| monthly SMA10, month-average | 4.3% | 0.43 | −19% | 1.5 |
| daily SMA200 | 3.1% | 0.36 | −16% | **7.7** |
| daily SMA200 ±2% band | 4.4% | **0.49** | −16% | 1.9 |

Over 100 synthetic markets the ranking is the same: the banded daily filter beats month-end in 64% of
seeds, plain daily in 55%, and month-average in 39%.

**Lesson.** Month-end closes are at least as good as the averages our long backtests used, so the
real-world rule is not weaker than the research version. Checking daily without a band triples trading
for nothing. **Recommended implementation: monthly check on month-end closes, or a daily 200-day SMA with a ±2% band.**

`python scripts/today.py` prints what the frozen rules hold now. As of 2026-09: stocks +7.1% above their
10-month average and bonds +1.5% above, so both are IN. 60/40+SMA10 = 60% stocks / 40% bonds. The value tilt = 48/52
(its CAPE input is stale since 2023 and would be even lower at today's CAPE ≈ 41).

---

## Experiment 14: An inflation-regime guard for the bond sleeve
`exp14_inflation_regime.py` · train 1957–1990, test 1991–2026, CPI lagged 2 months

**Hypothesis.** Cutting bonds to cash when inflation is high and rising avoids the regimes where bonds
stop hedging (Exp 4).

| Excess Sharpe | train 1957–90 | test 1991–2026 |
|---|---|---|
| 60/40 | 0.35 | 0.85 |
| 60/40+SMA10 [frozen] | **0.28** | **1.01** |
| best guard on train (6%, rising) | 0.37 | 1.01 |

* The guard looked good on train (+0.09) and added nothing on test (+0.004). It is active in only 3% of
  test months, and the bond trend filter already exits bonds when rising inflation pushes yields up. **Rejected.**
* **New finding:** in 1957–1990 the frozen trend filter had a *lower* excess Sharpe than plain 60/40. All of its
  risk-adjusted advantage comes from 1991–2026 (and its drawdown benefit holds in both). So results depend on
  the era: an edge that is real over 70 years can be absent for 30. Size expectations accordingly.

---

## Experiment 15: Kelly leverage only where it is "safe" (incl. ridge and a neural net)
`exp15_kelly_safe_leverage.py` · 1957–2026 · Kelly fraction / cap / vol target chosen on TRAIN 1957–1990 by
"highest CAGR with max DD ≥ −25%", TEST 1991–2026 reported once · leverage costs cash + 1%

**Idea.** Multi-asset Kelly w = λ·Σ⁻¹·μ levers calm assets and calm periods. A trend gate sets μ = 0 below
the 10-month SMA. Ways to estimate μ: K1 literature priors (equity premium 5%, term premium 1.5%), K2 valuation
(excess CAPE yield), K3 ridge regression, K4 neural-net ensemble (walk-forward, 12 features), and K5 = K1 with a
volatility floor.

**ML diagnostic first:** out-of-sample R² vs the historical mean is negative for every model
(ridge −1.3% / −0.8%, NN −2.4% / −0.6% for stocks / bonds). Sign hit-rates of 53–56% equal the base rate of up-months.
**The models have no forecasting skill**, the Goyal–Welch result again. (The first NN version had R² of −917%:
an unscaled target made it extrapolate wildly. A good reminder that NNs fail loudly on financial data.)

| 1957–2026 | CAGR train | CAGR **test** | CAGR full | max DD full | excess Sharpe full | avg gross |
|---|---|---|---|---|---|---|
| 60/40 | 8.7% | 9.0% | 8.9% | −27% | 0.59 | 1.00 |
| 60/40+SMA10 | 7.6% | 8.4% | 8.0% | −15% | 0.63 | 0.73 |
| **60/40+SMA10 ×1.5** (train-picked) | 8.0% | **10.9%** | **9.5%** | **−23%** | 0.60 | 1.10 |
| **60/40+SMA10 vol-target 10%** (train-picked) | 8.9% | **12.4%** | **10.7%** | **−24%** | 0.57 | 1.46 |
| K1 Kelly priors | 5.9% | 5.8% | 5.8% | −9% | 0.50 | 0.41 |
| K2 Kelly valuation | 6.6% | 6.4% | 6.5% | −24% | 0.40 | 0.71 |
| K3 Kelly ridge | 10.0% | 13.7% | 11.9% | **−38%** | 0.60 | 1.14 |
| K4 Kelly neural net | 8.6% | 8.5% | 8.5% | **−51%** | 0.40 | 0.77 |
| K5 Kelly priors + vol floor | 7.0% | 9.5% | 8.3% | −19% | 0.54 | 1.10 |

* Kelly with priors levered bonds about 4×. Rising rates in 1957–81 forced the train-selected fraction down to 0.05,
  leaving an under-invested portfolio. **Low volatility ≠ low risk.** The vol floor (K5) fixes part of it.
* Ridge and NN broke the drawdown budget out of sample (−38%, −51%). Their train-period CAGR was not skill.
* Vol-targeting ("lever only when calm") vs constant leverage with the same volatility: CAGR 10.7% vs 10.8%,
  P(higher CAGR) 40%, P(smaller DD) 62%. **Timing the leverage adds nothing reliable.**
* **Financing decides it.** Trend-filtered 60/40 at 2×: 11.4% (free), 10.5% (+1.5%), 9.6% (+3%), 8.3% (+5%), vs
  8.9% for plain 60/40. Futures-level financing is required for leverage to be worth it.

**Answer to "can we beat 8.6% a year?":** yes, robustly. 60/40+SMA10 at 1.5× gives 9.5%/yr since 1957 and 10.9% on the
untouched test period with a −23% max DD; ~2× or a 10% vol target gives ~10.7%/yr with −24 to −30%. The gain comes
from *how much* leverage, made tolerable by the trend filter. No forecasting model (ridge, NN, Kelly μ-estimates)
added anything. And at today's valuations all of these numbers should be expected to be lower.

---

## Experiment 16 — 2×/3× daily-reset ETFs vs margin (`scripts/experiments/exp16_etf_vs_margin.py`)
* A 2× ETF (SSO) costs about the same as cheap margin and beats retail margin (cash + 1.5% or more).
* A 3× ETF (UPRO) beat 3× margin in 83% of the bootstrap simulations. Its daily-reset drag L(L−1)/2·σ² is smaller than the margin spread.
* **Leverage up to 3× is deployable without a margin account**, so `research.evaluate` now reports a `deployable_etf` flag.

## Improvement loop, rounds 1–5 (`scripts/loop.py`, `loop/checkpoints.json`)
* **New champion: recovery leverage.** It holds the S&P 500 at 2× above its 10-month average, at 3× while the index is more than 15% below its 36-month high, and in cash otherwise.
  * Parameters were chosen on 1957–1990: train 11.6%/yr, untouched 1991–2026 test 19.0%/yr, max drawdown −49%.
  * The plain 2× trend version tests at 18.1%, with a −41% loss on the crash-day stress test.
* **These are leveraged numbers from strong periods.** The S&P 500 itself made 11.4%/yr over 1991–2026 and about 10.7%/yr since 1957.
* At today's valuations (CAPE ≈ 41) the forward 10-year median is about 4.3%/yr for the champion and for the S&P 500. The champion's average wealth is higher (1.89× vs 1.61×), and its chance of a 10-year loss is about 20%.

## 5-year full-cost simulation (`algotrading/execution.py`, `scripts/simulate_last5.py`)
* Setup: $100k from 2021-10 to 2026-09, trading real tickers.
* Every cost is charged:
  * $1 commission per order;
  * spread and slippage;
  * the SEC fee and FINRA TAF on sales;
  * ETF expense ratios;
  * leveraged-ETF financing and daily-reset drag;
  * the cash sweep fee.

| Strategy | End value | CAGR | Max drawdown | Trades | All costs |
|---|---|---|---|---|---|
| Champion (SSO; the 3× boost never triggered) | $247,322 | 19.9% | −22% | 5 | $24,187 (trading costs $319) |
| S&P 500 buy & hold (SPY) | $184,899 | 13.1% | −19% | 1 | small |
| 60/40 (SPY/IEF) | $143,325 | 7.5% | −19% | 18 | $692 |

* Almost all costs sit inside the funds (financing and expense ratio). Trading costs are negligible.
* Pre-tax. The 2025 April whipsaw (sell, then rebuy two months later) cost most of that year's gain.

---

## Loop rounds 6–8: exit signals, bonds when out, and the overfitting audit
**Round 6.** Exiting only when *both* the 10-month average and 12-month momentum turn down kept recovery leverage invested through whipsaws:
- test result: 21.8% vs 19.0%;
- trend switches: 125 vs 187.

Holding 2× Treasuries instead of cash when out raised training growth to 13.6%. In 2022 it failed: the exit came late and bonds fell with stocks. In the full-cost 5-year simulation it made 16.6%/yr against 19.9% for the old rule.

**Round 7.** Holding bonds only while they trend up cut 2022 from −35% to −30%, but cost 1.3 points a year in training. Rejected.

**Overfitting audit (prompted by the user).**
- **Trials.** 219 variants have been tried against the same 1991–2026 test.
- **Walk-forward selection.** Running the selection itself in real time (each year, pick the best strategy on data known then) earned 14.9%/yr on 1991–2026, against 22.5% for the hand-picked champion.
- **Significance.** The probability that the champion's Sharpe truly beats the S&P's is 0.86, below the 0.95 bar.
- **Unseen eras.** In 1880–1956, never used to choose anything, the recovery family made only 8.7%/yr in 1930–56, no better than the S&P. Its "3× after a crash" boost was caught in the false recoveries of 1930–32. Trend and valuation leverage made 12–19% in the same era.

**Round 8: ensemble.** Equal thirds of 3× trend, valuation-scaled leverage (≤3×) and recovery+, with no fitted weights:

| | 1880–1929 | 1930–1956 | 1957–1990 | 1991–2026 | Worst era | Forward median | Forward average wealth | P(10-year loss) |
|---|---|---|---|---|---|---|---|---|
| Ensemble (new champion) | 15.4% | 14.6% | 13.1% | 21.3% | **13.1%** | 4.7% | 1.98× | 17.5% |
| Recovery+ | 14.0% | 8.7% | 13.6% | 22.5% | 8.7% | 6.8% | 2.30× | 14.5% |
| S&P 500 | 8.3% | 8.8% | 10.0% | 11.4% | 8.3% | 4.3% | 1.61× | 18% |

5-year full-cost simulation for the ensemble: $100k → $247,966 (19.9%/yr), max drawdown −26%, 47 trades costing $398 in total.

**Lesson.** A strategy found by searching many ideas on one history is partly a description of that history. Defences:
- a different, untouched era;
- fixed equal-weight combinations of independent, published effects;
- judging by the worst era, not the best;
- the forward paper record, which is the only truly unseen data.

---

## Experiment 17: daily trading vs the monthly approach (`scripts/experiments/exp17_daily_trading.py`)
Data: DJIA daily 2007–2016, a price index with no dividends, the only daily data reachable offline. All rules are published ones with no fitted parameters. Signals trade one day after they fire.

| Rule (cost per side) | 0bp | 1bp | 5bp | Trades/yr | Max DD |
|---|---|---|---|---|---|
| Buy & hold | 4.8% | 4.7% | 4.7% | 0 | −54% |
| 200-day average | 3.0% | 3.0% | 2.6% | 4 | −16% |
| 20-day momentum | 3.2% | 2.9% | 1.8% | 13 | −26% |
| 1-day reversal | 2.4% | 1.1% | −4.1% | 66 | −23% |
| RSI(2) (Connors) | 0.9% | 0.7% | 0.2% | 7 | −14% |
| Turn of the month | 0.1% | −0.2% | −1.1% | 12 | −30% |
| Intraday only | 6.1% | 0.9% | −17.6% | 252 | −51% |

* **No daily rule beat buy & hold even with free trading,** except "intraday only", whose edge disappears at 0.3bp per side.
  * The overnight/intraday split is unreliable on index data, because index opens are stale.
* **Costs scale with turnover:**
  * 66 trades a year at 5bp costs about 6.6% a year, more than the whole market return;
  * the monthly champion trades about 10 times a year, costing well under 0.1% a year.
* The daily 200-day average behaves like the monthly 10-month rule:
  * less return in this window (it sat out the 2009 rebound);
  * a much smaller drawdown (−16% vs −54%).
* **Why:** short-horizon predictability in index prices is tiny (Exp 6), and it is where the professionals compete hardest:
  * market makers and high-frequency firms;
  * colocated servers;
  * rebates for providing liquidity.

  A retail account pays the spread on every round trip. Short-term gains are also taxed at higher rates in many countries (e.g. the US).
* **Where daily data can still help:** faster crash exits for the monthly rules (an open direction). Testing that needs daily SPY/IEF bars, which come with the Alpaca connection.

---

## Loop round 9: two published whipsaw cures, neither beats the champion
| | 1880–1929 | 1930–56 | 1957–90 | 1991–2026 | Max DD |
|---|---|---|---|---|---|
| Growth-trend timing 2× (earnings) | 9.8% | 14.7% | 7.9% | 20.2% | −75% |
| Multi-horizon trend 3× (1/3/12 m) | 10.4% | 16.4% | 9.5% | 20.2% | −59% |
| Ensemble with growth-trend sleeve | 14.2% | 16.1% | 11.8% | 22.4% | −55% |
| Ensemble + multi-horizon (4 sleeves) | 14.4% | 15.6% | 12.3% | 21.2% | −54% |
| **Champion ensemble** | 15.4% | 14.6% | **13.1%** | 21.3% | −53% |

* **Growth-trend timing failed in 1973–74.** In an inflationary bust, nominal earnings kept growing, so the "economy is weakening" gate never opened and the rule stayed invested at 2× through the crash.
* **Multi-horizon trend was worse on one market.** The 1-month signal adds noise. Its published edge comes from 67 markets, not one.
* **228 variants tried so far.** The walk-forward selection estimate is unchanged: 14.3%/yr since 1957.

---

## Experiment 18: a daily crash exit for the 3× trend sleeve (`scripts/experiments/exp18_daily_crash_exit.py`)
Daily data 2007–2026: DJIA to 2016 chained to the FRED S&P 500, T-bills from FRED, 5bp per switch, 3× daily-reset exposure.

| 3× rule | CAGR | Max DD | 2008 | 2020 | 2022 | 2007–16 | 2017–26 |
|---|---|---|---|---|---|---|---|
| Monthly 10-month average (current sleeve) | 14.1% | −64% | −3.8% | +51.6% | −54.2% | 11.1% | 17.2% |
| Daily 200-day (Gayed-Bilello) | 15.9% | −55% | −8.0% | +35.7% | −42.6% | 7.6% | 25.0% |
| Daily 200-day ±2% band | 16.5% | −58% | −3.5% | −8.6% | −41.0% | 11.7% | 21.6% |
| Monthly + daily emergency exit | 11.7% | −56% | −3.8% | +33.2% | −42.6% | 7.0% | 16.8% |

* No rule dominates.
* The banded daily rule beat the monthly one in both halves. It cut the 2022 loss, but lost money in 2020 because it sold during the crash and missed the rebound.
* This covers one era only, so the champion is unchanged. The rule runs as a new real-time simulation account (checked daily) to collect unseen evidence.

---

## Experiment 19: Japan, a market none of the rules were designed on (`scripts/experiments/exp19_japan_out_of_sample.py`)
Data: Nikkei 225, 1950–2026, from FRED. Month-ends, dividends added at 1.5%/yr, US T-bill cash. Same rules as in the US, nothing re-fitted.

| Rule | 1950–89 boom | 1990–2012 bust | 2013–26 | 1950–2026 | Max DD |
|---|---|---|---|---|---|
| Buy & hold 1× | 17.6% | −4.2% | 15.7% | 10.3% | −75% |
| Buy & hold 3× | 29.9% | −31.6% | 34.0% | 7.8% | −100% |
| 10-month average 1× | 14.7% | 0.3% | 11.9% | 9.7% | **−38%** |
| 10-month average 2× | 21.4% | −4.3% | 19.4% | 12.7% | −77% |
| 10-month average 3× | 24.8% | −10.5% | 24.4% | 12.9% | **−94%** |
| Recovery+ | 24.2% | −12.0% | 18.6% | 11.1% | −97% |
| Champion-like mix* | 22.6% | −6.5% | 19.4% | 12.5% | −87% |
| Lower-leverage mix* (trend 2×, recovery 1–1.5×) | 17.9% | −2.0% | 14.6% | 11.0% | −61% |

\*The valuation sleeve is replaced by a neutral 1× trend sleeve, because there is no Japanese earnings data.

* **The trend filter travels.** Out of sample it halved the drawdown and broke even through the bust.
* **3× leverage does not travel.** In a 20-year sideways bust, whipsaws at 3× compound toward ruin.

Lower-leverage ensemble on US history:
* era growth 12.5% / 12.9% / 11.7% / 15.8%;
* worst era 11.7% (champion 13.1%);
* max DD −39% (−53%);
* forward P(10-year loss) 8% (18%).

The choice between them is about risk appetite. Both now run in the real-time simulation.

---

## Experiment 20: the US rules on 12 other markets (`scripts/experiments/exp20_twelve_markets.py`)
Data: OECD monthly share prices (FRED) for Germany, the UK, France, Italy, Canada, Australia, the Netherlands, Switzerland, Sweden, Spain, Korea and Japan, mostly from the 1950s to 2026. Dividends added at 3%/yr, US T-bill cash, nothing re-fitted.

| Rule | Median growth | Worst country | Median max DD | Worst max DD | DD worse than −80% |
|---|---|---|---|---|---|
| Buy & hold 1× | 9.4% | 7.2% | −59% | −69% | 0 of 12 |
| Trend 1× | 8.6% | 7.3% | −39% | −48% | 0 |
| Trend 2× | 10.7% | 7.5% | −75% | −88% | 3 |
| Trend 3× | 12.0% | 4.5% | −94% | wiped out (Australia) | 12 |
| Recovery+ | 11.2% | 8.1% | −80% | −92% | 6 |
| Champion-like mix* | 11.2% | 7.9% | −75% | −89% | 2 |
| Lower-leverage mix* | 9.7% | 7.9% | −52% | −66% | 0 |

\*The valuation sleeve is replaced by a neutral 1× trend sleeve, because there is no earnings data for these markets.

* **The trend filter travels.** It halved drawdowns in every market.
* **The champion's structure beats buy & hold out of sample** in 11 of 12 markets, by +1.8 points a year at the median. The cost is drawdowns that are deep everywhere.
* **3× on a single sleeve is not survivable.** On monthly data it went to zero in Australia's 1987 crash.
* **The ensemble survived everywhere.** With daily-reset ETFs, a sleeve can lose at most what is in it.
* **Lower leverage costs median growth (9.7% vs 11.2%)** but has the same worst-country growth and much shallower falls.

---

## Experiment 21: a global trend sleeve (`scripts/experiments/exp21_global_trend.py`)
Setup: 13 markets (OECD monthly data via FRED), each held while above its own 10-month average, equal weight, rules fixed in advance.

| Sleeve / mix | 1961–90 | 1991–2026 | 1961–2026 | Max DD |
|---|---|---|---|---|
| US trend 3× | 9.8% | 17.2% | 13.8% | −71% |
| 13-market trend 1× | 10.0% | 8.3% | 9.1% | −23% |
| 13-market trend 3× | 14.3% | 16.4% | 15.4% | −65% |
| Champion (US trend sleeve) | 13.8% | 21.2% | 17.8% | −53% |
| Champion with global 3× trend sleeve | 14.7% | 18.8% | 16.9% | −50% |

* **Trend plus leverage is not a US artefact:** diversified across 13 markets it earns more with shallower falls.
* **It does not replace the US trend sleeve.** It loses in 1991–2026, so it fails "must not lose anywhere".
* **It is hard to implement:** 3× on 12 foreign markets needs futures or margin.

---

## Data fix: series the Shiller file stopped updating in 2023
**What was stale.** The source file stopped updating several series after 2023 and the loader carried the last values forward:
* the 10-year yield, CPI and CAPE stopped in 2023-09;
* earnings stopped in 2023-06.

**Effects:**
* bonds and cash used a frozen 4.09% yield, while the real 10-year yield rose to 5.0%;
* the valuation sleeve saw CAPE 30.8 instead of about 41, so it held 1.27× stocks instead of 0.99×.

**Fix.** `data/fred_monthly.csv` holds GS10, CPIAUCSL, TB3MS and CP from FRED, and `data._extend_with_fred` splices them in. The recomputed CAPE matches Shiller's within 0.2% (median, 1900–2023).

**Results after rerunning everything:**
* champion worst era 13.1% (unchanged), 1991–2026 21.2% (was 21.3%);
* 5-year simulation: champion 19.6%/yr (was 19.9%), 60/40 6.9% (was 7.5%).

---

## Experiment 22: how much leverage? (`scripts/experiments/exp22_leverage_frontier_13_markets.py`)
| Champion scaled by | Median growth (13 markets) | Avg log growth | Worst market | Worst max DD | US worst era |
|---|---|---|---|---|---|
| 0.33× | 7.5% | 7.8% | 6.7% | −32% | 7.0% |
| 0.50× | 8.7% | 9.2% | 7.5% | −52% | 9.4% |
| 0.67× | 9.7% | 10.4% | 7.9% | −67% | 11.4% |
| 0.83× | 10.6% | 11.4% | **8.1%** | −78% | 12.3% |
| 1.00× (champion) | **11.2%** | **12.2%** | 7.9% | −85% | **13.0%** |

* **Expected growth in an unseen market is still rising at the champion's leverage,** so the champion is not past the growth-optimal (Kelly) point. No market was ruined at any scale.
* **The worst market does best at about 0.83×.** That is the protective choice if the owner wants one.

---

## Experiment 23: country momentum rotation (`scripts/experiments/exp23_country_momentum.py`)
Rule fixed in advance: hold the top 4 of 13 markets by 12-1 month return, equal weight, each only while above its 10-month average.

| | 1961–90 | 1991–2026 | 1961–2026 | Max DD |
|---|---|---|---|---|
| All 13, trend 1× | 9.8% | 8.3% | 8.9% | −23% |
| Top 4 by momentum, trend 1× | 13.8% | 9.0% | 11.2% | −23% |
| All 13, trend 3× | 13.5% | 16.4% | 15.1% | −65% |
| Top 4 by momentum, trend 3× | 23.9% | 17.1% | 20.2% | −67% |

* **The edge holds up but has decayed.** Excess over all 13 markets: +4.0 points a year in 1961–90, +0.8 in 1991–2013, +0.7 in 2014–26 (after publication).
* **It is robust to the number of markets held** (top 2 to 6).
* **Not a champion candidate yet:** there is no data before 1960, the post-publication edge is small, and it needs 12 country ETFs.

---

## Experiment 24: what if markets stop trending? (`scripts/experiments/exp24_no_trend_world.py`)
Setup: 200 synthetic 70-year histories resampled from 1957–2026. Shown as median growth (5th percentile).

| | No structure (i.i.d.) | Structure kept (24-month blocks) |
|---|---|---|
| S&P 1× | 10.6% (8.2%) | 10.7% (7.9%) |
| S&P 3× buy & hold | 17.7% (9.2%) | 18.0% (8.3%) |
| Trend 3× sleeve | 13.9% (5.3%) | 17.5% (10.1%) |
| Champion | 13.3% (7.3%) | 16.3% (10.8%) |

* **Most of the champion's excess comes from leverage.** Timing (trend and recovery) adds the rest, and pays only if market structure persists.
* **With structure,** the champion has the best bad case of all.
* **Without structure,** it lags plain leverage by about 3 points a year, but still beats the S&P in 84% of paths.
* **Caveat:** resampling from 1957 onward leaves out 1929–32, when 3× buy & hold was wiped out.

---

## Taxes: the 5-year simulation after a 25% capital-gains tax
| | Pre-tax per year | After tax, sold at the end | Tax paid along the way |
|---|---|---|---|
| Champion | 19.6% | 15.7% | $10,372 |
| S&P 500 buy & hold | 13.1% | 10.4% | $0 (deferred) |
| 60/40 | 6.9% | 5.4% | $559 |

* **Switching costs tax.** Each switch realizes gains, which costs the champion about 3.9 points a year against 2.7 for buy & hold.
* **The lead survives tax:** about +5 points a year after tax.
* **A tax-deferred account would remove the drag.**

---

## Open questions / next experiments
* More breadth (commodities, currencies, international), which is where trend following's real edge lives.
  Needs data access.
* Month-end prices instead of monthly averages, to measure the real-time version of the filter.
* Firm-level news + LLM scoring (Exp 5 follow-up).
* Paper trading the frozen rule through `scripts/rebalance.py` (needs broker keys + network access; see docs/AUTOMATION.md).
