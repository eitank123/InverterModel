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

## Open questions / next experiments
* **Exp 3: Is it the trend or the volatility scaling?** Reproduce Kim, Tse & Wald's
  critique: compare TSMOM with and without vol scaling, and buy & hold with vol scaling.
* **Exp 4: A second real asset.** Build a 10-year Treasury total-return series from
  Shiller's bond yields (price ≈ −duration × Δyield + carry). Then test 60/40 versus
  60/40 + trend, which is the real use case for trend following: crisis diversification.
* **Real daily data** (ETFs, futures) needs network access to a data provider (e.g. Yahoo Finance).
* **Event-driven engine** for paper trading (Roadmap phase 5).
