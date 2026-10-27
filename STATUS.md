# Project status

_Last updated: 2026-10-27 (after Experiments 6–11)_

## Where we are
**Roadmap:** Phase 1 (simulator) ✅ · Phase 2 (research hygiene) ✅ · Phase 3 (real data) mostly ✅ ·
Phase 4 (math / signal processing) started.

| Area | State |
|---|---|
| Backtester | Single asset + portfolios, costs, cash interest, leverage with borrow spread, any bar frequency |
| Statistics | Variance ratio, Hurst, Newey-West OLS, out-of-sample R², block bootstrap, Kelly, Probabilistic/Deflated Sharpe |
| Data | S&P 500 + 10y bonds monthly 1871–2026 (bonds reliable only from 1953) · CAPE/earnings to 2023 · DJIA daily 2007–16 · 50k headlines |
| Experiments | 11, all reproducible (`scripts/experiments/`) |
| Tests | 42 passing |

## The honest bottom line
1. **Markets are close to a random walk.** The only structure that survived testing is a weak trend at
   about 12 months and valuation mean reversion at about 10 years (Exp 6, 7).
2. **The trend filter (hold each asset only above its 10-month average) is crash insurance, not alpha.**
   Since 1957: drawdown −15% vs −27% for 60/40 (robust: 100% of bootstrap resamples), excess Sharpe
   0.63 vs 0.59 (not significant). It wins in slow bear markets and loses in V-shaped rebounds.
3. **At today's valuations (CAPE ≈ 41) every candidate is expected to earn about 4–5%/yr nominal over 10
   years.** The candidates differ in downside: P(10-year loss) is 23% for the S&P 500 and 2.7% for the filtered 60/40.
4. **Leverage is a dial, not an edge.** Kelly says 3.6× on stocks; history says 4× is ruin.

## Annual profits (simulated, monthly data)
| | 2024 | 2025 | 2026 YTD | last 12m | 10y p.a. |
|---|---|---|---|---|---|
| S&P 500 | +30.3% | +15.8% | +13.2% | +18.3% | 15.5% |
| 60/40 | +19.3% | +11.2% | +9.2% | +12.6% | 9.6% |
| 60/40 + SMA-10 (frozen) | +19.1% | +3.6% | +9.2% | +12.6% | 8.0% |
| 60/40 + SMA-10 ×1.5 | +27.6% | +3.6% | +12.4% | +17.2% | 11.0% |
| Professional trend funds (SG Trend Index) | ≈ +2.6% | ≈ +2.4% | n/a | n/a | n/a |

## Corrections made this round
* "P(trend filter beats 60/40) = 99.5%" was withdrawn. It was inflated by fake-smooth pre-1953 bond data and by
  counting cash interest in Sharpe. See research log, Exp 11.
* Bond "momentum" was carry (Exp 6).

## Blockers
* **Network:** Yahoo/Stooq/FRED/news sites are blocked; only GitHub and PyPI work. Without more markets
  (commodities, currencies, international) we can't test where trend following really earns.
* **Anthropic API key:** for `news.ClaudeScorer`.

## Next steps
1. Month-end (not averaged) prices to measure the real-time filter. Needs daily data.
2. Breadth: 20+ markets.
3. Event-driven engine and paper trading of the frozen rule.
