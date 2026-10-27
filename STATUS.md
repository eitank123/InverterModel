# Project status

_Last updated: 2026-10-27_

## Where we are
**Roadmap:** Phase 1 (simulator) ✅ · Phase 2 (research hygiene) mostly ✅ · Phase 3 (real data) in progress.

| Area | State |
|---|---|
| Backtester | Single asset + portfolios, costs, cash interest/leverage, any bar frequency |
| Validation | Lookahead check, walk-forward, permutation/null tests, Probabilistic & Deflated Sharpe |
| Data | S&P 500 + 10y bonds monthly 1871–2026 · DJIA daily 2007–16 · 50k headlines 2008–16 · synthetic generators |
| News | Source list, RSS reader, finance-dictionary scorer, Claude scorer (untested live: no API key) |
| Tests | 32 passing |

## Best candidate so far (not yet validated out-of-sample)
**60/40 stocks/bonds, each asset held only while above its 10-month moving average, else cash.**
1871–2026, trading one month late, 10 bp costs: Sharpe 1.14 vs 0.90 for plain 60/40,
max drawdown −32% vs −61%, CAGR 7.0% vs 7.7%. Rule fixed as of today; re-test on new data only.

## What did *not* work (and why that is useful)
* 1-month trend signals on monthly-average prices: an artifact of how the data is built (Exp 1).
* Vol-scaled, levered trend on just stocks + bonds: lost 51% in 1966–82, too little breadth (Exp 4).
* World-news headline sentiment for the index: no predictive power at any horizon (Exp 5).

## Blockers
* **Network:** Yahoo Finance, Stooq, news sites and Wikipedia are blocked from the sandbox; only
  GitHub and PyPI work. Allowing those hosts in the environment's network settings unlocks
  daily ETF/futures data and live news.
* **Anthropic API key:** needed to run `news.ClaudeScorer`.

## Next steps (in order)
1. More breadth: commodities, currencies, international indices (needs data access).
2. Firm-level news: SEC 8-K filings + LLM scoring on post-cutoff headlines.
3. Event-driven engine (bar-by-bar) so the same code can paper trade.
4. Paper trading the frozen 60/40 + SMA-10 rule with a broker paper account (e.g. Alpaca).
