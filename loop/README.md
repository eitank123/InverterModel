# The improvement loop

Goal: the highest **expected long-run growth**. Lower confidence and bigger swings are
acceptable; ruin is not.

## One round
1. **Checkpoint review** (every round): read `leaderboard.csv`, `checkpoints.json` and the
   research log. Search the literature for an approach the loop has *not* tried yet. Write a
   checkpoint entry: what was found and what the next round tries, and why.
2. **New candidates**: add an `iteration_N()` to `scripts/loop.py`. Each candidate is a
   strategy builder plus a parameter grid.
3. **Evaluate** (`python scripts/loop.py --iter N --report --frontier`):
   * parameters chosen on 1957–1990 only, by highest growth;
   * judged on the untouched 1991–2026 test;
   * crash stress: the worst historical one-day falls applied to the exposure held at the time
     (a loss of 100% or more is ruin);
   * forward view: 10-year simulations at today's valuations (median, average wealth, chance of loss);
   * two financing levels (cash + 0.5% futures-like, cash + 1.5% margin) and a "fits a 2× margin
     account" flag.
4. **Real-time simulation** (`python scripts/live_sim.py`): play-money accounts that started
   2026-10-23 advance by every new real trading day (FRED data), with every cost charged.
   Nothing is backfilled. (`scripts/paper_trade.py` keeps the older monthly ledger.)
5. **Dashboard** (`python scripts/export_dashboard.py`): writes `loop/dashboard/*.json`, which are
   uploaded to the Algo Trading Lab dashboard.
6. Commit and push.

## Rules that stop the loop from fooling itself
The strategy will trade data nobody has seen, so it must not be fitted to particular past years.
* Parameters are chosen on 1957–1990 only, and only from short grids of round, published values
  (10-month average, 12-month momentum, 2×/3×).
* **Champion rule (since checkpoint 7): best WORST era.** A strategy is judged in four eras:
  1880–1929 and 1930–1956 (never used to choose anything), 1957–1990 (train) and 1991–2026
  (test, which has been looked at by many rounds and is now partly "used"). A challenger must not
  lose to the incumbent in any era (`research.beats_everywhere`) and must not raise the chance of
  ruin.
* Prefer combining independent effects with fixed equal weights over picking the single best fit.
* Count every variant tried. The walk-forward selection return (`research.walk_forward_selection`:
  each year pick the best strategy on data known then) is the honest estimate of what the loop
  itself delivers. Backtest numbers of the hand-picked champion are an upper bound.
* The only truly unseen data is the forward paper record (from 2026-09).
* **Out-of-sample market check (since checkpoint 10):** report each champion candidate's testable
  sleeves on Japan 1950–2026 (`scripts/experiments/exp19_japan_out_of_sample.py`). The trend
  filter worked there; 3× leverage came close to ruin (−94%).
* **12-market check (since checkpoint 11):** `scripts/experiments/exp20_twelve_markets.py` runs the
  US rules on 12 other markets. Candidates report median growth, worst-country growth and worst
  drawdown there, and anything that goes to zero in any market is excluded.
* Directions already explored (see checkpoints) are not re-run unless something changed (new data,
  a new constraint, a bug fix).

## Directions explored so far
Leverage on trend-filtered portfolios · stock/bond mixes · Gayed-Bilello levered S&P ·
volatility targeting · trend-strength leverage · valuation-scaled Kelly · leveraged bonds ·
≤2× deployable variants · drawdown brake · levered dual momentum · Sell-in-May seasonality (watch list) ·
recovery leverage · exit signals (SMA10 / 12-month momentum / either / both) · bonds as the risk-off asset ·
bond-trend gate · equal-weight ensembles · growth-trend timing (earnings) · multi-horizon (1/3/12-month) trend · daily 200-day exits (Exp 18, one era only; live account) · Japan out-of-sample test (Exp 19) · 12-market test (Exp 20) · global 13-market trend sleeve (Exp 21) · leverage frontier across 13 markets (Exp 22) · country momentum rotation (Exp 23).

## Open directions (candidates for coming rounds)
Live earnings data for the valuation sleeve ·
deployable 2x global sleeve via country ETFs, momentum-ranked (Exp 23; needs ETF data) ·
synthetic-market tests (does the ensemble earn on markets with no trend / no mean reversion?) ·
turn-of-month effects ·
lifecycle de-leveraging (Ayres-Nalebuff) · combining valuation and trend leverage ·
more markets once data access is allowed.
