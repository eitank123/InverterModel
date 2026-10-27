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
4. **Paper trading** (`python scripts/paper_trade.py`): forward accounts advance by every newly
   completed month. Nothing is backfilled.
5. **Dashboard** (`python scripts/export_dashboard.py`): writes `loop/dashboard/*.json`, which are
   uploaded to the Algo Trading Lab dashboard.
6. Commit and push.

## Rules that stop the loop from fooling itself
* The test period is never used to choose parameters.
* A new idea must beat the current champion on the test period **and** on the forward median, or
  give a clearly higher average wealth at a similar chance of loss, before it replaces the champion.
* Directions already explored (see checkpoints) are not re-run unless something changed (new data,
  a new constraint, a bug fix).

## Directions explored so far
Leverage on trend-filtered portfolios · stock/bond mixes · Gayed-Bilello levered S&P ·
volatility targeting · trend-strength leverage · valuation-scaled Kelly · leveraged bonds ·
≤2× deployable variants · drawdown brake · levered dual momentum · Sell-in-May seasonality (watch list).

## Open directions (candidates for coming rounds)
Daily-reset leveraged ETFs (volatility decay) vs futures ·
turn-of-month effects ·
lifecycle de-leveraging (Ayres-Nalebuff) · combining valuation and trend leverage ·
more markets once data access is allowed.
