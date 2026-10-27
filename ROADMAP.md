# Learning roadmap: from zero to algorithmic trading

Assumes you know probability, linear algebra, signals & systems, and Python. Most of
the work is learning not to fool yourself. The math is the easy part for you.

## Phase 0: Market basics (1–2 weeks, in parallel with Phase 1)
- How an exchange works: order book, bid/ask spread, market vs. limit orders, slippage.
- Asset classes: equities, ETFs, futures, FX, crypto. What "going short" and leverage mean.
- Returns: simple vs. log returns, compounding, why volatility drags down growth.
- Reading: *Trading and Exchanges* (Larry Harris), chapters 1–6. Or Ernest Chan,
  *Quantitative Trading*, for a faster, practical start.

## Phase 1: Simulation environment ✅ (this repo)
- Synthetic markets where the ground truth is known (GBM, OU, regime switching).
- Backtester with explicit timing and costs. Metrics. Lookahead detection.
- **Exercise:** run `scripts/run_demo.py`. Explain *why* each strategy wins or loses in
  each market. Then change `cost_bps` to 10 and see which strategies survive.

## Phase 2: Research hygiene (2–3 weeks), the most important phase
- Overfitting and multiple testing: `scripts/overfitting_demo.py`. Read about the
  Deflated Sharpe Ratio (Bailey & López de Prado).
- Train/test split, walk-forward, and why you can only use the test set **once**.
- Survivorship bias, lookahead bias, data snooping, and regime changes.
- Reading: López de Prado, *Advances in Financial Machine Learning*, chapters 7, 11, 12.
- **Exercise:** add a test showing that a strategy tuned on a random walk has an
  out-of-sample Sharpe near zero across many seeds.

## Phase 3: Real data and classic strategies (3–4 weeks)
- Download daily data for liquid ETFs (SPY, QQQ, TLT, GLD) using `load_yahoo`.
- Reproduce known effects: time-series momentum (Moskowitz, Ooi & Pedersen 2012),
  volatility targeting, and the 60/40 portfolio.
- Extend the framework to **multiple assets** (portfolio weights instead of a single
  position). This is the next coding milestone.
- Pairs trading / cointegration (Engle-Granger, Johansen), where your OU knowledge applies.

## Phase 4: Signal processing and ML (ongoing, plays to your EE strengths)
- Filters as indicators: an EMA is a first-order IIR low-pass filter. Look at phase
  lag vs. noise rejection, and Kalman filters for dynamic hedge ratios.
- Spectral and wavelet analysis of returns (and why markets are mostly noise).
- Feature engineering, then linear/ridge models before trees and before deep learning.
  Use purged k-fold cross-validation for time series.
- Position sizing: Kelly criterion, risk parity, drawdown control.

## Phase 5: Paper trading (1–3 months)
- Event-driven engine: bars arrive one at a time, the same code runs in backtest and live.
- Broker API with a paper account: Alpaca (US stocks, free), Interactive Brokers,
  or a crypto exchange testnet.
- Compare paper fills against backtest assumptions. The gap is your real cost model.

## Phase 6: Small live capital (only after all of the above)
- Use money you can afford to lose. Add risk limits, a kill switch, logging and monitoring.
- Expect live Sharpe to be roughly half the backtest Sharpe.

## Ground rules
1. A backtest Sharpe above 2 on daily data is almost always a bug or overfitting.
2. Always include costs. Many strategies only work before costs.
3. Decide on a hypothesis before you run the backtest, not after looking at the results.
4. Fewer parameters are better. Simple and robust beats complex and fragile.
