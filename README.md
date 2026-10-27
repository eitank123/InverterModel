# Algo_trading

A small, readable backtesting framework for learning algorithmic trading. It's built
to be read and changed, not used as a black box.

* **[STATUS.md](STATUS.md)**: where the project stands, best result so far, next steps
* **[ROADMAP.md](ROADMAP.md)**: the learning plan
* **[docs/lessons/](docs/lessons/)**: market concepts, introduced as experiments need them
* **[docs/research_log.md](docs/research_log.md)**: every experiment: hypothesis, result, lesson
* **[docs/AUTOMATION.md](docs/AUTOMATION.md)**: from backtest to paper trading to live trading

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                 # 50 tests, ~5 s
python scripts/today.py             # what the frozen rules hold right now
python scripts/rebalance.py         # monthly rebalance (offline paper demo; --broker alpaca for real API)
python scripts/run_demo.py          # strategies on 3 synthetic markets -> results/*.png
python scripts/overfitting_demo.py  # how parameter search fools you
python scripts/experiments/exp01_sp500_trend.py      # trend timing on 155 years of S&P 500
python scripts/experiments/exp02_diversification.py  # breadth, parameter risk, costs
python scripts/experiments/exp03_vol_scaling.py      # trend signal vs volatility scaling
python scripts/experiments/exp04_stocks_bonds.py     # 60/40 and trend, 1871-2026
python scripts/experiments/exp05_news_sentiment.py   # do news headlines predict the market?
python scripts/experiments/exp06_random_walk.py      # variance ratios: is it a random walk?
python scripts/experiments/exp07_valuation.py        # CAPE, Goyal-Welch out-of-sample test
python scripts/experiments/exp08_kelly_leverage.py   # Kelly sizing and leverage
python scripts/experiments/exp09_robustness_and_profits.py  # annual profits, bootstrap, 10y outlook (~1.5 min)
python scripts/experiments/exp10_improve_filter.py   # whipsaw fixes, train/test split
python scripts/experiments/exp11_clean_bond_era.py   # re-check on reliable bond data
python scripts/experiments/exp12_insurance_or_exposure.py  # filter vs vol-matched control
python scripts/experiments/exp13_implementation.py   # month-end vs average vs daily (banded)
python scripts/experiments/exp14_inflation_regime.py # inflation guard for bonds, train/test
python scripts/experiments/exp15_kelly_safe_leverage.py  # Kelly / ridge / neural-net leverage (~4 min first run)
```

## Layout

| File | What it does |
|---|---|
| `algotrading/data.py` | Synthetic markets with known structure (GBM random walk, Ornstein-Uhlenbeck mean reversion, regime-switching trends, multi-asset panels); 155 years of S&P 500 total return (`load_shiller_sp500`), 10-year bonds from yields (`bond_total_return`), daily DJIA 2007–16, GARCH generator; CSV and Yahoo loaders |
| `algotrading/strategy.py` | Single-asset strategies: buy & hold, SMA crossover, Faber SMA-10, momentum, trend ensemble, z-score mean reversion, vol targeting, `MonthlySMA`, `BandedSMA`, `AverageSignal`. Portfolios: `PerAsset` (incl. 60/40-style budgets), `TrendFilter`, `ValueTilt`, `InverseVol`, `PortfolioVolTarget`, `Levered`, `Blend`, `KellyPortfolio` |
| `algotrading/backtest.py` | Vectorised backtester for one asset or a portfolio: explicit timing, transaction costs, interest on cash/leverage |
| `algotrading/metrics.py` | Sharpe, Sortino, CAGR, max drawdown, turnover, Probabilistic and Deflated Sharpe; any bar frequency |
| `algotrading/news.py` | Curated news sources, RSS/Atom reader, Loughran-McDonald and Claude headline scorers, daily sentiment index, `SentimentStrategy` |
| `algotrading/stats.py` | Variance ratio, Hurst exponent, Newey-West OLS, out-of-sample R², stationary block bootstrap, Kelly fraction |
| `algotrading/ml.py` | Point-in-time macro features, walk-forward forecasting, small regularised neural-net ensemble |
| `algotrading/broker.py` | Broker interface, local `PaperBroker`, `AlpacaBroker`, rebalancer with safety limits, the frozen rule's live signal |
| `algotrading/validation.py` | Lookahead detector, grid search, walk-forward, Monte-Carlo Sharpe null distribution |
| `scripts/experiments/` | Reproducible research experiments (see the research log) |
| `data/` | Shiller S&P 500 monthly dataset (public domain, via github.com/datasets/s-and-p-500) |
| `tests/` | Correctness tests, including timing, costs and lookahead |

## The core model

A strategy is a **causal map** from prices up to and including bar *t* to a target
position *p_t* ∈ [-1, 1] (as a fraction of capital). The backtester then computes:

```
r_strategy[t] = p[t-1] * r_asset[t]  -  cost * |p[t] - p[t-1]|
```

For an EE student, this is a discrete-time causal filter driving a system whose output
is P&L. Most backtest bugs come from breaking causality (lookahead), and
`validation.check_no_lookahead` checks for it.

## Writing a strategy

```python
import algotrading as at

class MyStrategy(at.Strategy):
    def __init__(self, window=50):
        self.window = window
    def positions(self, prices):
        return (prices > prices.rolling(self.window).mean()).astype(float)

px = at.data.regime_switching(seed=0)
assert at.validation.check_no_lookahead(MyStrategy(), px["close"])
res = at.run_backtest(px, MyStrategy(), cost_bps=2)
print(res)
```

Real data: `at.data.load_yahoo("SPY")` (needs `pip install yfinance`) or `at.data.load_csv(path)`.
