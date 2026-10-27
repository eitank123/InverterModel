"""Vectorised backtester with transaction costs, for one asset or a portfolio.

Timing convention (the single most important thing to get right):

    bar t close:   strategy sees prices[:t+1] and outputs target weights w_t
                   we trade from w_{t-1} to w_t, paying cost * |w_t - w_{t-1}|
    bar t+1 close: we earn w_t . r_{t+1}, where r_{t+1} = P_{t+1}/P_t - 1

So the return on bar t is

    w_{t-1} . r_t  +  (1 - sum(w_{t-1})) * cash_t  -  cost * sum|w_t - w_{t-1}|

The middle term matters: capital you don't invest sits in cash and earns interest,
and capital you borrow (sum of weights > 1) pays it. Ignoring it makes "sit in cash"
strategies look worse than they are when rates are high, and leverage look free.

Trading at the same close you computed the signal on is a slight idealisation; set
``execution_lag=1`` to trade one bar later (more conservative).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import metrics
from .strategy import PortfolioStrategy, Strategy


@dataclass
class BacktestResult:
    name: str
    prices: pd.Series | pd.DataFrame
    positions: pd.Series | pd.DataFrame   # weight held during each bar (after lag)
    gross_returns: pd.Series
    costs: pd.Series
    returns: pd.Series                    # net of costs, including cash interest
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return metrics.equity_curve(self.returns)

    def __repr__(self) -> str:
        s = self.stats
        return (f"<{self.name}: Sharpe {s['sharpe']:.2f}, CAGR {s['cagr']:.1%}, "
                f"MaxDD {s['max_drawdown']:.1%}>")


def _cash_series(cash: float | pd.Series, index: pd.Index, ppy: int) -> pd.Series:
    """Per-bar cash return from an annual rate (float or Series of annual rates)."""
    if isinstance(cash, pd.Series):
        return (cash.reindex(index).ffill().fillna(0.0) / ppy)
    return pd.Series(cash / ppy, index=index)


def run_backtest(prices: pd.Series | pd.DataFrame, strategy: Strategy | PortfolioStrategy,
                 cost_bps: float = 2.0, execution_lag: int = 0,
                 cash_rate: float | pd.Series = 0.0, ppy: int | None = None,
                 name: str | None = None) -> BacktestResult:
    """Backtest ``strategy`` on ``prices``.

    prices:    Series (one asset), DataFrame with a ``close`` column (one asset), or a
               DataFrame with one column per asset (portfolio; needs a PortfolioStrategy).
               Use *total-return* prices (dividends reinvested) whenever you can.
    cost_bps:  one-way cost per unit of weight traded, in basis points (1bp = 0.01%).
               Covers commission + half the bid/ask spread + slippage. 1-5bp is realistic
               for liquid ETFs/futures, much more for small caps or crypto.
    cash_rate: annual interest earned on uninvested capital (and paid on leverage).
               A float, or a Series of annual rates (e.g. T-bill yield over time).
    """
    if isinstance(prices, pd.DataFrame) and "close" in prices.columns:
        prices = prices["close"]
    ppy = ppy or metrics.infer_ppy(prices.index)

    if isinstance(prices, pd.DataFrame):
        if not isinstance(strategy, PortfolioStrategy):
            raise TypeError("multi-asset prices need a PortfolioStrategy (e.g. PerAsset)")
        target = strategy.weights(prices).reindex(prices.index).fillna(0.0)
    else:
        target = strategy.positions(prices).reindex(prices.index).fillna(0.0)
    if execution_lag:
        target = target.shift(execution_lag).fillna(0.0)

    asset_ret = prices.pct_change().fillna(0.0)
    held = target.shift(1).fillna(0.0)                 # weight during bar t
    trades = target.diff().abs().fillna(target.abs())
    invested = held
    if isinstance(prices, pd.DataFrame):
        trades = trades.sum(axis=1)
        gross = (held * asset_ret).sum(axis=1)
        invested = held.sum(axis=1)
    else:
        gross = held * asset_ret
    cash = _cash_series(cash_rate, prices.index, ppy) * (1 - invested)
    costs = trades * cost_bps / 1e4
    net = gross + cash - costs

    return BacktestResult(
        name=name or repr(strategy),
        prices=prices,
        positions=held,
        gross_returns=gross,
        costs=costs,
        returns=net,
        stats=metrics.summary(net, held, ppy),
    )


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    """Table of stats, one row per backtest."""
    return pd.DataFrame({r.name: r.stats for r in results}).T


def plot(results: list[BacktestResult], path: str | None = None, title: str = ""):
    """Equity curves (log scale) + drawdowns. Saves to ``path`` if given."""
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for r in results:
        ax1.plot(r.equity, label=f"{r.name}  (SR {r.stats['sharpe']:.2f})", lw=1.2)
        ax2.plot(metrics.drawdown(r.returns), lw=1)
    ax1.set_yscale("log")
    ax1.set_ylabel("equity (log)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_title(title)
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
