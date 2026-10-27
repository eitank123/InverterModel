"""Vectorised single-asset backtester with transaction costs.

Timing convention (the single most important thing to get right):

    bar t close:   strategy sees prices[:t+1] and outputs target position p_t
                   we trade from p_{t-1} to p_t, paying cost * |p_t - p_{t-1}|
    bar t+1 close: we earn p_t * r_{t+1}, where r_{t+1} = P_{t+1}/P_t - 1

So strategy return on bar t is  p_{t-1} * r_t - cost * |p_t - p_{t-1}|.
Trading at the same close you computed the signal on is a slight idealisation; set
``execution_lag=1`` to trade one bar later (more conservative).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import metrics
from .strategy import Strategy


@dataclass
class BacktestResult:
    name: str
    prices: pd.Series
    positions: pd.Series      # position held during each bar (after lag)
    gross_returns: pd.Series
    costs: pd.Series
    returns: pd.Series        # net of costs
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return metrics.equity_curve(self.returns)

    def __repr__(self) -> str:
        s = self.stats
        return (f"<{self.name}: Sharpe {s['sharpe']:.2f}, CAGR {s['cagr']:.1%}, "
                f"MaxDD {s['max_drawdown']:.1%}>")


def run_backtest(prices: pd.Series | pd.DataFrame, strategy: Strategy,
                 cost_bps: float = 2.0, execution_lag: int = 0,
                 name: str | None = None) -> BacktestResult:
    """Backtest ``strategy`` on ``prices``.

    cost_bps: one-way cost per unit of position traded, in basis points (1bp = 0.01%).
              Covers commission + half the bid/ask spread + slippage. 1-5bp is realistic
              for liquid ETFs/futures, much more for small caps or crypto.
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices["close"]
    target = strategy.positions(prices).reindex(prices.index).fillna(0.0)
    if execution_lag:
        target = target.shift(execution_lag).fillna(0.0)

    asset_ret = prices.pct_change().fillna(0.0)
    held = target.shift(1).fillna(0.0)              # position during bar t
    trades = target.diff().abs().fillna(target.abs())
    costs = trades * cost_bps / 1e4
    gross = held * asset_ret
    net = gross - costs

    return BacktestResult(
        name=name or repr(strategy),
        prices=prices,
        positions=held,
        gross_returns=gross,
        costs=costs,
        returns=net,
        stats=metrics.summary(net, held),
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
