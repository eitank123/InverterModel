"""A small, readable backtesting framework for learning algorithmic trading."""
from . import data, metrics, validation
from .backtest import BacktestResult, compare, plot, run_backtest
from .strategy import (BuyAndHold, Momentum, SMACrossover, Strategy, VolTarget,
                       ZScoreMeanReversion)

__all__ = [
    "data", "metrics", "validation", "run_backtest", "compare", "plot", "BacktestResult",
    "Strategy", "BuyAndHold", "SMACrossover", "Momentum", "ZScoreMeanReversion", "VolTarget",
]
