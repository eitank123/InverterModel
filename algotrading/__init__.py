"""A small, readable backtesting framework for learning algorithmic trading."""
from . import data, metrics, news, validation
from .backtest import BacktestResult, compare, plot, run_backtest
from .strategy import (Blend, BuyAndHold, Momentum, PerAsset, PortfolioStrategy,
                       PortfolioVolTarget, PriceAboveSMA, SMACrossover, Strategy,
                       TrendEnsemble, VolTarget, ZScoreMeanReversion)

__all__ = [
    "data", "metrics", "news", "validation", "run_backtest", "compare", "plot", "BacktestResult",
    "Strategy", "BuyAndHold", "SMACrossover", "Momentum", "ZScoreMeanReversion", "VolTarget",
    "PriceAboveSMA", "TrendEnsemble", "PortfolioStrategy", "PerAsset", "PortfolioVolTarget",
    "Blend",
]
