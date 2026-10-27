"""A small, readable backtesting framework for learning algorithmic trading."""
from . import data, metrics, news, stats, validation
from .backtest import BacktestResult, compare, plot, run_backtest
from .strategy import (AverageSignal, BandedSMA, Blend, BuyAndHold, InverseVol, Levered, Momentum, MonthlySMA, PerAsset, PortfolioStrategy,
                       PortfolioVolTarget, PriceAboveSMA, SMACrossover, Strategy,
                       TrendEnsemble, TrendFilter, ValueTilt, VolTarget,
                       ZScoreMeanReversion)

__all__ = [
    "data", "metrics", "news", "stats", "validation", "run_backtest", "compare", "plot", "BacktestResult",
    "Strategy", "BuyAndHold", "SMACrossover", "Momentum", "ZScoreMeanReversion", "VolTarget",
    "PriceAboveSMA", "TrendEnsemble", "PortfolioStrategy", "PerAsset", "PortfolioVolTarget",
    "Blend", "Levered", "TrendFilter", "AverageSignal", "ValueTilt", "InverseVol",
    "MonthlySMA", "BandedSMA",
]
