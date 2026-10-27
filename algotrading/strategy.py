"""Strategies.

A strategy is a *causal filter*: it maps prices up to and including bar t to a target
position for bar t, in [-1, 1] (fraction of capital; negative = short).

The backtester is responsible for the timing: a position decided at the close of bar t
earns the return from t to t+1. So a strategy must only use data available at t, and
must never call things like ``.shift(-1)``. ``validation.check_no_lookahead`` tests this.

To write your own strategy, subclass ``Strategy`` and implement ``positions``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .metrics import infer_ppy


class Strategy(ABC):
    @abstractmethod
    def positions(self, prices: pd.Series) -> pd.Series:
        """Return target position per bar (same index as ``prices``)."""

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


class BuyAndHold(Strategy):
    """Always long. The benchmark every strategy must beat (after costs!)."""

    def positions(self, prices: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=prices.index)


class SMACrossover(Strategy):
    """Trend following: long when fast moving average > slow one, else flat (or short)."""

    def __init__(self, fast: int = 20, slow: int = 100, allow_short: bool = False):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast, self.slow, self.allow_short = fast, slow, allow_short

    def positions(self, prices: pd.Series) -> pd.Series:
        fast = prices.rolling(self.fast).mean()
        slow = prices.rolling(self.slow).mean()
        down = -1.0 if self.allow_short else 0.0
        pos = pd.Series(np.where(fast > slow, 1.0, down), index=prices.index)
        return pos.where(slow.notna(), 0.0)


class Momentum(Strategy):
    """Time-series momentum: position = sign of the return over the last ``lookback`` bars."""

    def __init__(self, lookback: int = 60, allow_short: bool = True):
        self.lookback, self.allow_short = lookback, allow_short

    def positions(self, prices: pd.Series) -> pd.Series:
        past_ret = prices.pct_change(self.lookback)
        pos = np.sign(past_ret).fillna(0.0)
        return pos if self.allow_short else pos.clip(lower=0.0)


class ZScoreMeanReversion(Strategy):
    """Mean reversion on the z-score of log price vs. its rolling mean.

    Enter short when z > entry, long when z < -entry; exit when |z| < exit.
    The entry/exit band (hysteresis) avoids flipping in and out on noise, which would
    be eaten by transaction costs. Note this is *stateful*, so we use a loop.
    """

    def __init__(self, lookback: int = 20, entry: float = 1.5, exit: float = 0.5):
        self.lookback, self.entry, self.exit = lookback, entry, exit

    def positions(self, prices: pd.Series) -> pd.Series:
        logp = np.log(prices)
        mean = logp.rolling(self.lookback).mean()
        std = logp.rolling(self.lookback).std()
        z = ((logp - mean) / std).to_numpy()
        pos = np.zeros(len(z))
        current = 0.0
        for t, zt in enumerate(z):
            if np.isnan(zt):
                current = 0.0
            elif current == 0.0:
                if zt > self.entry:
                    current = -1.0
                elif zt < -self.entry:
                    current = 1.0
            elif abs(zt) < self.exit:
                current = 0.0
            pos[t] = current
        return pd.Series(pos, index=prices.index)


class VolTarget(Strategy):
    """Wrap another strategy and scale its position to target a constant volatility.

    position = base_position * target_vol / realised_vol, capped at ``max_leverage``.
    A simple and very common risk-management layer.
    """

    def __init__(self, base: Strategy, target_vol: float = 0.10, lookback: int = 20,
                 max_leverage: float = 2.0):
        self.base, self.target_vol = base, target_vol
        self.lookback, self.max_leverage = lookback, max_leverage

    def positions(self, prices: pd.Series) -> pd.Series:
        ppy = infer_ppy(prices.index)
        realised = prices.pct_change().rolling(self.lookback).std() * np.sqrt(ppy)
        scale = (self.target_vol / realised).clip(upper=self.max_leverage).fillna(0.0)
        return self.base.positions(prices) * scale


class PriceAboveSMA(Strategy):
    """Faber (2007) timing rule: long when price > its ``window``-bar moving average,
    else in cash. Classic version: monthly bars, window=10 (~ the 200-day average)."""

    def __init__(self, window: int = 10):
        self.window = window

    def positions(self, prices: pd.Series) -> pd.Series:
        sma = prices.rolling(self.window).mean()
        return (prices > sma).astype(float).where(sma.notna(), 0.0)


class TrendEnsemble(Strategy):
    """Average of time-series momentum signals over several lookbacks.

    Hurst, Ooi & Pedersen ("A Century of Evidence on Trend-Following", AQR) use an
    equal mix of 1-, 3- and 12-month lookbacks. Averaging hedges *parameter risk*:
    we don't know which horizon trends at, so we hold a bit of each. The output is in
    [-1, 1] and naturally scales down when the horizons disagree.
    Lookbacks are in bars: (21, 63, 252) for daily data, (1, 3, 12) for monthly.
    """

    def __init__(self, lookbacks: tuple[int, ...] = (21, 63, 252), allow_short: bool = True):
        self.lookbacks, self.allow_short = tuple(lookbacks), allow_short

    def positions(self, prices: pd.Series) -> pd.Series:
        sigs = [np.sign(prices.pct_change(lb)) for lb in self.lookbacks]
        # Stay flat until the longest lookback has data.
        ready = prices.pct_change(max(self.lookbacks)).notna()
        pos = (sum(sigs) / len(sigs)).where(ready, 0.0)
        return pos if self.allow_short else pos.clip(lower=0.0)


# --- Portfolios ----------------------------------------------------------------------

class PortfolioStrategy(ABC):
    """Maps a DataFrame of prices (one column per asset) to a DataFrame of weights."""

    @abstractmethod
    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Target weight per asset per bar. sum(|w|) is gross exposure (leverage)."""

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


class PerAsset(PortfolioStrategy):
    """Run a single-asset strategy on every column and give each an equal 1/N budget."""

    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = pd.DataFrame({c: self.strategy.positions(prices[c]) for c in prices.columns})
        return w / prices.shape[1]


class PortfolioVolTarget(PortfolioStrategy):
    """Scale a portfolio's weights so its *combined* realised volatility hits a target.

    Diversification lowers portfolio vol below the per-asset vol; this layer levers it
    back up so a better Sharpe turns into more return at the same risk.
    """

    def __init__(self, base: PortfolioStrategy, target_vol: float = 0.10,
                 lookback: int = 60, max_leverage: float = 3.0):
        self.base, self.target_vol = base, target_vol
        self.lookback, self.max_leverage = lookback, max_leverage

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = self.base.weights(prices)
        ppy = infer_ppy(prices.index)
        # Returns the base portfolio *would have* earned, using only past weights.
        port_ret = (w.shift(1) * prices.pct_change()).sum(axis=1)
        realised = port_ret.rolling(self.lookback).std() * np.sqrt(ppy)
        scale = (self.target_vol / realised).clip(upper=self.max_leverage)
        scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gross = w.abs().sum(axis=1)
        scale = np.minimum(scale, self.max_leverage / gross.where(gross > 0, np.inf))
        return w.mul(scale, axis=0)
