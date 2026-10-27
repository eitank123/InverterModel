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
    """Run a single-asset strategy on every column and give each a fixed budget.

    budgets: {column: weight}; default equal 1/N. ``PerAsset(BuyAndHold(),
    {"stocks": .6, "bonds": .4})`` is the classic 60/40 portfolio, rebalanced every bar.
    """

    def __init__(self, strategy: Strategy, budgets: dict[str, float] | None = None):
        self.strategy, self.budgets = strategy, budgets

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = pd.DataFrame({c: self.strategy.positions(prices[c]) for c in prices.columns})
        if self.budgets is None:
            return w / prices.shape[1]
        return w * pd.Series(self.budgets).reindex(w.columns).fillna(0.0)


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


class Blend(PortfolioStrategy):
    """Sum of several portfolio strategies with fixed mixing weights,
    e.g. ``Blend({core_60_40: 0.8, trend_sleeve: 0.2})``."""

    def __init__(self, parts: dict[PortfolioStrategy, float]):
        self.parts = parts

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        return sum(s.weights(prices) * k for s, k in self.parts.items())

    def __repr__(self) -> str:
        return "Blend(" + ", ".join(f"{k}*{s!r}" for s, k in self.parts.items()) + ")"


class Levered(PortfolioStrategy):
    """Scale every weight of a base portfolio by a constant leverage factor."""

    def __init__(self, base: PortfolioStrategy, leverage: float):
        self.base, self.leverage = base, leverage

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self.base.weights(prices) * self.leverage


class TrendFilter(PortfolioStrategy):
    """Multiply each asset's base weight by a single-asset filter in [0, 1]
    (e.g. ``PriceAboveSMA(10)``): keep the position while trending, cash otherwise."""

    def __init__(self, base: PortfolioStrategy, filt: Strategy):
        self.base, self.filt = base, filt

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = self.base.weights(prices)
        f = pd.DataFrame({c: self.filt.positions(prices[c]) for c in prices}, index=prices.index)
        return w * f


class AverageSignal(Strategy):
    """Average of several single-asset strategies' positions (e.g. SMA 6/8/10/12):
    partial exposure when they disagree, which softens whipsaw."""

    def __init__(self, strategies: list[Strategy]):
        self.strategies = strategies

    def positions(self, prices: pd.Series) -> pd.Series:
        return sum(s.positions(prices) for s in self.strategies) / len(self.strategies)


class ValueTilt(PortfolioStrategy):
    """Stock/bond split from a valuation percentile in [0, 1] (1 = cheap stocks):
    stock weight = lo + (hi - lo) * percentile. The signal must be point-in-time."""

    def __init__(self, signal: pd.Series, lo: float = 0.3, hi: float = 0.9,
                 stocks: str = "stocks", bonds: str = "bonds"):
        self.signal, self.lo, self.hi, self.stocks, self.bonds = signal, lo, hi, stocks, bonds

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.signal.reindex(prices.index).ffill().fillna(0.5)
        ws = self.lo + (self.hi - self.lo) * p
        return pd.DataFrame({self.stocks: ws, self.bonds: 1 - ws}, index=prices.index)

    def __repr__(self) -> str:
        return f"ValueTilt(lo={self.lo}, hi={self.hi})"


class InverseVol(PortfolioStrategy):
    """Risk parity (naive): weights proportional to 1 / trailing volatility, summing to 1."""

    def __init__(self, lookback: int = 36):
        self.lookback = lookback

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        vol = prices.pct_change().rolling(self.lookback).std()
        inv = 1 / vol
        return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)


class MonthlySMA(Strategy):
    """SMA filter evaluated once a month on month-end or month-average prices, held
    for the following month. Uses only data up to each month's last trading day."""

    def __init__(self, window: int = 10, how: str = "end"):
        self.window, self.how = window, how

    def positions(self, prices: pd.Series) -> pd.Series:
        g = prices.groupby(prices.index.to_period("M"))
        monthly = g.last() if self.how == "end" else g.mean()
        last_day = g.apply(lambda s: s.index[-1])
        sig = (monthly > monthly.rolling(self.window).mean()).astype(float)
        sig.index = pd.DatetimeIndex(last_day.values)
        return sig.reindex(prices.index).ffill().fillna(0.0)


class BandedSMA(Strategy):
    """Daily SMA filter with hysteresis: enter above sma*(1+band), exit below sma*(1-band)."""

    def __init__(self, window: int = 200, band: float = 0.02):
        self.window, self.band = window, band

    def positions(self, prices: pd.Series) -> pd.Series:
        sma = prices.rolling(self.window).mean().to_numpy()
        p = prices.to_numpy()
        pos = np.zeros(len(p))
        cur = 0.0
        for t in range(len(p)):
            if np.isnan(sma[t]):
                cur = 0.0
            elif cur == 0.0 and p[t] > sma[t] * (1 + self.band):
                cur = 1.0
            elif cur == 1.0 and p[t] < sma[t] * (1 - self.band):
                cur = 0.0
            pos[t] = cur
        return pd.Series(pos, index=prices.index)


class KellyPortfolio(PortfolioStrategy):
    """Multi-asset fractional Kelly: w = fraction * Sigma^-1 mu, long-only, gross <= cap.

    mu:   DataFrame of *annual excess-return forecasts* per asset, indexed by the date the
          forecast is made (must be point-in-time). NaN or <= 0 means "no bet".
    Sigma is the trailing ``cov_lookback``-bar covariance of returns, annualised;
    ``vol_floor`` ({asset: annual vol}) stops Kelly from treating a calm period as safe.
    Kelly levers assets in inverse proportion to their variance, so calm assets
    (bonds) get the most leverage, and calm periods get more than turbulent ones.
    """

    def __init__(self, mu: pd.DataFrame, fraction: float = 0.5, cov_lookback: int = 36,
                 cap: float = 2.0, vol_floor: dict[str, float] | None = None):
        self.mu, self.fraction, self.cov_lookback, self.cap = mu, fraction, cov_lookback, cap
        self.vol_floor = vol_floor

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        ppy = infer_ppy(prices.index)
        rets = prices.pct_change()
        mu = self.mu.reindex(prices.index).ffill()[prices.columns]
        out = np.zeros(prices.shape)
        for t in range(self.cov_lookback, len(prices)):
            m = mu.iloc[t].to_numpy()
            if np.isnan(m).all():
                continue
            m = np.nan_to_num(m).clip(min=0.0)
            if not m.any():
                continue
            cov = rets.iloc[t - self.cov_lookback + 1: t + 1].cov().to_numpy() * ppy
            if self.vol_floor:
                # "Calm" is not "safe": never assume an asset is less risky than its floor.
                vol = np.sqrt(np.diag(cov))
                floor = np.array([self.vol_floor.get(c, 0.0) for c in prices.columns])
                scale = np.maximum(vol, floor) / np.where(vol > 0, vol, 1.0)
                cov = cov * np.outer(scale, scale)
            w = self.fraction * np.linalg.solve(cov + 1e-8 * np.eye(len(m)), m)
            w = w.clip(min=0.0)
            w[m == 0] = 0.0
            gross = w.sum()
            if gross > self.cap:
                w *= self.cap / gross
            out[t] = w
        return pd.DataFrame(out, index=prices.index, columns=prices.columns)

    def __repr__(self) -> str:
        return (f"KellyPortfolio(fraction={self.fraction}, cov_lookback={self.cov_lookback}, "
                f"cap={self.cap}, vol_floor={self.vol_floor})")
