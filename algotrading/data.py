"""Price data: synthetic generators (for controlled experiments) and loaders (for real data).

All functions return a DataFrame indexed by business-day dates with a ``close`` column.

Why synthetic data first? Because you *know the ground truth*. On a pure random walk
(GBM) no strategy has a real edge, so any positive Sharpe you find there is luck or a
bug. On a mean-reverting (OU) series a mean-reversion strategy *should* win. That makes
synthetic data the unit test for your research process.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _index(n: int, start: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def gbm(n: int = 2520, mu: float = 0.07, sigma: float = 0.20, s0: float = 100.0,
        seed: int | None = None, start: str = "2015-01-01") -> pd.DataFrame:
    """Geometric Brownian Motion: dS/S = mu dt + sigma dW (annualised mu, sigma).

    Log returns are i.i.d. normal -> past prices carry no information about the future.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    log_ret[0] = 0.0
    close = s0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame({"close": close}, index=_index(n, start))


def ornstein_uhlenbeck(n: int = 2520, theta: float = 5.0, sigma: float = 0.20,
                       s0: float = 100.0, mean: float | None = None,
                       seed: int | None = None, start: str = "2015-01-01") -> pd.DataFrame:
    """Mean-reverting log price: dx = theta (m - x) dt + sigma dW, with x = log(S).

    ``theta`` is the (annualised) speed of reversion; half-life = ln(2)/theta years.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    m = np.log(s0 if mean is None else mean)
    x = np.empty(n)
    x[0] = np.log(s0)
    # Exact discretisation of the OU process.
    a = np.exp(-theta * dt)
    noise_sd = sigma * np.sqrt((1 - a**2) / (2 * theta))
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = m + a * (x[t - 1] - m) + noise_sd * eps[t]
    return pd.DataFrame({"close": np.exp(x)}, index=_index(n, start))


def regime_switching(n: int = 2520, mus: tuple[float, ...] = (0.30, -0.30),
                     sigmas: tuple[float, ...] = (0.12, 0.30), p_stay: float = 0.997,
                     s0: float = 100.0, seed: int | None = None,
                     start: str = "2015-01-01") -> pd.DataFrame:
    """GBM whose drift/vol follow a hidden Markov chain (e.g. bull / bear regimes).

    With a persistent chain (p_stay close to 1) trends last for months, which is the
    kind of structure trend-following strategies try to exploit. The hidden regime is
    returned in the ``regime`` column so you can check what your strategy "saw".
    """
    rng = np.random.default_rng(seed)
    k = len(mus)
    dt = 1 / TRADING_DAYS
    regime = np.empty(n, dtype=int)
    regime[0] = 0
    for t in range(1, n):
        if rng.random() < p_stay:
            regime[t] = regime[t - 1]
        else:
            regime[t] = rng.choice([r for r in range(k) if r != regime[t - 1]])
    mu = np.asarray(mus)[regime]
    sig = np.asarray(sigmas)[regime]
    log_ret = (mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * rng.standard_normal(n)
    log_ret[0] = 0.0
    close = s0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame({"close": close, "regime": regime}, index=_index(n, start))


def load_csv(path: str, date_col: str = "Date", close_col: str = "Close") -> pd.DataFrame:
    """Load a CSV of daily prices (e.g. downloaded from Yahoo Finance / Stooq)."""
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    return df.rename(columns={close_col: "close"})[["close"]].dropna()


def load_yahoo(ticker: str, start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Download adjusted daily closes via ``yfinance`` (optional dependency)."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install yfinance to download real market data") from e
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns a column per ticker
        close = close.iloc[:, 0]
    return pd.DataFrame({"close": close}).dropna()
