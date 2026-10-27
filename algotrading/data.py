"""Price data: synthetic generators (for controlled experiments) and loaders (for real data).

All functions return a DataFrame indexed by business-day dates with a ``close`` column.

Why synthetic data first? Because you *know the ground truth*. On a pure random walk
(GBM) no strategy has a real edge, so any positive Sharpe you find there is luck or a
bug. On a mean-reverting (OU) series a mean-reversion strategy *should* win. That makes
synthetic data the unit test for your research process.
"""
from __future__ import annotations

import os

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


def garch(n: int = 2520, mu: float = 0.07, omega: float = 2e-6, alpha: float = 0.09,
          beta: float = 0.89, s0: float = 100.0, seed: int | None = None,
          start: str = "2015-01-01") -> pd.DataFrame:
    """GARCH(1,1) returns: constant expected return, *clustered* volatility.

    sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}   (daily units)
    Returns themselves are unpredictable, but volatility is predictable, which is
    a stylised fact of every real market ("volatility clustering").
    Long-run daily vol = sqrt(omega / (1 - alpha - beta)) (defaults: ~1%/day, 16%/yr).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    var = omega / (1 - alpha - beta)
    r = np.zeros(n)
    sig = np.zeros(n)
    for t in range(1, n):
        var = omega + alpha * r[t - 1] ** 2 + beta * var
        sig[t] = np.sqrt(var)
        r[t] = mu / TRADING_DAYS + sig[t] * z[t]
    close = s0 * np.cumprod(1 + r)
    return pd.DataFrame({"close": close, "sigma": sig * np.sqrt(TRADING_DAYS)},
                        index=_index(n, start))


def panel(generator=regime_switching, n_assets: int = 10, seed: int = 0,
          **kwargs) -> pd.DataFrame:
    """Independent synthetic assets side by side: one ``close`` column per asset."""
    cols = {f"asset_{i:02d}": generator(seed=seed * 1000 + i, **kwargs)["close"]
            for i in range(n_assets)}
    return pd.DataFrame(cols)


SHILLER_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
SHILLER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_shiller_monthly.csv")


def load_shiller_sp500(path: str | None = None) -> pd.DataFrame:
    """Monthly S&P 500 since 1871 (Robert Shiller's dataset, via datasets/s-and-p-500).

    Columns:
      close      price index (no dividends)
      tr         total-return index: dividends reinvested -> what an investor really earned
      dividend   annualised dividend per share
      long_rate  10-year government bond yield, as a decimal (0.05 = 5%)
      cpi        consumer price index

    Caveats you must know before trusting results on this data:
      * Prices up to mid-2023 are *monthly averages of daily closes*, not month-end
        closes. Averaging smooths the series and creates artificial positive
        autocorrelation in monthly returns, which flatters short-lookback momentum.
        Use ``execution_lag=1`` as a guard.
      * After 2023-06 only the price is published; dividends and rates are carried
        forward from the last known value (dividend *yield* held constant).
    """
    path = path or SHILLER_PATH
    if not os.path.exists(path):
        path = SHILLER_URL
    raw = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    df = pd.DataFrame({"close": raw["SP500"]})
    div = raw["Dividend"].replace(0.0, np.nan)
    last = div.last_valid_index()
    yield_last = div[last] / df.loc[last, "close"]
    div[div.index > last] = df.loc[div.index > last, "close"] * yield_last
    df["dividend"] = div.ffill()
    df["long_rate"] = raw["Long Interest Rate"].replace(0.0, np.nan).ffill() / 100
    df["cpi"] = raw["Consumer Price Index"].replace(0.0, np.nan).ffill()
    monthly_tr = (df["close"] + df["dividend"] / 12) / df["close"].shift(1) - 1
    df["tr"] = df["close"].iloc[0] * (1 + monthly_tr.fillna(0.0)).cumprod()
    return df


def bond_total_return(yields: pd.Series, maturity: float = 10.0) -> pd.Series:
    """Total-return index of a constant-maturity government bond, built from yields.

    Each month we buy a new ``maturity``-year par bond (coupon = current yield), hold it
    one month, and reprice it at next month's yield. Return = coupon income + price
    change. Price change ~ -duration * change in yield: when rates rise, bonds fall.
    ``yields`` are annual decimals, monthly frequency.
    """
    y0 = yields.shift(1)
    y1 = yields
    c = y0 / 2                                  # semi-annual coupon of a par bond
    n = int(maturity * 2)
    t = np.arange(1, n + 1)[None, :] - 1 / 6    # cash-flow times (half-years) one month later
    disc = (1 + y1.to_numpy()[:, None] / 2) ** (-t)
    price = c.to_numpy()[:, None] * disc
    price[:, -1] += disc[:, -1]
    # Discounting from one month in already includes the month's accrued coupon.
    ret = pd.Series(price.sum(axis=1) - 1.0, index=yields.index)
    return (1 + ret.fillna(0.0)).cumprod()


DJIA_URL = ("https://raw.githubusercontent.com/jalajthanaki/stock_price_prediction/"
            "master/data/DJIA_data.csv")


def load_djia_daily(cache_dir: str | None = None) -> pd.DataFrame:
    """Daily Dow Jones Industrial Average OHLC, 2006-12 to 2016-12 (index level, no
    dividends: dividend yield ~2%/yr is missing). Downloaded once, then cached."""
    cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(cache_dir, "djia_daily_2007_2016.csv")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        pd.read_csv(DJIA_URL).to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df.rename(columns={"adj_close": "close", "close": "raw_close"})


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
