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


FRED_MONTHLY_PATH = os.path.join(os.path.dirname(SHILLER_PATH), "fred_monthly.csv")
FRED_MONTHLY_SERIES = {"GS10": "10-year Treasury yield (%)", "CPIAUCSL": "CPI, all urban",
                       "TB3MS": "3-month T-bill (%)", "CP": "Corporate profits after tax (quarterly)"}


def save_fred_monthly(path: str | None = None) -> pd.DataFrame:
    """Download the monthly FRED series used to keep the Shiller file current (scripts/update_data.py)."""
    cols = {}
    for sid in FRED_MONTHLY_SERIES:
        s = load_fred(sid)
        s.index = s.index.to_period("M").to_timestamp()
        cols[sid] = s
    df = pd.DataFrame(cols)
    df.index.name = "Date"
    df.to_csv(path or FRED_MONTHLY_PATH)
    return df


def _extend_with_fred(df: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """Fill what the Shiller file stopped publishing (rates, CPI, earnings, CAPE after 2023)
    from FRED, splicing at the last month each Shiller series was really updated.

    * long_rate: FRED GS10 (the same 10-year constant-maturity yield Shiller uses).
    * cpi: FRED CPIAUCSL rescaled to Shiller's CPI at the splice month.
    * earnings: last Shiller earnings x growth of US corporate profits after tax (FRED CP),
      lagged 5 months for publication. CP grew slower than S&P EPS in 2013-23 (buybacks),
      so this errs toward a HIGHER CAPE, i.e. less valuation leverage.
    * cape: recomputed like Shiller (real price / 10-year average real earnings) wherever his
      value is missing.
    """
    df = df.copy()
    fred = fred.reindex(df.index.union(fred.index)).sort_index()

    def last_real(col):
        s = df[col].dropna()
        return s[s.diff().fillna(1) != 0].index[-1]

    lr = last_real("long_rate")
    gs10 = fred["GS10"].reindex(df.index) / 100
    df.loc[df.index > lr, "long_rate"] = gs10[df.index > lr]
    df["long_rate"] = df["long_rate"].ffill()
    lc = last_real("cpi")
    cpi_f = fred["CPIAUCSL"].reindex(df.index).ffill()
    if lc in cpi_f.index and pd.notna(cpi_f[lc]):
        df.loc[df.index > lc, "cpi"] = (cpi_f * df.loc[lc, "cpi"] / cpi_f[lc])[df.index > lc]
    df["cpi"] = df["cpi"].ffill()
    le = df["earnings"].last_valid_index()
    cp = fred["CP"].dropna()
    cp = cp.reindex(fred.index).ffill().shift(5).reindex(df.index).ffill()
    if pd.notna(cp.get(le)):
        df.loc[df.index > le, "earnings"] = (df.loc[le, "earnings"] * cp / cp[le])[df.index > le]
    real_p = df["close"] * df["cpi"].iloc[-1] / df["cpi"]
    real_e = df["earnings"] * df["cpi"].iloc[-1] / df["cpi"]
    cape = real_p / real_e.rolling(120).mean()
    df["cape"] = df["cape"].fillna(cape.where(df.index > df["cape"].last_valid_index()))
    return df


def load_shiller_sp500(path: str | None = None) -> pd.DataFrame:
    """Monthly S&P 500 since 1871 (Robert Shiller's dataset, via datasets/s-and-p-500).

    Columns:
      close      price index (no dividends)
      tr         total-return index: dividends reinvested -> what an investor really earned
      dividend   annualised dividend per share
      long_rate  10-year government bond yield, as a decimal (0.05 = 5%)
      cpi        consumer price index
      earnings   annualised earnings per share (NaN after 2023-06)
      cape       Shiller CAPE = real price / 10-year average real earnings (NaN when unknown)

    Caveats you must know before trusting results on this data:
      * Prices up to mid-2023 are *monthly averages of daily closes*, not month-end
        closes. Averaging smooths the series and creates artificial positive
        autocorrelation in monthly returns, which flatters short-lookback momentum.
        Use ``execution_lag=1`` as a guard.
      * After 2023-06 only the price is published; the dividend *yield* is held constant.
        Rates, CPI, earnings and CAPE after 2023 come from FRED (``data/fred_monthly.csv``,
        written by scripts/update_data.py; see ``_extend_with_fred``). Without that file
        they are carried forward from 2023.
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
    df["earnings"] = raw["Earnings"].replace(0.0, np.nan)
    df["cape"] = raw["PE10"].replace(0.0, np.nan)
    monthly_tr = (df["close"] + df["dividend"] / 12) / df["close"].shift(1) - 1
    df["tr"] = df["close"].iloc[0] * (1 + monthly_tr.fillna(0.0)).cumprod()
    if path == SHILLER_PATH and os.path.exists(FRED_MONTHLY_PATH):
        fred = pd.read_csv(FRED_MONTHLY_PATH, index_col=0, parse_dates=True)
        df = _extend_with_fred(df, fred)
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


def load_fred(series: str) -> pd.Series:
    """One daily series from FRED (St. Louis Fed), e.g. "SP500" (S&P 500 price, last ~10 years)
    or "DGS10" (10-year Treasury yield, %). Public CSV endpoint, no key needed."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().rename(series)


def fred_daily_proxies(div_yield: float = 0.012, duration: float = 7.5,
                       sp500: pd.Series | None = None, dgs10: pd.Series | None = None) -> pd.DataFrame:
    """Daily total-return proxies from FRED: S&P 500 price plus a steady dividend accrual, and a
    7-10y Treasury index built from the 10-year yield (carry minus duration x yield change).
    Good enough for monthly SIGNALS; not a substitute for traded prices."""
    sp = sp500 if sp500 is not None else load_fred("SP500")
    y = (dgs10 if dgs10 is not None else load_fred("DGS10")) / 100
    # The yield is often published a day later than the S&P close: carry the last value forward
    # (a later yield change is then booked on the day it becomes known).
    days = sp.index[sp.index >= y.index[0]]
    sp, y = sp[days], y.reindex(y.index.union(days)).ffill()[days]
    stocks = sp * (1 + div_yield / 252) ** np.arange(len(days))
    bonds = (1 + (y.shift(1) / 252 - duration * y.diff()).fillna(0.0)).cumprod()
    return pd.DataFrame({"stocks": stocks, "bonds": bonds}, index=days)


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
