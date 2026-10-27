"""Statistical tools for testing what market returns actually look like.

* Variance ratio (Lo & MacKinlay 1988): is the price a random walk at horizon q?
* Hurst exponent: long memory (H > 0.5 trending, H < 0.5 mean-reverting).
* OLS with Newey-West standard errors: regressions on overlapping horizons.
* Out-of-sample R^2 (Campbell & Thompson 2008): does a predictor beat the
  historical mean *in real time*?
* Stationary block bootstrap (Politis & Romano 1994): confidence intervals that keep
  the autocorrelation and volatility clustering of returns.
* Kelly fraction: growth-optimal leverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --- Random-walk tests ---------------------------------------------------------------

def variance_ratio(log_returns: pd.Series | np.ndarray, q: int) -> tuple[float, float]:
    """Lo-MacKinlay variance ratio VR(q) and its heteroskedasticity-robust z-statistic.

    VR(q) = Var(q-period return) / (q * Var(1-period return)).
    Random walk: VR = 1. VR > 1: positive autocorrelation (trends).
    VR < 1: negative autocorrelation (mean reversion). Uses overlapping q-period sums.
    VR(q) - 1 ~= 2 * sum_{k=1}^{q-1} (1 - k/q) * rho_k, a weighted sum of autocorrelations.
    """
    x = np.asarray(log_returns, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mu = x.mean()
    d = x - mu
    var1 = (d @ d) / (n - 1)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    rq = cs[q:] - cs[:-q] - q * mu
    m = q * (n - q + 1) * (1 - q / n)
    varq = (rq @ rq) / m
    vr = varq / var1
    # Heteroskedasticity-consistent variance of VR (Lo & MacKinlay 1988, eq. 19-20).
    d2 = d**2
    denom = (d2.sum()) ** 2
    theta = 0.0
    for k in range(1, q):
        delta = n * (d2[k:] @ d2[:-k]) / denom
        theta += (2 * (q - k) / q) ** 2 * delta
    # Var(VR) ~ theta / n under the null.
    z = (vr - 1) * np.sqrt(n / theta) if theta > 0 else np.nan
    return float(vr), float(z)


def hurst_exponent(log_returns: pd.Series | np.ndarray, min_window: int = 8) -> float:
    """Hurst exponent from rescaled-range (R/S) analysis.

    H = 0.5 random walk, > 0.5 persistent, < 0.5 anti-persistent. R/S is biased upward
    in small samples (Anis-Lloyd), so compare with H estimated on simulated random walks.
    """
    x = np.asarray(log_returns, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    sizes = np.unique(np.floor(np.logspace(np.log10(min_window), np.log10(n // 4), 12)).astype(int))
    rs = []
    for s in sizes:
        k = n // s
        chunks = x[: k * s].reshape(k, s)
        dev = np.cumsum(chunks - chunks.mean(axis=1, keepdims=True), axis=1)
        r = dev.max(axis=1) - dev.min(axis=1)
        sd = chunks.std(axis=1, ddof=1)
        ok = sd > 0
        rs.append(np.mean(r[ok] / sd[ok]))
    slope = np.polyfit(np.log(sizes), np.log(rs), 1)[0]
    return float(slope)


# --- Regression and forecasting ------------------------------------------------------

def ols_newey_west(y: pd.Series, X: pd.DataFrame, lags: int) -> pd.DataFrame:
    """OLS with Newey-West (HAC) standard errors.

    Needed when regressing *overlapping* multi-period returns (e.g. 10-year returns
    sampled monthly): consecutive observations share 119 of 120 months, so plain OLS
    t-stats are wildly overstated.
    """
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    Y = df["_y"].to_numpy()
    Xm = np.column_stack([np.ones(len(df)), df.drop(columns="_y").to_numpy()])
    names = ["const"] + list(X.columns)
    XtX_inv = np.linalg.inv(Xm.T @ Xm)
    beta = XtX_inv @ Xm.T @ Y
    u = Y - Xm @ beta
    Xu = Xm * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        G = Xu[lag:].T @ Xu[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (u @ u) / ((Y - Y.mean()) @ (Y - Y.mean()))
    out = pd.DataFrame({"coef": beta, "se_nw": se, "t_nw": beta / se}, index=names)
    out.attrs["r2"] = float(r2)
    out.attrs["n"] = len(df)
    return out


def oos_r2(y: pd.Series, x: pd.Series, horizon: int, min_train: int = 240,
           nonneg: bool = False) -> tuple[float, pd.DataFrame]:
    """Out-of-sample R^2 of predicting ``y`` (the ``horizon``-period future return,
    already aligned to the forecast date) with ``x``, re-estimated each period on data
    available at that time only, vs. the expanding historical mean.

    R2_oos = 1 - sum (y - yhat_model)^2 / sum (y - yhat_mean)^2.
    > 0 means the predictor beat the naive mean in real time. Training only uses
    observations whose outcome was already known (t - horizon), avoiding overlap leakage.
    ``nonneg`` applies the Campbell-Thompson restriction: forecasts can't be below 0.
    """
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    preds = []
    for i in range(min_train + horizon, len(df)):
        train = df.iloc[: i - horizon]          # outcomes known at time i
        b = np.polyfit(train["x"], train["y"], 1)
        f_model = np.polyval(b, df["x"].iloc[i])
        f_mean = train["y"].mean()
        if nonneg:
            f_model = max(f_model, 0.0)
        preds.append((df.index[i], df["y"].iloc[i], f_model, f_mean))
    p = pd.DataFrame(preds, columns=["date", "y", "model", "mean"]).set_index("date")
    r2 = 1 - ((p.y - p.model) ** 2).sum() / ((p.y - p["mean"]) ** 2).sum()
    return float(r2), p


# --- Bootstrap -----------------------------------------------------------------------

def stationary_bootstrap(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Indices for one stationary-bootstrap resample (random block lengths ~ Geometric)."""
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = rng.integers(n)
        length = rng.geometric(1 / mean_block)
        take = np.arange(start, start + length) % n
        idx[i:i + length] = take[: n - i]
        i += length
    return idx


def bootstrap_stat(returns: pd.DataFrame | pd.Series, stat, n_boot: int = 1000,
                   mean_block: float = 12.0, seed: int = 0) -> np.ndarray:
    """Distribution of ``stat(resampled_returns)`` under the stationary bootstrap.
    Resample rows jointly so cross-asset/strategy correlation is preserved."""
    rng = np.random.default_rng(seed)
    arr = returns.to_numpy()
    out = []
    for _ in range(n_boot):
        idx = stationary_bootstrap(len(arr), mean_block, rng)
        sample = arr[idx]
        if isinstance(returns, pd.DataFrame):
            sample = pd.DataFrame(sample, columns=returns.columns)
        else:
            sample = pd.Series(sample)
        out.append(stat(sample))
    return np.asarray(out)


# --- Position sizing -----------------------------------------------------------------

def kelly_fraction(excess_returns: pd.Series) -> float:
    """Growth-optimal leverage f* = mu / sigma^2 (per-period excess return moments).

    Maximises E[log wealth] for i.i.d. returns (Kelly 1956, Thorp). Expected log growth at
    leverage f is g(f) = r + f*mu - f^2 sigma^2 / 2, so g peaks at f* and drops to the
    cash rate at 2 f*. Half-Kelly keeps 75% of the growth with half the volatility.
    """
    r = excess_returns.dropna()
    return float(r.mean() / r.var())
