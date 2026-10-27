"""Performance metrics computed from a series of per-bar strategy returns.

Every annualised metric takes ``ppy`` (periods per year): 252 for daily bars, 52 for
weekly, 12 for monthly. If omitted it is inferred from the DatetimeIndex.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def infer_ppy(index: pd.Index) -> int:
    """Guess bars per year from the median spacing of a DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return TRADING_DAYS
    days = float(np.median(np.diff(index.values).astype("timedelta64[D]").astype(float)))
    if days <= 4:
        return TRADING_DAYS
    if days <= 10:
        return 52
    if days <= 40:
        return 12
    if days <= 100:
        return 4
    return 1


def _ppy(returns: pd.Series, ppy: int | None) -> int:
    return ppy if ppy is not None else infer_ppy(returns.index)


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series, ppy: int | None = None) -> float:
    years = len(returns) / _ppy(returns, ppy)
    growth = (1 + returns).prod()
    return float(growth ** (1 / years) - 1) if years > 0 and growth > 0 else float("nan")


def annual_vol(returns: pd.Series, ppy: int | None = None) -> float:
    return float(returns.std() * np.sqrt(_ppy(returns, ppy)))


def sharpe(returns: pd.Series, risk_free: float = 0.0, ppy: int | None = None) -> float:
    """Annualised Sharpe ratio. Rule of thumb: >1 good, >2 suspicious (check for bugs!).

    ``risk_free`` is an annual rate. Strictly, Sharpe uses returns *in excess of cash*;
    with cash at 0 we get the "raw" Sharpe, which flatters strategies when rates are high.
    """
    p = _ppy(returns, ppy)
    excess = returns - risk_free / p
    sd = excess.std()
    return float(excess.mean() / sd * np.sqrt(p)) if sd > 0 else 0.0


def sortino(returns: pd.Series, ppy: int | None = None) -> float:
    p = _ppy(returns, ppy)
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean())
    return float(returns.mean() / downside * np.sqrt(p)) if downside > 0 else 0.0


def drawdown(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def calmar(returns: pd.Series, ppy: int | None = None) -> float:
    mdd = max_drawdown(returns)
    return float(cagr(returns, ppy) / abs(mdd)) if mdd < 0 else float("nan")


def hit_rate(returns: pd.Series) -> float:
    active = returns[returns != 0]
    return float((active > 0).mean()) if len(active) else float("nan")


# --- Statistical significance -------------------------------------------------------

def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0,
                         ppy: int | None = None) -> float:
    """P(true Sharpe > ``sr_benchmark``) given the sample (Bailey & López de Prado 2012).

    Accounts for track-record length, skewness and fat tails: a negatively skewed,
    fat-tailed strategy needs a longer record to prove the same Sharpe. Sharpe ratios
    in and out are annualised; internally the formula works per-period.
    """
    p = _ppy(returns, ppy)
    r = returns.dropna()
    n = len(r)
    if n < 3 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()                     # per-period
    sr_star = sr_benchmark / np.sqrt(p)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0                # pandas gives excess kurtosis
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(_N.cdf((sr - sr_star) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_std: float, sr_mean: float = 0.0) -> float:
    """Expected best Sharpe among ``n_trials`` *skill-less* strategies.

    ``sr_std`` is the standard deviation of the Sharpe ratios across your trials
    (same units as the result, e.g. annualised). Uses the extreme-value approximation
    from Bailey & López de Prado (2014).
    """
    if n_trials <= 1:
        return sr_mean
    z1 = _N.inv_cdf(1 - 1 / n_trials)
    z2 = _N.inv_cdf(1 - 1 / (n_trials * np.e))
    return float(sr_mean + sr_std * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sharpes_std: float,
                    ppy: int | None = None) -> float:
    """Deflated Sharpe Ratio: probability the selected strategy's Sharpe beats the best
    one you would expect from luck after trying ``n_trials`` variants.

    Rule of thumb: DSR > 0.95 is evidence of a real effect. Be honest with n_trials:
    count every variant you looked at, not just the ones in the final table.
    """
    hurdle = expected_max_sharpe(n_trials, trial_sharpes_std)
    return probabilistic_sharpe(returns, hurdle, ppy)


def summary(returns: pd.Series, positions: pd.Series | pd.DataFrame | None = None,
            ppy: int | None = None) -> dict[str, float]:
    p = _ppy(returns, ppy)
    out = {
        "total_return": total_return(returns),
        "cagr": cagr(returns, p),
        "ann_vol": annual_vol(returns, p),
        "sharpe": sharpe(returns, ppy=p),
        "sortino": sortino(returns, p),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns, p),
        "hit_rate": hit_rate(returns),
        "psr": probabilistic_sharpe(returns, 0.0, p),
    }
    if positions is not None:
        gross = positions.abs().sum(axis=1) if isinstance(positions, pd.DataFrame) else positions.abs()
        trades = positions.diff().abs()
        if isinstance(trades, pd.DataFrame):
            trades = trades.sum(axis=1)
        out["exposure"] = float(gross.mean())
        out["ann_turnover"] = float(trades.sum() / len(positions) * p)
    return out
