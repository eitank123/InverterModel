"""Performance metrics computed from a series of per-bar (daily) strategy returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series) -> float:
    years = len(returns) / TRADING_DAYS
    growth = (1 + returns).prod()
    return float(growth ** (1 / years) - 1) if years > 0 and growth > 0 else float("nan")


def annual_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio. Rule of thumb: >1 good, >2 suspicious (check for bugs!)."""
    excess = returns - risk_free / TRADING_DAYS
    sd = excess.std()
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def sortino(returns: pd.Series) -> float:
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean())
    return float(returns.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else 0.0


def drawdown(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def calmar(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    return float(cagr(returns) / abs(mdd)) if mdd < 0 else float("nan")


def hit_rate(returns: pd.Series) -> float:
    active = returns[returns != 0]
    return float((active > 0).mean()) if len(active) else float("nan")


def summary(returns: pd.Series, positions: pd.Series | None = None) -> dict[str, float]:
    out = {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "ann_vol": annual_vol(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns),
        "hit_rate": hit_rate(returns),
    }
    if positions is not None:
        out["exposure"] = float((positions != 0).mean())
        out["ann_turnover"] = float(positions.diff().abs().sum() / len(positions) * TRADING_DAYS)
    return out
