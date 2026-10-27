"""Tools to avoid fooling yourself: lookahead checks, parameter search, walk-forward,
and a Monte-Carlo null distribution for the Sharpe ratio.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd

from . import data, metrics
from .backtest import run_backtest
from .strategy import Strategy


def check_no_lookahead(strategy: Strategy, prices: pd.Series, n_checks: int = 5,
                       seed: int = 0) -> bool:
    """Verify positions[:t] don't change when the future (prices[t:]) is scrambled.

    If a strategy peeks at the future, altering future prices changes past positions.
    """
    rng = np.random.default_rng(seed)
    full = strategy.positions(prices)
    n = len(prices)
    for t in rng.integers(n // 4, n - 1, size=n_checks):
        tampered = prices.copy()
        tampered.iloc[t:] = tampered.iloc[t:].to_numpy() * rng.uniform(0.5, 1.5, n - t)
        pos = strategy.positions(tampered)
        if not np.allclose(full.iloc[:t].to_numpy(), pos.iloc[:t].to_numpy(), equal_nan=True):
            return False
    return True


def grid_search(make: Callable[..., Strategy], grid: dict[str, list], prices: pd.Series,
                metric: str = "sharpe", **bt_kwargs) -> pd.DataFrame:
    """Backtest every parameter combination; returns a table sorted by ``metric``."""
    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        try:
            strat = make(**params)
        except ValueError:          # invalid combination, e.g. fast >= slow
            continue
        res = run_backtest(prices, strat, **bt_kwargs)
        rows.append({**params, **res.stats})
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


def walk_forward(make: Callable[..., Strategy], grid: dict[str, list], prices: pd.Series,
                 train_bars: int = 756, test_bars: int = 252, metric: str = "sharpe",
                 **bt_kwargs) -> pd.DataFrame:
    """Re-optimise on a rolling train window, then trade the *next* unseen test window.

    The concatenated test-window returns are an honest(er) estimate of live performance.
    Returns a DataFrame with the chosen params and out-of-sample return per window,
    plus the stitched OOS return series in ``.attrs['oos_returns']``.
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices["close"]
    windows, oos = [], []
    start = 0
    while start + train_bars + test_bars <= len(prices):
        train = prices.iloc[start:start + train_bars]
        table = grid_search(make, grid, train, metric, **bt_kwargs)
        # Read per column (not table.iloc[0]) so int params don't get upcast to float.
        params = {k: _py(table[k].iloc[0]) for k in grid}
        # Run on train+test so indicators are warmed up, then keep only the test part.
        span = prices.iloc[start:start + train_bars + test_bars]
        res = run_backtest(span, make(**params), **bt_kwargs)
        test_ret = res.returns.iloc[train_bars:]
        oos.append(test_ret)
        windows.append({"test_start": test_ret.index[0], **params,
                        f"is_{metric}": table[metric].iloc[0],
                        f"oos_{metric}": metrics.summary(test_ret)[metric]})
        start += test_bars
    out = pd.DataFrame(windows)
    out.attrs["oos_returns"] = pd.concat(oos) if oos else pd.Series(dtype=float)
    return out


def sharpe_null_distribution(strategy: Strategy, n_paths: int = 200, n_bars: int = 2520,
                             seed: int = 0, **bt_kwargs) -> np.ndarray:
    """Sharpe ratios of ``strategy`` on random-walk (no-edge) paths.

    Compare your real Sharpe against this: if it's not far in the right tail, it's
    indistinguishable from luck.
    """
    out = np.empty(n_paths)
    for i in range(n_paths):
        px = data.gbm(n_bars, mu=0.0, seed=seed + i)["close"]
        out[i] = run_backtest(px, strategy, **bt_kwargs).stats["sharpe"]
    return out


def _py(x):
    """numpy scalar -> python scalar (so ints stay ints for window sizes)."""
    return x.item() if hasattr(x, "item") else x
