"""Research loop harness: candidates, a standard evaluation, and a persistent leaderboard.

Objective (set by the user): maximise EXPECTED LONG-RUN GROWTH (compound return), accepting
higher volatility and lower confidence. Constraints are only the ones that make growth
real: no ruin, realistic financing, parameters chosen out of sample.

Protocol for every candidate:
  * free parameters are chosen on TRAIN (1957-1990) by highest CAGR;
  * the chosen setting is reported on TEST (1991-2026, never used for choosing);
  * a crash stress applies the worst historical one-day falls (which monthly data hides)
    to the leveraged exposure held at the time: a loss >= 100% is ruin;
  * forward expected growth: block bootstrap of history with means shifted to today's
    valuation-implied returns.
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import data, metrics, stats
from .backtest import run_backtest
from .strategy import PortfolioStrategy

TRAIN, TEST, FULL = ("1957", "1990"), ("1991", "2026"), ("1957", "2026")

# Largest one-day S&P 500 / DJIA falls (approximate, %), which monthly averages smooth away.
CRASH_DAYS = {"1929-10": -0.129, "1987-10": -0.205, "2008-10": -0.090, "2020-03": -0.120}


@dataclass
class Candidate:
    name: str
    family: str
    build: Callable[..., PortfolioStrategy]          # build(**params) -> strategy
    grid: dict[str, list] = field(default_factory=dict)
    note: str = ""


@dataclass
class Context:
    px: pd.DataFrame
    cash: pd.Series
    financing: float = 0.005          # borrow spread over cash (futures-like)
    cost_bps: float = 10.0

    @classmethod
    def default(cls, financing: float = 0.005) -> "Context":
        df = data.load_shiller_sp500()
        px = pd.DataFrame({"stocks": df["tr"], "bonds": data.bond_total_return(df["long_rate"])})
        cash = (df["long_rate"] - 0.015).clip(lower=0)
        return cls(px, cash, financing)

    def run(self, strategy: PortfolioStrategy):
        return run_backtest(self.px, strategy, cost_bps=self.cost_bps, execution_lag=1,
                            cash_rate=self.cash, borrow_spread=self.financing)


def crash_stress(positions: pd.DataFrame) -> float:
    """Worst single-day loss implied by the stock exposure held in historical crash months."""
    worst = 0.0
    for month, drop in CRASH_DAYS.items():
        if month in positions.index.strftime("%Y-%m"):
            w = positions.loc[month, "stocks"].iloc[0] if "stocks" in positions else 0.0
            worst = min(worst, w * drop)
    return worst


def period_stats(r: pd.Series, cash: pd.Series) -> dict:
    ex = r - cash.reindex(r.index) / 12
    return {"cagr": metrics.cagr(r, 12), "vol": metrics.annual_vol(r, 12),
            "max_dd": metrics.max_drawdown(r), "excess_sharpe": metrics.sharpe(ex, ppy=12),
            "worst_year": float(((1 + r).groupby(r.index.year).prod() - 1).min())}


def evaluate(c: Candidate, ctx: Context) -> tuple[dict, pd.Series]:
    """Choose params on TRAIN by CAGR (ruin excluded), report everything."""
    best, best_cagr = {}, -np.inf
    keys = list(c.grid)
    for values in itertools.product(*(c.grid[k] for k in keys)) if keys else [()]:
        params = dict(zip(keys, values))
        r = ctx.run(c.build(**params)).returns
        tr = r[TRAIN[0]:TRAIN[1]]
        if (1 + tr).min() <= 0:
            continue
        g = metrics.cagr(tr, 12)
        if np.isfinite(g) and g > best_cagr:
            best, best_cagr = params, g
    res = ctx.run(c.build(**best))
    r = res.returns
    row = {"name": c.name, "family": c.family, "params": json.dumps(best), "note": c.note,
           "financing": ctx.financing, "avg_gross": float(res.positions[FULL[0]:].sum(axis=1).mean()),
           "max_gross": float(res.positions[FULL[0]:].sum(axis=1).max()),
           "crash_day_loss": crash_stress(res.positions)}
    for label, (a, b) in {"train": TRAIN, "test": TEST, "full": FULL}.items():
        for k, v in period_stats(r[a:b], ctx.cash).items():
            row[f"{label}_{k}"] = v
    row["ruin"] = bool(row["crash_day_loss"] <= -1.0 or (1 + r[FULL[0]:]).min() <= 0)
    # Deployable with an ordinary margin account (Reg-T: about 2x overnight); more needs futures.
    row["deployable"] = bool(row["max_gross"] <= 2.0 + 1e-9)
    return row, r


def forward_growth(strategy: PortfolioStrategy, ctx: Context, n_paths: int = 300,
                   horizon: int = 120, seed: int = 0) -> dict:
    """Expected 10-year growth at today's valuations: block bootstrap of 1957+ history
    (stocks, bonds, cash jointly) with stock mean 5%/yr (CAPE ~41), bonds = current yield."""
    df = data.load_shiller_sp500()
    hist = pd.DataFrame({"stocks": ctx.px["stocks"].pct_change(), "bonds": ctx.px["bonds"].pct_change(),
                         "cash": ctx.cash / 12})["1957":].dropna()
    target = pd.Series({"stocks": 0.05, "bonds": df["long_rate"].iloc[-1],
                        "cash": max(df["long_rate"].iloc[-1] - 0.015, 0)})
    arr = (hist + (target - hist.mean() * 12) / 12).to_numpy()
    rng = np.random.default_rng(seed)
    warm = ctx.px.iloc[-24:]
    out = []
    for _ in range(n_paths):
        sim = arr[stats.stationary_bootstrap(len(arr), 24, rng)[:horizon]]
        dates = pd.date_range(warm.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
        path = pd.concat([warm, pd.DataFrame(np.cumprod(1 + sim[:, :2], axis=0) * warm.iloc[-1].to_numpy(),
                                             index=dates, columns=["stocks", "bonds"])])
        c = pd.Series(np.r_[np.full(len(warm), sim[0, 2]), sim[:, 2]] * 12, index=path.index)
        r = run_backtest(path, strategy, cost_bps=ctx.cost_bps, execution_lag=1, cash_rate=c,
                         borrow_spread=ctx.financing).returns.iloc[-horizon:]
        out.append(float((1 + r).prod()))
    w = np.array(out)
    g = np.where(w > 0, w, 1e-9) ** (12 / horizon) - 1
    return {"fwd_median_cagr": float(np.median(g)), "fwd_mean_cagr": float(np.mean(g)),
            "fwd_p5_cagr": float(np.percentile(g, 5)), "fwd_p_loss": float((w < 1).mean()),
            "fwd_mean_wealth_x": float(w.mean())}


class Leaderboard:
    """CSV-backed table of every evaluated candidate (one row per name + financing)."""

    def __init__(self, path: str):
        self.path = path
        self.df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    def add(self, row: dict, iteration: int):
        row = {**row, "iteration": iteration}
        if not self.df.empty:
            same = (self.df["name"] == row["name"]) & (self.df["financing"] == row["financing"])
            self.df = self.df[~same]
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.df.sort_values("test_cagr", ascending=False).to_csv(self.path, index=False)

    def top(self, n: int = 10, survivors_only: bool = True) -> pd.DataFrame:
        d = self.df[~self.df["ruin"].astype(bool)] if survivors_only else self.df
        return d.sort_values("test_cagr", ascending=False).head(n)
