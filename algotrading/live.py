"""Running the research strategies on live market data.

The strategies need long monthly histories (10-month averages, 36-month highs, a 10-year
valuation median), but a broker only serves a few years of daily bars. ``splice_monthly``
chains the broker's month-end closes onto the long Shiller-based history, so the signal
always uses the full record and the most recent months come from real quotes.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import data
from .execution import etf_weights


def completed_month_ends(closes: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Last close of each month, dropping the current month while it is still running."""
    asof = pd.Timestamp(asof or pd.Timestamp.today()).normalize()
    me = closes.groupby(closes.index.to_period("M")).tail(1)
    last = me.index[-1]
    finished = (asof.to_period("M") > last.to_period("M")
                or last.normalize() == last + pd.offsets.BMonthEnd(0))
    return me if finished else me.iloc[:-1]


def splice_monthly(history: pd.DataFrame, month_ends: pd.DataFrame) -> pd.DataFrame:
    """Extend monthly index levels (columns stocks/bonds, month-start dates) with live
    month-end closes (same columns). The live data must overlap the history by at least one
    month: from that anchor month on, the live returns replace the history's."""
    live = month_ends.copy()
    live.index = live.index.to_period("M").to_timestamp()           # month-start dates
    anchor = live.index[0]
    if anchor not in history.index:
        raise ValueError("live data must start inside the history (one month overlap)")
    growth = live / live.iloc[0]
    spliced = growth * history.loc[anchor]
    return pd.concat([history[history.index < anchor], spliced])


def shiller_history() -> tuple[pd.DataFrame, pd.Series]:
    df = data.load_shiller_sp500()
    px = pd.DataFrame({"stocks": df["tr"], "bonds": data.bond_total_return(df["long_rate"])})
    return px, (df["long_rate"] - 0.015).clip(lower=0)


def target_etf_weights(strategy, monthly: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """Strategy exposure on the latest completed month -> ETF weights (SPY/UPRO/IEF/UST)."""
    w = strategy.weights(monthly).iloc[-1]
    exposure = {"stocks": float(w.get("stocks", 0.0)), "bonds": float(w.get("bonds", 0.0))}
    return etf_weights(exposure["stocks"], exposure["bonds"]), exposure


def log_run(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
