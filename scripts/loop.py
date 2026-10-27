"""The improvement loop. Each iteration adds candidate families; every candidate is
evaluated the same way (algotrading/research.py) and lands on loop/leaderboard.csv.

    python scripts/loop.py --iter 1            # run one iteration's candidates
    python scripts/loop.py --iter all          # rerun everything
    python scripts/loop.py --report            # print the leaderboard + export dashboard data

Objective: highest expected long-run growth (CAGR) on the untouched TEST period
1991-2026, parameters chosen on TRAIN 1957-1990, ruin excluded. See loop/journal.md for
the checkpoint reviews that decide what each new iteration explores.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from algotrading.research import Candidate, Context, Leaderboard  # noqa: E402

warnings.filterwarnings("ignore")
LOOP = os.path.join(os.path.dirname(__file__), "..", "loop")
SMA10 = at.PriceAboveSMA(10)


def mix(stock_w):
    return at.PerAsset(at.BuyAndHold(), {"stocks": stock_w, "bonds": 1 - stock_w})


def filtered(stock_w=0.6, window=10):
    return at.TrendFilter(mix(stock_w), at.PriceAboveSMA(window))


class TrendScaledLeverage(at.PortfolioStrategy):
    """Leverage proportional to trend strength: L * average of SMA signals (6..12 months)."""

    def __init__(self, stock_w, lev):
        self.stock_w, self.lev = stock_w, lev
        self.sig = at.AverageSignal([at.PriceAboveSMA(w) for w in (6, 8, 10, 12)])

    def weights(self, prices):
        return at.TrendFilter(mix(self.stock_w), self.sig).weights(prices) * self.lev


def iteration_1():
    """Baseline families: how far does leverage on trend-filtered portfolios go?"""
    return [
        Candidate("60/40", "benchmark", lambda: mix(0.6)),
        Candidate("S&P 500", "benchmark", lambda: mix(1.0)),
        Candidate("60/40+SMA10 xL", "levered trend 60/40", lambda lev: at.Levered(filtered(0.6), lev),
                  {"lev": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]}),
        Candidate("mix+SMA10 xL", "levered trend mix", lambda w, lev: at.Levered(filtered(w), lev),
                  {"w": [0.4, 0.6, 0.8, 1.0], "lev": [1.0, 1.5, 2.0, 2.5, 3.0]}),
        Candidate("S&P+SMA10 xL (Gayed-Bilello)", "levered trend stocks",
                  lambda lev: at.Levered(filtered(1.0), lev), {"lev": [1.0, 1.25, 1.5, 2.0, 3.0]},
                  note="Leverage for the Long Run: levered S&P above its MA, T-bills below"),
        Candidate("60/40+SMA10 vol-target", "vol targeting",
                  lambda tv, cap: at.PortfolioVolTarget(filtered(0.6), tv, 12, cap),
                  {"tv": [0.08, 0.10, 0.12, 0.15, 0.20], "cap": [2.0, 3.0, 4.0]}),
        Candidate("trend-strength leverage", "trend-scaled leverage",
                  lambda w, lev: TrendScaledLeverage(w, lev),
                  {"w": [0.6, 0.8, 1.0], "lev": [1.5, 2.0, 3.0]}),
    ]


class ValuationKelly(at.PortfolioStrategy):
    """Stock exposure = clip(fraction * expected excess return / variance, 0, cap), gated by trend.

    Expected nominal stock return ~ real earnings yield (1/CAPE, published with a 6-month lag)
    + trailing 10-year inflation (Campbell-Shiller / Gordon growth logic). Excess = minus cash.
    Variance: trailing 12-month realised, floored at 15%/yr vol (calm != safe).
    Bonds: a fixed sleeve, gated by their own trend filter.
    """

    def __init__(self, fraction, cap, bond_w=0.0, gate=True, ctx=None):
        self.fraction, self.cap, self.bond_w, self.gate = fraction, cap, bond_w, gate
        df = at.data.load_shiller_sp500()
        infl10 = (df["cpi"] / df["cpi"].shift(120)) ** (1 / 10) - 1
        cash = (df["long_rate"] - 0.015).clip(lower=0)
        ey = (1 / df["cape"].shift(6)).ffill()
        self.premium = (ey + infl10.shift(2) - cash)

    def weights(self, prices):
        s = prices["stocks"]
        vol = (s.pct_change().rolling(12).std() * np.sqrt(12)).clip(lower=0.15)
        prem = self.premium.reindex(prices.index).ffill()
        lev = (self.fraction * prem / vol**2).clip(0, self.cap).fillna(0.0)
        if self.gate:
            lev = lev * SMA10.positions(s)
        bonds = self.bond_w * SMA10.positions(prices["bonds"])
        return pd.DataFrame({"stocks": lev, "bonds": bonds}, index=prices.index)


def iteration_2():
    """Checkpoint 1: leverage must follow the expected premium; widen leverage grids."""
    return [
        Candidate("valuation-Kelly stocks", "valuation Kelly",
                  lambda f, cap: ValuationKelly(f, cap), {"f": [0.25, 0.5, 0.75, 1.0], "cap": [2.0, 3.0, 4.0]},
                  note="leverage = Kelly with earnings-yield premium, trend-gated"),
        Candidate("valuation-Kelly + bonds", "valuation Kelly",
                  lambda f, cap, b: ValuationKelly(f, cap, b),
                  {"f": [0.25, 0.5, 0.75, 1.0], "cap": [2.0, 3.0], "b": [0.4, 0.8]}),
        Candidate("valuation-Kelly ungated", "valuation Kelly",
                  lambda f, cap: ValuationKelly(f, cap, 0.0, gate=False),
                  {"f": [0.25, 0.5, 0.75, 1.0], "cap": [2.0, 3.0]}),
        Candidate("S&P+SMA10 xL wide", "levered trend stocks",
                  lambda lev: at.Levered(filtered(1.0), lev), {"lev": [2.0, 3.0, 3.5, 4.0, 5.0]}),
        Candidate("bonds+SMA10 xL", "levered trend bonds",
                  lambda lev: at.Levered(at.TrendFilter(mix(0.0), SMA10), lev),
                  {"lev": [1.0, 2.0, 3.0, 4.0]}, note="lever only the 'safe' asset"),
        Candidate("stocks 1x + levered bonds, trend", "levered trend mix",
                  lambda b: at.TrendFilter(at.PerAsset(at.BuyAndHold(), {"stocks": 1.0, "bonds": b}), SMA10),
                  {"b": [0.5, 1.0, 1.5, 2.0]}, note="100% stocks plus a levered bond overlay"),
    ]


class CappedValuationLeverage(at.PortfolioStrategy):
    """Trend-filtered stock/bond mix whose total leverage follows the expected stock premium:
    L = clip(k * premium / premium_median, lo, cap). Cheap market -> more leverage."""

    def __init__(self, stock_w, k, lo, cap):
        self.stock_w, self.k, self.lo, self.cap = stock_w, k, lo, cap
        self.prem = ValuationKelly(1, 1).premium

    def weights(self, prices):
        prem = self.prem.reindex(prices.index).ffill()
        med = prem.expanding(120).median()                   # point-in-time typical premium
        lev = (self.k * prem / med).clip(self.lo, self.cap).fillna(1.0)
        return filtered(self.stock_w).weights(prices).mul(lev, axis=0)


def iteration_3():
    """Checkpoint 2: what is the best strategy that an ordinary margin account can hold (<= 2x)?"""
    return [
        Candidate("S&P+SMA10 x2 (deployable)", "deployable levered trend",
                  lambda lev: at.Levered(filtered(1.0), lev), {"lev": [1.5, 1.75, 2.0]}),
        Candidate("mix+SMA10 x2 (deployable)", "deployable levered trend",
                  lambda w: at.Levered(filtered(w), 2.0), {"w": [0.6, 0.7, 0.8, 0.9, 1.0]}),
        Candidate("valuation-scaled leverage <=2x", "valuation leverage",
                  lambda w, k: CappedValuationLeverage(w, k, 0.5, 2.0),
                  {"w": [0.6, 0.8, 1.0], "k": [1.0, 1.5, 2.0]}),
        Candidate("vol-target <=2x", "vol targeting",
                  lambda w, tv: at.PortfolioVolTarget(filtered(w), tv, 12, 2.0),
                  {"w": [0.6, 0.8, 1.0], "tv": [0.12, 0.15, 0.20]}),
        Candidate("trend-strength leverage <=2x", "trend-scaled leverage",
                  lambda w: TrendScaledLeverage(w, 2.0), {"w": [0.6, 0.8, 1.0]}),
    ]


ITERATIONS = {1: iteration_1, 2: iteration_2, 3: iteration_3}


def run(iters, financings=(0.005, 0.015)):
    board = Leaderboard(os.path.join(LOOP, "leaderboard.csv"))
    for fin in financings:
        ctx = Context.default(financing=fin)
        for it in iters:
            for c in ITERATIONS[it]():
                row, _ = at.research.evaluate(c, ctx)
                board.add(row, it)
                print(f"[iter {it} | financing +{fin:.1%}] {c.name:34s} params {row['params']:28s} "
                      f"train {row['train_cagr']:6.1%}  TEST {row['test_cagr']:6.1%}  "
                      f"DD {row['full_max_dd']:5.0%}  crash-day {row['crash_day_loss']:5.0%}"
                      f"{'  RUIN' if row['ruin'] else ''}")
    board.save()
    return board


def report(board, n_forward=6):
    """Print the top of the leaderboard and export dashboard data files."""
    top = board.top(15)
    cols = ["name", "params", "financing", "train_cagr", "test_cagr", "full_cagr", "full_max_dd",
            "full_worst_year", "crash_day_loss", "avg_gross"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(top[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    export(board, n_forward)


def build_from_row(row):
    """Rebuild a leaderboard entry's strategy from its name + params."""
    for it, fn in ITERATIONS.items():
        for c in fn():
            if c.name == row["name"]:
                return c.build(**json.loads(row["params"]))
    raise KeyError(row["name"])


def export(board, n_forward=6):
    """Write loop/dashboard/*.csv: leaderboard, equity curves, yearly returns, forward view."""
    out = os.path.join(LOOP, "dashboard")
    os.makedirs(out, exist_ok=True)
    lb = board.df.copy()
    lb["rank"] = lb["test_cagr"].rank(ascending=False, method="first").astype(int)
    lb.sort_values("rank").to_csv(os.path.join(out, "leaderboard.csv"), index=False)
    # Curves for the benchmarks + the top survivors at futures-like financing.
    cheap = board.df[(board.df["financing"] == 0.005) & ~board.df["ruin"].astype(bool)]
    best_per_family = (cheap[cheap["family"] != "benchmark"].sort_values("test_cagr", ascending=False)
                       .groupby("family").head(1).nlargest(n_forward, "test_cagr"))
    picks = pd.concat([cheap[cheap["family"] == "benchmark"], best_per_family]).drop_duplicates("name")
    # Different families can reach the same strategy (e.g. 100% stocks via "mix"): keep one.
    picks = picks.loc[picks[["test_cagr", "full_cagr"]].round(6).drop_duplicates().index]
    ctx = Context.default(0.005)
    curves, yearly, fwd = {}, {}, []
    for _, row in picks.iterrows():
        label = f"{row['name']} {row['params']}" if row["params"] != "{}" else row["name"]
        s = build_from_row(row)
        r = ctx.run(s).returns["1957":]
        curves[label] = (1 + r).cumprod()
        yearly[label] = (1 + r).groupby(r.index.year).prod() - 1
        f = at.research.forward_growth(s, ctx, n_paths=200)
        fwd.append({"strategy": label, **f, "test_cagr": row["test_cagr"],
                    "full_max_dd": row["full_max_dd"]})
    eq = pd.DataFrame(curves)
    eq.index.name = "date"
    eq.reset_index().melt(id_vars="date", var_name="strategy", value_name="growth_of_1").to_csv(
        os.path.join(out, "equity.csv"), index=False)
    yr = pd.DataFrame(yearly)
    yr.index.name = "year"
    yr.reset_index().melt(id_vars="year", var_name="strategy", value_name="return").to_csv(
        os.path.join(out, "yearly.csv"), index=False)
    pd.DataFrame(fwd).to_csv(os.path.join(out, "forward.csv"), index=False)
    print("\nForward 10-year view at today's valuations (futures-like financing):")
    print(pd.DataFrame(fwd)[["strategy", "fwd_median_cagr", "fwd_mean_cagr", "fwd_p5_cagr",
                             "fwd_p_loss", "fwd_mean_wealth_x"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def frontier(n_paths=300):
    """The expected-value frontier: leverage vs historical and forward outcomes."""
    out = os.path.join(LOOP, "dashboard")
    os.makedirs(out, exist_ok=True)
    ctx = Context.default(0.005)
    rows = []
    for base_name, base in [("S&P 500 + SMA10", filtered(1.0)), ("60/40 + SMA10", filtered(0.6)),
                            ("S&P 500 (no filter)", mix(1.0))]:
        for lev in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            s = at.Levered(base, lev)
            res = ctx.run(s)
            r = res.returns["1957":]
            f = at.research.forward_growth(s, ctx, n_paths=n_paths)
            rows.append({"base": base_name, "leverage": lev,
                         "hist_cagr_1957": at.metrics.cagr(r, 12),
                         "hist_test_cagr": at.metrics.cagr(r["1991":], 12),
                         "hist_max_dd": at.metrics.max_drawdown(r),
                         "crash_day_loss": at.research.crash_stress(res.positions),
                         **f})
            print(f"{base_name:20s} x{lev:3.1f}  hist {rows[-1]['hist_cagr_1957']:6.1%}  "
                  f"fwd median {f['fwd_median_cagr']:6.1%}  E[wealth] {f['fwd_mean_wealth_x']:.2f}x  "
                  f"P(loss) {f['fwd_p_loss']:.0%}  p5 {f['fwd_p5_cagr']:6.1%}  crash-day {rows[-1]['crash_day_loss']:5.0%}")
    pd.DataFrame(rows).to_csv(os.path.join(out, "frontier.csv"), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iter", default=None, help="iteration number or 'all'")
    p.add_argument("--report", action="store_true")
    p.add_argument("--frontier", action="store_true")
    a = p.parse_args()
    board = None
    if a.iter:
        iters = list(ITERATIONS) if a.iter == "all" else [int(a.iter)]
        board = run(iters)
    if a.report:
        report(board or Leaderboard(os.path.join(LOOP, "leaderboard.csv")))
    if a.frontier:
        frontier()
