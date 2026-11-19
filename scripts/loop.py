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
# The current champion (changed only by a checkpoint decision; see loop/checkpoints.json).
# Since checkpoint 7 the champion is chosen by its WORST era (1880-1929, 1930-1956, 1957-1990,
# 1991-2026), not by the 1991-2026 test alone (see loop/README.md).
CHAMPION = {"name": "ensemble, trend sleeve gated by unemployment", "params": "{}"}
PREVIOUS_CHAMPION = {"name": "ensemble: trend + value + recovery", "params": '{"trend": 3.0}'}
SMA10 = at.PriceAboveSMA(10)


def mix(stock_w):
    return at.PerAsset(at.BuyAndHold(), {"stocks": stock_w, "bonds": 1 - stock_w})


def filtered(stock_w=0.6, window=10, band=0.0):
    sig = at.BandedSMA(window, band) if band else at.PriceAboveSMA(window)
    return at.TrendFilter(mix(stock_w), sig)


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

    def __init__(self, stock_w, k, lo, cap, band=0.0):
        self.stock_w, self.k, self.lo, self.cap, self.band = stock_w, k, lo, cap, band
        self.prem = ValuationKelly(1, 1).premium

    def weights(self, prices):
        prem = self.prem.reindex(prices.index).ffill()
        med = prem.expanding(120).median()                   # point-in-time typical premium
        lev = (self.k * prem / med).clip(self.lo, self.cap).fillna(1.0)
        return filtered(self.stock_w, band=self.band).weights(prices).mul(lev, axis=0)


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


class DrawdownBrake(at.PortfolioStrategy):
    """Base leverage L; after the strategy's own equity falls more than ``dd`` from its peak,
    cut to ``safe`` leverage until a new high (uses only past returns)."""

    def __init__(self, base, lev, dd, safe=1.0):
        self.base, self.lev, self.dd, self.safe = base, lev, dd, safe

    def weights(self, prices):
        w = self.base.weights(prices)
        r = (w.shift(1) * prices.pct_change()).sum(axis=1).to_numpy()
        lev = np.empty(len(r))
        eq, peak, cur = 1.0, 1.0, self.lev
        for t in range(len(r)):
            eq *= 1 + cur * r[t]                           # equity at the leverage actually held
            peak = max(peak, eq)
            cur = self.safe if eq / peak - 1 < -self.dd else self.lev
            lev[t] = cur
        return w.mul(lev, axis=0)


class DualMomentumLevered(at.PortfolioStrategy):
    """Each month hold the stronger of stocks or bonds at leverage L if its 12-month return beats
    cash, else cash (Antonacci's dual momentum, levered)."""

    def __init__(self, lev, lookback=12):
        self.lev, self.lookback = lev, lookback
        df = at.data.load_shiller_sp500()
        self.cash = (df["long_rate"] - 0.015).clip(lower=0)

    def weights(self, prices):
        mom = prices.pct_change(self.lookback)
        hurdle = self.cash.reindex(prices.index).ffill() * self.lookback / 12
        best = mom.fillna(-np.inf).idxmax(axis=1)
        ok = mom.max(axis=1) > hurdle
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for c in prices.columns:
            w.loc[(best == c) & ok, c] = self.lev
        return w.where(mom.notna().all(axis=1), 0.0)


class SeasonalLeverage(at.PortfolioStrategy):
    """Lever the base in November-April, hold ``summer`` leverage May-October (Bouman & Jacobsen
    2002, 'Sell in May'). The weight decided at month t is held in month t+2 (execution lag 1),
    so the calendar is shifted accordingly."""

    def __init__(self, base, winter, summer):
        self.base, self.winter, self.summer = base, winter, summer

    def weights(self, prices):
        held_month = (prices.index + pd.DateOffset(months=2)).month
        lev = np.where(np.isin(held_month, [11, 12, 1, 2, 3, 4]), self.winter, self.summer)
        return self.base.weights(prices).mul(lev, axis=0)


def iteration_4():
    """Checkpoint 3 open directions: drawdown brakes, dual momentum, seasonality, valuation x trend."""
    return [
        Candidate("S&P+SMA10 xL + drawdown brake", "drawdown brake",
                  lambda lev, dd: DrawdownBrake(filtered(1.0), lev, dd),
                  {"lev": [2.0, 3.0], "dd": [0.15, 0.25, 0.35]}),
        Candidate("dual momentum xL", "dual momentum",
                  lambda lev: DualMomentumLevered(lev), {"lev": [1.0, 1.5, 2.0, 3.0]},
                  note="Antonacci (2014): strongest of stocks/bonds if it beats cash"),
        Candidate("seasonal leverage S&P+SMA10", "seasonality",
                  lambda w_, s_: SeasonalLeverage(filtered(1.0), w_, s_),
                  {"w_": [2.0, 3.0], "s_": [0.5, 1.0, 2.0]}, note="Sell in May (Bouman & Jacobsen 2002)"),
        Candidate("valuation-scaled leverage <=3x", "valuation leverage",
                  lambda w, k: CappedValuationLeverage(w, k, 0.5, 3.0),
                  {"w": [0.6, 1.0], "k": [1.5, 2.0, 3.0]}),
    ]


class VarianceManaged(at.PortfolioStrategy):
    """Moreira & Muir (2017): stock weight = c / trailing variance (not volatility), capped.
    c is set so the weight equals ``target_lev`` at the long-run variance (16% vol)."""

    def __init__(self, target_lev, cap, window=3, gate=True):
        self.target_lev, self.cap, self.window, self.gate = target_lev, cap, window, gate

    def weights(self, prices):
        s = prices["stocks"]
        var = (s.pct_change().rolling(self.window).var() * 12).clip(lower=0.04**2)
        w = (self.target_lev * 0.16**2 / var).clip(upper=self.cap).fillna(0.0)
        if self.gate:
            w = w * SMA10.positions(s)
        return pd.DataFrame({"stocks": w, "bonds": 0.0}, index=prices.index)


def iteration_5():
    """Checkpoint 4: variance management (published version) and recovery leverage."""
    return [
        Candidate("variance-managed stocks", "variance managed",
                  lambda lev, cap, win: VarianceManaged(lev, cap, win),
                  {"lev": [1.0, 1.5, 2.0], "cap": [2.0, 3.0], "win": [3, 6]},
                  note="Moreira & Muir (2017): weight ~ 1 / variance"),
        Candidate("variance-managed ungated", "variance managed",
                  lambda lev, cap: VarianceManaged(lev, cap, 3, gate=False),
                  {"lev": [1.0, 1.5, 2.0], "cap": [2.0, 3.0]}),
        Candidate("recovery leverage", "recovery leverage",
                  lambda base, boost, depth: at.RecoveryLeverage(base, boost, depth),
                  {"base": [1.0, 2.0], "boost": [2.0, 3.0], "depth": [0.15, 0.25, 0.35]},
                  note="extra leverage after deep falls once the trend turns up"),
        Candidate("recovery leverage <=2x", "recovery leverage",
                  lambda base, depth: at.RecoveryLeverage(base, 2.0, depth),
                  {"base": [1.0, 1.5], "depth": [0.15, 0.25, 0.35]}),
    ]


class RecoveryMix(at.PortfolioStrategy):
    """Stock/bond version (checkpoint 5): trend-filtered mix at ``base`` leverage, ``boost``
    after deep stock falls."""

    def __init__(self, stock_w, base, boost, depth=0.15):
        self.f, self.base, self.boost, self.depth = filtered(stock_w), base, boost, depth

    def weights(self, prices):
        s = prices["stocks"]
        deep = (s / s.rolling(36, min_periods=12).max() - 1) < -self.depth
        lev = np.where(deep, self.boost, self.base)
        return self.f.weights(prices).mul(lev, axis=0)


def iteration_6():
    """Checkpoint 5: exit signal choice, bonds as the risk-off asset, stock/bond recovery."""
    return [
        Candidate("recovery, exit signal", "recovery leverage",
                  lambda signal: at.RecoveryLeverage(2.0, 3.0, 0.15, signal=signal),
                  {"signal": ["sma10", "tsmom12", "either", "both"]},
                  note="SMA10 vs 12-month momentum vs combinations"),
        Candidate("recovery, bonds when out", "recovery leverage",
                  lambda off: at.RecoveryLeverage(2.0, 3.0, 0.15, off_lev=off),
                  {"off": [0.0, 1.0, 2.0, 3.0]},
                  note="risk-off asset = Treasuries (Alpha Architect 2016)"),
        Candidate("recovery mix", "recovery leverage",
                  lambda w, base, boost: RecoveryMix(w, base, boost),
                  {"w": [0.6, 0.8], "base": [2.0, 3.0], "boost": [3.0, 4.0]}),
        Candidate("recovery+ (signal, bonds when out)", "recovery leverage",
                  lambda signal, off: at.RecoveryLeverage(2.0, 3.0, 0.15, signal=signal, off_lev=off),
                  {"signal": ["sma10", "either"], "off": [0.0, 1.0, 2.0]},
                  note="stay in while SMA10 or 12m momentum is up; Treasuries when out"),
    ]


def iteration_7():
    """Checkpoint 6: the champion's weak spot is bonds falling with stocks (2022). Gate the
    risk-off bonds on their own trend."""
    return [
        Candidate("recovery+ bond-gated", "recovery leverage",
                  lambda signal, off: at.RecoveryLeverage(2.0, 3.0, 0.15, signal=signal, off_lev=off,
                                                          off_gate=True),
                  {"signal": ["sma10", "either"], "off": [1.0, 2.0, 3.0]},
                  note="Treasuries when out of stocks only if bonds trend up (Faber 2007 per-asset rule)"),
    ]


def ensemble(trend_lev=2.0, band=0.0):
    """Equal thirds of three different, published effects; no selection between them:
    trend (Moskowitz-Ooi-Pedersen 2012; Faber 2007), valuation (Campbell & Shiller 1988),
    post-crash mean reversion with trend confirmation (Poterba & Summers 1988)."""
    return at.Blend({at.Levered(filtered(1.0, band=band), trend_lev): 1 / 3,
                     CappedValuationLeverage(1.0, 2.0, 0.5, 3.0, band=band): 1 / 3,
                     at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0, band=band): 1 / 3})


def ensemble_lower(trend_lev=2.0, rec_base=1.0, rec_boost=1.5, off=1.0):
    """Exp 19 (Japan): the same three sleeves at lower leverage."""
    return at.Blend({at.Levered(filtered(1.0), trend_lev): 1 / 3,
                     CappedValuationLeverage(1.0, 2.0, 0.5, 3.0): 1 / 3,
                     at.RecoveryLeverage(rec_base, rec_boost, 0.15, signal="either", off_lev=off): 1 / 3})


def iteration_8():
    """Checkpoint 7 (user: 'don't depend on past years'): stop picking the single best fit.
    Fixed equal-weight ensembles of independent effects, judged on their WORST era."""
    return [
        Candidate("ensemble: trend + value + recovery", "ensemble",
                  lambda trend: ensemble(trend), {"trend": [2.0, 3.0]},
                  note="equal thirds; no weights fitted"),
        Candidate("ensemble: trend + value", "ensemble",
                  lambda: at.Blend({at.Levered(filtered(1.0), 2.0): 0.5,
                                    CappedValuationLeverage(1.0, 2.0, 0.5, 3.0): 0.5})),
    ]


class GrowthTrend(at.PortfolioStrategy):
    """Growth-trend timing (Philosophical Economics 2016): obey the 10-month price trend only
    while the economy weakens (12-month earnings growth < 0, lagged 3 months for reporting);
    otherwise stay invested. ``lev`` x S&P 500 when in, cash when out.
    Note: Shiller earnings end 2023-06; after that the gate stays open (always invested)."""

    def __init__(self, lev):
        self.lev = lev
        e = at.data.load_shiller_sp500()["earnings"].dropna()
        self.weak = (e / e.shift(12) - 1).shift(3) < 0

    def weights(self, prices):
        s = prices["stocks"]
        weak = self.weak.reindex(prices.index).fillna(False).astype(bool)
        on = ~weak | SMA10.positions(s).astype(bool)
        return pd.DataFrame({"stocks": np.where(on, self.lev, 0.0), "bonds": 0.0}, index=prices.index)


class MultiHorizonTrend(at.PortfolioStrategy):
    """Hurst, Ooi & Pedersen (2017): equal-weight 1-, 3- and 12-month time-series momentum;
    S&P exposure = lev x share of the three signals that are positive."""

    def __init__(self, lev):
        self.lev = lev

    def weights(self, prices):
        s = prices["stocks"]
        votes = sum((s / s.shift(k) - 1 > 0).astype(float) for k in (1, 3, 12)) / 3
        return pd.DataFrame({"stocks": votes * self.lev, "bonds": 0.0}, index=prices.index)


def iteration_9():
    """Checkpoint 7 + reading: two published ways to cut trend whipsaw, alone and as ensemble sleeves."""
    return [
        Candidate("growth-trend timing xL", "growth-trend",
                  lambda lev: GrowthTrend(lev), {"lev": [2.0, 3.0]},
                  note="Philosophical Economics (2016): trend exit only when earnings growth < 0"),
        Candidate("multi-horizon trend xL", "multi-horizon trend",
                  lambda lev: MultiHorizonTrend(lev), {"lev": [2.0, 3.0]},
                  note="Hurst, Ooi & Pedersen (2017): 1/3/12-month TSMOM, equal weight"),
        Candidate("ensemble: growth-trend + value + recovery", "ensemble",
                  lambda: at.Blend({GrowthTrend(3.0): 1 / 3, CappedValuationLeverage(1.0, 2.0, 0.5, 3.0): 1 / 3,
                                    at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0): 1 / 3})),
        Candidate("ensemble: 4 sleeves (+ multi-horizon)", "ensemble",
                  lambda: at.Blend({at.Levered(filtered(1.0), 3.0): 0.25, CappedValuationLeverage(1.0, 2.0, 0.5, 3.0): 0.25,
                                    at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0): 0.25,
                                    MultiHorizonTrend(3.0): 0.25})),
    ]


class RiskParityBlend(at.PortfolioStrategy):
    """Sleeves weighted by inverse trailing 36-month volatility of their own (simulated) returns,
    weights summing to 1 (Qian 2005; Asness, Frazzini & Pedersen 2012). No fitted parameters."""

    def __init__(self, sleeves, window=36):
        self.sleeves, self.window = sleeves, window

    def weights(self, prices):
        r = prices.pct_change().fillna(0.0)
        ws = [s.weights(prices) for s in self.sleeves]
        # each sleeve's own return, earned with its weights from the month before
        rets = pd.concat([(w.shift(1) * r).sum(axis=1) for w in ws], axis=1)
        vol = rets.rolling(self.window, min_periods=12).std().shift(1)
        inv = (1 / vol).replace([np.inf, -np.inf], np.nan)
        k = inv.div(inv.sum(axis=1), axis=0).fillna(1 / len(ws))
        return sum(w.mul(k[i], axis=0) for i, w in enumerate(ws))


def iteration_10():
    """Checkpoint 15: sleeve weighting (risk parity) and a 4th independent effect (volatility timing)."""
    def sleeves():
        return [at.Levered(filtered(1.0), 3.0), CappedValuationLeverage(1.0, 2.0, 0.5, 3.0),
                at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0)]
    return [
        Candidate("ensemble, risk-parity sleeve weights", "ensemble", lambda: RiskParityBlend(sleeves()),
                  note="inverse 36-month volatility of each sleeve"),
        Candidate("ensemble + variance-managed (4 sleeves)", "ensemble",
                  lambda: at.Blend({s: 0.25 for s in sleeves() + [VarianceManaged(2.0, 3.0, 3)]}),
                  note="Moreira & Muir (2017) as a 4th independent effect"),
    ]


def iteration_11():
    """Checkpoint 20: the round-6 'either' exit signal (10-month average OR 12-month momentum
    up), which helped recovery+, applied to the ensemble's 3x trend sleeve."""
    return [
        Candidate("ensemble, trend sleeve on 'either' signal", "ensemble",
                  lambda: at.Blend({at.RecoveryLeverage(3.0, 3.0, 0.15, signal="either"): 1 / 3,
                                    CappedValuationLeverage(1.0, 2.0, 0.5, 3.0): 1 / 3,
                                    at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0): 1 / 3}),
                  note="3x while S&P above 10-month average OR up over 12 months"),
        Candidate("trend 3x on 'either' signal", "levered trend stocks",
                  lambda: at.RecoveryLeverage(3.0, 3.0, 0.15, signal="either")),
    ]


def iteration_12():
    """Checkpoint 26 (Exp 33-34): the live system reads the trend from month-end closes, which
    whipsawed out of the market in April 2025 and April 2026. A 2% band around the 10-month
    average (round; used before in Exp 13/18) on every sleeve's trend signal."""
    return [
        Candidate("ensemble, 2% band on the trend signals", "ensemble", lambda: ensemble(3.0, band=0.02),
                  note="same three sleeves; enter above SMA10 +2%, exit below SMA10 -2%"),
    ]


class InflationRegimeScaled(at.PortfolioStrategy):
    """Neville, Draaisma, Funnell, Harvey & Van Hemert (2021), 'The Best Strategies for
    Inflationary Times': an inflation regime is year-on-year US CPI at or above 5% and still
    accelerating. Equities lose in real terms there and borrowing costs climb, so leverage is
    cut to ``scale`` x the base weights (CPI lagged one month for its publication delay)."""

    def __init__(self, base, scale=0.5, need_rising=True):
        self.base, self.scale, self.need_rising = base, scale, need_rising
        cpi = at.data.load_shiller_sp500()["cpi"]
        yoy = cpi / cpi.shift(12) - 1
        reg = yoy >= 0.05
        if need_rising:
            reg &= yoy > yoy.shift(6)
        self.regime = reg.shift(1).fillna(False).astype(bool)

    def weights(self, prices):
        w = self.base.weights(prices)
        reg = self.regime.reindex(prices.index).ffill().fillna(False).astype(bool)
        return w.mul(np.where(reg, self.scale, 1.0), axis=0)


def iteration_13():
    """Checkpoint 33 (Exp 41, Neville et al. 2021): cut the champion's leverage in inflation
    regimes (CPI >= 5% and accelerating). Round, published threshold; 2 variants."""
    return [
        Candidate("ensemble, half leverage in inflation regimes", "ensemble",
                  lambda: InflationRegimeScaled(ensemble(3.0), 0.5, True),
                  note="CPI y/y >= 5% and above its level 6 months earlier"),
        Candidate("ensemble, half leverage when CPI >= 5%", "ensemble",
                  lambda: InflationRegimeScaled(ensemble(3.0), 0.5, False),
                  note="level only, no acceleration test"),
    ]


class MacroGateScaled(at.PortfolioStrategy):
    """Scale the base weights by ``scale`` while a monthly macro warning is on (lagged one month
    for publication; no warning where the data do not exist yet)."""

    def __init__(self, base, warning, scale=0.5):
        self.base, self.scale = base, scale
        self.warning = warning.shift(1).fillna(False).astype(bool)

    def weights(self, prices):
        w = self.base.weights(prices)
        on = self.warning.reindex(prices.index, method="ffill").fillna(False).astype(bool)
        return w.mul(np.where(on, self.scale, 1.0), axis=0)


def _monthly(series):
    x = at.data.load_fred(series)
    return x.groupby(x.index.to_period("M")).last().to_timestamp()


def curve_inverted():
    """Estrella & Mishkin (1996/98): 10-year minus 3-month yield below zero warns of recession
    12 months ahead. Long rate from Shiller (GS10 after 1953), T-bill from FRED (1934+)."""
    long = at.data.load_shiller_sp500()["long_rate"] * 100
    long.index = long.index.to_period("M").to_timestamp()
    tb = _monthly("TB3MS")
    spread = (long.reindex(tb.index) - tb).dropna()
    return spread < 0


def credit_widening():
    """Credit-spread trend (Gilchrist & Zakrajsek 2012: credit spreads lead downturns): Moody's
    Baa minus Aaa yield above its own 10-month average, the same round rule as the price trend."""
    sp = (_monthly("BAA") - _monthly("AAA")).dropna()
    return sp > sp.rolling(10).mean()


def iteration_14():
    """Checkpoint 34 (Exp 42): two macro warnings not yet tried - yield-curve inversion and a
    widening credit spread - each halving the champion's leverage. Published, round rules; 2 variants."""
    return [
        Candidate("ensemble, half leverage when the yield curve is inverted", "ensemble",
                  lambda: MacroGateScaled(ensemble(3.0), curve_inverted(), 0.5),
                  note="10y - 3m < 0 (Estrella-Mishkin); data from 1934"),
        Candidate("ensemble, half leverage while credit spreads widen", "ensemble",
                  lambda: MacroGateScaled(ensemble(3.0), credit_widening(), 0.5),
                  note="Baa - Aaa above its 10-month average; data from 1919"),
    ]


class AnchorLeverage(at.PortfolioStrategy):
    """Li & Yu (2012, JFE): nearness to the 52-week high predicts HIGHER market returns
    (under-reaction to good news), nearness to the all-time high LOWER returns. Trend-filtered
    S&P 500 at ``hi`` x when the price index is within 5% of its 52-week high (and, if
    ``avoid_ath``, more than 5% below its all-time high), ``lo`` x otherwise. Uses the Shiller
    price index (month-end), so the anchors use only past closes."""

    def __init__(self, hi=3.0, lo=1.0, avoid_ath=False):
        self.hi, self.lo, self.avoid_ath = hi, lo, avoid_ath
        p = at.data.load_shiller_sp500()["close"]
        near52 = p / p.rolling(12, min_periods=12).max() >= 0.95
        cond = near52
        if avoid_ath:
            cond &= p / p.cummax() < 0.95
        self.lev = pd.Series(np.where(cond, hi, lo), index=p.index)

    def weights(self, prices):
        lev = self.lev.reindex(prices.index).ffill().fillna(self.lo)
        return filtered(1.0).weights(prices).mul(lev, axis=0)


def ensemble4(fourth):
    """The champion's three sleeves plus a fourth, in equal quarters."""
    return at.Blend({at.Levered(filtered(1.0), 3.0): 0.25,
                     CappedValuationLeverage(1.0, 2.0, 0.5, 3.0): 0.25,
                     at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0): 0.25,
                     fourth: 0.25})


def iteration_15():
    """Checkpoint 35 (Exp 43): psychological anchors (Li & Yu 2012) as a 4th equal-weight sleeve.
    Round 5% nearness thresholds, 3x/1x; 2 variants."""
    return [
        Candidate("ensemble + 52-week-high anchor sleeve", "ensemble",
                  lambda: ensemble4(AnchorLeverage(3.0, 1.0, False)),
                  note="4th sleeve: 3x within 5% of the 52-week high, else 1x (trend-filtered)"),
        Candidate("ensemble + Li-Yu anchor sleeve (52-week high, not all-time high)", "ensemble",
                  lambda: ensemble4(AnchorLeverage(3.0, 1.0, True)),
                  note="3x near the 52-week high but >5% below the all-time high, else 1x"),
    ]


class IntermediateMomentum(at.Strategy):
    """Novy-Marx (2012, JFE) 'Is momentum really momentum?': returns from 12 to 7 months ago
    predict better than the most recent 6 months. Long when the 12-7 month return is positive."""

    def positions(self, px):
        r = px.shift(6) / px.shift(12) - 1
        return (r > 0).astype(float).where(r.notna(), 1.0)


class VoteSignal(at.Strategy):
    """Majority of SMA10, 12-1 month momentum and 12-7 month momentum (three horizons, one vote each)."""

    def positions(self, px):
        a = at.PriceAboveSMA(10).positions(px)
        b = ((px.shift(1) / px.shift(12) - 1) > 0).astype(float)
        c = IntermediateMomentum().positions(px)
        return ((a + b + c) >= 2).astype(float)


def ensemble_with_trend_signal(sig):
    return at.Blend({at.Levered(at.TrendFilter(mix(1.0), sig), 3.0): 1 / 3,
                     CappedValuationLeverage(1.0, 2.0, 0.5, 3.0): 1 / 3,
                     at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0): 1 / 3})


def iteration_16():
    """Checkpoint 37 (Exp 45): intermediate-horizon momentum (Novy-Marx 2012) as the trend
    sleeve's signal, alone and in a 3-horizon majority vote. 2 variants."""
    return [
        Candidate("ensemble, trend sleeve on 12-7 month momentum", "ensemble",
                  lambda: ensemble_with_trend_signal(IntermediateMomentum())),
        Candidate("ensemble, trend sleeve on a 3-horizon vote", "ensemble",
                  lambda: ensemble_with_trend_signal(VoteSignal()),
                  note="SMA10, 12-1 and 12-7 month momentum; 2 of 3"),
    ]


class MacroGatedTrend(at.PortfolioStrategy):
    """Growth-trend timing (Philosophical Economics 2016) with real-economy data: obey the 10-month
    trend exit only while the economy weakens; otherwise stay invested at ``lev``. Before the data
    exist the exit is always obeyed (= the plain trend sleeve). ``weak`` is lagged one month."""

    def __init__(self, weak, lev=3.0):
        self.lev = lev
        self.weak = weak.shift(1)

    def weights(self, prices):
        s = prices["stocks"]
        weak = self.weak.reindex(prices.index, method="ffill").fillna(True).astype(bool)
        on = ~weak | SMA10.positions(s).astype(bool)
        return pd.DataFrame({"stocks": np.where(on, self.lev, 0.0), "bonds": 0.0}, index=prices.index)


def unemployment_rising():
    u = _monthly("UNRATE")
    return u > u.rolling(12).mean()


def production_falling():
    ip = _monthly("INDPRO")
    return ip / ip.shift(12) - 1 < 0


def ensemble_trend_sleeve(sleeve):
    return at.Blend({sleeve: 1 / 3, CappedValuationLeverage(1.0, 2.0, 0.5, 3.0): 1 / 3,
                     at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0): 1 / 3})


def ensemble_unemployment():
    """Champion since checkpoint 38: the ensemble with its trend sleeve gated by unemployment
    (exit on the 10-month trend only while UNRATE is above its 12-month average)."""
    return ensemble_trend_sleeve(MacroGatedTrend(unemployment_rising()))


def iteration_17():
    """Checkpoint 38 (Exp 46): growth-trend timing with unemployment (1948+) and industrial
    production (1919+) instead of earnings; the trend sleeve exits only in a weakening economy. 2 variants."""
    return [
        Candidate("ensemble, trend sleeve gated by unemployment", "ensemble",
                  lambda: ensemble_trend_sleeve(MacroGatedTrend(unemployment_rising())),
                  note="exit only while UNRATE > its 12-month average"),
        Candidate("ensemble, trend sleeve gated by industrial production", "ensemble",
                  lambda: ensemble_trend_sleeve(MacroGatedTrend(production_falling())),
                  note="exit only while INDPRO is below its level a year earlier"),
    ]


class GatedValuationLeverage(CappedValuationLeverage):
    """The valuation sleeve with the same unemployment gate: its trend exit counts only while
    unemployment is rising; otherwise it holds stocks at the valuation-set leverage."""

    def __init__(self, weak):
        super().__init__(1.0, 2.0, 0.5, 3.0)
        self.weak = weak.shift(1)

    def weights(self, prices):
        prem = self.prem.reindex(prices.index).ffill()
        med = prem.expanding(120).median()
        lev = (self.k * prem / med).clip(self.lo, self.cap).fillna(1.0)
        weak = self.weak.reindex(prices.index, method="ffill").fillna(True).astype(bool)
        on = ~weak | SMA10.positions(prices["stocks"]).astype(bool)
        return pd.DataFrame({"stocks": np.where(on, lev, 0.0), "bonds": 0.0}, index=prices.index)


def iteration_18():
    """Checkpoint 41 (Exp 49): extend the champion's unemployment gate to the valuation sleeve
    (the recovery sleeve already holds Treasuries when out). Same rule, no new parameter. 1 variant."""
    w = unemployment_rising()
    return [
        Candidate("ensemble, trend and valuation sleeves gated by unemployment", "ensemble",
                  lambda: at.Blend({MacroGatedTrend(w): 1 / 3, GatedValuationLeverage(w): 1 / 3,
                                    at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0): 1 / 3})),
    ]


def sahm_rule():
    """Sahm (2019): recession signal when the 3-month average unemployment rate is 0.5 points or
    more above its low of the previous 12 months."""
    u = _monthly("UNRATE")
    a3 = u.rolling(3).mean()
    return a3 - a3.shift(1).rolling(12).min() >= 0.5


def claims_rising():
    """Weekly initial jobless claims (ICSA, 1967+): month-end 4-week average above its 52-week
    average - the same rising-unemployment test on a faster series. Before 1967: UNRATE gate."""
    c = at.data.load_fred("ICSA")
    w = c.rolling(4).mean() > c.rolling(52).mean()
    m = w.groupby(w.index.to_period("M")).last().to_timestamp()
    return pd.concat([unemployment_rising()[:m.index[0] - pd.Timedelta(days=1)], m])


def iteration_19():
    """Checkpoint 42 (Exp 50): two published alternatives to the champion's unemployment test.
    2 variants."""
    return [
        Candidate("ensemble, trend sleeve gated by the Sahm rule", "ensemble",
                  lambda: ensemble_trend_sleeve(MacroGatedTrend(sahm_rule()))),
        Candidate("ensemble, trend sleeve gated by jobless claims", "ensemble",
                  lambda: ensemble_trend_sleeve(MacroGatedTrend(claims_rising()))),
    ]


class BearAwareGatedTrend(MacroGatedTrend):
    """Exp 53: the unemployment-gated trend sleeve, but the trend exit is also obeyed in a bear
    market (price more than 20% below its 12-month high - the conventional definition), so a deep
    fall without rising unemployment (Japan 1990-2012) is not held at 3x."""

    def weights(self, prices):
        s = prices["stocks"]
        weak = self.weak.reindex(prices.index, method="ffill").fillna(True).astype(bool)
        bear = s / s.rolling(12, min_periods=1).max() < 0.80
        on = ~(weak | bear) | SMA10.positions(s).astype(bool)
        return pd.DataFrame({"stocks": np.where(on, self.lev, 0.0), "bonds": 0.0}, index=prices.index)


def iteration_20():
    """Checkpoint 44 (Exp 53): unemployment gate plus a bear-market override. 1 variant."""
    return [Candidate("ensemble, unemployment-gated trend with a bear-market override", "ensemble",
                      lambda: ensemble_trend_sleeve(BearAwareGatedTrend(unemployment_rising())),
                      note="trend exit obeyed if unemployment is rising OR price < 80% of its 12-month high")]


def half_half():
    """Exp 54: half the old champion, half the new one (a fixed 50/50 hedge against choosing the
    wrong trend-sleeve rule): the trend sleeve is 3x plain on half its money, 3x gated on the other."""
    return ensemble_trend_sleeve(at.Blend({at.Levered(filtered(1.0), 3.0): 0.5,
                                           MacroGatedTrend(unemployment_rising()): 0.5}))


def iteration_21():
    """Checkpoint 45 (Exp 54): 50/50 old and new trend sleeves. 1 variant."""
    return [Candidate("ensemble, trend sleeve half plain, half unemployment-gated", "ensemble", half_half)]


ITERATIONS = {1: iteration_1, 2: iteration_2, 3: iteration_3, 4: iteration_4, 5: iteration_5,
              6: iteration_6, 7: iteration_7, 8: iteration_8, 9: iteration_9, 10: iteration_10,
              11: iteration_11, 12: iteration_12, 13: iteration_13, 14: iteration_14, 15: iteration_15, 16: iteration_16, 17: iteration_17, 18: iteration_18, 19: iteration_19, 20: iteration_20, 21: iteration_21}


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
    # Always include the champion and the best strategy an ordinary (<=2x) margin account can hold.
    deploy = cheap[cheap["deployable"].astype(bool) & (cheap["family"] != "benchmark")].nlargest(1, "test_cagr")
    champ = cheap[(cheap["name"] == CHAMPION["name"]) & (cheap["params"] == CHAMPION["params"])]
    picks = pd.concat([cheap[cheap["family"] == "benchmark"], champ, deploy, best_per_family]).drop_duplicates("name")
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
