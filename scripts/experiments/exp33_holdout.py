"""Experiment 33: a clean split - choose with data up to 2024, test on 2025-2026 once.

Owner's request: 'do all the trading up until 2024 and have 2025/6 to test on (like validation
and test)'. Every rule here has fixed, published, round parameters, so nothing is re-fitted; what the
split checks is the CHOICE between rules and whether their edge survives in the newest data.
Honest caveat: 2025-2026 was inside the 1991-2026 era used in earlier rounds, so it is not perfectly
unseen. The only fully unseen data is the real-time simulation (from 2026-10-23).

  * train: 1927-2014 (CRSP monthly, three eras 1927-56 / 1957-90 / 1991-2014)
  * validation: 2015-2024 - the champion rule (best worst-era over train + validation) picks ONE
  * test: 2025-01 to the latest month - reported once, for every candidate
Part B replays 2025-01-02 to the latest close DAY BY DAY on REAL ETF prices (Yahoo Finance: SPY,
UPRO, IEF, the sector SPDRs, MTUM, VTV) with every trading cost and margin interest, exactly as the
play-money accounts trade.

    python scripts/experiments/exp33_holdout.py
"""
import os
import shutil
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import algotrading as at  # noqa: E402
import exp31_combinations as e  # noqa: E402
from algotrading.execution import INSTRUMENTS, REAL_ETFS, SECTORS, CostModel, etf_weights  # noqa: E402
from algotrading.live import completed_month_ends, shiller_history, splice_monthly  # noqa: E402
from algotrading.live_sim import LiveSimAccount  # noqa: E402
from loop import ensemble, ensemble_lower  # noqa: E402

# ---------------- Part A: monthly, CRSP --------------------------------------------------------
D = e.LEGS["diversified (ind mom + stock mom + value)"]
lower_W = e.mi(ensemble_lower().weights(e.idx))


def from_W(W, leg=e.MKT):
    W = W.shift(1).reindex(e.DATES)
    g = W["stocks"] + W["bonds"]
    return (W["stocks"] * leg.reindex(e.DATES) + W["bonds"] * e.RB.reindex(e.DATES) + (1 - g) * e.RF.reindex(e.DATES)
            - (g - 1).clip(lower=0) * 0.005 / 12 - W.diff().abs().sum(axis=1).fillna(0) * 0.001)


CANDS = {
    "S&P 500 buy & hold": e.MKT.reindex(e.DATES),
    "60/40": (0.6 * e.MKT + 0.4 * e.RB).reindex(e.DATES),
    "champion (3x trend + value + recovery)": e.portfolio(e.MKT),
    "lower-leverage ensemble": from_W(lower_W),
    "trend sleeve alone 3x": e.portfolio(e.MKT, sleeves=("trend",)),
    "diversified leg x1": e.portfolio(D),
    "diversified leg x2/3 (max 2x)": e.portfolio(D, scale=2 / 3),
    "industry-momentum leg x1": e.portfolio(e.LEGS["industry momentum"]),
}
SPLITS = {"train 1927-56": ("1927", "1956"), "train 1957-90": ("1957", "1990"), "train 1991-2014": ("1991", "2014"),
          "validation 2015-24": ("2015", "2024"), "TEST 2025-26": ("2025", "2026")}


def dd(r):
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min())


rows = {}
for k, r in CANDS.items():
    row = {s: at.metrics.cagr(r[a:b], 12) for s, (a, b) in SPLITS.items()}
    row["worst (train+val)"] = min(v for s, v in row.items() if not s.startswith("TEST"))
    row["max DD to 2024"] = dd(r[:"2024"])
    row["TEST max DD"] = dd(r["2025":])
    rows[k] = row
out = pd.DataFrame(rows).T
pick = out["worst (train+val)"].idxmax()
pd.set_option("display.width", 250)
print(f"Part A (monthly, CRSP, all costs; test months {e.DATES[e.DATES >= '2025'][0]:%Y-%m} to {e.DATES[-1]:%Y-%m}):\n")
print(out.to_string(float_format=lambda v: f"{v:.1%}"))
print(f"\nChoice made with data to 2024 (best worst era over train + validation): {pick}")
print(f"  its TEST result: {out.loc[pick, 'TEST 2025-26']:.1%}/yr vs S&P {out.loc['S&P 500 buy & hold', 'TEST 2025-26']:.1%}, "
      f"current champion {out.loc['champion (3x trend + value + recovery)', 'TEST 2025-26']:.1%}")

# ---------------- Part B: daily, real ETF prices, 2025-01-02 onward ---------------------------
real = at.data.etf_closes(["SPY", "UPRO"] + list(REAL_ETFS), rng="5y")
tb = at.data.load_fred("DTB3") / 100
cash = tb.reindex(real.index).ffill().bfill()
closes = at.data.fred_daily_proxies(sp500=at.data.sp500_with_fallback())
history, _ = shiller_history()
INS = {**REAL_ETFS, "SPY": INSTRUMENTS["SPY"], "UPRO": INSTRUMENTS["UPRO"]}


def exposure(strategy):
    cache = {}

    def fn(d):
        m = d.strftime("%Y-%m")
        if m not in cache:
            me = completed_month_ends(closes[:d], asof=d)
            w = strategy.weights(splice_monthly(history, me)).iloc[-1]
            cache[m] = (float(w.get("stocks", 0.0)), float(w.get("bonds", 0.0)))
        return cache[m]
    return fn


champ, lower = exposure(ensemble(3.0)), exposure(ensemble_lower())


def etf_rule(expo):
    def fn(d):
        s, b = expo(d)
        return etf_weights(s, b), {"stocks": s, "bonds": b}, f"stocks {s:.2f}x bonds {b:.2f}x"
    return fn


def diversified(d):
    s, b = champ(d)
    s, b = s * 2 / 3, b * 2 / 3
    me = real[SECTORS][real.index < d.to_period("M").to_timestamp()]
    me = me.groupby(me.index.to_period("M")).tail(1)
    mom = (me.iloc[-2] / me.iloc[-13] - 1).dropna()
    top = list(mom.sort_values(ascending=False).index[:3])
    w = {"MTUM": s / 3, "VTV": s / 3, **{t: s / 9 for t in top}}
    if b:
        w["IEF"] = b
    return {k: v for k, v in w.items() if v > 1e-9}, {"stocks": s, "bonds": b}, f"top {top}"


RULES = {"S&P 500 (SPY)": (lambda d: ({"SPY": 1.0}, {"stocks": 1.0}, "hold"), CostModel()),
         "60/40 (SPY/IEF)": (lambda d: ({"SPY": 0.6, "IEF": 0.4}, {"stocks": 0.6}, "60/40"), CostModel()),
         "champion (SPY/UPRO/IEF)": (etf_rule(champ), CostModel()),
         "lower-leverage ensemble": (etf_rule(lower), CostModel()),
         "diversified x2/3 (sectors+MTUM+VTV, margin)": (diversified, CostModel(swap_spread=0.015))}
tmp = tempfile.mkdtemp()
px = real[real.index >= "2025-01-02"]
print(f"\nPart B (daily, real ETF prices, every cost, {px.index[0]:%Y-%m-%d} to {px.index[-1]:%Y-%m-%d}):\n")
rows = {}
for name, (fn, costs) in RULES.items():
    a = LiveSimAccount.load_or_create(os.path.join(tmp, f"{len(rows)}.json"), "x", name, "", px.index[0].strftime("%Y-%m-%d"))
    a.advance(px, cash, fn, costs, instruments=INS)
    eq = pd.Series({pd.Timestamp(h["date"]): h["equity"] for h in a.st["equity_history"]})
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    rows[name] = {"2025": eq[:"2025"].iloc[-1] / 1e5 - 1, "2026 to date": eq.iloc[-1] / eq[:"2025"].iloc[-1] - 1,
                  "total": eq.iloc[-1] / 1e5 - 1, "per year": (eq.iloc[-1] / 1e5) ** (1 / yrs) - 1,
                  "max DD": dd(r), "trades": len(a.st["trades"]),
                  "costs paid": sum(a.st["fees"].values())}
shutil.rmtree(tmp)
out = pd.DataFrame(rows).T
print(out.to_string(formatters={**{c: "{:+.1%}".format for c in ["2025", "2026 to date", "total", "per year", "max DD"]},
                                "trades": "{:.0f}".format, "costs paid": "${:,.0f}".format}))
