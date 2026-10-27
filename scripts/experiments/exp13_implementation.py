"""Experiment 13: how should the trend filter be implemented in practice?

The research so far used Shiller's monthly *average* prices. A real trader sees
month-end closes, or daily closes. Variants:
  * monthly SMA-10 on month-END closes  (what Faber actually tested)
  * monthly SMA-10 on month-AVERAGE prices (what our long backtests used)
  * daily SMA-200 (checked every day: faster exits, more whipsaw)
  * daily SMA-200 with a +/-2% band (hysteresis: only switch on a clear cross)
Traded on daily returns in every case.

  A. Real DJIA daily 2007-2016 (one decade, includes 2008: an anecdote, not proof)
  B. 100 synthetic daily regime-switching markets (where the answer is measurable)

    python scripts/experiments/exp13_implementation.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", "{:.3f}".format)


VARIANTS = {
    "buy & hold": at.BuyAndHold(),
    "monthly SMA10, month-end": at.MonthlySMA(10, "end"),
    "monthly SMA10, month-avg": at.MonthlySMA(10, "avg"),
    "daily SMA200": at.PriceAboveSMA(200),
    "daily SMA200 +/-2% band": at.BandedSMA(200, 0.02),
}
for name, s in VARIANTS.items():
    if name != "buy & hold":
        assert at.validation.check_no_lookahead(s, at.data.gbm(1500, seed=1)["close"]), name


def stats(r, pos):
    return {"cagr": at.metrics.cagr(r), "sharpe": at.metrics.sharpe(r),
            "max_dd": at.metrics.max_drawdown(r), "switches_per_yr": pos.diff().abs().sum() / len(r) * 252}


print("A. DJIA daily 2007-2016 (price only, lag 1 day, 2bp, cash 0.5%)")
dj = at.data.load_djia_daily()["close"]
rows = {}
for name, s in VARIANTS.items():
    res = at.run_backtest(dj, s, cost_bps=2, execution_lag=1, cash_rate=0.005)
    rows[name] = stats(res.returns, res.positions)
print(pd.DataFrame(rows).T)

print("\nB. 100 synthetic daily markets, 30 years each (regime switching, 2bp, cash 2%)")
rows = []
for seed in range(100):
    p = at.data.regime_switching(n=252 * 30, mus=(0.15, -0.15), sigmas=(0.12, 0.25),
                                 p_stay=0.997, seed=seed)["close"]
    for name, s in VARIANTS.items():
        res = at.run_backtest(p, s, cost_bps=2, execution_lag=1, cash_rate=0.02)
        rows.append({"variant": name, **stats(res.returns, res.positions)})
syn = pd.DataFrame(rows)
print(syn.groupby("variant").mean())
piv = syn.assign(seed=np.repeat(range(100), len(VARIANTS))).pivot(index="seed", columns="variant", values="sharpe")
print("\n   P(Sharpe beats 'monthly SMA10, month-end') across seeds:")
print((piv.sub(piv["monthly SMA10, month-end"], axis=0) > 0).mean().round(2).to_string())
