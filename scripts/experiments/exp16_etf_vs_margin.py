"""Experiment 16: how to hold 2x in practice: a daily-reset leveraged ETF or a margin account?

* Leveraged ETF: leverage reset to exactly 2x every day; expense ratio ~0.9%/yr plus
  financing inside the fund (~cash + spread). Daily reset loses about L(L-1)/2 * variance
  per year in choppy markets and gains in steady trends.
* Margin: buy 2x at the start of each month and let leverage drift until the next
  rebalance; you pay the broker's margin rate on the loan.
Both follow the champion's signal (S&P/DJIA above its 10-month average at month end).

    python scripts/experiments/exp16_etf_vs_margin.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import algotrading as at  # noqa: E402

L = 2.0


def simulate(prices: pd.Series, cash: float, etf_fee: float, etf_spread: float,
             margin_spread: float) -> dict:
    """Daily simulation of both implementations of the same monthly signal."""
    sig = at.MonthlySMA(10, "end").positions(prices).shift(1).fillna(0.0)   # trade next day
    r = prices.pct_change().fillna(0.0).to_numpy()
    on = sig.to_numpy()
    month = prices.index.to_period("M")
    new_month = np.r_[True, month[1:] != month[:-1]]
    dc = cash / 252
    etf = margin = 1.0
    shares = loan = 0.0              # margin account: shares value in "price units" and loan
    out_etf, out_margin = [], []
    px = prices.to_numpy()
    for t in range(len(r)):
        # ETF: daily reset
        if on[t]:
            etf *= 1 + L * r[t] - (L - 1) * (dc + etf_spread / 252) - etf_fee / 252
        else:
            etf *= 1 + dc
        # Margin: value today's holding, then set tomorrow's position at today's close:
        # back to L on the first trading day of a month, or when the signal flips.
        if t > 0:
            loan *= 1 + dc + margin_spread / 252
            margin = shares * px[t] - loan if shares else margin * (1 + dc)
        nxt = on[t + 1] if t + 1 < len(r) else on[t]
        rebalance = t == 0 or (t + 1 < len(r) and new_month[t + 1]) or nxt != on[t]
        if rebalance:
            if nxt:
                shares, loan = L * margin / px[t], (L - 1) * margin
            else:
                shares, loan = 0.0, 0.0
        out_etf.append(etf)
        out_margin.append(margin)
    idx = prices.index
    e, m = pd.Series(out_etf, idx), pd.Series(out_margin, idx)
    bh = prices / prices.iloc[0]
    yrs = len(idx) / 252
    g = lambda s: s.iloc[-1] ** (1 / yrs) - 1  # noqa: E731
    return {"etf_cagr": g(e), "margin_cagr": g(m), "unlevered_index_cagr": g(bh),
            "etf_max_dd": at.metrics.max_drawdown(e.pct_change().fillna(0)),
            "margin_max_dd": at.metrics.max_drawdown(m.pct_change().fillna(0))}


print("A. Real DJIA daily 2007-2016 (price only), cash 1%")
dj = at.data.load_djia_daily()["close"]
for label, fee, es, ms in [("ETF 0.9% fee vs margin +1.5%", 0.009, 0.005, 0.015),
                           ("ETF 0.9% fee vs margin +5% (retail)", 0.009, 0.005, 0.05)]:
    res = simulate(dj, 0.01, fee, es, ms)
    print(f"   {label:38s} ETF {res['etf_cagr']:6.1%} (DD {res['etf_max_dd']:.0%})   "
          f"margin {res['margin_cagr']:6.1%} (DD {res['margin_max_dd']:.0%})   index {res['unlevered_index_cagr']:.1%}")

print("\nB. 200 synthetic 20-year daily markets (GARCH, 7% drift, ~16% vol), cash 3%")
rows = []
for seed in range(200):
    p = at.data.garch(n=252 * 20, mu=0.09, seed=seed)["close"]
    rows.append(simulate(p, 0.03, 0.009, 0.005, 0.015))
syn = pd.DataFrame(rows)
print(f"   mean CAGR: ETF {syn.etf_cagr.mean():.2%}, margin(+1.5%) {syn.margin_cagr.mean():.2%}; "
      f"ETF better in {(syn.etf_cagr > syn.margin_cagr).mean():.0%} of markets; "
      f"median gap {(syn.etf_cagr - syn.margin_cagr).median():+.2%}/yr")
