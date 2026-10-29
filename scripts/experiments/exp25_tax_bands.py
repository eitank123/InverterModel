"""Experiment 25: tax-aware rebalancing. Does trading less (wider no-trade bands) keep more
after tax?

The champion is run through the full execution simulator (ETFs, every cost, 25% capital-gains
tax with losses carried forward) over 1957-2026, with the ETF prices rebuilt from the monthly
index data. The only thing varied is the no-trade band: a holding is traded only when it is
more than X% of the account away from its target (full exits always happen). Round values
fixed in advance: 2% (current), 5%, 10%, 20%. Buy & hold is shown for reference.

    python scripts/experiments/exp25_tax_bands.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from algotrading.execution import INSTRUMENTS, CostModel, etf_prices, etf_weights, simulate  # noqa: E402
from loop import ensemble  # noqa: E402

START, TAX = "1957-01-01", 0.25
df = at.data.load_shiller_sp500()
idx = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
cash = (df["long_rate"] - 0.015).clip(lower=0)
costs = CostModel()
prices = etf_prices(idx.pct_change(), cash, INSTRUMENTS, costs)[START:]
W = ensemble(3.0).weights(idx)


def champion(d):
    return etf_weights(W.loc[d, "stocks"], W.loc[d, "bonds"]), "ensemble"


rows = {}
years = len(prices) / 12
for label, fn, band in [("S&P 500 buy & hold", lambda d: ({"SPY": 1.0}, "hold"), 0.02)] + \
        [(f"champion, band {b:.0%}", champion, b) for b in (0.02, 0.05, 0.10, 0.20)]:
    pre = simulate(fn, prices, cash, 100_000, costs, min_trade_frac=band)
    post = simulate(fn, prices, cash, 100_000, costs, min_trade_frac=band, tax_rate=TAX)
    rows[label] = {"trades/yr": len(pre["trades"]) / years,
                   "pre-tax CAGR": (pre["final"] / 1e5) ** (1 / years) - 1,
                   "after-tax CAGR (sold at end)": (post["liquidation_after_tax"] / 1e5) ** (1 / years) - 1,
                   "tax paid along the way ($k)": post["fees"]["taxes_paid"] / 1e3}
out = pd.DataFrame(rows).T
pd.set_option("display.width", 200)
print(f"{START[:4]}-2026, $100k, all costs, {TAX:.0%} capital-gains tax:\n")
print(out.to_string(formatters={"trades/yr": "{:.1f}".format, "pre-tax CAGR": "{:.2%}".format,
                                "after-tax CAGR (sold at end)": "{:.2%}".format,
                                "tax paid along the way ($k)": "{:,.0f}".format}))
