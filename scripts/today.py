"""What do the frozen rules say to hold right now?

    python scripts/today.py

Uses the latest month in the Shiller dataset (prices are monthly averages, so this is an
approximation of a month-end signal). Not investment advice: a research readout.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402

df = at.data.load_shiller_sp500()
px = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
last = px.index[-1]
print(f"Data through {last:%Y-%m} (S&P 500 = {df['close'].iloc[-1]:,.0f}, 10y yield = "
      f"{df['long_rate'].iloc[-1]:.2%})\n")

for asset in px:
    p = px[asset]
    sma = p.rolling(10).mean().iloc[-1]
    gap = p.iloc[-1] / sma - 1
    print(f"{asset:7s} price vs 10-month average: {gap:+.1%} -> {'IN' if gap > 0 else 'OUT (cash)'}")

infl10 = (df["cpi"] / df["cpi"].shift(120)) ** (1 / 10) - 1
ecy = 1 / df["cape"].shift(6) - (df["long_rate"] - infl10).shift(6)
pct = ecy.expanding(min_periods=240).rank(pct=True).ffill()
print(f"\nLast known CAPE {df['cape'].dropna().iloc[-1]:.1f} ({df['cape'].dropna().index[-1]:%Y-%m}); "
      f"excess CAPE yield percentile vs history: {pct.iloc[-1]:.0%} (low = expensive stocks)")

core = at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4})
rules = {
    "60/40 + SMA-10 (frozen)": at.TrendFilter(core, at.PriceAboveSMA(10)),
    "value tilt + SMA-10": at.TrendFilter(at.ValueTilt(pct), at.PriceAboveSMA(10)),
}
print("\nTarget weights for next month:")
for name, s in rules.items():
    w = s.weights(px).iloc[-1]
    print(f"  {name:26s} stocks {w['stocks']:.0%}, bonds {w['bonds']:.0%}, cash {1 - w.sum():.0%}")
