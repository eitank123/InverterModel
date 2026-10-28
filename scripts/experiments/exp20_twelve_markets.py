"""Experiment 20: the US-designed rules on 12 other stock markets (out of sample).

Exp 19 found that 3x leverage came close to ruin in Japan. Is that a one-off? FRED carries OECD
monthly share-price indices (SPASTT01<cc>M661N): Germany, UK, France, Italy, Canada,
Australia, Netherlands, Switzerland, Sweden, Spain, Korea, Japan, mostly from the 1950s-60s.
Monthly AVERAGE price indices (like Shiller, so trades lag one month), dividends added at
3%/yr, cash = US 3-month T-bill (a USD-hedged approximation), leverage financed at cash +
0.5%, 10bp per unit traded. Same rules and parameters as the US champion's sleeves.

    python scripts/experiments/exp20_twelve_markets.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from loop import filtered, mix  # noqa: E402

COUNTRIES = {"DE": "Germany", "GB": "UK", "FR": "France", "IT": "Italy", "CA": "Canada",
             "AU": "Australia", "NL": "Netherlands", "CH": "Switzerland", "SE": "Sweden",
             "ES": "Spain", "KR": "Korea", "JP": "Japan"}
tb = at.data.load_fred("DTB3") / 100
tbm = tb.groupby(tb.index.to_period("M")).mean()
tbm.index = tbm.index.to_timestamp()


def standin_mix(trend, rec):
    """Champion structure; valuation sleeve replaced by a neutral 1x trend sleeve (no earnings data)."""
    return at.Blend({at.Levered(filtered(1.0), trend): 1 / 3, filtered(1.0): 1 / 3,
                     at.RecoveryLeverage(rec, rec * 1.5, 0.15, signal="either"): 1 / 3})


RULES = {
    "B&H 1x": mix(1.0), "B&H 2x": at.Levered(mix(1.0), 2.0), "B&H 3x": at.Levered(mix(1.0), 3.0),
    "trend 1x": filtered(1.0), "trend 2x": at.Levered(filtered(1.0), 2.0),
    "trend 3x": at.Levered(filtered(1.0), 3.0),
    "recovery+ 2/3x": at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either"),
    "champion-like mix": standin_mix(3.0, 2.0),
    "lower-leverage mix": standin_mix(2.0, 1.0),
}

cagr, mdd = {}, {}
for cc, name in COUNTRIES.items():
    s = at.data.load_fred(f"SPASTT01{cc}M661N")
    s.index = s.index.to_period("M").to_timestamp()
    tr = s * (1 + 0.03 / 12) ** np.arange(len(s))
    px = pd.DataFrame({"stocks": tr, "bonds": tr})
    cash = tbm.reindex(px.index).ffill().fillna(0.015)
    for rule, strat in RULES.items():
        r = at.run_backtest(px, strat, cost_bps=10, execution_lag=1, cash_rate=cash,
                            borrow_spread=0.005).returns.iloc[12:]
        cagr[(name, rule)] = at.metrics.cagr(r, 12)
        mdd[(name, rule)] = at.metrics.max_drawdown(r)

C = pd.Series(cagr).unstack()[list(RULES)]
D = pd.Series(mdd).unstack()[list(RULES)]
pd.set_option("display.width", 250)
print("Growth per year (each country's full history):")
print(C.to_string(float_format=lambda v: f"{v:.1%}"))
print("\nMax drawdown:")
print(D.to_string(float_format=lambda v: f"{v:.0%}"))
summary = pd.DataFrame({
    "median growth": C.median(), "worst-country growth": C.min(),
    "median max DD": D.median(), "worst max DD": D.min(),
    "countries with DD worse than -80%": (D < -0.80).sum(),
    "countries beating B&H 1x": (C.gt(C["B&H 1x"], axis=0)).sum(),
})
print("\nSummary across 12 markets:")
print(summary.to_string(formatters={"median growth": "{:.1%}".format, "worst-country growth": "{:.1%}".format,
                                    "median max DD": "{:.0%}".format, "worst max DD": "{:.0%}".format}))
