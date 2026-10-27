"""Full-cost simulation of the last 5 years: $100,000 traded through real ETFs.

    python scripts/simulate_last5.py

Every trade pays commission, spread + slippage and US regulatory fees; ETF prices are net
of expense ratios, leveraged-ETF financing and daily-reset drag; idle cash earns the cash
rate minus a fund fee. Outputs loop/dashboard/sim_*.json for the dashboard and prints a summary.

Caveats: monthly data (Shiller averages until mid-2023, month values after), so trades happen
monthly at the next month's price; ETF prices are reconstructed from index returns, not quotes.
Results are pre-tax.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from algotrading.execution import CostModel, etf_prices, simulate  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "loop", "dashboard")
START, START_CASH = "2021-10-01", 100_000.0

df = at.data.load_shiller_sp500()
idx = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
cash = (df["long_rate"] - 0.015).clip(lower=0)
costs = CostModel()
prices_all = etf_prices(idx.pct_change(), cash, at.execution.INSTRUMENTS, costs,
                        start_price={"SPY": 650.0, "SSO": 110.0, "UPRO": 105.0, "IEF": 95.0})
prices = prices_all[START:]
sma = idx["stocks"].rolling(10).mean()
high36 = idx["stocks"].rolling(36, min_periods=12).max()


def champion(d):
    s = idx["stocks"].loc[d]
    gap = s / sma.loc[d] - 1
    below_high = s / high36.loc[d] - 1
    if gap <= 0:
        return {}, f"S&P {gap:+.1%} vs its 10-month average: out, hold cash"
    if below_high < -0.15:
        return {"UPRO": 1.0}, f"S&P {below_high:.0%} below its 3-year high but trending up ({gap:+.1%}): 3x"
    return {"SSO": 1.0}, f"S&P {gap:+.1%} above its 10-month average: 2x"


def margin_champion(d):
    gap = idx["stocks"].loc[d] / sma.loc[d] - 1
    return ({"SSO": 1.0}, f"S&P {gap:+.1%} above its 10-month average: 2x") if gap > 0 else \
        ({}, f"S&P {gap:+.1%} vs its 10-month average: out, hold cash")


STRATS = {
    "Champion: recovery leverage (SSO/UPRO)": champion,
    "Margin champion as ETF: 2x trend (SSO)": margin_champion,
    "Benchmark: 60/40 (SPY/IEF)": lambda d: ({"SPY": 0.6, "IEF": 0.4}, "monthly rebalance to 60/40"),
    "Benchmark: S&P 500 buy & hold (SPY)": lambda d: ({"SPY": 1.0}, "buy and hold"),
}
# Cross-check: the signal must match the research implementation.
w_lib = at.RecoveryLeverage(2.0, 3.0, 0.15).weights(idx[: prices.index[-1]])["stocks"][START:]
w_here = pd.Series({d: {"UPRO": 3.0, "SSO": 2.0}.get(next(iter(champion(d)[0]), ""), 0.0)
                    for d in prices.index})
assert (w_lib.reindex(w_here.index) == w_here).all(), "simulation signal differs from the library"

summary, equity_rows, trade_rows, yearly_rows = [], [], [], []
for name, fn in STRATS.items():
    res = simulate(fn, prices, cash, START_CASH, costs)
    eq, tr, fees = res["equity"], res["trades"], res["fees"]
    years = len(eq) / 12
    cagr = (res["final"] / START_CASH) ** (1 / years) - 1
    trading_costs = fees["commission"] + fees["spread_slippage"] + fees["regulatory"]
    summary.append({"strategy": name, "start": eq.index[0], "end": eq.index[-1], "start_value": START_CASH,
                    "end_value": round(res["final"], 2), "profit": round(res["final"] - START_CASH, 2),
                    "cagr": cagr, "max_drawdown": float((eq["equity"] / eq["equity"].cummax() - 1).min()),
                    "trades": int(len(tr)), "commissions": round(fees["commission"], 2),
                    "spread_slippage": round(fees["spread_slippage"], 2),
                    "regulatory_fees": round(fees["regulatory"], 2),
                    "trading_costs_total": round(trading_costs, 2),
                    "fund_costs_embedded": round(fees["fund_costs_embedded"], 2),
                    "all_costs": round(trading_costs + fees["fund_costs_embedded"], 2),
                    "realized_pl": round(float(tr["realized_pl"].sum()) if len(tr) else 0.0, 2)})
    for d, r in eq.iterrows():
        equity_rows.append({"strategy": name, "date": d, "equity": r["equity"], "pl": r["pl"]})
    for _, t in tr.iterrows():
        trade_rows.append({"strategy": name, **t.to_dict()})
    e = eq["equity"]
    e.index = pd.to_datetime(e.index)
    prev = START_CASH
    for y, g in e.groupby(e.index.year):
        yearly_rows.append({"strategy": name, "year": int(y), "start_value": round(prev, 2),
                            "end_value": round(g.iloc[-1], 2), "pl": round(g.iloc[-1] - prev, 2),
                            "return": g.iloc[-1] / prev - 1})
        prev = g.iloc[-1]

os.makedirs(OUT, exist_ok=True)
for fname, rows in [("sim_summary", summary), ("sim_equity", equity_rows), ("sim_trades", trade_rows),
                    ("sim_yearly", yearly_rows)]:
    with open(os.path.join(OUT, f"{fname}.json"), "w") as f:
        json.dump(rows, f, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
s = pd.DataFrame(summary).set_index("strategy")
print(f"$100,000 from {s['start'].iloc[0]} to {s['end'].iloc[0]}, all costs included (pre-tax):\n")
print(s[["end_value", "profit", "cagr", "max_drawdown", "trades", "trading_costs_total",
         "fund_costs_embedded", "all_costs"]].to_string(float_format=lambda v: f"{v:,.3f}"))
print("\nYearly P/L ($):")
y = pd.DataFrame(yearly_rows).pivot(index="year", columns="strategy", values="pl")
print(y.to_string(float_format=lambda v: f"{v:,.0f}"))
print("\nChampion trades:")
t = pd.DataFrame(trade_rows)
print(t[t.strategy.str.startswith("Champion")][["date", "side", "symbol", "shares", "price", "amount",
      "commission", "spread_slippage", "regulatory_fees", "realized_pl", "reason"]].to_string(index=False))
