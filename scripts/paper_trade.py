"""Advance the forward paper-trading accounts by every newly completed month.

    python scripts/paper_trade.py            # offline: Shiller month data as SPY / IEF stand-ins

Accounts start on the latest month available when first created and only move forward:
this is the "real life" record, not a backtest. When a broker is connected
(docs/AUTOMATION.md), the same strategies run through scripts/rebalance.py instead.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from algotrading.paper import PaperAccount  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "loop", "paper")
SYMBOLS = {"stocks": "SPY", "bonds": "IEF"}


def filtered(stock_w):
    return at.TrendFilter(at.PerAsset(at.BuyAndHold(), {"stocks": stock_w, "bonds": 1 - stock_w}),
                          at.PriceAboveSMA(10))


ACCOUNTS = {
    "champion_deployable": ("Champion (deployable): S&P 500 + SMA10 x2",
                            "S&P 500 held at 2x leverage while above its 10-month average, cash otherwise. "
                            "Fits an ordinary margin account.", at.Levered(filtered(1.0), 2.0), 0.015, 2.0),
    "champion_research": ("Research champion: 60/40 + SMA10 x3",
                          "60/40 with each asset trend-filtered, at 3x leverage. Needs futures-level financing.",
                          at.Levered(filtered(0.6), 3.0), 0.005, 3.0),
    "benchmark_6040": ("Benchmark: 60/40", "60% stocks, 40% bonds, rebalanced monthly, no leverage.",
                       at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4}), 0.0, 1.0),
}


def main():
    df = at.data.load_shiller_sp500()
    px = pd.DataFrame({"stocks": df["tr"], "bonds": at.data.bond_total_return(df["long_rate"])})
    # Stand-in ETF prices: the total-return indices rescaled to SPY ~ 650 and IEF ~ 95 today.
    etf = pd.DataFrame({"SPY": px["stocks"] / px["stocks"].iloc[-1] * 650,
                        "IEF": px["bonds"] / px["bonds"].iloc[-1] * 95})
    cash = (df["long_rate"] - 0.015).clip(lower=0)
    for key, (name, desc, strategy, spread, max_gross) in ACCOUNTS.items():
        acct = PaperAccount.load_or_create(
            os.path.join(ROOT, f"{key}.json"), name=name, description=desc, symbols=SYMBOLS,
            start=px.index[-1].strftime("%Y-%m-%d"), cash=100_000.0)
        months = px.index if acct.state.last_month is None else \
            px.index[px.index.strftime("%Y-%m") > acct.state.last_month]
        if acct.state.last_month is None:
            months = px.index[-1:]
        for m in months:
            hist = px.loc[:m]
            w = strategy.weights(hist).iloc[-1].to_dict()
            acct.step(m, etf.loc[m].to_dict(), w, float(cash.loc[m]), spread, max_gross=max_gross + 0.01)
            print(f"{m:%Y-%m} {name:45s} equity {acct.state.equity_history[-1]['equity']:>12,.2f}  "
                  f"targets {acct.state.targets}")
        acct.save()


if __name__ == "__main__":
    main()
