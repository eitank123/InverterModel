"""Real-time forward simulation of the strategies (play money, real market data).

    python scripts/live_sim.py

Reads the newest daily closes from FRED (S&P 500, 10-year Treasury yield, 3-month T-bill),
advances every account by the trading days it has not seen yet, and saves the record in
loop/live_sim/*.json. No money and no broker are involved. ETF prices (SPY, UPRO, IEF, UST)
are built from the real daily index moves net of each fund's leverage financing and expense
ratio; fills are at the daily close with spread, slippage, commission and regulatory fees.
Accounts start on the latest available close when first created and are never backfilled.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import algotrading as at  # noqa: E402
from algotrading.execution import CostModel, etf_weights  # noqa: E402
from algotrading.live import completed_month_ends, shiller_history, splice_monthly  # noqa: E402
from algotrading.live_sim import LiveSimAccount, daily_etf_returns  # noqa: E402
from loop import ensemble  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "loop", "live_sim")
ANCHOR = "2026-10-23"                       # price levels: SPY = S&P/10 there; others fixed
LEVELS = {"UPRO": 100.0, "IEF": 95.0, "UST": 45.0}

sp, y10, tb = at.data.load_fred("SP500"), at.data.load_fred("DGS10"), at.data.load_fred("DTB3")
closes = at.data.fred_daily_proxies(sp500=sp, dgs10=y10)
cash = (tb / 100).reindex(closes.index).ffill()
costs = CostModel()
growth = (1 + daily_etf_returns(closes, cash, costs)).cumprod()
anchor = pd.Timestamp(ANCHOR)
etf = growth / growth.loc[anchor] * pd.Series({"SPY": sp.loc[anchor] / 10, **LEVELS})
history, _ = shiller_history()


def from_strategy(strategy):
    def fn(d):
        me = completed_month_ends(closes[:d], asof=d)
        monthly = splice_monthly(history, me)
        w = strategy.weights(monthly).iloc[-1]
        exp = {"stocks": float(w.get("stocks", 0.0)), "bonds": float(w.get("bonds", 0.0))}
        return etf_weights(exp["stocks"], exp["bonds"]), exp, \
            f"signal from month-ends to {me.index[-1]:%Y-%m}: stocks {exp['stocks']:.2f}x, bonds {exp['bonds']:.2f}x"
    return fn


def fixed(weights, reason):
    return lambda d: (weights, {"stocks": weights.get("SPY", 0.0), "bonds": weights.get("IEF", 0.0)}, reason)


ACCOUNTS = {
    "champion_ensemble": ("Champion: ensemble (3x trend + valuation + recovery+)",
                          "Equal thirds of three published effects: S&P 500 at 3x above its 10-month average; "
                          "leverage 0.5-3x set by valuation; recovery+. Traded through SPY/UPRO and IEF/UST.",
                          from_strategy(ensemble(3.0))),
    "challenger_recovery_plus": ("Challenger: recovery+ (2x/3x, 2x Treasuries when out)",
                                 "2x S&P while above its 10-month average or up over 12 months, 3x after >15% "
                                 "falls; 2x Treasuries otherwise.",
                                 from_strategy(at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0))),
    "previous_recovery": ("Previous champion: recovery leverage",
                          "2x S&P above its 10-month average, 3x after >15% falls, cash otherwise.",
                          from_strategy(at.RecoveryLeverage(2.0, 3.0, 0.15))),
    "benchmark_sp500": ("Benchmark: S&P 500 buy & hold", "100% SPY.", fixed({"SPY": 1.0}, "buy and hold")),
    "benchmark_6040": ("Benchmark: 60/40", "60% SPY, 40% IEF, rebalanced monthly.",
                       fixed({"SPY": 0.6, "IEF": 0.4}, "monthly rebalance to 60/40")),
}

print(f"Data: FRED S&P 500 to {sp.index[-1]:%Y-%m-%d}, 10y yield to {y10.index[-1]:%Y-%m-%d}, "
      f"T-bill to {tb.index[-1]:%Y-%m-%d}")
for key, (name, desc, fn) in ACCOUNTS.items():
    acct = LiveSimAccount.load_or_create(os.path.join(ROOT, f"{key}.json"), key, name, desc,
                                         start_date=etf.index[-1].strftime("%Y-%m-%d"))
    n = acct.advance(etf, cash, fn, costs)
    acct.save()
    st = acct.st
    eq = st["equity_history"][-1]["equity"] if st["equity_history"] else st["start_equity"]
    print(f"{name:55s} {st['last_date']}  +{n} days  equity {eq:>12,.2f} "
          f"({eq / st['start_equity'] - 1:+.2%})  holdings {{{', '.join(f'{k}: {v:,.1f}' for k, v in st['holdings'].items())}}}")
