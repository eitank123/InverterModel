"""Trade the current champion on the real market (Alpaca paper account by default).

    python scripts/live_trade.py                       # offline: history only, local paper broker
    python scripts/live_trade.py --broker alpaca       # live quotes, PAPER account, dry run
    python scripts/live_trade.py --broker alpaca --execute          # paper orders sent
    python scripts/live_trade.py --broker alpaca --live --execute   # REAL money (asks first)

Needs ALPACA_API_KEY / ALPACA_SECRET_KEY in the environment and network access to
*.alpaca.markets. Safe to run often: the signal only changes when a month completes, and
drifts under 2% of equity are not traded. Every run is appended to loop/live/runs.jsonl.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import algotrading as at  # noqa: E402
from algotrading.broker import Limits  # noqa: E402
from algotrading.live import completed_month_ends, log_run, shiller_history, splice_monthly, \
    target_etf_weights  # noqa: E402
from loop import CHAMPION, ensemble  # noqa: E402

LOG = os.path.join(os.path.dirname(__file__), "..", "loop", "live", "runs.jsonl")
# Index proxies used for the SIGNAL (total-return adjusted): S&P 500 and 7-10y Treasuries.
SIGNAL = {"stocks": "SPY", "bonds": "IEF"}
TRADED = ["SPY", "UPRO", "IEF", "UST"]

p = argparse.ArgumentParser()
p.add_argument("--broker", choices=["paper", "alpaca"], default="paper")
p.add_argument("--execute", action="store_true", help="actually send orders")
p.add_argument("--live", action="store_true", help="use the LIVE (real money) account")
a = p.parse_args()

strategy = ensemble(3.0)
history, _ = shiller_history()
if a.broker == "alpaca":
    if a.live and a.execute and input("LIVE account, real money. Type 'yes' to continue: ") != "yes":
        sys.exit("aborted")
    broker = at.broker.AlpacaBroker(paper=not a.live)
    start = str(history.index[-1] - pd.DateOffset(months=60))[:10]
    closes = broker.daily_closes(list(SIGNAL.values()), start=start).rename(
        columns={v: k for k, v in SIGNAL.items()})
    month_ends = completed_month_ends(closes)
    monthly = splice_monthly(history, month_ends)
    source = f"history to {month_ends.index[0]:%Y-%m}, live {SIGNAL} month-ends to {month_ends.index[-1]:%Y-%m-%d}"
else:
    monthly = history
    prices = {"SPY": 650.0, "UPRO": 105.0, "IEF": 95.0, "UST": 45.0}
    broker = at.broker.PaperBroker(cash=100_000, prices=prices)
    a.execute = True                                   # nothing real can happen here
    source = "offline history only (no broker connection)"

targets, exposure = target_etf_weights(strategy, monthly)
print(f"Champion: {CHAMPION['name']} {CHAMPION['params']}")
print(f"Signal data: {source}")
print(f"Exposure: stocks {exposure['stocks']:.2f}x, bonds {exposure['bonds']:.2f}x  ->  ETF weights {targets}")
equity_before = broker.equity()
print(f"Equity before: {equity_before:,.2f}  positions: {broker.positions()}")
orders = []
try:
    for order, oid in at.broker.rebalance(broker, targets, Limits(max_gross_leverage=1.0 + 1e-6,
                                                                   max_order_frac=1.01), execute=a.execute):
        print(f"  {order.side.upper():4s} {order.qty:,.4f} {order.symbol}  -> {oid}")
        orders.append({"symbol": order.symbol, "side": order.side, "qty": order.qty, "id": oid})
    status = "ok"
except RuntimeError as e:                              # market closed, account blocked
    print(f"Not trading: {e}")
    status = str(e)
if a.broker != "paper":                               # only real-market runs go in the record
    log_run(LOG, {"time": pd.Timestamp.now(tz="UTC").isoformat(), "broker": a.broker,
                  "account": "live" if a.live else "paper", "executed": a.execute, "status": status,
                "signal_source": source, "exposure": exposure, "targets": targets,
                "equity": equity_before, "positions": broker.positions(), "orders": orders})
