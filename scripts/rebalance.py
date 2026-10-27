"""Monthly rebalance of the frozen rule through a broker.

    python scripts/rebalance.py                         # offline demo, local paper broker
    python scripts/rebalance.py --broker alpaca         # Alpaca PAPER account, dry run
    python scripts/rebalance.py --broker alpaca --execute          # paper orders sent
    python scripts/rebalance.py --broker alpaca --live --execute   # REAL money

Alpaca needs ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment, and network access
to *.alpaca.markets. Run it after the close on the last trading day of the month (the
signal uses completed months only, so running a few days later is fine too).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402
from algotrading.broker import Limits  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--broker", choices=["paper", "alpaca"], default="paper")
p.add_argument("--stock", default="SPY", help="stock ETF (S&P 500)")
p.add_argument("--bond", default="IEF", help="bond ETF (7-10y Treasuries)")
p.add_argument("--leverage", type=float, default=1.0)
p.add_argument("--execute", action="store_true", help="actually send orders")
p.add_argument("--live", action="store_true", help="use the LIVE (real money) account")
a = p.parse_args()

limits = Limits(max_gross_leverage=max(1.0, a.leverage) + 1e-9)
if a.broker == "alpaca":
    if a.live and a.execute:
        if input("LIVE account, real money. Type 'yes' to continue: ") != "yes":
            sys.exit("aborted")
    broker = at.broker.AlpacaBroker(paper=not a.live)
    closes = broker.daily_closes([a.stock, a.bond], start=str(pd.Timestamp.today() - pd.DateOffset(months=16))[:10])
else:
    # Offline demo: Shiller-based stand-ins for SPY/IEF (monthly), local paper broker.
    df = at.data.load_shiller_sp500()
    closes = pd.DataFrame({a.stock: df["tr"] / df["tr"].iloc[-1] * 650,
                           a.bond: at.data.bond_total_return(df["long_rate"]).pipe(lambda s: s / s.iloc[-1] * 95)})
    broker = at.broker.PaperBroker(cash=100_000, prices=closes.iloc[-1].to_dict())
    a.execute = True                     # nothing real can happen here

weights = at.broker.frozen_rule_weights(closes, a.stock, a.bond, a.leverage,
                                        asof=closes.index[-1] + pd.offsets.MonthBegin())
print(f"Signal from month-end closes through {closes.index[-1].date()}: target weights {weights}")
print(f"Equity before: {broker.equity():,.2f}  positions: {broker.positions()}")
for order, oid in at.broker.rebalance(broker, weights, limits, execute=a.execute):
    print(f"  {order.side.upper():4s} {order.qty:,.4f} {order.symbol}  -> {oid}")
if a.broker == "paper":
    px = broker.last_prices(list(weights))
    held = {s: f"{q * px[s] / broker.equity():.1%}" for s, q in broker.positions().items()}
    print(f"Equity after: {broker.equity():,.2f}  weights now: {held}")
