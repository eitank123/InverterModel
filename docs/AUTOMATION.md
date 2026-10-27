# Going from research to automatic trading

The code path exists (`algotrading/broker.py`, `scripts/rebalance.py`) and is tested with a
local paper broker and a fake Alpaca client. What remains needs *your* accounts and settings.

## Stage 1: Offline (works today)
```bash
python scripts/rebalance.py          # local paper broker, Shiller stand-in prices
python scripts/today.py              # what the rule holds right now
```

## Stage 2: Paper trading (fake money, real market). Do this for 3–6 months.
1. Open an Alpaca account and generate **paper** API keys (or use Interactive Brokers' paper account,
   which needs a different adapter: ask me to write it). Check that the broker accepts residents of your country.
2. Give the code the keys as environment variables, never in files or git:
   `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`. In a Claude Code cloud environment, add them as environment secrets.
3. Allow network access to `paper-api.alpaca.markets` and `data.alpaca.markets`
   (and `api.alpaca.markets` for live). In this cloud environment: open the environment menu in the
   session's title bar → Edit → Network access, and add these hosts.
4. Dry run, then send paper orders for the **current champion** (the ensemble). The signal joins the
   long history to live SPY/IEF month-end closes; trades go through SPY, UPRO, IEF and UST:
   ```bash
   python scripts/live_trade.py --broker alpaca             # shows orders, sends nothing
   python scripts/live_trade.py --broker alpaca --execute   # sends PAPER orders
   ```
   Every real-market run is appended to `loop/live/runs.jsonl`. (`scripts/rebalance.py` still runs
   the original frozen 60/40 rule.)
5. Scheduling is already done: the improvement-loop routine runs every 4 hours, and once the keys
   and network access are in place it runs the paper command above each time. Rerunning is safe:
   the signal only changes when a month completes, and drifts under 2% are not traded.
6. Each month, compare paper fills with what the backtest assumed (prices, costs, timing).
   **The gap is your real cost model.**

## Stage 3: Live (real money)
Only after paper trading behaves as expected:
* Start small. The rule's worst historical year at 1.0× was about −10%, and live is always worse than backtests.
* `--live --execute` asks you to type `yes`. Keep a human confirmation step for at least the first year.
* Leverage: do **not** use expensive margin. Above ~3% over cash it destroys the benefit (Exp 15 G).
  Use 1.0× until you have futures or low-cost financing.
* Add monitoring: an alert if an order fails, if positions differ from targets, or if the account drops more than X%.

## Can Claude place the trades?
Technically, yes. A scheduled session can run `scripts/rebalance.py --execute` with your keys.
The safer setup is: Claude runs the analysis and a **dry run** monthly and sends you the proposed orders,
and you approve the actual execution. That keeps a human in the loop for real money.

*This project is research and education, not investment advice. Taxes, account types and regulations
depend on where you live.*
