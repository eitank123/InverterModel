# Instructions for Claude (Algo_trading project)

Standing instructions from the owner. Every session and every scheduled round follows them.

* **Always keep the improvement loop running.** Test, analyze, read new articles or papers,
  implement new methods and retest, without asking permission. The loop process is in
  `loop/README.md`, and the routine "Algo trading improvement loop" runs it every 4 hours. If that
  routine is missing or disabled, recreate it.
* **Objective:** the highest *expected* long-run growth. Lower confidence and bigger swings are
  acceptable. Ruin is not.
* **No fitting to past years.** The strategy will trade unseen, real-time data:
  * use round, published parameters only;
  * choose the champion by its WORST era (1880–1929, 1930–56, 1957–90, 1991–2026);
  * prefer fixed equal-weight combinations of independent effects;
  * count every variant tried.
* **Costs:** every simulation includes all trading costs and fund costs.
* **Git:** commit and push to `main` on github.com/eitank123/InverterModel.
* **Dashboard:** refresh it after each round: https://claude.ai/artifact/X77L8R6nfhADmsfDaS7oYt
* **Paper trading:** when `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are set and *.alpaca.markets
  is reachable, every round runs `python scripts/live_trade.py --broker alpaca --execute` (PAPER
  account) and reports the account value and orders.
* **Secrets and real money:** never put API keys in code or git. Real-money orders need the
  owner's explicit approval (docs/AUTOMATION.md).
* **Reporting:** report results honestly, including when an idea fails or a number is an upper bound.
