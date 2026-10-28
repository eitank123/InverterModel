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
* **Real-time simulation (owner's choice: no real money, no broker for now):** every round runs
  `python scripts/live_sim.py`. It advances the play-money accounts in `loop/live_sim/` by the
  newest real trading days from FRED; report their values and trades. Never send broker orders
  unless the owner asks again.
* **Reports:**
  * a daily report (Tue–Sat, routine "Algo trading daily report");
  * a weekly report with a learning review (Saturday, routine "Algo trading weekly report").

  Both use `python scripts/report.py daily|weekly`, write to `reports/`, and appear on the
  dashboard's Reports tab. The fleet of strategies in `scripts/live_sim.py` runs side by side.
  New ideas get NEW accounts, and existing accounts are never edited.
* **Secrets and real money:** never put API keys in code or git. Real-money orders need the
  owner's explicit approval (docs/AUTOMATION.md).
* **Reporting:** report results honestly, including when an idea fails or a number is an upper bound.
