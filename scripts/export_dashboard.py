"""Write the dashboard's data files (JSON row lists) into loop/dashboard/.

    python scripts/export_dashboard.py

Inputs: loop/leaderboard.csv, loop/dashboard/{equity,yearly,forward,frontier}.csv (from
scripts/loop.py --report --frontier), loop/checkpoints.json, loop/paper/*.json.
"""
import glob
import json
import os
from datetime import datetime, timezone

import pandas as pd

sys_path = os.path.join(os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), "..", "loop")
OUT = os.path.join(ROOT, "dashboard")


def dump(name, rows):
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(rows, f, default=str)
    print(f"{name}.json: {len(rows)} rows")


def records(df):
    return json.loads(df.to_json(orient="records", double_precision=6))


def main():
    lb = pd.read_csv(os.path.join(ROOT, "leaderboard.csv"))
    lb = lb.sort_values("test_cagr", ascending=False).reset_index(drop=True)
    lb["id"] = lb["name"] + " | " + lb["params"] + " | +" + (lb["financing"] * 100).round(1).astype(str) + "%"
    import importlib.util
    spec = importlib.util.spec_from_file_location("loop", os.path.join(sys_path, "loop.py"))
    loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loop)
    lb["is_champion"] = (lb["name"] == loop.CHAMPION["name"]) & (lb["params"] == loop.CHAMPION["params"]) \
        & (lb["financing"] == 0.005)
    lb["deployable_etf"] = lb.get("deployable_etf", lb["max_gross"] <= 3.0 + 1e-9)
    cols = ["id", "name", "family", "params", "financing", "iteration", "is_champion", "deployable_etf",
            "train_cagr", "test_cagr",
            "full_cagr", "full_max_dd", "full_worst_year", "crash_day_loss", "avg_gross", "max_gross",
            "deployable", "ruin", "test_excess_sharpe", "note"]
    dump("leaderboard", records(lb[cols]))

    eq = pd.read_csv(os.path.join(OUT, "equity.csv"))
    eq = eq[pd.to_datetime(eq["date"]).dt.month.isin([1, 4, 7, 10])]        # quarterly points
    dump("equity", records(eq))
    yr = pd.read_csv(os.path.join(OUT, "yearly.csv"))
    dump("yearly", records(yr[yr["year"] >= 2000]))
    dump("forward", records(pd.read_csv(os.path.join(OUT, "forward.csv"))))
    dump("frontier", records(pd.read_csv(os.path.join(OUT, "frontier.csv"))))
    with open(os.path.join(ROOT, "checkpoints.json")) as f:
        dump("checkpoints", json.load(f))

    eq_rows, pos_rows, trade_rows, acct_rows = [], [], [], []
    # The trading desk shows the real-time simulation (loop/live_sim) once it exists; the older
    # monthly paper ledger (loop/paper) is the fallback.
    live = sorted(glob.glob(os.path.join(ROOT, "live_sim", "*.json")))
    for path in live or sorted(glob.glob(os.path.join(ROOT, "paper", "*.json"))):
        with open(path) as f:
            st = json.load(f)
        key = os.path.basename(path)[:-5]
        last = st["equity_history"][-1]
        prices = st.get("prices") or {t["symbol"]: t["price"] for t in st["trades"]}
        for h in st["equity_history"]:
            eq_rows.append({"account": st["name"], "date": h["date"], "equity": h["equity"]})
        invested = 0.0
        for sym in sorted(set(st["holdings"]) | set(st["targets"])):
            q = st["holdings"].get(sym, 0.0)
            px = prices.get(sym)
            value = q * px if px else 0.0
            invested += value
            pos_rows.append({"account": st["name"], "symbol": sym, "qty": round(q, 4), "price": px,
                             "value": round(value, 2), "weight": value / last["equity"] if last["equity"] else 0,
                             "target_weight": st["targets"].get(sym, 0.0)})
        for t in st["trades"]:
            trade_rows.append({"account": st["name"], **{k: v for k, v in t.items() if k != "shares"},
                               "notional": round(t["qty"] * t["price"], 2)})
        acct_rows.append({"key": key, "account": st["name"], "description": st["description"],
                          "start": st["start"], "as_of": last["date"], "equity": last["equity"],
                          "start_equity": 100000.0, "return_since_start": last["equity"] / 100000.0 - 1,
                          "cash": round(st["cash"], 2), "gross_exposure": invested / last["equity"],
                          "trading_costs": round(sum(st.get("fees", {}).values()), 2),
                          "fund_costs": round(st.get("fund_costs", 0.0), 2)})
    dump("paper_equity", eq_rows)
    dump("paper_positions", pos_rows)
    dump("paper_trades", trade_rows)
    dump("paper_accounts", acct_rows)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    last_day = max((r["as_of"] for r in acct_rows), default="")
    dump("trading_status", [
        {"item": "Mode", "value": "Real-time simulation (play money)", "state": "ok",
         "detail": "Accounts started 2026-10-23 with $100,000 each and advance one real trading day at a time; "
                   "nothing is backfilled. No broker and no real money."},
        {"item": "Price source", "value": f"FRED daily closes to {last_day}", "state": "ok",
         "detail": "S&P 500, 10-year Treasury yield and 3-month T-bill from the St. Louis Fed. ETF prices "
                   "(SPY, UPRO, IEF, UST) are built from the real index moves net of fund costs."},
        {"item": "Costs", "value": "All charged", "state": "ok",
         "detail": "Commission, spread + slippage, SEC/FINRA fees, expense ratios, leveraged-ETF financing, "
                   "cash sweep fee."},
        {"item": "Broker connection", "value": "Not used", "state": "ok",
         "detail": "Owner's choice: simulation only. Alpaca paper keys would add real quotes and fills (docs/AUTOMATION.md)."},
        {"item": "Last export", "value": now, "state": "ok", "detail": "scripts/export_dashboard.py"},
    ])


if __name__ == "__main__":
    main()
