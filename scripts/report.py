"""Daily / weekly report on the real-time simulation and the research loop.

    python scripts/report.py daily      # reports/daily/YYYY-MM-DD.md
    python scripts/report.py weekly     # reports/weekly/YYYY-Www.md

Reads loop/live_sim/*.json (run scripts/live_sim.py first), FRED market data and the loop's
checkpoints. Also writes loop/dashboard/reports.json (every report, newest first) for the
dashboard's Reports tab.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
LIVE = os.path.join(ROOT, "loop", "live_sim")
REPORTS = os.path.join(ROOT, "reports")


def pct(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{d}%}"


def load_accounts():
    accts = []
    for path in sorted(glob.glob(os.path.join(LIVE, "*.json"))):
        with open(path) as f:
            st = json.load(f)
        eq = pd.Series({pd.Timestamp(h["date"]): h["equity"] for h in st["equity_history"]})
        accts.append((st, eq))
    return accts


def market_section(asof):
    sp = at.data.load_fred("SP500")[:asof]
    y10 = at.data.load_fred("DGS10")[:asof]
    tb = at.data.load_fred("DTB3")[:asof]
    me = sp.groupby(sp.index.to_period("M")).tail(1)
    done = me if sp.index[-1] == sp.index[-1] + pd.offsets.BMonthEnd(0) else me.iloc[:-1]
    sma10 = done.iloc[-10:].mean()
    mom12 = done.iloc[-1] / done.iloc[-13] - 1
    high = sp.iloc[-756:].max()
    lines = [
        "## Market",
        f"* S&P 500: {sp.iloc[-1]:,.2f} on {sp.index[-1]:%a %d %b %Y} "
        f"(day {pct(sp.iloc[-1] / sp.iloc[-2] - 1)}, week {pct(sp.iloc[-1] / sp.asof(sp.index[-1] - pd.Timedelta(days=7)) - 1)}, "
        f"month to date {pct(sp.iloc[-1] / done.iloc[-1] - 1) if done.index[-1].month != sp.index[-1].month else 'n/a'})",
        f"* 10-year Treasury yield {y10.iloc[-1]:.2f}% · 3-month T-bill {tb.iloc[-1]:.2f}%",
        f"* Trend signals (last completed month, {done.index[-1]:%b %Y}): S&P {pct(done.iloc[-1] / sma10 - 1, 1)} vs its "
        f"10-month average, {pct(mom12, 1)} over 12 months; today {pct(sp.iloc[-1] / sma10 - 1, 1)} vs that average "
        f"and {pct(sp.iloc[-1] / high - 1, 1)} from its 3-year high.",
    ]
    gap_today = sp.iloc[-1] / sma10 - 1
    if abs(gap_today) < 0.03:
        lines.append(f"* **Watch:** the S&P is within 3% of its 10-month average; the trend signal could flip at "
                     "the next month-end.")
    nxt = (sp.index[-1] + pd.offsets.BMonthBegin(1)).strftime("%a %d %b")
    lines.append(f"* Next scheduled rebalance: first trading day of next month ({nxt}).")
    return lines, sp


def accounts_table(accts, since):
    rows = []
    for st, eq in accts:
        if eq.empty:
            continue
        before = eq[eq.index < since]
        base = before.iloc[-1] if len(before) else st["start_equity"]
        dd = float((eq / eq.cummax() - 1).min())
        rows.append({"account": st["name"], "equity": eq.iloc[-1], "period": eq.iloc[-1] / base - 1,
                     "since_start": eq.iloc[-1] / st["start_equity"] - 1, "max_dd": dd,
                     "stocks": st.get("exposure", {}).get("stocks", 0.0),
                     "bonds": st.get("exposure", {}).get("bonds", 0.0),
                     "costs": sum(st.get("fees", {}).values()) + st.get("fund_costs", 0.0),
                     "trades": [t for t in st["trades"] if pd.Timestamp(t["date"]) >= since]})
    return sorted(rows, key=lambda r: r["since_start"], reverse=True)


def render_table(rows, period_label):
    out = [f"| # | Account | Equity | {period_label} | Since start | Max DD | Stocks | Bonds | Costs paid |",
           "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        out.append(f"| {i} | {r['account']} | ${r['equity']:,.0f} | {pct(r['period'])} | {pct(r['since_start'])} | "
                   f"{r['max_dd']:.1%} | {r['stocks']:.2f}x | {r['bonds']:.2f}x | ${r['costs']:,.0f} |")
    return out


def trades_lines(rows):
    out = []
    for r in rows:
        for t in r["trades"]:
            out.append(f"* {t['date']} · {r['account']}: {t['side'].upper()} {t['qty']:,.2f} {t['symbol']} @ "
                       f"{t['price']:,.2f} (${t['amount']:,.0f}; fees ${t['commission'] + t['spread_slippage'] + t['regulatory_fees']:,.2f}"
                       f"{'; realized ' + f'${t['realized_pl']:,.0f}' if t['side'] == 'sell' else ''}) — {t['reason']}")
    return out or ["* No trades in this period."]


def learning_lines(accts, start, end):
    """Weekly: is each account behaving as its model says it should? Its week's move relative
    to the S&P's should match its stock exposure (e.g. a 3x account moves about 3x)."""
    out = []
    for st, eq in accts:
        if len(eq) < 2:
            continue
        w = eq[(eq.index >= start) & (eq.index <= end)]
        prev = eq[eq.index < start]
        if w.empty or prev.empty:
            continue
        r_week = w.iloc[-1] / prev.iloc[-1] - 1
        out.append((st["name"], r_week))
    if not out:
        return ["* Not enough real-time history yet for a behaviour check (needs one full week)."]
    sp_week = dict(out).get("Benchmark: S&P 500 buy & hold")
    lines = []
    for name, r in sorted(out, key=lambda x: -x[1]):
        beta = r / sp_week if sp_week else float("nan")
        lines.append(f"* {name}: week {pct(r)}" + (f", {beta:.1f}x the S&P's move" if sp_week and abs(sp_week) > 0.002 else ""))
    return lines


def write(kind, title, body, key, date):
    os.makedirs(os.path.join(REPORTS, kind), exist_ok=True)
    path = os.path.join(REPORTS, kind, f"{key}.md")
    text = "\n".join([f"# {title}", ""] + body) + "\n"
    with open(path, "w") as f:
        f.write(text)
    index = []
    for p in sorted(glob.glob(os.path.join(REPORTS, "*", "*.md")), reverse=True):
        with open(p) as f:
            t = f.read()
        k = os.path.basename(os.path.dirname(p))
        index.append({"id": f"{k}/{os.path.basename(p)[:-3]}", "period": k,
                      "title": t.splitlines()[0].lstrip("# "), "text": t})
    index.sort(key=lambda r: r["title"].split("·")[-1].strip(), reverse=True)
    with open(os.path.join(ROOT, "loop", "dashboard", "reports.json"), "w") as f:
        json.dump(index, f)
    print(text)
    print(f"-> {os.path.relpath(path, ROOT)}")


def main(kind):
    accts = load_accounts()
    asof = max(eq.index[-1] for _, eq in accts if not eq.empty)
    mkt, sp = market_section(asof)
    with open(os.path.join(ROOT, "loop", "checkpoints.json")) as f:
        cps = json.load(f)
    sys.path.insert(0, os.path.dirname(__file__))
    from loop import CHAMPION  # noqa: E402
    if kind == "daily":
        since = asof
        rows = accounts_table(accts, since)
        body = [f"Data through the {asof:%a %d %b %Y} close (FRED). Play money, every cost charged. "
                f"{len(rows)} accounts running since {min(st['start'] for st, _ in accts)}.", ""] + mkt + \
               ["", "## Accounts (ranked by return since start)"] + render_table(rows, "Day") + \
               ["", "## Trades today"] + trades_lines(rows)
        best, worst = rows[0], rows[-1]
        body += ["", "## Notes",
                 f"* Leader since start: {best['account']} ({pct(best['since_start'])}); last: "
                 f"{worst['account']} ({pct(worst['since_start'])}).",
                 f"* Champion: {CHAMPION['name']} {CHAMPION['params']}. Research loop: {len(cps)} checkpoints, "
                 f"latest #{cps[-1]['checkpoint']} ({cps[-1]['date']})."]
        write("daily", f"Daily report · {asof:%Y-%m-%d}", body, f"{asof:%Y-%m-%d}", asof)
    else:
        start = asof - pd.Timedelta(days=asof.weekday())          # Monday of the data week
        rows = accounts_table(accts, start)
        week_cps = [c for c in cps if pd.Timestamp(c["date"]) >= start - pd.Timedelta(days=7)]
        body = [f"Week of {start:%d %b} to {asof:%d %b %Y} (FRED closes). Play money, every cost charged.", ""] + mkt + \
               ["", "## Accounts (ranked by return since start)"] + render_table(rows, "Week") + \
               ["", "## Trades this week"] + trades_lines(rows) + \
               ["", "## Behaviour check (real time vs what the backtest implies)"] + learning_lines(accts, start, asof) + \
               ["", "## Research loop this week"] + \
               ([f"* Checkpoint {c['checkpoint']} ({c['date']}): {c['decision']}" for c in week_cps] or
                ["* No new checkpoints."]) + \
               ["", f"Champion: {CHAMPION['name']} {CHAMPION['params']}. Real-time results this short are mostly "
                "noise: a strategy's long-run edge shows only over many months; the weekly check looks for "
                "behaviour that contradicts the model (e.g. a 3x account not moving about 3x the S&P)."]
        iso = asof.isocalendar()
        write("weekly", f"Weekly report · {iso.year}-W{iso.week:02d} · {asof:%Y-%m-%d}", body,
              f"{iso.year}-W{iso.week:02d}", asof)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "daily")
