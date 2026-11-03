"""Experiment 32: is there an edge in Polymarket prediction markets?

Prediction-market prices should equal probabilities. The best-documented deviation is the
favorite-longshot bias: long shots are overpriced and favorites underpriced (horse racing:
Griffith 1949, Thaler & Ziemba 1988; prediction markets: Snowberg & Wolfers 2010). If Polymarket
shares it, buying favorites earns more than their price implies.

Data: Polymarket's public APIs (Gamma for markets, CLOB for daily price history), resolved binary
markets that ended 2024-01-01 to 2026-10-31 with at least $250k volume (liquid enough to trade).
For each market the YES price 7 and 30 days before its last trade is compared with the outcome.
Pre-set rule: 7 days before resolution, buy whichever side costs 0.80-0.95 (the favorite), hold to
resolution; 1 cent of spread + slippage per share. Return per day held is compared with T-bills.
Caveats: markets are chosen by their final volume (known only afterwards); daily 'price' is the last
trade, not a quote you could surely fill; resolution disputes are ignored. Variants counted: 3.

    python scripts/experiments/exp32_polymarket.py      (downloads ~1-2k price histories once)
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CACHE = os.path.join(ROOT, "data", "polymarket")
H = {"User-Agent": "algotrading-research/0.1"}


def get(url, tries=4):
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=30).read())
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def markets():
    path = os.path.join(CACHE, "markets.json")
    if os.path.exists(path):
        return json.load(open(path))
    out = []
    for off in range(0, 8000, 100):
        page = get("https://gamma-api.polymarket.com/markets?closed=true&limit=100&offset=%d&order=volumeNum"
                   "&ascending=false&end_date_min=2024-01-01&end_date_max=2026-10-31&volume_num_min=250000" % off)
        if not page:
            break
        for m in page:
            try:
                prices = [float(x) for x in json.loads(m["outcomePrices"])]
                outs = json.loads(m["outcomes"])
                toks = json.loads(m["clobTokenIds"])
            except Exception:
                continue
            if len(outs) != 2 or sorted(prices) != [0.0, 1.0]:
                continue                                         # only cleanly resolved binaries
            out.append({"id": m["id"], "q": m["question"], "end": m.get("endDate"), "vol": m.get("volumeNum"),
                        "yes_token": toks[0], "yes_won": prices[0] == 1.0, "outcomes": outs})
        if len(page) < 100:
            break
    json.dump(out, open(path, "w"))
    return out


def history(tok):
    path = os.path.join(CACHE, f"h_{tok[:24]}.json")
    if os.path.exists(path):
        return json.load(open(path))
    h = get(f"https://clob.polymarket.com/prices-history?market={tok}&interval=max&fidelity=1440")
    h = h.get("history", []) if h else []
    json.dump(h, open(path, "w"))
    return h


if __name__ == "__main__":
    ms = markets()
    print(f"{len(ms)} resolved binary markets with >= $250k volume, ended 2024-2026")
    rows = []
    for i, m in enumerate(ms):
        h = history(m["yes_token"])
        if len(h) < 35:
            continue
        p = pd.Series([x["p"] for x in h], index=pd.to_datetime([x["t"] for x in h], unit="s"))
        last = p.index[-1]
        for lag in (7, 30):
            before = p[p.index <= last - pd.Timedelta(days=lag)]
            if len(before):
                rows.append({"id": m["id"], "lag": lag, "p_yes": float(before.iloc[-1]), "yes_won": m["yes_won"],
                             "end": last, "q": m["q"]})
        if i % 200 == 0:
            print(f"  {i} markets read", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CACHE, "snapshots.csv"), index=False)

    bins = [0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
    for lag in (7, 30):
        d = df[df.lag == lag].copy()
        d["bucket"] = pd.cut(d["p_yes"], bins, include_lowest=True)
        t = d.groupby("bucket", observed=True).agg(n=("yes_won", "size"), price=("p_yes", "mean"), won=("yes_won", "mean"))
        t["edge (won - price)"] = t["won"] - t["price"]
        t["se"] = np.sqrt(t["won"] * (1 - t["won"]) / t["n"])
        print(f"\nCalibration {lag} days before resolution ({len(d)} markets): YES price vs how often YES won")
        print(t.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nRule: 7 days out, buy the side priced 0.80-0.95, hold to resolution, 1 cent cost per share")
    d = df[df.lag == 7].copy()
    fav_yes = d["p_yes"].between(0.80, 0.95)
    fav_no = (1 - d["p_yes"]).between(0.80, 0.95)
    price = np.where(fav_yes, d["p_yes"], 1 - d["p_yes"]) + 0.01
    won = np.where(fav_yes, d["yes_won"], ~d["yes_won"])
    sel = fav_yes | fav_no
    r = (won[sel] / price[sel] - 1)
    print(f"  {sel.sum()} bets; won {won[sel].mean():.1%} at an average cost of {price[sel].mean():.3f}; "
          f"average return per bet {r.mean():+.2%} (se {r.std() / np.sqrt(sel.sum()):.2%}) over ~7 days")
    print(f"  T-bills over 7 days: about +0.08%. Worst case: a lost bet loses ~100% of that stake.")
    for lab, mask in [("long shots (YES priced 0.02-0.15), buying YES", d["p_yes"].between(0.02, 0.15))]:
        rr = d.loc[mask, "yes_won"] / (d.loc[mask, "p_yes"] + 0.01) - 1
        print(f"  {lab}: {mask.sum()} bets, average return {rr.mean():+.1%} (se {rr.std() / np.sqrt(mask.sum()):.1%})")
