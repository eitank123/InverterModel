"""Forward paper test of the one Polymarket pattern Exp 32 found (play money, no real bets).

Exp 32: on 1,252 resolved liquid markets (2024-2026), YES contracts priced 15-30 cents about a week
before the end won 39% of the time. One of 16 buckets standing out is what chance produces, so
the honest test is forward. The rule was fixed BEFORE any of these bets:
  every run, for each OPEN binary market ending in 5-9 days with >= $100k volume SO FAR (Exp 32 used
  final volume >= $250k, which is only known afterwards) and a YES ask of
  0.15-0.30, paper-buy $100 of YES at the current ask + 0.5 cent (slippage); one bet per market;
  hold to resolution. Resolved bets are settled at 1 or 0.
Ledger: loop/polymarket_paper.json. Nothing is ever sent to Polymarket.

    python scripts/polymarket_paper.py
"""
import json
import os
import urllib.request

import pandas as pd

PATH = os.path.join(os.path.dirname(__file__), "..", "loop", "polymarket_paper.json")
H = {"User-Agent": "algotrading-research/0.1"}
STAKE, SLIP = 100.0, 0.005


def get(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=30).read())


def main():
    led = json.load(open(PATH)) if os.path.exists(PATH) else {"rule": __doc__.split("\n\n")[1], "bets": []}
    now = pd.Timestamp.now(tz="UTC")
    lo, hi = (now + pd.Timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"), (now + pd.Timedelta(days=9)).strftime("%Y-%m-%dT%H:%M:%SZ")
    have = {b["id"] for b in led["bets"]}
    placed = 0
    for off in range(0, 2000, 100):
        page = get(f"https://gamma-api.polymarket.com/markets?closed=false&active=true&limit=100&offset={off}"
                   f"&end_date_min={lo}&end_date_max={hi}&volume_num_min=100000")
        for m in page:
            try:
                outs = json.loads(m["outcomes"])
                ask = float(m.get("bestAsk") or "nan")
            except Exception:
                continue
            if len(outs) != 2 or m["id"] in have or not (0.15 <= ask <= 0.30) or not m.get("acceptingOrders", True):
                continue
            px = ask + SLIP
            led["bets"].append({"id": m["id"], "question": m["question"], "end": m["endDate"], "placed": now.isoformat(),
                                "price": round(px, 4), "shares": round(STAKE / px, 4), "stake": STAKE, "status": "open"})
            have.add(m["id"])
            placed += 1
        if len(page) < 100:
            break
    settled = 0
    for b in led["bets"]:
        if b["status"] != "open":
            continue
        try:
            m = get(f"https://gamma-api.polymarket.com/markets/{b['id']}")
        except Exception:
            continue
        if m.get("closed"):
            prices = [float(x) for x in json.loads(m["outcomePrices"])]
            if sorted(prices) == [0.0, 1.0]:
                b["status"] = "won" if prices[0] == 1.0 else "lost"
                b["payout"] = round(b["shares"] * prices[0], 2)
                b["settled"] = now.isoformat()
                settled += 1
    done = [b for b in led["bets"] if b["status"] != "open"]
    staked = sum(b["stake"] for b in done)
    pnl = sum(b["payout"] - b["stake"] for b in done)
    led["summary"] = {"bets": len(led["bets"]), "open": len(led["bets"]) - len(done), "settled": len(done),
                      "won": sum(b["status"] == "won" for b in done), "staked_settled": staked, "pnl": round(pnl, 2),
                      "return_per_bet": round(pnl / staked, 4) if staked else None,
                      "avg_price_settled": round(sum(b["price"] for b in done) / len(done), 4) if done else None,
                      "updated": now.isoformat()}
    json.dump(led, open(PATH, "w"), indent=1)
    print(f"Polymarket paper test: {placed} new bets, {settled} settled; {led['summary']}")


if __name__ == "__main__":
    main()
