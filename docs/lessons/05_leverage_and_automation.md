# Lesson 5: Leverage in practice, and connecting to a real market

## 1. Why "Kelly on safe assets" is subtle
Multi-asset Kelly sizes positions as w = λ·Σ⁻¹·μ. Σ⁻¹ scales with 1/variance, so Kelly piles
leverage onto whatever *looks* calmest. With a 1.5% expected term premium and 6% volatility,
full Kelly wants about 4× in bonds.

Exp 15 showed what that does: from 1957 to 1981 yields rose from 3% to 15%, and leveraged bonds
were the worst place to be. **Low volatility is not low risk.** The risk of bonds is inflation regimes,
which trailing volatility does not see. Fixes that helped: a trend gate (no bet below the 10-month average)
and a volatility *floor* (never assume bonds are calmer than 7%/yr). Neither beat simple constant leverage
on the trend-filtered 60/40.

## 2. Ways to get leverage, and what they really cost
Exp 15 G: the trend-filtered 60/40 at 2× earns ~11.4%/yr if borrowing is free, and 8.3% if it costs
cash + 5%. **The financing cost decides whether leverage is worth anything.**

| Instrument | How leverage works | Typical cost over cash | Catches |
|---|---|---|---|
| **Futures** (Micro E-mini S&P 500, 10-Year T-Note) | Post ~5–10% margin, control the full notional | ≈ 0.2–0.5% (implied in the futures price) | Contract sizes are large (one 10Y T-Note ≈ $100k face), quarterly rolls, a futures-enabled account |
| **Broker margin loan** | Borrow cash against your ETFs | from ~1–1.5% (low-cost brokers, larger balances) to 5–10%+ (many retail brokers) | Margin calls in crashes, rates vary by broker and balance |
| **Leveraged ETFs** (2× S&P, 2× long Treasuries) | The fund borrows inside, resets daily | expense ratio ~0.9% plus the financing inside the fund | Daily reset causes *volatility decay* in choppy markets. Fine for days, risky for years |
| **Box spreads** (options) | Synthetic loan at ≈ T-bill rates | ≈ 0.3–0.6% | Advanced; needs options approval |

Check what is available and allowed where you live. Some countries restrict retail access to
US-domiciled ETFs or to derivatives.

## 3. The recommended configuration (from Exp 15)
* **Unlevered (1.0×):** 60/40 + SMA-10 on SPY/IEF. 8.0%/yr since 1957, max DD −15%.
* **Moderate (1.5×):** 9.5%/yr since 1957 (10.9% in 1991–2026), max DD −23%. Needs cheap financing.
* **Aggressive (≈2×, or a 10% vol target):** ~10.7%/yr, max DD −24% to −30%. Only with futures-level financing.

At today's valuations (CAPE ≈ 41) expect less than history: perhaps 4–5%/yr unlevered, so
roughly 5–7% at 1.5×, before any surprise. Leverage multiplies the bad years too.

## 4. Connecting to the market
Yes, it can be fully automatic. `algotrading/broker.py` already contains the pieces:

```
prices -> frozen rule -> target weights -> plan_rebalance -> safety checks -> broker orders
```

* `AlpacaBroker`: US stocks/ETFs, commission-free, with a **paper account** (fake money, real prices) and an API.
  Interactive Brokers is the other common choice, especially outside the US (API via TWS / IB Gateway).
* Safety built in: dry-run by default, leverage cap, fat-finger guard (no order larger than 75% of equity),
  small drifts ignored, no orders while the market is closed, sells before buys,
  typed confirmation for the live account.
* The rule trades **once a month**, so it doesn't need a server running 24/7. A scheduled job
  (cron, a cloud function, or a scheduled Claude Code session) on the first trading day of each month is enough.

The step-by-step guide is in [`docs/AUTOMATION.md`](../AUTOMATION.md).
