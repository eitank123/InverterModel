# Lesson 1: How markets work (the minimum you need before trading anything)

## 1. What a price actually is
A stock exchange runs an **order book**. For each stock it keeps two queues:

```
        ASKS (sellers)            price    size
                                  100.03    800
                                  100.02    500   <- best ask
        ---------------- spread = 0.02 ----------------
                                  100.00    300   <- best bid
                                   99.99   1200
        BIDS (buyers)
```

* A **limit order** says "buy 100 shares at ≤ 100.00". It waits in the book and *provides liquidity*.
* A **market order** says "buy 100 shares now at any price". It *takes liquidity* and fills against the best ask (100.02).
* The "price" you see on a chart is usually the **last trade** or the **mid** ((bid+ask)/2 = 100.01).
  You can never trade at the mid with a market order. Buying and immediately selling costs the
  whole spread. That is why every backtest needs `cost_bps`.

**Transaction costs = commission + half-spread + market impact (+ slippage).**
* Liquid US ETF (SPY): spread ≈ 0.01/500 ≈ 0.2 bp, so total cost is about 1 bp.
* Small-cap stock: 10 to 50 bp.
* Retail crypto: 10 to 100 bp.
* Market impact grows with your size, roughly ∝ σ·√(your volume / market volume).

## 2. Who is on the other side of your trade?
Every trade has a buyer and a seller, so trading is zero-sum before costs and negative-sum
after them. Your profit is someone else's loss. The participants:

| Who | Why they trade | Are they "beatable"? |
|---|---|---|
| Market makers / HFT | Earn the spread, manage inventory | No. They are faster than you, by microseconds. |
| Pension funds, insurers | Rebalance huge portfolios, often on fixed dates | They don't care about small price differences, which creates patterns. |
| Central banks, corporate hedgers | Policy goals, hedging FX/commodity exposure | They are not profit-seeking. This is one explanation for trends. |
| Quant funds / CTAs | Systematic strategies (trend, carry, value…) | They compete with you for the same signals. |
| Retail investors | Many reasons, often emotional | Behavioural biases are the other source of patterns. |

**A strategy only makes money if you can say who pays you and why they will keep doing it.**
This is the "economic rationale" test. A backtest without a story is probably data mining.

## 3. Returns: the details that matter
* **Simple return** r = P₁/P₀ − 1 compounds as ∏(1+r). **Log return** ℓ = ln(P₁/P₀) adds over time.
  For EE intuition, log returns turn a multiplicative system into an additive one.
* **Volatility drag:** growth rate ≈ μ − σ²/2. Two strategies with the same average return
  but different volatility compound to different wealth. Lower vol is worth real money.
* **Total return vs. price return:** stocks pay **dividends**. In our 155-year S&P 500
  data the *price* grew 4.9%/yr while the *total return* (dividends reinvested) was 9.4%/yr.
  **Half of all stock returns came from dividends.** A backtest on price-only data
  understates what buy & hold earned and flatters strategies that sit in cash.
* **Cash is not zero:** money not invested earns the interest rate (T-bills). Strategies
  that are often out of the market look much better or worse depending on that rate.
  In Experiment 1 this assumption alone changed the result.
* **Nominal vs. real:** 9.4% nominal minus ~2.1% inflation (CPI 1871–2026) leaves ~7.2% real.

## 4. Risk-adjusted performance
* **Sharpe ratio** = (mean excess return) / (volatility), annualised by √(periods/year).
  It is the signal-to-noise ratio of your P&L. An EE analogy: SR² · T is like SNR · bandwidth-time.
  The *t-statistic* of the mean return is ≈ SR · √years. So a Sharpe of 0.5 needs about 16 years to
  reach t = 2. **This is why you cannot tell skill from luck on short track records.**
* Typical values: stock market ≈ 0.3 to 0.5 (excess of cash). A good diversified hedge-fund strategy ≈ 0.5 to 1.0.
  Above 2 on daily data is either HFT/market making or a bug.
* **Max drawdown**: the worst peak-to-trough loss. US stocks lost 82% in 1929–32.
  Losing 50% needs +100% to recover. Most people abandon a strategy during a drawdown,
  so in practice drawdown matters more than Sharpe.

## 5. Going short, leverage, and futures
* **Short selling:** borrow a share, sell it, buy it back later (hopefully cheaper). Loss
  is unbounded and you pay a borrow fee.
* **Leverage:** weights summing to more than 1. You borrow the difference and pay interest on it. Our
  backtester charges `cash_rate` on borrowed money for this reason.
* **Futures:** contracts to buy/sell later at a price agreed now. You post only ~5 to 10% margin,
  so leverage and shorting are cheap and symmetric. This is why professional trend followers
  ("CTAs", "managed futures") trade futures across stocks, bonds, currencies and commodities.

## 6. The market efficiency question
The **Efficient Market Hypothesis** says prices already reflect available information, so
patterns should be arbitraged away. The realistic view (Lo's *Adaptive Markets*) is that
small, persistent edges exist when they are:
1. **Risk premia**: you get paid to bear a risk others avoid (e.g. the equity premium, carry).
2. **Behavioural**: other participants' biases (under-reaction then over-reaction leads to trends).
3. **Structural**: non-profit-seeking flows (hedgers, central banks, index rebalancing).
4. **Too small or costly** for big funds to exploit (capacity limits).

Trend following, our first real strategy, is thought to come from (2) and (3). See the
[research log](../research_log.md) for what we found when we tested it.
