# Project status

_Last updated: 2026-10-27 (after loop round 8: overfitting audit, ensemble champion)_

> **Champion (round 8):** equal thirds of 3× S&P trend, valuation-scaled leverage and recovery+.
> It is chosen by its worst era across 1880–2026, not by the best backtest:
> - eras: 15.4% / 14.6% / 13.1% / 21.3%;
> - 5-year full-cost: 19.6%/yr with a −26% max drawdown (after the 2023+ data fix); 15.7%/yr after 25% capital-gains tax (S&P 500: 13.1% / 10.4%);
> - forward 10-year median about 4.7%/yr.
>
> Real-time selection would have earned about 15%/yr on 1991–2026, not the champion's 21%. Treat backtest numbers as upper bounds.
>
> **Out-of-sample warning (Exp 19, Japan 1950–2026):** the trend filter worked there, but 3× leverage came close to ruin (−94%). A champion-like mix fell −87%.
> A lower-leverage ensemble fell −61% in Japan. On US history it has a worst era of 11.7% vs 13.1% and a max drawdown of −39% vs −53%. It runs in the real-time simulation for comparison.

> **Reality check on returns.** The stock market's long-run average is about 10%/yr (9.4% since 1871, 10.7% since 1957).
> The 18–20% figures in this repo are **2–3× leveraged** results from unusually strong periods (1991–2026, 2021–2026).
> They come with −45 to −50% drawdowns. At today's valuations the expected 10-year return is about 4–5%/yr for every candidate,
> including the champion. Taxes are not included.

## Where we are
**Roadmap:** Phase 1 (simulator) ✅ · Phase 2 (research hygiene) ✅ · Phase 3 (real data) mostly ✅ ·
Phase 4 (math / signal processing) started.

| Area | State |
|---|---|
| Backtester | Single asset + portfolios, costs, cash interest, leverage with borrow spread, any bar frequency |
| Statistics | Variance ratio, Hurst, Newey-West OLS, out-of-sample R², block bootstrap, Kelly, Probabilistic/Deflated Sharpe |
| Data | S&P 500 + 10y bonds monthly 1871–2026 (bonds reliable only from 1953) · CAPE/earnings to 2023 · DJIA daily 2007–16 · 50k headlines |
| Experiments | 15, all reproducible (`scripts/experiments/`); `scripts/today.py` prints current signals |
| Tests | 50 passing |
| Trading | `broker.py`: paper broker, Alpaca adapter, safety limits; `scripts/rebalance.py` (offline demo works; live needs keys + network) |

## The honest bottom line
1. **Markets are close to a random walk.** The only structure that survived testing is a weak trend at
   about 12 months and valuation mean reversion at about 10 years (Exp 6, 7).
2. **The trend filter (hold each asset only above its 10-month average) is crash insurance, not alpha.**
   Since 1957: drawdown −15% vs −27% for 60/40 (robust: 100% of bootstrap resamples), excess Sharpe
   0.63 vs 0.59 (not significant). It wins in slow bear markets and loses in V-shaped rebounds.
   Against a vol-matched constant-exposure control it still has smaller drawdowns (≈90% bootstrap
   probability), and synthetic tests show that happens only when markets trend (Exp 12).
   Implement it on month-end closes, or with a daily SMA-200 and a ±2% band (Exp 13).
   Its risk-adjusted edge is era-dependent: absent in 1957–1990, present in 1991–2026 (Exp 14).
3. **At today's valuations (CAPE ≈ 41) every candidate is expected to earn about 4–5%/yr nominal over 10
   years.** The candidates differ in downside: P(10-year loss) is 23% for the S&P 500 and 2.7% for the filtered 60/40.
4. **Leverage is a dial, not an edge.** Kelly says 3.6× on stocks; history says 4× is ruin.
   60/40+SMA10 at 1.5× → 9.5%/yr since 1957 (10.9% on the untouched 1991–2026 test), max DD −23%.
   Kelly sizing, ridge regression and a neural net added nothing (negative out-of-sample R², Exp 15).
   Leverage only pays with cheap financing (futures, or about 1–1.5% over cash).

## Annual profits (simulated, monthly data)
| | 2024 | 2025 | 2026 YTD | last 12m | 10y p.a. |
|---|---|---|---|---|---|
| S&P 500 | +30.3% | +15.8% | +13.2% | +18.3% | 15.5% |
| 60/40 | +19.3% | +11.2% | +9.2% | +12.6% | 9.6% |
| 60/40 + SMA-10 (frozen) | +19.1% | +3.6% | +9.2% | +12.6% | 8.0% |
| 60/40 + SMA-10 ×1.5 | +27.6% | +3.6% | +12.4% | +17.2% | 11.0% |
| Professional trend funds (SG Trend Index) | ≈ +2.6% | ≈ +2.4% | n/a | n/a | n/a |

## Current signal (2026-09)
Stocks and bonds are both above their 10-month averages, so the frozen rule holds 60% stocks / 40% bonds (`python scripts/today.py`).

## Corrections made this round
* "P(trend filter beats 60/40) = 99.5%" was withdrawn. It was inflated by fake-smooth pre-1953 bond data and by
  counting cash interest in Sharpe. See research log, Exp 11.
* Bond "momentum" was carry (Exp 6).

## Blockers
* **Network:** Yahoo/Stooq/FRED/news sites are blocked; only GitHub and PyPI work. Without more markets
  (commodities, currencies, international) we can't test where trend following really earns.
* **Anthropic API key:** for `news.ClaudeScorer`.

## Next steps
1. Paper trading the frozen rule via Alpaca or IBKR (docs/AUTOMATION.md).
2. Breadth: 20+ markets (needs data access).
3. Firm-level news + LLM scoring.
