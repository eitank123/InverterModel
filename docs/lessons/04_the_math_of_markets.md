# Lesson 4: The math of markets: what is predictable, and how much to bet

This lesson collects the math behind Experiments 6–11. Every formula here was tested
on 155 years of data in `scripts/experiments/`.

## 1. Prices as a stochastic process
**Model.** Log price xₜ = ln Pₜ follows a random walk with drift: xₜ = xₜ₋₁ + μ + σεₜ, with εₜ i.i.d.
In continuous time this is geometric Brownian motion, dP/P = μ dt + σ dW. By Itô's lemma,
d(ln P) = (μ − σ²/2) dt + σ dW.

**Consequence (volatility drag).** The *median* investor's wealth grows at μ − σ²/2, not μ.
A strategy with 10% arithmetic mean and 20% volatility compounds at only ~8%. Lower vol
is worth real money even at the same mean.

**Why "random walk" is the default.** If returns were predictable from past prices, traders
would trade until they weren't (Samuelson 1965: properly anticipated prices fluctuate
randomly). Any predictability we find must be small, or compensation for risk, or costly to
exploit.

## 2. Testing the random walk: the variance ratio
If increments are uncorrelated, variance grows linearly with horizon: Var(rₜ + … + rₜ₋q₊₁) = q·Var(rₜ).
The **variance ratio**

    VR(q) = Var(q-period return) / (q · Var(1-period return)) ≈ 1 + 2 Σₖ₌₁^{q−1} (1 − k/q) ρₖ

is a weighted sum of autocorrelations ρₖ (Lo & MacKinlay 1988). VR > 1 means trending,
VR < 1 means mean-reverting. In EE terms, VR(q) − 1 is the low-frequency part of the
return spectrum compared with a white-noise spectrum: a random walk has a flat return spectrum.

**What we found (Exp 6):** the textbook z-test "rejects" the random walk with z = +7.
This is entirely an artifact of monthly *averaged* prices. Averaging a random walk gives
ρ₁ = 0.25 (Working 1960). Against a correctly simulated null:
* S&P 500: random walk at almost every horizon. Borderline **trend at 12 months**
  (VR 1.66 vs 95% band 1.65), consistent with time-series momentum.
* **Long-horizon mean reversion** (VR(120) = 0.64) only before 1945 (Poterba & Summers 1988;
  Fama & French 1988). Not after.
* Bonds look strongly "trending", but that is **carry**. A bond's return ≈ its yield, and yields
  persist for decades. Strip out carry and bond prices are a random walk.

## 3. Predicting with valuation: the Campbell–Shiller identity
Log return identity (Campbell & Shiller 1988): with dₜ − pₜ the log dividend yield,

    dₜ − pₜ ≈ Σⱼ ρʲ (rₜ₊₁₊ⱼ − Δdₜ₊₁₊ⱼ) + const

A high price relative to fundamentals *must* be followed by low returns or high dividend
growth. Historically it has been followed by low returns. The CAPE (price / 10-year average real
earnings) is the smoothed version.

**Two statistical traps:**
1. **Overlapping observations.** Monthly samples of 10-year returns share 119/120 months, so OLS
   t-stats are inflated about 10×. Use **Newey–West** standard errors (`stats.ols_newey_west`).
2. **In-sample ≠ real time.** Goyal & Welch (2008) showed nearly all predictors fail to beat the
   historical mean *out of sample*. Measure R²_OOS = 1 − Σ(y − ŷ_model)² / Σ(y − ŷ_mean)², with
   the model refit each month on data available then (`stats.oos_r2`).

**What we found (Exp 7):** in-sample, CAPE explains 24–29% of 10-year real returns
(Newey–West t ≈ −4.6). Out of sample it is unstable: +2–3% R² at 1 year, negative at 5 years,
positive at 10 years only with the Campbell–Thompson "no negative forecast" restriction.
**At today's CAPE ≈ 41 the regression implies ≈ 0.6%/yr real stock returns over the next decade.**
That is a central estimate with very wide error bands, not a forecast to bet on.

## 4. How much to bet: Kelly
To maximise long-run growth E[ln W], choose leverage f to maximise

    g(f) = r + f·μ − f²σ²/2   →   f* = μ/σ²,   g* = r + SR²/2

For GBM at a fraction λ of Kelly, P(wealth ever falls to x) = x^(2/λ − 1):

| Bet size | P(ever halve) | P(ever lose 80%) |
|---|---|---|
| Full Kelly | 50% | 20% |
| Half Kelly | 12.5% | 0.8% |
| Quarter Kelly | 0.8% | ~0% |

**What we found (Exp 8):**
* Full-sample Kelly says 3.6× leverage on stocks. At 4× the 1929 crash wipes you out.
  Fat tails and gaps break the GBM assumption.
* Estimated on trailing 30-year windows, "optimal" leverage ranged 1.0–9.4×. Following the
  estimate *reduced* Sharpe in every case.
* Practical rule: choose leverage by **maximum tolerable drawdown**, not by Kelly. Kelly is
  an upper bound you never want to reach.

## 5. Combining signals: breadth and correlation
* **Fundamental law** (Grinold 1989): IR ≈ IC · √breadth. Many independent small edges beat one
  big one (Exp 2: Sharpe 1.1 → 3.4 from 1 to 10 independent markets).
* Two strategies with Sharpe S₁ and S₂ and correlation ρ: the optimal mix has
  S² = (S₁² + S₂² − 2ρS₁S₂) / (1 − ρ²). Negative correlation is worth a lot. Asness, Moskowitz &
  Pedersen (2013) find value and momentum at ρ ≈ −0.5. Our *timing* versions on one market had
  ρ ≈ −0.05: little diversification between them.

## 6. Measuring honestly
* **Raw vs. excess Sharpe.** Sharpe must use returns *above cash*. A strategy that sits in cash
  during high-rate decades earns that interest and looks better on raw Sharpe. In Exp 11 the
  trend filter's raw Sharpe gain (1.12 → 1.34) shrank to 0.59 → 0.63 in excess terms.
* **Probabilistic / Deflated Sharpe** (Bailey & López de Prado): account for track-record length,
  skew, fat tails and the number of variants tried.
* **Stationary block bootstrap** (Politis & Romano 1994): resample blocks of months to get
  confidence intervals that respect autocorrelation and volatility clustering.
* **Data provenance:** monthly averages create fake momentum (Exp 1, 6), and interpolated
  pre-1953 bond yields create fake-smooth bonds with Sharpe 5–9 (Exp 10, 11). Know how your data was built.
