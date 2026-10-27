# Lesson 2: Bonds, portfolios and diversification

## 1. What a bond is
A government bond is a loan to the state. A 10-year bond with a 4% coupon pays 2% of face value
every six months for 10 years, then returns the face value. Its **price** today is the
present value of those cash flows, discounted at the current market **yield** *y*:

    P = Σ c/2 · (1 + y/2)^(−k)  +  (1 + y/2)^(−20)

The EE view: a bond is a known cash-flow "signal". Its price is that signal convolved with a
discount kernel, and the yield is the one parameter that moves.

* **Yields up → prices down.** Sensitivity is the **duration** D (≈ 8 years for a 10-year bond
  at 4%): ΔP/P ≈ −D · Δy. A 1% rise in rates costs ≈ 8% of the bond's value. That is why 2022
  (yields 1.5% → 4%) was the worst year for bonds in our data.
* **Bond return ≈ starting yield + price change.** Over a full cycle the starting yield is the
  best single predictor of a bond's return. Stocks have no equivalent anchor.
* Our `data.bond_total_return` does this repricing month by month from Shiller's yield series.
  Writing it from first principles (and catching the double-counted coupon bug) is the best way
  to learn it.

## 2. Why combine stocks and bonds
Portfolio variance for weights w₁, w₂, vols σ₁, σ₂ and correlation ρ:

    σ_p² = w₁²σ₁² + w₂²σ₂² + 2 w₁w₂ ρ σ₁σ₂

If ρ < 1 the portfolio is less risky than the weighted average of its parts. That is
**diversification**, the only free lunch in finance. Two traps:
* **ρ moves.** Stock-bond correlation was −0.33 in 2000–2020, which is why 60/40 looked
  magical. It was +0.28 in 1968–99 and +0.33 since 2021. When *inflation* is the shock, both
  fall together (rates up → bonds down; costs up and valuations down → stocks down).
* **Capital weights ≠ risk weights.** A 60/40 portfolio gets **95% of its risk from stocks**
  (Exp 4), because stocks are ~3× as volatile. "Balanced" in dollars is very unbalanced in risk.
  **Risk parity** sizes positions by 1/volatility so each asset contributes equal risk.
  Usually it then levers up to reach a useful return.

## 3. Rebalancing
A 60/40 drifts as prices move. Rebalancing back to 60/40 sells what went up and buys what went
down: a built-in mean-reversion trade. It costs turnover, so in practice it is done monthly or
quarterly, or only when weights drift by more than ~5%.

## 4. Leverage is a tool, not a free lunch
Levering a high-Sharpe, low-return portfolio (e.g. risk parity) can beat an unlevered
stock portfolio *on paper*. In reality:
* You pay interest on borrowed money. Our backtester charges `cash_rate` on leverage.
* Losses compound: at 3× leverage a 33% fall in the underlying wipes you out.
* Correlations jump to 1 in a crisis, exactly when you need diversification.

In Exp 4 the vol-scaled trend portfolio levered bonds ~2.3× and lost 51% in 1966–82.

## 5. What a professional "trend following" fund actually is
A CTA / managed-futures fund trades **futures** on 50–150 markets: equity indices, government bonds,
currencies, commodities. It is long or short each one by its trend, sized so each contributes
similar risk, with the total scaled to ~10–15% vol. The return comes from **breadth** (Exp 2:
Sharpe ∝ √N) and from **crisis convexity** (it gets short in long, slow bear markets). With
only stocks and bonds you get a weak version of it.
