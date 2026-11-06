"""Real-time forward simulation of the strategies (play money, real market data).

    python scripts/live_sim.py

Reads the newest daily closes from FRED (S&P 500, 10-year Treasury yield, 3-month T-bill),
advances every account by the trading days it has not seen yet, and saves the record in
loop/live_sim/*.json. No money and no broker are involved. ETF prices (SPY, UPRO, IEF, UST)
are built from the real daily index moves net of each fund's leverage financing and expense
ratio; fills are at the daily close with spread, slippage, commission and regulatory fees.
Accounts start on the latest available close when first created and are never backfilled.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import algotrading as at  # noqa: E402
from algotrading.execution import CostModel, etf_weights  # noqa: E402
from algotrading.live import completed_month_ends, shiller_history, splice_monthly  # noqa: E402
from algotrading.live_sim import LiveSimAccount, daily_etf_returns  # noqa: E402
from loop import build_from_row, ensemble, ensemble_lower, filtered  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "loop", "live_sim")
ANCHOR = "2026-10-23"                       # price levels: SPY = S&P/10 there; others fixed
LEVELS = {"UPRO": 100.0, "IEF": 95.0, "UST": 45.0}

sp, y10, tb = at.data.sp500_with_fallback(), at.data.load_fred("DGS10"), at.data.load_fred("DTB3")
closes = at.data.fred_daily_proxies(sp500=sp, dgs10=y10)
cash = (tb / 100).reindex(closes.index).ffill()
costs = CostModel()
growth = (1 + daily_etf_returns(closes, cash, costs)).cumprod()
anchor = pd.Timestamp(ANCHOR)
etf = growth / growth.loc[anchor] * pd.Series({"SPY": sp.loc[anchor] / 10, **LEVELS})
history, _ = shiller_history()


def from_strategy(strategy):
    def fn(d):
        me = completed_month_ends(closes[:d], asof=d)
        monthly = splice_monthly(history, me)
        w = strategy.weights(monthly).iloc[-1]
        exp = {"stocks": float(w.get("stocks", 0.0)), "bonds": float(w.get("bonds", 0.0))}
        return etf_weights(exp["stocks"], exp["bonds"]), exp, \
            f"signal from month-ends to {me.index[-1]:%Y-%m}: stocks {exp['stocks']:.2f}x, bonds {exp['bonds']:.2f}x"
    return fn


def fixed(weights, reason):
    return lambda d: (weights, {"stocks": weights.get("SPY", 0.0), "bonds": weights.get("IEF", 0.0)}, reason)


ACCOUNTS = {
    "champion_ensemble": ("Champion: ensemble (3x trend + valuation + recovery+)",
                          "Equal thirds of three published effects: S&P 500 at 3x above its 10-month average; "
                          "leverage 0.5-3x set by valuation; recovery+. Traded through SPY/UPRO and IEF/UST.",
                          from_strategy(ensemble(3.0))),
    "challenger_recovery_plus": ("Challenger: recovery+ (2x/3x, 2x Treasuries when out)",
                                 "2x S&P while above its 10-month average or up over 12 months, 3x after >15% "
                                 "falls; 2x Treasuries otherwise.",
                                 from_strategy(at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0))),
    "previous_recovery": ("Previous champion: recovery leverage",
                          "2x S&P above its 10-month average, 3x after >15% falls, cash otherwise.",
                          from_strategy(at.RecoveryLeverage(2.0, 3.0, 0.15))),
    "benchmark_sp500": ("Benchmark: S&P 500 buy & hold", "100% SPY.", fixed({"SPY": 1.0}, "buy and hold")),
    "benchmark_6040": ("Benchmark: 60/40", "60% SPY, 40% IEF, rebalanced monthly.",
                       fixed({"SPY": 0.6, "IEF": 0.4}, "monthly rebalance to 60/40")),
}

# The fleet: the best (train-picked) member of every strategy family the loop has tested, run
# side by side so the real-time record can tell them apart. Added 2026-10-27; never edited
# afterwards (new ideas get NEW accounts). (key, title, leaderboard name, params)
FLEET = [
    ("trend_3x", "Trend: S&P 500 + SMA10 at 3x", "S&P+SMA10 xL (Gayed-Bilello)", '{"lev": 3.0}'),
    ("trend_2x", "Trend: S&P 500 + SMA10 at 2x", "mix+SMA10 x2 (deployable)", '{"w": 1.0}'),
    ("trend_strength_3x", "Trend strength: leverage scaled by 4 averages", "trend-strength leverage",
     '{"w": 1.0, "lev": 3.0}'),
    ("trend_6040_4x", "Trend 60/40 at 4x (needs margin)", "60/40+SMA10 xL", '{"lev": 4.0}'),
    ("drawdown_brake_3x", "Trend 3x + drawdown brake", "S&P+SMA10 xL + drawdown brake", '{"lev": 3.0, "dd": 0.35}'),
    ("seasonal_3x", "Seasonal: 3x Nov-Apr, 1x May-Oct (trend-filtered)", "seasonal leverage S&P+SMA10",
     '{"w_": 3.0, "s_": 1.0}'),
    ("variance_managed", "Variance-managed stocks (Moreira-Muir)", "variance-managed stocks",
     '{"lev": 2.0, "cap": 3.0, "win": 3}'),
    ("vol_target_6040", "Vol-target 12% on trend 60/40 (cap 4x)", "60/40+SMA10 vol-target", '{"tv": 0.12, "cap": 4.0}'),
    ("valuation_3x", "Valuation-scaled leverage (0.5-3x)", "valuation-scaled leverage <=3x", '{"w": 1.0, "k": 2.0}'),
    ("valuation_kelly", "Valuation Kelly (cap 3x, no trend filter)", "valuation-Kelly ungated", '{"f": 1.0, "cap": 3.0}'),
    ("dual_momentum", "Dual momentum (Antonacci)", "dual momentum xL", '{"lev": 1.0}'),
    ("recovery_mix", "Recovery mix: 80/20 trend, 2x, 4x after falls", "recovery mix",
     '{"w": 0.8, "base": 2.0, "boost": 4.0}'),
    ("ensemble_trend_value", "Ensemble: 2x trend + valuation", "ensemble: trend + value", "{}"),
]
for key, title, name, params in FLEET:
    ACCOUNTS[key] = (title, f"Loop strategy '{name}' {params}, best of its family (parameters picked on "
                     "1957-1990).", from_strategy(build_from_row({"name": name, "params": params})))
ACCOUNTS["frozen_6040_sma10"] = ("Frozen rule: 60/40 + SMA10 (no leverage)",
                                 "The original research rule: 60/40, each asset held only above its 10-month "
                                 "average.", from_strategy(filtered(0.6)))



def daily_band_200(lev=3.0, band=0.02):
    """Exp 18: S&P at ``lev`` while above its 200-day average (+/-2% band), T-bills otherwise.
    Gayed & Bilello (2016) 'Leverage for the Long Run', banded as in Exp 13. Added 2026-10-27."""
    state = {"on": None}

    def fn(d):
        s = sp[:d]
        sma = s.iloc[-200:].mean()
        p = s.iloc[-1]
        if state["on"] is None:
            state["on"] = p > sma
        elif state["on"] and p < sma * (1 - band):
            state["on"] = False
        elif not state["on"] and p > sma * (1 + band):
            state["on"] = True
        w = etf_weights(lev, 0.0) if state["on"] else {}
        return w, {"stocks": lev if state["on"] else 0.0, "bonds": 0.0}, \
            f"S&P {p / sma - 1:+.1%} vs 200-day average: {'in' if state['on'] else 'out'}"
    return fn


DAILY = {"daily_band_3x": ("Daily trend: 3x above the 200-day average (±2% band)",
                           "Gayed-Bilello daily rule with a 2% band (Exp 18): UPRO while the S&P is above its "
                           "200-day average, T-bills otherwise; checked every day.", daily_band_200())}
ACCOUNTS.update(DAILY)


def news_tone_overlay(lev=2.0, low=1.0, z_cut=-1.0):
    """Exp 28: S&P at ``lev``, cut to ``low`` while the 5-day average tone of 'stock market' news
    (GDELT) is more than 1 standard deviation below its trailing 250-day mean (Tetlock 2007).
    The backtest (2018-2026) LOST to plain 2x (17.7% vs 21.2%/yr; p = 0.66 against shifted news):
    this account is a forward test the owner asked for, not a recommendation. Added 2026-11-03."""
    tone = at.news.update_gdelt_tone()
    z = at.news.tone_zscore(tone)

    def fn(d):
        zz = z[z.index < d]                                # news through the previous day only
        zv = float(zz.iloc[-1]) if len(zz) and pd.notna(zz.iloc[-1]) else 0.0
        L = low if zv < z_cut else lev
        return etf_weights(L, 0.0), {"stocks": L, "bonds": 0.0}, \
            f"news tone z {zv:+.1f} (through {zz.index[-1]:%d %b}): {L:.0f}x" if len(zz) else f"no news data: {L:.0f}x"
    return fn


NEWS = {"news_tone_2x": ("News tone: 2x S&P, 1x when the news turns gloomy",
                         "Exp 28 forward test: 2x the S&P, cut to 1x while the 5-day news tone (GDELT, "
                         "'stock market', English) is 1 standard deviation below its 250-day mean. "
                         "Lost to plain 2x in the 2018-2026 backtest.", news_tone_overlay())}
DAILY.update(NEWS)
ACCOUNTS.update(NEWS)
BANDS = {"champion_band10": 0.10}
ACCOUNTS["champion_band10"] = (
    "Champion, tax-aware (trades only beyond a 10% band)",
    "Same signals as the champion ensemble; a holding is traded only when it drifts more than 10% of "
    "the account from target (Exp 25: +0.4 points a year after a 25% tax over 1957-2026). Added 2026-10-29.",
    from_strategy(ensemble(3.0)))
ACCOUNTS["ensemble_lower_leverage"] = (
    "Ensemble, lower leverage (2x trend, 1-1.5x recovery)",
    "Exp 19: same three sleeves as the champion at lower leverage, after the Japan test (champion-like "
    "mix -87% there, this one about -61%). Added 2026-10-28.", from_strategy(ensemble_lower()))
ACCOUNTS["champion_band2"] = (
    "Champion with a 2% band on its trend signals",
    "Exp 34: the live system reads the trend from month-end closes, which whipsawed out of the market in "
    "April 2025 and April 2026. Same three sleeves; a sleeve leaves only 2% below its 10-month average and "
    "re-enters 2% above it. CRSP 1927-2026 on month-end closes: worst era 14.0% vs 11.8%. Added 2026-11-03.",
    from_strategy(ensemble(3.0, band=0.02)))

try:                                                         # grow the headline archive (Exp 28)
    print(f"Headlines archived: {len(at.news.collect_headlines()):,}")
except Exception as e:
    print(f"Headline feeds unavailable: {type(e).__name__}")
print(f"Data: FRED S&P 500 to {sp.index[-1]:%Y-%m-%d}, 10y yield to {y10.index[-1]:%Y-%m-%d}, "
      f"T-bill to {tb.index[-1]:%Y-%m-%d}")
for key, (name, desc, fn) in ACCOUNTS.items():
    acct = LiveSimAccount.load_or_create(os.path.join(ROOT, f"{key}.json"), key, name, desc,
                                         start_date=etf.index[-1].strftime("%Y-%m-%d"))
    if key in DAILY and acct.st["last_date"]:                # rebuild the band state up to the last day
        for d in sp.index[(sp.index >= pd.Timestamp(acct.st["start"])) & (sp.index <= pd.Timestamp(acct.st["last_date"]))]:
            fn(d)
    n = acct.advance(etf, cash, fn, costs, daily=key in DAILY, band=BANDS.get(key, 0.02))
    acct.save()
    st = acct.st
    eq = st["equity_history"][-1]["equity"] if st["equity_history"] else st["start_equity"]
    print(f"{name:55s} {st['last_date']}  +{n} days  equity {eq:>12,.2f} "
          f"({eq / st['start_equity'] - 1:+.2%})  holdings {{{', '.join(f'{k}: {v:,.1f}' for k, v in st['holdings'].items())}}}")


# --- Accounts on REAL ETF prices (Yahoo Finance), Exp 31 ----------------------------------
from algotrading.execution import REAL_ETFS, SECTORS  # noqa: E402

champ_signal = from_strategy(ensemble(3.0))


def diversified_two_thirds(scale=2 / 3):
    """Exp 31 lead challenger: the champion's stock/bond exposure x2/3 (never above 2x), with the
    stock part in equal thirds: industry momentum (top 3 of the 11 sector SPDRs by 12-1 month
    return, Moskowitz-Grinblatt 1999), stock momentum (MTUM, Jegadeesh-Titman 1993) and value
    (VTV, Fama-French 1992). Bonds through IEF. Leverage via a margin loan. Added 2026-11-03."""
    def fn(d):
        _, exp, why = champ_signal(d)
        s, b = exp["stocks"] * scale, exp["bonds"] * scale
        me = real[SECTORS][real.index < d.to_period("M").to_timestamp()]
        me = me.groupby(me.index.to_period("M")).tail(1)              # completed month-ends
        mom = (me.iloc[-2] / me.iloc[-13] - 1).dropna()               # 12-1 months
        top = list(mom.sort_values(ascending=False).index[:3])
        w = {"MTUM": s / 3, "VTV": s / 3, **{t: s / 9 for t in top}}
        if b:
            w["IEF"] = b
        w = {k: v for k, v in w.items() if v > 1e-9}
        return w, {"stocks": s, "bonds": b}, \
            f"champion x2/3: stocks {s:.2f}x (sectors {', '.join(top)} + MTUM + VTV), bonds {b:.2f}x"
    return fn


REAL = {"diversified_two_thirds": ("Diversified leg x2/3 (sectors momentum + MTUM + VTV, max 2x)",
                                   "Exp 31 lead challenger: the champion's signals at 2/3 of its leverage, stock part "
                                   "split equally into industry momentum (top 3 sector SPDRs), stock momentum (MTUM) and "
                                   "value (VTV). Real ETF prices from Yahoo Finance; margin loan at the T-bill + 1.5%.",
                                   diversified_two_thirds())}
RP_ASSETS = ["SPY", "EFA", "IEF", "LQD", "GLD"]


def champion_plus_risk_parity(tv=0.15):
    """Exp 35: 50% the champion (SPY/UPRO/IEF) + 50% risk parity with trend filters over US stocks,
    developed-market stocks, Treasuries, corporate bonds and gold (Asness-Frazzini-Pedersen 2012
    weights; Faber 2007 10-month filters), scaled to 15% volatility, max 3x. Added 2026-11-03."""
    import numpy as np

    def fn(d):
        w_ch, exp, _ = champ_signal(d)
        px = real[RP_ASSETS][real.index < d.to_period("M").to_timestamp()]
        me = px.groupby(px.index.to_period("M")).tail(1)
        r = me.pct_change().iloc[-12:]
        vol = r.std() * np.sqrt(12)
        base = (1 / vol) / (1 / vol).sum()
        pv = float(np.sqrt(base @ r.cov() @ base * 12))
        on = me.iloc[-1] > me.iloc[-10:].mean()
        rp = (base * min(tv / pv, 3.0)).where(on, 0.0)
        w = {k: 0.5 * v for k, v in w_ch.items()}
        for k, v in rp.items():
            if v > 1e-9:
                w[k] = w.get(k, 0.0) + 0.5 * float(v)
        return w, {"stocks": exp["stocks"] / 2 + 0.5 * float(rp["SPY"] + rp["EFA"]),
                   "bonds": exp["bonds"] / 2 + 0.5 * float(rp["IEF"] + rp["LQD"])}, \
            "50% champion + 50% risk parity: " + ", ".join(f"{k} {v:.2f}" for k, v in rp.items())
    return fn


btc = at.data.load_fred("CBBTCUSD").dropna()


def champion_plus_bitcoin(k=0.05):
    """Exp 36: 95% the champion + 5% Bitcoin (IBIT spot ETF) while Bitcoin is above its 200-day
    average, T-bills otherwise (checked daily). Weak evidence (2015-2026 only). Added 2026-11-03."""
    def fn(d):
        w_ch, exp, _ = champ_signal(d)
        b = btc[btc.index < d]
        on = b.iloc[-1] > b.iloc[-200:].mean()
        w = {s: (1 - k) * v for s, v in w_ch.items()}
        if on:
            w["IBIT"] = k
        return w, {"stocks": (1 - k) * exp["stocks"], "bonds": (1 - k) * exp["bonds"]}, \
            f"champion 95% + Bitcoin {'in' if on else 'out'} (BTC {b.iloc[-1] / b.iloc[-200:].mean() - 1:+.0%} vs 200-day)"
    return fn


REAL["champion_risk_parity"] = ("50/50 champion + multi-asset risk parity (15% vol)",
                                "Exp 35: half the champion, half risk parity with trend filters over SPY, EFA, IEF, LQD "
                                "and GLD at 15% volatility (max 3x). Real ETF prices; margin at T-bill + 1.5%.",
                                champion_plus_risk_parity())
REAL["champion_bitcoin5"] = ("Champion + 5% Bitcoin trend (IBIT)",
                             "Exp 36: 95% the champion, 5% in the IBIT Bitcoin ETF while Bitcoin is above its 200-day "
                             "average. Real ETF prices. Weak evidence: only 2015-2026 exists.", champion_plus_bitcoin())
champ_band_signal = from_strategy(ensemble(3.0, band=0.02))


def core_satellite():
    """Exp 37 round 4 (chosen on 1972-1999 only): 2/3 the champion with the 2% band + 1/3 a
    satellite at 2x made of equal thirds: risk parity + trend (10% vol), time-series momentum
    (10% vol per asset) and gold above its 10-month average; SPY/EFA/IEF/LQD/GLD. Added 2026-11-03."""
    import numpy as np

    def fn(d):
        w_ch, exp, _ = champ_band_signal(d)
        px = real[RP_ASSETS][real.index < d.to_period("M").to_timestamp()]
        me = px.groupby(px.index.to_period("M")).tail(1)
        r = me.pct_change().iloc[-12:]
        vol = r.std() * np.sqrt(12)
        on = me.iloc[-1] > me.iloc[-10:].mean()
        base = (1 / vol) / (1 / vol).sum()
        pv = float(np.sqrt(base @ r.cov() @ base * 12))
        rp = (base * min(0.10 / pv, 3.0)).where(on, 0.0)
        rf12 = float(cash[cash.index < d].iloc[-252:].mean())
        ts = ((me.iloc[-1] / me.iloc[-13] - 1 > rf12).astype(float) * 0.10 / vol / 5)
        ts = ts / max(ts.sum() / 3, 1.0)
        gold = pd.Series(0.0, index=RP_ASSETS)
        gold["GLD"] = float(on["GLD"])
        sat = 2 * (rp + ts + gold) / 3
        w = {k: 2 / 3 * v for k, v in w_ch.items()}
        for k, v in sat.items():
            if v > 1e-9:
                w[k] = w.get(k, 0.0) + float(v) / 3
        return w, {"stocks": 2 / 3 * exp["stocks"] + float(sat["SPY"] + sat["EFA"]) / 3,
                   "bonds": 2 / 3 * exp["bonds"] + float(sat["IEF"] + sat["LQD"]) / 3}, \
            "2/3 champion (2% band) + 1/3 satellite: " + ", ".join(f"{k} {v:.2f}" for k, v in sat.items())
    return fn


REAL["core_satellite"] = ("One algorithm: 2/3 champion + 1/3 multi-asset satellite (2x)",
                          "Exp 37 round 4, chosen on 1972-1999 only: the banded champion plus a satellite of risk parity, "
                          "time-series momentum and gold trend at 2x. Real ETF prices; margin at T-bill + 1.5%.",
                          core_satellite())
DAILY_REAL = {"champion_bitcoin5"}
real = at.data.etf_closes(list(REAL_ETFS))
margin_costs = CostModel(swap_spread=0.015)
print(f"Real ETF prices (Yahoo Finance) to {real.index[-1]:%Y-%m-%d}")
# Data guard: an unlevered ETF moving more than 40% in a day is almost always a share split the
# feed has not adjusted yet (MTUM 272.86 -> 62.60 on 2026-11-05, no split event in the feed) or a
# bad tick. Accounts holding such a fund wait at the day before until the price is confirmed.
day_ret = real[list(REAL_ETFS)].pct_change()
suspect = {s: day_ret.index[(day_ret[s].abs() > 0.40 * REAL_ETFS[s].leverage)].tolist() for s in REAL_ETFS}
suspect = {s: d for s, d in suspect.items() if d}
for s, d in suspect.items():
    print(f"DATA GUARD: {s} moved {day_ret.loc[d[-1], s]:+.0%} on {d[-1]:%Y-%m-%d} - probable unadjusted split/bad tick")
for key, (name, desc, fn) in REAL.items():
    acct = LiveSimAccount.load_or_create(os.path.join(ROOT, f"{key}.json"), key, name, desc,
                                         start_date=real.index[-1].strftime("%Y-%m-%d"))
    px = real[list(REAL_ETFS)].dropna()
    held = set(acct.st["holdings"]) | set(acct.st.get("targets", {}))
    cut = [d for s, ds in suspect.items() if s in held for d in ds if acct.st["last_date"] is None or d > pd.Timestamp(acct.st["last_date"])]
    if acct.st["last_date"] and pd.Timestamp(acct.st["last_date"]) in px.index:   # split re-basing
        for sym, q in list(acct.st["holdings"].items()):
            was, now = acct.st["prices"].get(sym), float(px.loc[pd.Timestamp(acct.st["last_date"]), sym])
            if was and not (0.67 < was / now < 1.5):
                ratio = was / now
                acct.st["holdings"][sym] = q * ratio
                acct.st["prices"][sym] = now
                acct.st.setdefault("corporate_actions", []).append(
                    {"date": acct.st["last_date"], "symbol": sym, "share_ratio": round(ratio, 6),
                     "note": "feed re-based past prices (split): shares scaled, value unchanged"})
                print(f"  {name}: {sym} re-based by the feed (x{ratio:.3f}); shares scaled, value unchanged")
    if cut:
        px = px[px.index < min(cut)]
        print(f"  {name}: waiting at {px.index[-1]:%Y-%m-%d} (holds a fund with a suspect price)")
    n = acct.advance(px, cash, fn, margin_costs, instruments=REAL_ETFS, daily=key in DAILY_REAL)
    acct.save()
    st = acct.st
    eq = st["equity_history"][-1]["equity"] if st["equity_history"] else st["start_equity"]
    print(f"{name:55s} {st['last_date']}  +{n} days  equity {eq:>12,.2f} "
          f"({eq / st['start_equity'] - 1:+.2%})  holdings {{{', '.join(f'{k}: {v:,.1f}' for k, v in st['holdings'].items())}}}")

try:                                                       # Polymarket forward paper test (Exp 32)
    import polymarket_paper
    polymarket_paper.main()
except Exception as e:
    print(f"Polymarket paper test skipped: {type(e).__name__}")
