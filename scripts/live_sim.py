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

sp, y10, tb = at.data.load_fred("SP500"), at.data.load_fred("DGS10"), at.data.load_fred("DTB3")
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
ACCOUNTS["ensemble_lower_leverage"] = (
    "Ensemble, lower leverage (2x trend, 1-1.5x recovery)",
    "Exp 19: same three sleeves as the champion at lower leverage, after the Japan test (champion-like "
    "mix -87% there, this one about -61%). Added 2026-10-28.", from_strategy(ensemble_lower()))

print(f"Data: FRED S&P 500 to {sp.index[-1]:%Y-%m-%d}, 10y yield to {y10.index[-1]:%Y-%m-%d}, "
      f"T-bill to {tb.index[-1]:%Y-%m-%d}")
for key, (name, desc, fn) in ACCOUNTS.items():
    acct = LiveSimAccount.load_or_create(os.path.join(ROOT, f"{key}.json"), key, name, desc,
                                         start_date=etf.index[-1].strftime("%Y-%m-%d"))
    if key in DAILY and acct.st["last_date"]:                # rebuild the band state up to the last day
        for d in sp.index[(sp.index >= pd.Timestamp(acct.st["start"])) & (sp.index <= pd.Timestamp(acct.st["last_date"]))]:
            fn(d)
    n = acct.advance(etf, cash, fn, costs, daily=key in DAILY)
    acct.save()
    st = acct.st
    eq = st["equity_history"][-1]["equity"] if st["equity_history"] else st["start_equity"]
    print(f"{name:55s} {st['last_date']}  +{n} days  equity {eq:>12,.2f} "
          f"({eq / st['start_equity'] - 1:+.2%})  holdings {{{', '.join(f'{k}: {v:,.1f}' for k, v in st['holdings'].items())}}}")
