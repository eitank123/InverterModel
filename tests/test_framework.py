import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402

ALL = [at.BuyAndHold(), at.SMACrossover(10, 50), at.SMACrossover(10, 50, allow_short=True),
       at.Momentum(20), at.ZScoreMeanReversion(20), at.VolTarget(at.Momentum(20))]


@pytest.fixture
def prices():
    return at.data.gbm(n=1000, seed=0)["close"]


def test_generators_shape_and_positive():
    for df in [at.data.gbm(500, seed=1), at.data.ornstein_uhlenbeck(500, seed=1),
               at.data.regime_switching(500, seed=1)]:
        assert len(df) == 500 and (df["close"] > 0).all()
        assert isinstance(df.index, pd.DatetimeIndex)


def test_gbm_statistics():
    df = at.data.gbm(252 * 200, mu=0.10, sigma=0.20, seed=0)
    r = np.log(df["close"]).diff().dropna()
    assert abs(r.std() * np.sqrt(252) - 0.20) < 0.01


def test_buy_and_hold_matches_asset(prices):
    res = at.run_backtest(prices, at.BuyAndHold(), cost_bps=0)
    assert np.isclose(res.equity.iloc[-1], prices.iloc[-1] / prices.iloc[0])


def test_costs_charged_once_on_entry(prices):
    res = at.run_backtest(prices, at.BuyAndHold(), cost_bps=10)
    assert np.isclose(res.costs.sum(), 10 / 1e4)


def test_position_timing_no_same_bar_return():
    # Price jumps on bar 5; a strategy that goes long ON bar 5 must not earn that jump.
    px = pd.Series([100.0] * 5 + [200.0] * 5, index=pd.bdate_range("2020-01-01", periods=10))

    class LongFromBar5(at.Strategy):
        def positions(self, prices):
            return pd.Series([0.0] * 5 + [1.0] * 5, index=prices.index)

    res = at.run_backtest(px, LongFromBar5(), cost_bps=0)
    assert np.isclose(res.returns.sum(), 0.0)


@pytest.mark.parametrize("strategy", ALL, ids=repr)
def test_no_lookahead(strategy, prices):
    assert at.validation.check_no_lookahead(strategy, prices)


def test_lookahead_detector_catches_cheater(prices):
    class Cheater(at.Strategy):
        def positions(self, prices):
            return np.sign(prices.pct_change().shift(-1)).fillna(0.0)

    assert not at.validation.check_no_lookahead(Cheater(), prices)
    assert at.run_backtest(prices, Cheater()).stats["sharpe"] > 5  # too good to be true


def test_mean_reversion_wins_on_ou():
    px = at.data.ornstein_uhlenbeck(2520, theta=20, sigma=0.25, seed=3)
    assert at.run_backtest(px, at.ZScoreMeanReversion(20)).stats["sharpe"] > 1


def test_metrics_known_values():
    r = pd.Series([0.1, -0.5, 0.2])
    assert np.isclose(at.metrics.total_return(r), 1.1 * 0.5 * 1.2 - 1)
    assert np.isclose(at.metrics.max_drawdown(r), -0.5)


def test_walk_forward_runs(prices):
    wf = at.validation.walk_forward(lambda lookback: at.Momentum(lookback),
                                    {"lookback": [10, 40]}, prices,
                                    train_bars=400, test_bars=200)
    assert len(wf) == 3 and len(wf.attrs["oos_returns"]) == 600


# --- Portfolio, cash, frequency and significance ---------------------------------------

def test_infer_ppy():
    assert at.metrics.infer_ppy(pd.bdate_range("2020-01-01", periods=50)) == 252
    assert at.metrics.infer_ppy(pd.date_range("2000-01-01", periods=50, freq="MS")) == 12


def test_flat_strategy_earns_cash():
    px = at.data.gbm(252 * 5, seed=0)["close"]

    class Flat(at.Strategy):
        def positions(self, prices):
            return pd.Series(0.0, index=prices.index)

    res = at.run_backtest(px, Flat(), cash_rate=0.05)
    assert abs(res.stats["cagr"] - 0.0513) < 0.002      # 5% compounded daily


def test_leverage_pays_cash_rate(prices):
    class Levered(at.Strategy):
        def positions(self, prices):
            return pd.Series(2.0, index=prices.index)

    a = at.run_backtest(prices, Levered(), cost_bps=0, cash_rate=0.0)
    b = at.run_backtest(prices, Levered(), cost_bps=0, cash_rate=0.05)
    assert np.allclose((a.returns - b.returns).iloc[1:], 0.05 / 252)


def test_portfolio_equal_weight_matches_average():
    px = at.data.panel(at.data.gbm, n_assets=3, seed=1, n=500)
    res = at.run_backtest(px, at.PerAsset(at.BuyAndHold()), cost_bps=0)
    expected = px.pct_change().fillna(0.0).mean(axis=1)
    assert np.allclose(res.returns.iloc[1:], expected.iloc[1:])
    assert np.isclose(res.stats["exposure"], 1.0, atol=0.01)


def test_portfolio_vol_target_is_causal():
    px = at.data.panel(at.data.regime_switching, n_assets=3, seed=2, n=800)
    strat = at.PortfolioVolTarget(at.PerAsset(at.TrendEnsemble((10, 50))), target_vol=0.1)
    full = strat.weights(px)
    t = 500
    tampered = px.copy()
    tampered.iloc[t:] *= 1.7
    assert np.allclose(full.iloc[:t], strat.weights(tampered).iloc[:t])


def test_portfolio_vol_target_hits_target():
    px = at.data.panel(at.data.regime_switching, n_assets=10, seed=3, n=2520)
    strat = at.PortfolioVolTarget(at.PerAsset(at.TrendEnsemble()), target_vol=0.10)
    vol = at.run_backtest(px, strat, cost_bps=0).stats["ann_vol"]
    assert 0.08 < vol < 0.12


@pytest.mark.parametrize("strategy", [at.PriceAboveSMA(10), at.TrendEnsemble((5, 20, 60))],
                         ids=repr)
def test_new_strategies_no_lookahead(strategy, prices):
    assert at.validation.check_no_lookahead(strategy, prices)


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(0)
    sims = rng.standard_normal((20000, 100)).max(axis=1).mean()
    assert abs(at.metrics.expected_max_sharpe(100, 1.0) - sims) < 0.05


def test_psr_and_dsr_behave():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2000-01-01", periods=2520)
    r = pd.Series(rng.normal(0.0004, 0.01, 2520), index=idx)    # true SR ~ 0.63
    psr_short = at.metrics.probabilistic_sharpe(r.iloc[:252])
    psr_long = at.metrics.probabilistic_sharpe(r)
    assert psr_long > psr_short and psr_long > 0.9
    assert at.metrics.deflated_sharpe(r, n_trials=1000, trial_sharpes_std=0.5) < psr_long


def test_shiller_data_sane():
    df = at.data.load_shiller_sp500()
    core = ["close", "tr", "dividend", "long_rate", "cpi"]
    assert df.index[0].year == 1871 and df[core].notna().all().all()
    assert 4 < df["cape"].min() and df["cape"].max() < 45     # 1999 peak was 44.2
    tr_cagr = at.metrics.cagr(df["tr"].pct_change().dropna())
    assert 0.08 < tr_cagr < 0.11     # ~9-10% nominal total return, per the literature


# --- Bonds, news ---------------------------------------------------------------------

def test_bond_constant_yield_earns_yield():
    idx = pd.date_range("2000-01-01", periods=121, freq="MS")
    tr = at.data.bond_total_return(pd.Series(0.05, index=idx))
    cagr = at.metrics.cagr(tr.pct_change().dropna())
    assert abs(cagr - 0.0506) < 0.001          # 5% semi-annual = 5.06% effective


def test_bond_loses_when_rates_jump():
    idx = pd.date_range("2000-01-01", periods=3, freq="MS")
    tr = at.data.bond_total_return(pd.Series([0.04, 0.04, 0.05], index=idx))
    r = tr.pct_change().iloc[-1]
    assert -0.085 < r < -0.07                  # ~ -duration (8.2) * 1% + carry


def test_lexicon_scorer_polarity():
    pytest.importorskip("pysentiment2")
    s = at.news.LexiconScorer().score(["Profits surge to record gains",
                                       "Bank faces fraud lawsuit and heavy losses",
                                       "The meeting is on Tuesday"])
    assert s[0] > 0 > s[1] and s[2] == 0


def test_sentiment_strategy_maps_news_to_trading_days():
    days = pd.bdate_range("2020-01-01", periods=400)
    news_days = pd.date_range("2020-01-01", periods=560, freq="D")    # includes weekends
    sent = pd.Series(np.sin(np.arange(560) / 20), index=news_days)
    px = pd.Series(100.0, index=days)
    strat = at.news.SentimentStrategy(sent, smooth=1, baseline=60)
    pos = strat.positions(px)
    assert pos.index.equals(days) and set(pos.unique()) <= {0.0, 1.0}
    # Future news must not change past positions.
    sent2 = sent.copy()
    sent2.iloc[300:] = -sent2.iloc[300:]
    pos2 = at.news.SentimentStrategy(sent2, smooth=1, baseline=60).positions(px)
    cutoff = news_days[300]
    assert pos[:cutoff - pd.Timedelta(days=1)].equals(pos2[:cutoff - pd.Timedelta(days=1)])


class _FakeClaude:
    """Stands in for anthropic.Anthropic(): returns one score per numbered line."""

    def __init__(self, wrong_count=False):
        self.wrong_count, self.calls = wrong_count, 0
        self.beta = self
        self.messages = self

    def create(self, **kw):
        import json
        from types import SimpleNamespace
        self.calls += 1
        n = kw["messages"][0]["content"].count("\n") + 1 + (1 if self.wrong_count else 0)
        text = json.dumps({"scores": [0.5] * (n - 1) + [-3.0]})
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=text)])


def test_claude_scorer_batches_and_clips():
    fake = _FakeClaude()
    s = at.news.ClaudeScorer(client=fake, batch_size=2).score(["a", "b", "c"])
    assert fake.calls == 2 and list(s) == [0.5, -1.0, -1.0]
    with pytest.raises(ValueError):
        at.news.ClaudeScorer(client=_FakeClaude(wrong_count=True)).score(["a"])


def test_fetch_rss_parses_rss_and_atom(tmp_path):
    rss = tmp_path / "f.xml"
    rss.write_text("""<rss><channel><item><title> Fed holds rates </title>
        <pubDate>Wed, 18 Sep 2024 18:00:00 GMT</pubDate><link>http://x</link></item>
        </channel></rss>""")
    atom = tmp_path / "a.xml"
    atom.write_text("""<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>8-K filed</title>
        <updated>2024-09-18T20:00:00Z</updated><link href="http://y"/></entry></feed>""")
    a = at.news.fetch_rss(rss.as_uri())
    b = at.news.fetch_rss(atom.as_uri())
    assert a.loc[0, "title"] == "Fed holds rates" and a.loc[0, "published"].hour == 18
    assert b.loc[0, "title"] == "8-K filed" and b.loc[0, "link"] == "http://y"


# --- Statistics ----------------------------------------------------------------------

def test_variance_ratio_size_and_power():
    rng = np.random.default_rng(0)
    zs = [at.stats.variance_ratio(rng.standard_normal(1000), 4)[1] for _ in range(300)]
    assert 0.01 < np.mean(np.abs(zs) > 1.96) < 0.10          # ~5% false alarms
    e = rng.standard_normal(5000)
    y = np.zeros(5000)
    for t in range(1, 5000):
        y[t] = 0.3 * y[t - 1] + e[t]
    vr, z = at.stats.variance_ratio(y, 2)
    assert abs(vr - 1.3) < 0.05 and z > 5                       # VR(2) = 1 + rho_1


def test_newey_west_wider_than_ols_on_overlapping_data():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.standard_normal(2000))
    y = r.rolling(60).sum().shift(-60)                           # overlapping 60-period sums
    # The correction matters when the predictor is persistent (like a valuation ratio).
    xs = np.zeros(2000)
    for t in range(1, 2000):
        xs[t] = 0.98 * xs[t - 1] + rng.standard_normal()
    x = pd.DataFrame({"x": xs})
    nw = at.stats.ols_newey_west(y, x, lags=60)
    plain = at.stats.ols_newey_west(y, x, lags=0)
    assert nw.loc["x", "se_nw"] > 3 * plain.loc["x", "se_nw"]


def test_oos_r2_negative_for_noise_predictor():
    rng = np.random.default_rng(2)
    y = pd.Series(rng.standard_normal(800) * 0.04 + 0.005)
    x = pd.Series(rng.standard_normal(800))
    r2, _ = at.stats.oos_r2(y, x, horizon=1, min_train=200)
    assert r2 < 0.01


def test_kelly_fraction():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.01, 0.05, 200000))
    assert abs(at.stats.kelly_fraction(r) - 4.0) < 0.2         # mu / sigma^2 = 0.01/0.0025


def test_borrow_spread_charged_only_on_leverage(prices):
    class Lev(at.Strategy):
        def __init__(self, f):
            self.f = f

        def positions(self, p):
            return pd.Series(self.f, index=p.index)

    for f, extra in [(0.5, 0.0), (2.0, 0.02 / 252)]:
        a = at.run_backtest(prices, Lev(f), cost_bps=0, cash_rate=0.03)
        b = at.run_backtest(prices, Lev(f), cost_bps=0, cash_rate=0.03, borrow_spread=0.02)
        assert np.allclose((a.returns - b.returns).iloc[1:], extra)


# --- Portfolio building blocks --------------------------------------------------------

@pytest.fixture
def two_assets():
    return at.data.panel(at.data.regime_switching, n_assets=2, seed=5, n=600).set_axis(
        ["stocks", "bonds"], axis=1)


def test_trend_filter_zeroes_weight_when_out(two_assets):
    core = at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4})
    w = at.TrendFilter(core, at.PriceAboveSMA(20)).weights(two_assets)
    sma_on = at.PriceAboveSMA(20).positions(two_assets["stocks"])
    assert np.allclose(w["stocks"], 0.6 * sma_on)


def test_average_signal_is_mean_of_parts(prices):
    parts = [at.PriceAboveSMA(w) for w in (10, 20, 40)]
    avg = at.AverageSignal(parts).positions(prices)
    assert np.allclose(avg, sum(p.positions(prices) for p in parts) / 3)
    assert at.validation.check_no_lookahead(at.AverageSignal(parts), prices)


def test_value_tilt_bounds(two_assets):
    sig = pd.Series(np.linspace(0, 1, len(two_assets)), index=two_assets.index)
    w = at.ValueTilt(sig, lo=0.3, hi=0.9).weights(two_assets)
    assert np.isclose(w["stocks"].iloc[0], 0.3) and np.isclose(w["stocks"].iloc[-1], 0.9)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_inverse_vol_favours_calmer_asset():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    px = pd.DataFrame({"calm": 100 * np.cumprod(1 + rng.normal(0, 0.005, 300)),
                       "wild": 100 * np.cumprod(1 + rng.normal(0, 0.02, 300))}, index=idx)
    w = at.InverseVol(60).weights(px).iloc[-1]
    assert np.isclose(w.sum(), 1.0) and w["calm"] > 3 * w["wild"]


def test_levered_scales_weights(two_assets):
    core = at.PerAsset(at.BuyAndHold(), {"stocks": 0.6, "bonds": 0.4})
    assert np.allclose(at.Levered(core, 1.5).weights(two_assets), 1.5 * core.weights(two_assets))


def test_monthly_and_banded_sma_are_causal_and_banding_cuts_trades():
    px = at.data.gbm(2520, seed=4)["close"]
    for s in (at.MonthlySMA(10, "end"), at.MonthlySMA(10, "avg"), at.BandedSMA(200, 0.02)):
        assert at.validation.check_no_lookahead(s, px)
    plain = at.PriceAboveSMA(200).positions(px).diff().abs().sum()
    banded = at.BandedSMA(200, 0.02).positions(px).diff().abs().sum()
    monthly = at.MonthlySMA(10, "end").positions(px)
    assert banded < plain
    # Monthly signal only changes on a month's last trading day.
    changes = monthly.index[monthly.diff().fillna(0) != 0]
    month_ends = px.groupby(px.index.to_period("M")).apply(lambda s: s.index[-1])
    assert set(changes) <= set(month_ends)


# --- Broker / rebalancing ---------------------------------------------------------------

from algotrading.broker import Limits, Order, PaperBroker, plan_rebalance  # noqa: E402


def test_paper_rebalance_reaches_targets_and_is_idempotent():
    b = PaperBroker(cash=100_000, prices={"SPY": 500.0, "IEF": 100.0}, cost_bps=0)
    at.broker.rebalance(b, {"SPY": 0.6, "IEF": 0.4}, execute=True)
    w = {s: q * b.prices[s] / b.equity() for s, q in b.positions().items()}
    assert abs(w["SPY"] - 0.6) < 1e-9 and abs(w["IEF"] - 0.4) < 1e-9
    assert at.broker.rebalance(b, {"SPY": 0.6, "IEF": 0.4}, execute=True) == []


def test_plan_sells_first_and_exits_dropped_symbols():
    orders = plan_rebalance({"SPY": 0.5}, 100_000, {"SPY": 50, "IEF": 300},
                            {"SPY": 500.0, "IEF": 100.0}, Limits(max_order_frac=1.0))
    assert orders[0] == Order("IEF", 300, "sell")
    assert orders[1].side == "buy" and orders[1].symbol == "SPY"


def test_plan_safety_limits():
    px = {"SPY": 500.0}
    with pytest.raises(ValueError, match="leverage"):
        plan_rebalance({"SPY": 2.0}, 100_000, {}, px, Limits(max_gross_leverage=1.5))
    with pytest.raises(ValueError, match="fat-finger"):
        plan_rebalance({"SPY": 1.0}, 100_000, {}, px, Limits(max_order_frac=0.5))
    with pytest.raises(ValueError, match="short"):
        plan_rebalance({"SPY": -0.5}, 100_000, {}, px)
    # Small drift is ignored.
    assert plan_rebalance({"SPY": 0.6}, 100_000, {"SPY": 119}, px) == []


def test_dry_run_sends_nothing():
    b = PaperBroker(cash=10_000, prices={"SPY": 500.0})
    out = at.broker.rebalance(b, {"SPY": 0.5}, execute=False)
    assert out and all(oid == "DRY-RUN" for _, oid in out) and b.positions() == {}


def test_frozen_rule_ignores_partial_month():
    days = pd.bdate_range("2024-01-01", "2025-03-14")
    up = pd.Series(np.linspace(100, 200, len(days)), index=days)
    closes = pd.DataFrame({"SPY": up, "IEF": up})
    closes.loc["2025-03-03":, "SPY"] = 50.0          # crash inside the unfinished month
    w = at.broker.frozen_rule_weights(closes, "SPY", "IEF", asof=pd.Timestamp("2025-03-14"))
    assert w == {"SPY": 0.6, "IEF": 0.4}            # decided on February's close
    w_end = at.broker.frozen_rule_weights(closes, "SPY", "IEF", asof=pd.Timestamp("2025-04-01"))
    assert w_end["SPY"] == 0.0                       # March completed: now out


def test_alpaca_adapter_builds_correct_requests():
    pytest.importorskip("alpaca")
    from types import SimpleNamespace
    from alpaca.trading.enums import OrderSide, TimeInForce
    sent = []
    fake = SimpleNamespace(
        submit_order=lambda req: sent.append(req) or SimpleNamespace(id="abc"),
        get_account=lambda: SimpleNamespace(equity="1000", trading_blocked=False, status="ACTIVE"),
        get_all_positions=lambda: [SimpleNamespace(symbol="SPY", qty="2.5")],
        get_clock=lambda: SimpleNamespace(is_open=False))
    b = at.broker.AlpacaBroker.__new__(at.broker.AlpacaBroker)
    b.trading = fake
    assert b.submit(Order("SPY", 1.23456, "sell")) == "abc"
    assert sent[0].symbol == "SPY" and sent[0].qty == 1.2346
    assert sent[0].side == OrderSide.SELL and sent[0].time_in_force == TimeInForce.DAY
    assert b.equity() == 1000.0 and b.positions() == {"SPY": 2.5} and b.account_ok()
    with pytest.raises(RuntimeError, match="closed"):
        at.broker.rebalance(b, {"SPY": 0.5}, execute=True)


def test_kelly_portfolio_levers_calm_asset_and_respects_cap():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    px = pd.DataFrame({"stocks": 100 * np.cumprod(1 + rng.normal(0.006, 0.045, 120)),
                       "bonds": 100 * np.cumprod(1 + rng.normal(0.003, 0.015, 120))}, index=idx)
    mu = pd.DataFrame({"stocks": 0.05, "bonds": 0.02}, index=idx)
    w = at.KellyPortfolio(mu, fraction=0.5, cap=2.0).weights(px).iloc[-1]
    assert w["bonds"] > w["stocks"] and abs(w.sum() - 2.0) < 1e-9
    floored = at.KellyPortfolio(mu, 0.5, cap=10, vol_floor={"bonds": 0.10}).weights(px).iloc[-1]
    assert floored["bonds"] < at.KellyPortfolio(mu, 0.5, cap=10).weights(px).iloc[-1]["bonds"]


# --- Improvement loop and paper ledger -------------------------------------------------

def test_crash_stress_uses_exposure_in_crash_month():
    idx = pd.date_range("1987-01-01", periods=12, freq="MS")
    pos = pd.DataFrame({"stocks": 2.0, "bonds": 0.0}, index=idx)
    assert np.isclose(at.research.crash_stress(pos), 2.0 * -0.205)
    pos.loc["1987-10", "stocks"] = 0.0                 # out of the market that month
    assert at.research.crash_stress(pos) == 0.0


def test_leaderboard_replaces_same_candidate(tmp_path):
    lb = at.research.Leaderboard(str(tmp_path / "lb.csv"))
    row = {"name": "x", "financing": 0.005, "test_cagr": 0.1, "ruin": False}
    lb.add(row, 1)
    lb.add({**row, "test_cagr": 0.2}, 2)
    assert len(lb.df) == 1 and lb.df.iloc[0]["test_cagr"] == 0.2
    lb.save()
    assert len(at.research.Leaderboard(str(tmp_path / "lb.csv")).df) == 1


def test_paper_account_moves_forward_with_interest(tmp_path):
    acct = at.paper.PaperAccount.load_or_create(
        str(tmp_path / "a.json"), name="t", description="", symbols={"stocks": "SPY", "bonds": "IEF"},
        start="2026-01-01", cash=100_000.0)
    m1, m2 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")
    acct.step(m1, {"SPY": 100.0, "IEF": 50.0}, {"stocks": 2.0, "bonds": 0.0}, 0.03, 0.01, cost_bps=0)
    assert np.isclose(acct.state.holdings["SPY"], 2000) and np.isclose(acct.state.cash, -100_000)
    acct.step(m2, {"SPY": 100.0, "IEF": 50.0}, {"stocks": 2.0, "bonds": 0.0}, 0.03, 0.01, cost_bps=0)
    # one month of borrowing at cash + spread = 4%/12 on 100k
    assert np.isclose(acct.state.equity_history[-1]["equity"], 100_000 - 100_000 * 0.04 / 12, atol=0.01)
    acct.save()
    again = at.paper.PaperAccount.load_or_create(str(tmp_path / "a.json"))
    assert again.state.last_month == "2026-02" and len(again.state.equity_history) == 2


def test_recovery_leverage_boosts_only_after_deep_fall_in_uptrend():
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    p = np.r_[np.linspace(100, 150, 24), np.linspace(150, 90, 12), np.linspace(90, 130, 24)]
    px = pd.DataFrame({"stocks": p, "bonds": 100.0}, index=idx)
    w = at.RecoveryLeverage(2.0, 3.0, 0.15).weights(px)["stocks"]
    assert w.iloc[23] == 2.0                      # uptrend, at the high
    assert w.iloc[30] == 0.0                      # falling: below its 10-month average
    rec = w.iloc[36:50]
    assert (rec == 3.0).any()                     # recovering while still >15% below the high
    assert w.iloc[-1] == 2.0                      # back near the high
    assert at.validation.check_no_lookahead(
        type("S", (at.Strategy,), {"positions": lambda self, s: at.RecoveryLeverage().weights(
            pd.DataFrame({"stocks": s, "bonds": s}))["stocks"]})(),
        at.data.gbm(600, seed=3)["close"])


def test_recovery_either_signal_and_bonds_when_out():
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    p = np.r_[np.linspace(100, 150, 24), np.linspace(150, 90, 12), np.linspace(90, 130, 24)]
    bonds = np.r_[np.linspace(100, 120, 40), np.linspace(120, 100, 20)]
    px = pd.DataFrame({"stocks": p, "bonds": bonds}, index=idx)
    base = at.RecoveryLeverage(2.0, 3.0, 0.15).weights(px)
    either = at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0).weights(px)
    both = at.RecoveryLeverage(2.0, 3.0, 0.15, signal="both").weights(px)
    # "either" is in whenever the champion is in; "both" only when the champion is in
    assert ((base["stocks"] > 0) <= (either["stocks"] > 0)).all()
    assert ((both["stocks"] > 0) <= (base["stocks"] > 0)).all()
    # never stocks and bonds together; bonds exactly when out of stocks (after warm-up)
    assert ((either["stocks"] > 0) & (either["bonds"] > 0)).sum() == 0
    out = either["stocks"].iloc[12:] == 0
    assert (either["bonds"].iloc[12:][out] == 2.0).all()
    gated = at.RecoveryLeverage(2.0, 3.0, 0.15, signal="either", off_lev=2.0, off_gate=True).weights(px)
    assert (gated["bonds"] <= either["bonds"]).all()
    assert at.validation.check_no_lookahead(
        type("S", (at.Strategy,), {"positions": lambda self, s: at.RecoveryLeverage(
            signal="either", off_lev=1.0).weights(pd.DataFrame({"stocks": s, "bonds": s}))["stocks"]})(),
        at.data.gbm(600, seed=4)["close"])


# --- Execution / full-cost simulation ---------------------------------------------------

def test_ledger_charges_every_fee_and_realizes_pl():
    from algotrading.execution import CostModel, Ledger
    led = Ledger(10_000.0, CostModel(commission_per_order=1.0, extra_slippage_bps=0.0))
    led.trade("2024-01-01", "SPY", 10, 500.0, spread_bps=10, reason="buy")
    assert np.isclose(led.cash, 10_000 - 5000 - 5.0 - 1.0)          # amount + 10bp spread + $1
    led.trade("2024-02-01", "SPY", -10, 600.0, spread_bps=10, reason="sell")
    reg = 6000 * 27.80 / 1e6 + 10 * 0.000166
    t = led.trades[-1]
    assert np.isclose(t["realized_pl"], (6000 - 6.0 - 1.0 - reg) - 5006.0, atol=0.01)
    assert led.shares == {} and np.isclose(led.cash, 10_000 + t["realized_pl"], atol=0.01)
    assert np.isclose(sum(led.fees.values()), 1 + 5 + 1 + 6 + reg)


def test_leveraged_etf_prices_include_costs_and_drag():
    from algotrading.execution import CostModel, Instrument, etf_prices
    idx = pd.date_range("2000-01-01", periods=121, freq="MS")
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"stocks": rng.normal(0.008, 0.045, 121)}, index=idx)
    r.iloc[0] = np.nan
    cash = pd.Series(0.02, index=idx)
    inst = {"X1": Instrument("X1", "stocks", 1.0, 0.0), "X2": Instrument("X2", "stocks", 2.0, 0.009)}
    p = etf_prices(r, cash, inst, CostModel())
    g = p.pct_change().dropna()
    naive = 2 * g["X1"]
    assert (g["X2"] < naive).all()                       # financing + fees + drag every month


def test_simulation_buy_and_hold_matches_prices_minus_costs():
    from algotrading.execution import simulate, INSTRUMENTS
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    px = pd.DataFrame({"SPY": np.linspace(100, 150, 24), "SSO": 1.0, "UPRO": 1.0, "IEF": 1.0}, index=idx)
    res = simulate(lambda d: ({"SPY": 1.0}, "hold"), px, pd.Series(0.0, index=idx), 10_000.0)
    assert len(res["trades"]) == 1                        # one buy, never rebalanced
    # 100 shares bought; the $1 + 3bp fees ($4) are paid from cash (a small debit with interest).
    expected = 100 * 150 - 4.0
    assert abs(res["final"] - expected) < 0.1


def test_walk_forward_selection_uses_only_past_years():
    from algotrading.research import beats_everywhere, walk_forward_selection
    idx = pd.date_range("1950-01-01", "1960-12-01", freq="MS")
    # "a" wins until 1957, "b" wins from 1957: the selector can only switch AFTER seeing it
    a = pd.Series(np.where(idx.year < 1957, 0.02, -0.01), index=idx)
    b = pd.Series(np.where(idx.year < 1957, 0.00, 0.03), index=idx)
    wf = walk_forward_selection(pd.DataFrame({"a": a, "b": b}), start=1957)
    assert (wf["1957"] == -0.01).all()            # 1957 picked with pre-1957 data only
    g = beats_everywhere(b, a)
    assert not g["passes"]


def test_etf_weights_reproduce_exposure_without_margin():
    from algotrading.execution import INSTRUMENTS, etf_weights
    for st, bd in [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (2.4, 0.67), (0.0, 2.0), (1.5, 0.5)]:
        w = etf_weights(st, bd)
        stocks = sum(v * INSTRUMENTS[k].leverage for k, v in w.items() if INSTRUMENTS[k].underlying == "stocks")
        bonds = sum(v * INSTRUMENTS[k].leverage for k, v in w.items() if INSTRUMENTS[k].underlying == "bonds")
        assert abs(stocks - st) < 1e-5 and abs(bonds - bd) < 1e-5
        if st + bd <= 3 * 1.0 and st <= 3 * (1 - min(bd, 1)):
            assert sum(w.values()) <= 1.0 + 1e-9          # fully funded: no margin loan


def test_splice_monthly_uses_live_returns_after_anchor():
    from algotrading.live import completed_month_ends, splice_monthly
    hist = pd.DataFrame({"stocks": np.arange(1.0, 25.0), "bonds": 1.0},
                        index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    days = pd.bdate_range("2021-06-01", "2022-03-15")
    live = pd.DataFrame({"stocks": np.linspace(100, 200, len(days)), "bonds": 50.0}, index=days)
    me = completed_month_ends(live, asof=pd.Timestamp("2022-03-15"))
    assert me.index[-1] == pd.Timestamp("2022-02-28")             # March still running: dropped
    out = splice_monthly(hist, me)
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    assert out.loc["2021-05-01", "stocks"] == hist.loc["2021-05-01", "stocks"]   # untouched before
    assert out.loc["2021-06-01", "stocks"] == hist.loc["2021-06-01", "stocks"]   # anchor
    r_live = me["stocks"].iloc[-1] / me["stocks"].iloc[-2] - 1
    assert abs(out["stocks"].iloc[-1] / out["stocks"].iloc[-2] - 1 - r_live) < 1e-12


def test_fred_proxies_from_given_series():
    idx = pd.bdate_range("2024-01-01", periods=300)
    sp = pd.Series(np.linspace(4000, 5000, 300), index=idx)
    y = pd.Series(4.0, index=idx)                      # flat 4% yield: bonds earn carry only
    px = at.data.fred_daily_proxies(sp500=sp, dgs10=y)
    assert px["stocks"].iloc[-1] / px["stocks"].iloc[0] > 5000 / 4000  # dividends added
    carry = px["bonds"].iloc[-1] / px["bonds"].iloc[0] - 1
    assert abs(carry - ((1 + 0.04 / 252) ** 299 - 1)) < 1e-9


def test_live_sim_moves_forward_only_and_rebalances_monthly(tmp_path):
    from algotrading.execution import CostModel
    from algotrading.live_sim import LiveSimAccount
    days = pd.bdate_range("2026-01-05", "2026-03-31")
    px = pd.DataFrame({"SPY": np.linspace(600, 660, len(days)), "UPRO": 100.0, "IEF": 95.0, "UST": 45.0},
                      index=days)
    cash = pd.Series(0.04, index=days)
    calls = []

    def target(d):
        calls.append(d)
        return {"SPY": 1.0}, {"stocks": 1.0, "bonds": 0.0}, "test"

    path = str(tmp_path / "a.json")
    acct = LiveSimAccount.load_or_create(path, "a", "A", "", "2026-01-05")
    assert acct.advance(px[:"2026-02-10"], cash, target, CostModel()) == len(px[:"2026-02-10"])
    acct.save()
    assert [d.strftime("%Y-%m") for d in calls] == ["2026-01", "2026-02"]   # once per month
    again = LiveSimAccount.load_or_create(path, "a", "A", "", "2026-01-05")
    assert again.advance(px[:"2026-02-10"], cash, target, CostModel()) == 0  # nothing new
    assert again.advance(px, cash, target, CostModel()) == len(px["2026-02-11":])
    eq = pd.Series({h["date"]: h["equity"] for h in again.st["equity_history"]})
    assert eq.index.is_unique and eq.iloc[-1] > 100_000 * 1.05            # followed SPY up
    assert again.st["trades"][0]["commission"] == 1.0                      # costs charged


def test_live_sim_daily_rule_trades_mid_month(tmp_path):
    from algotrading.execution import CostModel
    from algotrading.live_sim import LiveSimAccount
    days = pd.bdate_range("2026-01-05", "2026-01-30")
    px = pd.DataFrame({"SPY": 600.0, "UPRO": 100.0, "IEF": 95.0, "UST": 45.0}, index=days)
    flip = pd.Timestamp("2026-01-15")

    def target(d):
        return ({"UPRO": 1.0}, {"stocks": 3.0, "bonds": 0.0}, "in") if d < flip else ({}, {"stocks": 0.0, "bonds": 0.0}, "out")

    acct = LiveSimAccount.load_or_create(str(tmp_path / "d.json"), "d", "D", "", "2026-01-05")
    acct.advance(px, pd.Series(0.04, index=days), target, CostModel(), daily=True)
    sides = [(t["date"], t["side"]) for t in acct.st["trades"]]
    assert sides == [("2026-01-05", "buy"), ("2026-01-15", "sell")]


def test_simulate_charges_capital_gains_tax_on_realized_gains():
    from algotrading.execution import CostModel, simulate
    idx = pd.date_range("2020-01-01", periods=36, freq="MS")
    px = pd.DataFrame({"SPY": np.linspace(100, 200, 36), "SSO": 100.0, "UPRO": 100.0, "IEF": 100.0, "UST": 100.0},
                      index=idx)
    cash = pd.Series(0.0, index=idx)
    free = CostModel(commission_per_order=0, extra_slippage_bps=0, sec_fee_rate=0, finra_taf_per_share=0,
                     cash_fee=0)
    hold = lambda d: ({"SPY": 1.0}, "hold")                       # noqa: E731
    flip = lambda d: ({"SPY": 1.0} if d.month != 12 else {}, "flip")  # noqa: E731
    h = simulate(hold, px, cash, 100_000, free, tax_rate=0.25)
    f = simulate(flip, px, cash, 100_000, free, tax_rate=0.25)
    assert h["fees"]["taxes_paid"] == 0.0                         # buy & hold defers all tax
    assert f["fees"]["taxes_paid"] > 0                            # selling each December realizes gains
    assert h["liquidation_after_tax"] < h["final"]                # tax due when finally sold
