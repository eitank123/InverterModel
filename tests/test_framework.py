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
    assert df.index[0].year == 1871 and df.notna().all().all()
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
