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
