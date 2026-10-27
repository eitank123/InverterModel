"""Point-in-time features and walk-forward forecasting for return prediction.

The rule that matters more than the model: a forecast made at time t may only be
trained on (feature, outcome) pairs whose outcome was already known at t. With monthly
data and ``execution_lag=1`` a weight decided at t earns the return of bar t+2, so the
target is shifted by ``horizon=2`` and training stops ``horizon`` bars before t.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def macro_features(df: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Monthly features from the Shiller panel, each lagged to when it was publishable.

    Prices/yields: known at month end (lag 0). CPI: ~1 month publication lag (use 2).
    Earnings/CAPE: ~3-6 months (use 6).
    """
    s, b = px["stocks"], px["bonds"]
    infl = (df["cpi"] / df["cpi"].shift(12) - 1).shift(2)
    infl10 = (df["cpi"] / df["cpi"].shift(120)) ** (1 / 10) - 1
    f = pd.DataFrame(index=px.index)
    f["stk_gap_sma10"] = s / s.rolling(10).mean() - 1
    f["stk_ret_12m"] = s.pct_change(12)
    f["stk_ret_1m"] = s.pct_change()
    f["stk_vol_12m"] = s.pct_change().rolling(12).std() * np.sqrt(12)
    f["bnd_gap_sma10"] = b / b.rolling(10).mean() - 1
    f["bnd_ret_12m"] = b.pct_change(12)
    f["yield"] = df["long_rate"]
    f["yield_chg_12m"] = df["long_rate"] - df["long_rate"].shift(12)
    f["inflation_12m"] = infl
    f["real_yield"] = df["long_rate"] - infl10.shift(2)
    f["log_cape"] = np.log(df["cape"].shift(6)).ffill()
    f["ecy"] = (1 / df["cape"].shift(6) - (df["long_rate"] - infl10).shift(6)).ffill()
    return f


def walk_forward(model_factory: Callable[[], object], X: pd.DataFrame, y: pd.Series,
                 start: str, horizon: int = 2, min_train: int = 120,
                 refit_every: int = 12, train_from: str | None = None) -> pd.Series:
    """Expanding-window forecasts of ``y`` (already aligned to the decision date).

    At decision date t the model is fit on rows with index <= t - horizon (outcomes
    known), refit every ``refit_every`` bars, features standardised on the training set.
    Returns a Series of forecasts indexed by decision date (NaN before ``start``).
    """
    data = pd.concat([X, y.rename("_y")], axis=1)
    if train_from:
        data = data[train_from:]
    feats = data.drop(columns="_y")
    preds = pd.Series(np.nan, index=X.index)
    dates = feats.index[feats.index >= pd.Timestamp(start)]
    model, mu, sd, last_fit = None, None, None, -10**9
    for i, t in enumerate(dates):
        pos = feats.index.get_loc(t)
        if model is None or i - last_fit >= refit_every:
            train = data.iloc[: max(pos - horizon + 1, 0)].dropna()
            if len(train) < min_train:
                continue
            Xt = train.drop(columns="_y")
            mu, sd = Xt.mean(), Xt.std().replace(0, 1)
            model = model_factory()
            model.fit(((Xt - mu) / sd).to_numpy(), train["_y"].to_numpy())
            last_fit = i
        row = feats.loc[[t]]
        if row.isna().any(axis=1).iloc[0]:
            continue
        preds[t] = float(model.predict(((row - mu) / sd).to_numpy())[0])
    return preds


class EnsembleMLP:
    """Average of several small, heavily regularised sklearn MLPs (different seeds).

    The target is standardised internally: returns are tiny numbers (~0.005/month) with a
    huge noise-to-signal ratio, and an unscaled network happily extrapolates to absurd
    forecasts. Predictions are clipped to +/- ``clip_sd`` training standard deviations.
    """

    def __init__(self, hidden=(4,), alpha=1.0, seeds=(0, 1, 2, 3, 4), clip_sd=1.0):
        from sklearn.neural_network import MLPRegressor
        self.models = [MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=3000,
                                    learning_rate_init=1e-3, random_state=s) for s in seeds]
        self.clip_sd = clip_sd

    def fit(self, X, y):
        self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y)) or 1.0
        z = (np.asarray(y) - self.y_mu) / self.y_sd
        for m in self.models:
            m.fit(X, z)
        return self

    def predict(self, X):
        z = np.mean([m.predict(X) for m in self.models], axis=0).clip(-self.clip_sd, self.clip_sd)
        return self.y_mu + self.y_sd * z
