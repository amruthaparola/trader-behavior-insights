"""Model tests on simulated data where the true coefficient is known."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from sentiment_quant import config, models


def fake_market_day(beta: float, n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Daily series where ret_on_volume = beta * score_lag1/10 (in bps) + noise."""
    rng = np.random.default_rng(seed)
    score = np.clip(50 + np.cumsum(rng.normal(0, 4, n)), 1, 99)  # persistent, like F&G
    labels = pd.cut(score, [0, 25, 45, 55, 75, 100], labels=config.SENTIMENT_ORDER)
    ret_bps = beta * score / 10 + rng.normal(0, 20, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "ret_on_volume": ret_bps / 1e4,
            "sentiment_score_lag1": score,
            "sentiment_lag1": labels,
            "sentiment_lag2": pd.Series(labels).shift(1).bfill().to_numpy(),
        }
    )


def test_newey_west_lag_rule():
    assert models.newey_west_lags(100) == 4
    assert models.newey_west_lags(476) == 5


def test_winsorize_clips_only_the_tails():
    s = pd.Series(np.arange(1, 101, dtype=float))
    w = models.winsorize(s)
    assert w.min() > 1 and w.max() < 100 and (w.iloc[10:90] == s.iloc[10:90]).all()


def test_time_series_recovers_true_beta():
    res = models.time_series_models(fake_market_day(beta=5.0), winsor=False)
    a = res.loc["A: score"]
    assert a["coef"] == pytest.approx(5.0, abs=3 * a["se_hac"])
    assert a["p_hac"] < 0.01


def test_hac_se_larger_than_ols_for_autocorrelated_errors():
    rng = np.random.default_rng(1)
    n = 500
    x = np.cumsum(rng.normal(size=n))  # persistent regressor
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + rng.normal()  # AR(1) errors
    d = pd.DataFrame({"y": e, "x": x})
    ols = smf.ols("y ~ x", d).fit()
    hac = smf.ols("y ~ x", d).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    assert hac.bse["x"] > ols.bse["x"]


def test_oos_r2_positive_with_signal_negative_without():
    assert models.out_of_sample_r2(fake_market_day(beta=8.0))["r2_oos"] > 0
    # With beta = 0 the extra parameter only adds estimation noise.
    assert models.out_of_sample_r2(fake_market_day(beta=0.0, seed=4))["r2_oos"] < 0.01


def test_within_transformation_equals_dummy_fixed_effects():
    # Frisch-Waugh-Lovell: demeaning by group gives the same slope as group dummies.
    rng = np.random.default_rng(5)
    g = np.repeat(np.arange(10), 30)
    x = rng.normal(size=300) + g * 0.5
    y = 2.0 * x + g * 3.0 + rng.normal(size=300)
    d = pd.DataFrame({"y": y, "x": x, "g": g})
    dummies = smf.ols("y ~ x + C(g)", d).fit().params["x"]
    d["y_w"] = d.y - d.groupby("g").y.transform("mean")
    d["x_w"] = d.x - d.groupby("g").x.transform("mean")
    within = smf.ols("y_w ~ x_w", d).fit().params["x_w"]
    assert within == pytest.approx(dummies)
