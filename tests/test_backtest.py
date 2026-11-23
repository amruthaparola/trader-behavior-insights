"""Backtest tests: timing (no look-ahead), cost accounting, and equity maths."""

import numpy as np
import pandas as pd
import pytest

from sentiment_quant import backtest as bt
from sentiment_quant import data


@pytest.fixture(scope="module")
def frame():
    return bt.load_frame()


def test_signal_uses_previous_days_sentiment(frame):
    sent = data.load_sentiment().set_index("date")["sentiment_score"]
    for _, row in frame.sample(25, random_state=0).iterrows():
        assert row["sentiment_score_lag1"] == sent.loc[row["date"] - pd.Timedelta(days=1)]


def test_return_is_close_to_close_of_same_day(frame):
    r = frame.set_index("date")
    d = r.index[100]
    prev_close = r["close"].iloc[99]
    assert r.loc[d, "ret"] == pytest.approx(r.loc[d, "close"] / prev_close - 1)


def test_buy_and_hold_equals_price_ratio(frame):
    strat = bt.run(np.ones(len(frame)), frame["ret"], cost_bps=0)
    first_close = frame["close"].iloc[0] / (1 + frame["ret"].iloc[0])
    assert np.prod(1 + strat) == pytest.approx(frame["close"].iloc[-1] / first_close)


def test_costs_charged_on_turnover_including_first_entry():
    pos = np.array([1, 1, 0, -1, -1])
    ret = np.zeros(5)
    out = bt.run(pos, ret, cost_bps=10)
    # turnover: 1 (enter), 0, 1 (exit), 1 (go short), 0
    assert out == pytest.approx(-1e-3 * np.array([1, 0, 1, 1, 0]))


def test_short_position_earns_negative_of_return():
    out = bt.run(np.array([-1.0]), np.array([0.05]), cost_bps=0)
    assert out[0] == pytest.approx(-0.05)


def test_max_drawdown_pct_hand_path():
    # equity 1 -> 1.5 -> 0.75 -> 0.9 ; worst fall 1.5 -> 0.75 = 50%
    assert bt.max_drawdown_pct([0.5, -0.5, 0.2]) == pytest.approx(0.5)


def test_perfect_foresight_is_caught_by_timing_test():
    # A rule that peeks at today's return is "too good"; the circular-shift
    # null should reject it decisively. (Sanity check of the test itself.)
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.03, 1000)
    cheat = (ret > 0).astype(float)
    assert bt.timing_test(cheat, ret, cost_bps=0)["p"] < 0.01


def test_random_timing_is_not_significant():
    rng = np.random.default_rng(1)
    ret = rng.normal(0.001, 0.03, 1500)
    pos = np.repeat(rng.integers(0, 2, 50), 30).astype(float)  # persistent, unrelated
    assert bt.timing_test(pos, ret, cost_bps=0)["p"] > 0.05
