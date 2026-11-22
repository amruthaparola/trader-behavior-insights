"""Metric tests on hand-built series whose answers can be worked out on paper."""

import numpy as np
import pandas as pd
import pytest

from sentiment_quant import metrics as m


def test_sharpe_on_known_series():
    x = np.array([1.0, -1.0, 1.0, -1.0, 3.0])  # mean 0.6, sample std sqrt(2.8)
    assert m.sharpe(x, periods=1) == pytest.approx(0.6 / np.sqrt(2.8))
    assert m.sharpe(x) == pytest.approx(np.sqrt(365) * 0.6 / np.sqrt(2.8))


def test_sharpe_is_scale_invariant():
    x = np.random.default_rng(0).normal(0.1, 1, 500)
    assert m.sharpe(1000 * x) == pytest.approx(m.sharpe(x))


def test_max_drawdown_hand_path():
    # equity: 0 -> 10 -> 4 -> 12 -> 5 -> 7  ; worst fall is 12 -> 5 = 7
    assert m.max_drawdown([10, -6, 8, -7, 2]) == pytest.approx(7)
    # an immediate loss from the starting equity of 0 counts
    assert m.max_drawdown([-3, 1]) == pytest.approx(3)
    assert m.max_drawdown([1, 2, 3]) == 0


def test_sortino_infinite_without_losses_and_ge_sharpe_for_right_skew():
    assert m.sortino([1.0, 2.0, 0.5]) == np.inf
    x = np.array([-1.0] * 50 + [5.0] * 10 + [0.5] * 40)  # right-skewed
    assert m.sortino(x) > m.sharpe(x)


def test_cvar_is_at_least_var():
    x = np.random.default_rng(1).standard_t(3, 10_000)
    var, cvar = m.var_cvar(x, 0.95)
    assert cvar >= var > 0


def test_psr_is_one_half_at_zero_sharpe_and_rises_with_n():
    x = np.array([1.0, -1.0] * 50)  # exact zero mean
    assert m.probabilistic_sharpe(x) == pytest.approx(0.5)
    rng = np.random.default_rng(2)
    short, long = rng.normal(0.05, 1, 100), rng.normal(0.05, 1, 10_000)
    assert m.probabilistic_sharpe(long) > m.probabilistic_sharpe(short)


def test_sharpe_se_close_to_normal_theory():
    # For normal i.i.d. data Var(SR) ~ (1 + SR^2/2)/n  (per period)
    x = np.random.default_rng(3).normal(0.02, 1, 50_000)
    sr = x.mean() / x.std(ddof=1)
    expected = np.sqrt((1 + sr**2 / 2) / (len(x) - 1))
    assert m.sharpe_standard_error(x, periods=1) == pytest.approx(expected, rel=0.02)


def test_trade_summary_identities():
    t = m.trade_summary([10, 10, 10, -20, 0])
    assert t["hit_rate"] == pytest.approx(0.6)
    assert t["payoff_ratio"] == pytest.approx(0.5)
    assert t["profit_factor"] == pytest.approx(1.5)
    assert t["expectancy"] == pytest.approx(2.0)
    assert t["breakeven_hit_rate"] == pytest.approx(1 / 1.5)


def test_calendar_fills_inactive_days_with_zero():
    ad = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-04"]), "net_pnl": [5.0, -2.0]})
    s = m.calendar_daily_pnl(ad)
    assert len(s) == 4 and s.loc["2024-01-02"] == 0
