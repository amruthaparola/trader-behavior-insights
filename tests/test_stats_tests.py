"""Unit tests for the statistical tools, on synthetic data where the answer is known."""

import numpy as np
import pytest
from scipy import stats

from sentiment_quant import stats_tests as st


def test_kruskal_h_matches_scipy_without_ties():
    rng = np.random.default_rng(0)
    y = rng.standard_t(df=3, size=300)  # fat-tailed, continuous => no ties
    codes = rng.integers(0, 5, size=300)
    expected = stats.kruskal(*[y[codes == i] for i in range(5)]).statistic
    assert st.kruskal_h(y, codes) == pytest.approx(expected)


def test_bootstrap_ratio_point_estimate_and_ci_cover_it():
    num = np.array([1.0, 2.0, 3.0, 4.0])
    den = np.array([2.0, 2.0, 2.0, 2.0])
    out = st.cluster_bootstrap_ratio(num, den, n_boot=2000)
    assert out["estimate"] == pytest.approx(10 / 8)
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"]


def test_cluster_se_exceeds_iid_se_when_clusters_are_homogeneous():
    # 50 days x 100 trades. Within a day every trade has the same outcome
    # (perfect intra-day correlation), so the data holds only ~50 independent
    # observations, not 5000. Cluster SE should be ~sqrt(100)=10x the iid SE.
    rng = np.random.default_rng(1)
    day_outcome = rng.integers(0, 2, size=50).astype(float)
    n_per_day = np.full(50, 100.0)
    wins = day_outcome * n_per_day
    p = wins.sum() / n_per_day.sum()
    se_iid = np.sqrt(p * (1 - p) / n_per_day.sum())
    se_cluster = st.cluster_bootstrap_ratio(wins, n_per_day, n_boot=4000)["se"]
    assert 7 < se_cluster / se_iid < 13


def test_permutation_p_is_large_under_null_and_small_under_alternative():
    rng = np.random.default_rng(2)
    codes = np.repeat(np.arange(4), 50)
    null_y = rng.normal(size=200)
    alt_y = null_y + 1.5 * (codes == 3)
    assert st.permutation_pvalue(null_y, codes, n_perm=2000)["p"] > 0.05
    assert st.permutation_pvalue(alt_y, codes, n_perm=2000)["p"] < 0.01


def test_circular_shift_is_less_fooled_by_persistent_series_than_iid_shuffle():
    # Two INDEPENDENT but highly persistent series: a regime label that comes in
    # long runs, and an AR(1) "PnL" series. There is no true relationship, yet
    # i.i.d. shuffling tends to report one, because it ignores autocorrelation.
    rng = np.random.default_rng(3)
    n = 400
    codes = np.repeat(rng.integers(0, 3, size=n // 40), 40)  # runs of 40 days
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.97 * y[t - 1] + rng.normal()
    p_iid = st.permutation_pvalue(y, codes, n_perm=2000)["p"]
    p_circ = st.circular_shift_pvalue(y, codes)["p"]
    assert p_circ > p_iid


def test_cliffs_delta_extremes():
    assert st.cliffs_delta(np.array([5.0, 6.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)
    assert st.cliffs_delta(np.array([1.0, 2.0]), np.array([5.0, 6.0])) == pytest.approx(-1.0)
