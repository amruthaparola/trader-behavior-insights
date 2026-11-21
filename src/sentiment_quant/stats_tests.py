"""Dependence-aware hypothesis tests for "does performance differ by sentiment regime?".

The core problem with the original notebook
-------------------------------------------
Sentiment is a *daily* variable. Every trade on a given day shares the same
label, and trades on the same day are correlated (same market move, same
traders). So the information about "sentiment" is carried by ~476 days, not
~105k trades. Treating trades as i.i.d. shrinks standard errors by roughly
sqrt(design effect) and makes p-values absurdly small.

Tools in this module
--------------------
* ``naive_trade_level_tests``  - the notebook's ANOVA + chi-square, for contrast.
* ``cluster_bootstrap_ratio``  - regime statistic with a CI that resamples whole days.
* ``design_effect_table``      - how much i.i.d. SEs understate the truth.
* ``kruskal_h``                - rank-based ANOVA statistic (robust to fat tails).
* ``permutation_pvalue``       - shuffle regime labels across days.
* ``circular_shift_pvalue``    - rotate the label series; keeps autocorrelation.
* ``pairwise_regime_tests``    - Mann-Whitney + Cliff's delta + Holm correction.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from . import config

LABEL = "sentiment"


# --------------------------------------------------------------------------- #
# 0. What the notebook did (kept for comparison)
# --------------------------------------------------------------------------- #
def naive_trade_level_tests(closes: pd.DataFrame, label_col: str = LABEL) -> dict:
    """One-way ANOVA on per-trade return and chi-square on win/loss, treating every
    closing trade as an independent observation. Assumptions violated: independence
    (trades cluster by day) and, for ANOVA, normality (returns are fat-tailed)."""
    ret = closes["closed_pnl"] / closes["size_usd"]
    groups = [ret[closes[label_col] == g].values for g in config.SENTIMENT_ORDER]
    f_stat, p_anova = stats.f_oneway(*groups)
    table = pd.crosstab(closes[label_col], closes["is_win"])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(table)
    return {
        "n_observations": int(len(closes)),
        "anova_F": float(f_stat),
        "anova_p": float(p_anova),
        "chi2": float(chi2),
        "chi2_dof": int(dof),
        "chi2_p": float(p_chi2),
        "excess_kurtosis_of_returns": float(stats.kurtosis(ret)),
    }


# --------------------------------------------------------------------------- #
# 1. Cluster (day) bootstrap
# --------------------------------------------------------------------------- #
def daily_sums(closes: pd.DataFrame, label_col: str = LABEL) -> pd.DataFrame:
    """Collapse closing trades to one row per day: the sufficient statistics for
    ratio estimators (win rate = wins / closes, mean return = sum r / closes)."""
    df = closes.assign(
        ret=closes["closed_pnl"] / closes["size_usd"],
        win=closes["is_win"],
    )
    out = df.groupby("date").agg(
        n_closes=("ret", "size"), n_wins=("win", "sum"), sum_ret=("ret", "sum")
    )
    out[label_col] = df.groupby("date")[label_col].first()
    return out.reset_index()


def cluster_bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Bootstrap a ratio estimator sum(num)/sum(den) by resampling *clusters*.

    Each element of `numerator` / `denominator` is one cluster (one day). Resampling
    whole days keeps every within-day correlation intact, so the spread of the
    bootstrap distribution reflects the real uncertainty. Percentile CI.
    """
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    n = len(num)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(num.sum() / den.sum()),
        "se": float(boot.std(ddof=1)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_clusters": int(n),
    }


def regime_bootstrap_table(closes: pd.DataFrame, label_col: str = LABEL, n_boot: int = 5000) -> pd.DataFrame:
    """Per regime: win rate and mean trade return (bps) with 95% day-cluster CIs."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    d = daily_sums(closes, label_col)
    rows = []
    for regime in config.SENTIMENT_ORDER:
        g = d[d[label_col] == regime]
        wr = cluster_bootstrap_ratio(g["n_wins"], g["n_closes"], n_boot, rng=rng)
        mr = cluster_bootstrap_ratio(g["sum_ret"], g["n_closes"], n_boot, rng=rng)
        rows.append(
            {
                "regime": regime,
                "n_days": wr["n_clusters"],
                "n_closes": int(g["n_closes"].sum()),
                "win_rate": wr["estimate"],
                "win_rate_ci_low": wr["ci_low"],
                "win_rate_ci_high": wr["ci_high"],
                "mean_ret_bps": 1e4 * mr["estimate"],
                "mean_ret_ci_low_bps": 1e4 * mr["ci_low"],
                "mean_ret_ci_high_bps": 1e4 * mr["ci_high"],
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def design_effect_table(closes: pd.DataFrame, label_col: str = LABEL, n_boot: int = 5000) -> pd.DataFrame:
    """Compare the i.i.d. binomial SE of the win rate with the day-cluster SE.

    design effect  deff = (SE_cluster / SE_iid)^2
    effective n    n_eff = n_trades / deff   (how many *independent* trades the
                                              data is really worth)
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    d = daily_sums(closes, label_col)
    rows = []
    for regime in config.SENTIMENT_ORDER:
        g = d[d[label_col] == regime]
        n = g["n_closes"].sum()
        p = g["n_wins"].sum() / n
        se_iid = np.sqrt(p * (1 - p) / n)
        se_cl = cluster_bootstrap_ratio(g["n_wins"], g["n_closes"], n_boot, rng=rng)["se"]
        deff = (se_cl / se_iid) ** 2
        rows.append(
            {
                "regime": regime,
                "n_trades": int(n),
                "n_days": len(g),
                "se_iid": se_iid,
                "se_cluster": se_cl,
                "se_ratio": se_cl / se_iid,
                "design_effect": deff,
                "effective_n": n / deff,
            }
        )
    return pd.DataFrame(rows).set_index("regime")


# --------------------------------------------------------------------------- #
# 2. Global test: is there ANY difference across regimes? (day level)
# --------------------------------------------------------------------------- #
def kruskal_h(values: np.ndarray, codes: np.ndarray) -> float:
    """Kruskal-Wallis H from ranks (no tie correction): a rank-based one-way ANOVA.

    H = 12 / (N (N+1)) * sum_g n_g * (Rbar_g - (N+1)/2)^2

    Using ranks makes it insensitive to the fat tails that break ANOVA. Written
    out by hand (instead of scipy) so permutation loops are fast and vectorised.
    """
    return _h_from_ranks(stats.rankdata(values), np.asarray(codes))


def _h_from_ranks(ranks: np.ndarray, codes: np.ndarray) -> float:
    # Permuting labels never changes the ranks, so loops rank once and call this.
    n = len(ranks)
    k = codes.max() + 1
    n_g = np.bincount(codes, minlength=k)
    r_g = np.bincount(codes, weights=ranks, minlength=k)
    mask = n_g > 0
    mean_r = r_g[mask] / n_g[mask]
    return float(12.0 / (n * (n + 1)) * np.sum(n_g[mask] * (mean_r - (n + 1) / 2) ** 2))


def permutation_pvalue(values, codes, n_perm: int = 10000, rng=None) -> dict:
    """Shuffle regime labels across days. Valid if days were exchangeable - but
    regimes persist (today's label predicts tomorrow's), so this is optimistic."""
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    ranks, codes = stats.rankdata(values), np.asarray(codes)
    h_obs = _h_from_ranks(ranks, codes)
    null = np.array([_h_from_ranks(ranks, rng.permutation(codes)) for _ in range(n_perm)])
    return {"H": h_obs, "p": float((1 + np.sum(null >= h_obs)) / (1 + n_perm)), "n_null": n_perm}


def circular_shift_pvalue(values, codes, min_shift: int = 30) -> dict:
    """Rotate the label series against the PnL series by every shift k.

    Rotating keeps the *autocorrelation of both series* (runs of Fear days stay
    runs) while destroying their alignment. That is the right null for two
    persistent time series; i.i.d. shuffling is not. Shifts closer than
    `min_shift` to 0 are skipped because they barely break the alignment.
    """
    ranks, codes = stats.rankdata(values), np.asarray(codes)
    n = len(ranks)
    h_obs = _h_from_ranks(ranks, codes)
    shifts = range(min_shift, n - min_shift + 1)
    null = np.array([_h_from_ranks(ranks, np.roll(codes, k)) for k in shifts])
    return {"H": h_obs, "p": float((1 + np.sum(null >= h_obs)) / (1 + len(null))), "n_null": len(null)}


def global_tests(market_day: pd.DataFrame, value_col: str = "ret_on_volume", label_col: str = LABEL) -> dict:
    """Kruskal-Wallis (asymptotic chi^2), i.i.d. permutation and circular-shift
    p-values for the daily series `value_col` grouped by `label_col`."""
    m = market_day.dropna(subset=[value_col, label_col]).sort_values("date")
    codes = m[label_col].cat.codes.to_numpy()
    y = m[value_col].to_numpy()
    groups = [y[codes == i] for i in range(len(config.SENTIMENT_ORDER)) if np.any(codes == i)]
    kw = stats.kruskal(*groups)
    perm = permutation_pvalue(y, codes)
    circ = circular_shift_pvalue(y, codes)
    return {
        "n_days": int(len(m)),
        "kruskal_H": float(kw.statistic),
        "kruskal_p_asymptotic": float(kw.pvalue),
        "permutation_p": perm["p"],
        "circular_shift_p": circ["p"],
        "circular_shift_n_null": circ["n_null"],
    }


# --------------------------------------------------------------------------- #
# 3. Which regimes differ? Pairwise tests + multiple-testing correction
# --------------------------------------------------------------------------- #
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) - P(X < Y): a rank effect size in [-1, 1]. |d| < 0.15 is
    negligible, < 0.33 small, < 0.47 medium, else large (Romano et al.)."""
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic  # counts X>Y (+ half ties)
    return float(2 * u / (len(x) * len(y)) - 1)


def pairwise_regime_tests(
    market_day: pd.DataFrame, value_col: str = "ret_on_volume", label_col: str = LABEL
) -> pd.DataFrame:
    """All 10 regime pairs on daily data: Mann-Whitney p, Cliff's delta, Holm-adjusted p.

    Holm controls the family-wise error rate: with 10 tests at 5% each, the chance
    of at least one false positive is up to ~40% without correction.
    """
    m = market_day.dropna(subset=[value_col, label_col])
    rows = []
    for a, b in itertools.combinations(config.SENTIMENT_ORDER, 2):
        x = m.loc[m[label_col] == a, value_col].to_numpy()
        y = m.loc[m[label_col] == b, value_col].to_numpy()
        p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
        rows.append(
            {
                "pair": f"{a} vs {b}",
                "n_days_a": len(x),
                "n_days_b": len(y),
                "median_a_bps": 1e4 * np.median(x),
                "median_b_bps": 1e4 * np.median(y),
                "cliffs_delta": cliffs_delta(x, y),
                "p_raw": p,
            }
        )
    out = pd.DataFrame(rows).set_index("pair")
    out["p_holm"] = multipletests(out["p_raw"], method="holm")[1]
    out["significant_5pct_holm"] = out["p_holm"] < 0.05
    return out
