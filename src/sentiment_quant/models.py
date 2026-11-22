"""Predictive models: does sentiment known *before* day t help explain PnL on day t?

Every regressor here is lagged (t-1 or t-2), so it was public before the trading
day began - no look-ahead.

1. ``time_series_models``  daily book return on lagged sentiment, with OLS vs
   Newey-West (HAC) standard errors, a lagged-return control, and a time-trend
   control (the confounder Step 3 pointed to).
2. ``panel_models``        account-day returns with account fixed effects and
   clustered standard errors.
3. ``transition_table`` / ``change_direction_model``  the notebook's "Fear ->
   Extreme Fear" story, re-checked with cell counts and HAC inference.
4. ``out_of_sample_r2``    expanding-window forecasts vs the historical mean
   (Campbell & Thompson 2008). In-sample fit is not predictive power.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import config

Y = "ret_bps"  # daily net PnL per USD traded, in basis points


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def newey_west_lags(n: int) -> int:
    """Rule-of-thumb bandwidth floor(4 (n/100)^(2/9)) (Newey & West 1994)."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip to the [lower, upper] quantiles. Keeps every observation but stops a
    handful of extreme days from dominating a least-squares fit."""
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)


def prepare_daily(market_day: pd.DataFrame, winsor: bool = True) -> pd.DataFrame:
    """Daily regression frame (trading days only, in date order)."""
    d = market_day.sort_values("date").reset_index(drop=True).copy()
    d[Y] = 1e4 * d["ret_on_volume"]
    if winsor:
        d[Y] = winsorize(d[Y])
    d["ret_lag1"] = d[Y].shift(1)  # previous *trading* day's return
    d["score_lag1_10"] = d["sentiment_score_lag1"] / 10  # coefficient per 10 index points
    d["trend_years"] = (d["date"] - d["date"].min()).dt.days / 365
    # Quarter fixed effects. 2023 is sparse (2023Q2 has a single trading day, which
    # makes its dummy unidentifiable under HAC), so all of 2023 is one period.
    q = d["date"].dt.to_period("Q").astype(str)
    d["quarter"] = q.where(d["date"].dt.year > 2023, "2023")
    for col in ("sentiment_lag1", "sentiment_lag2"):
        d[col] = d[col].astype(str)
    return d.dropna(subset=["ret_lag1"])


def _coef_row(name: str, fit_ols, fit_hac, term: str) -> dict:
    return {
        "model": name,
        "term": term,
        "coef": fit_hac.params[term],
        "se_ols": fit_ols.bse[term],
        "se_hac": fit_hac.bse[term],
        "t_hac": fit_hac.tvalues[term],
        "p_hac": fit_hac.pvalues[term],
        "r2": fit_hac.rsquared,
        "n": int(fit_hac.nobs),
    }


# --------------------------------------------------------------------------- #
# 1. Time-series regressions
# --------------------------------------------------------------------------- #
TS_SPECS = {
    "A: score": f"{Y} ~ score_lag1_10",
    "B: score + AR(1)": f"{Y} ~ score_lag1_10 + ret_lag1",
    "C: score + AR(1) + trend": f"{Y} ~ score_lag1_10 + ret_lag1 + trend_years",
    "D: score + AR(1) + quarter FE": f"{Y} ~ score_lag1_10 + ret_lag1 + C(quarter)",
}


def time_series_models(market_day: pd.DataFrame, winsor: bool = True) -> pd.DataFrame:
    """r_t = a + b * S_{t-1}/10 + controls + e_t, with OLS and HAC standard errors.

    Why HAC: daily PnL is autocorrelated and its variance changes over time
    (heteroskedasticity). Both make plain OLS standard errors too small. Newey-West
    re-weights the residual autocovariances up to `maxlags` to correct for that.
    """
    d = prepare_daily(market_day, winsor)
    lags = newey_west_lags(len(d))
    rows = []
    for name, formula in TS_SPECS.items():
        ols = smf.ols(formula, d).fit()
        hac = smf.ols(formula, d).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        rows.append(_coef_row(name, ols, hac, "score_lag1_10"))
    out = pd.DataFrame(rows).set_index("model")
    out.attrs["hac_lags"] = lags
    return out


def regime_dummy_model(market_day: pd.DataFrame, winsor: bool = True) -> pd.DataFrame:
    """Yesterday's regime as dummies (baseline = Neutral), HAC SEs, plus a joint
    Wald test that all four regime coefficients are zero."""
    d = prepare_daily(market_day, winsor)
    formula = f"{Y} ~ C(sentiment_lag1, Treatment('Neutral')) + ret_lag1"
    hac = smf.ols(formula, d).fit(cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(d))})
    terms = [t for t in hac.params.index if t.startswith("C(sentiment_lag1")]
    table = pd.DataFrame(
        {"coef": hac.params[terms], "se_hac": hac.bse[terms], "p_hac": hac.pvalues[terms]}
    )
    table.index = [t.split("[T.")[1].rstrip("]") for t in terms]
    wald = hac.wald_test(" = ".join(terms) + " = 0", scalar=True)
    table.attrs["joint_p"] = float(wald.pvalue)
    return table


# --------------------------------------------------------------------------- #
# 2. Panel regression (account-day)
# --------------------------------------------------------------------------- #
def panel_models(account_day: pd.DataFrame) -> pd.DataFrame:
    """r_{i,t} = alpha_i + b * S_{t-1}/10 + e_{i,t}   (account fixed effects).

    alpha_i absorbs each trader's own average skill, so b is identified only from
    how the *same* trader does on high- vs low-sentiment days.
    SEs clustered (a) by account and (b) two-way by account and date: residuals
    are correlated within a trader over time and across traders on the same day.
    Caveat: 32 account clusters is on the low side for cluster asymptotics.
    """
    p = account_day[account_day["volume_usd"] > 0].copy()
    p[Y] = winsorize(1e4 * p["net_pnl"] / p["volume_usd"])
    p["score_lag1_10"] = p["sentiment_score_lag1"] / 10
    p = p.dropna(subset=[Y, "score_lag1_10"]).reset_index(drop=True)
    acct = pd.factorize(p["account"])[0]
    day = pd.factorize(p["date"])[0]

    # Fixed effects via the *within transformation*: subtract each account's own
    # mean from y and x. Numerically identical slope to adding 31 account dummies
    # (Frisch-Waugh-Lovell), but avoids a rank-deficient cluster covariance
    # (31 dummies vs 32 clusters). We don't adjust df for the absorbed means;
    # with 2,338 obs that is negligible.
    for col in (Y, "score_lag1_10"):
        p[f"{col}_w"] = p[col] - p.groupby("account")[col].transform("mean")

    rows = []
    for name, formula in {
        "pooled": f"{Y} ~ score_lag1_10",
        "account FE": f"{Y}_w ~ score_lag1_10_w",
    }.items():
        base = smf.ols(formula, p)
        ols = base.fit()
        for cname, kw in {
            "cluster account": {"groups": acct},
            "cluster account+date": {"groups": np.column_stack([acct, day])},
        }.items():
            fit = base.fit(cov_type="cluster", cov_kwds=kw)
            term = "score_lag1_10_w" if name == "account FE" else "score_lag1_10"
            row = _coef_row(f"{name} | {cname}", ols, fit, term)
            row["term"] = "score_lag1_10"
            rows.append(row)
    return pd.DataFrame(rows).set_index("model").rename(columns={"se_hac": "se_cluster", "t_hac": "t_cluster", "p_hac": "p_cluster"})


# --------------------------------------------------------------------------- #
# 3. Sentiment transitions
# --------------------------------------------------------------------------- #
def transition_table(market_day: pd.DataFrame) -> pd.DataFrame:
    """The notebook's view: previous-day regime -> today's regime, with the number
    of days behind each cell (the notebook showed means without counts)."""
    d = market_day.assign(ret_bps=1e4 * market_day["ret_on_volume"])
    g = d.groupby(["sentiment_lag1", "sentiment"], observed=True)["ret_bps"]
    return pd.DataFrame({"n_days": g.size(), "mean_ret_bps": g.mean(), "median_ret_bps": g.median()})


def change_direction_model(market_day: pd.DataFrame, winsor: bool = True) -> pd.DataFrame:
    """Predictive version: did the regime worsen / improve from t-2 to t-1, and
    does that forecast the return on day t? (HAC SEs, baseline = unchanged)."""
    d = prepare_daily(market_day, winsor)
    order = {r: i for i, r in enumerate(config.SENTIMENT_ORDER)}
    step = d["sentiment_lag1"].map(order) - d["sentiment_lag2"].map(order)
    d["change"] = np.select([step < 0, step > 0], ["worsened", "improved"], "unchanged")
    hac = smf.ols(f"{Y} ~ C(change, Treatment('unchanged')) + ret_lag1", d).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(d))}
    )
    terms = [t for t in hac.params.index if t.startswith("C(change")]
    out = pd.DataFrame({"coef": hac.params[terms], "se_hac": hac.bse[terms], "p_hac": hac.pvalues[terms]})
    out.index = [t.split("[T.")[1].rstrip("]") for t in terms]
    out["n_days"] = [int((d["change"] == k).sum()) for k in out.index]
    return out


# --------------------------------------------------------------------------- #
# 4. Out-of-sample R^2
# --------------------------------------------------------------------------- #
def out_of_sample_r2(market_day: pd.DataFrame, start_frac: float = 0.4) -> dict:
    """Expanding-window one-step-ahead forecasts.

    For each day t after the first `start_frac` of the sample: fit
    r = a + b*S_{t-1} on days < t only, forecast r_t, and compare with the naive
    forecast "historical mean of r up to t-1".

        R2_oos = 1 - sum (r_t - fhat_t)^2 / sum (r_t - rbar_t)^2

    R2_oos > 0 means the signal beats the mean out of sample; < 0 means it does
    worse than ignoring sentiment altogether.

    No look-ahead anywhere: outliers are clipped using quantiles of the
    *training window only* (winsorising on the full sample first would leak
    future information into past forecasts), and both forecasts are scored
    against the raw, unclipped outcome.
    """
    d = prepare_daily(market_day, winsor=False)
    y = d[Y].to_numpy()
    x = d["score_lag1_10"].to_numpy()
    start = int(len(d) * start_frac)
    err_model, err_mean = [], []
    for t in range(start, len(d)):
        lo, hi = np.quantile(y[:t], [0.01, 0.99])
        y_train = np.clip(y[:t], lo, hi)
        X = np.column_stack([np.ones(t), x[:t]])
        beta, *_ = np.linalg.lstsq(X, y_train, rcond=None)
        err_model.append(y[t] - (beta[0] + beta[1] * x[t]))
        err_mean.append(y[t] - y_train.mean())
    em, eb = np.array(err_model), np.array(err_mean)
    return {
        "n_forecasts": len(em),
        "first_forecast_date": str(d["date"].iloc[start].date()),
        "r2_oos": float(1 - np.sum(em**2) / np.sum(eb**2)),
        "rmse_model_bps": float(np.sqrt(np.mean(em**2))),
        "rmse_mean_bps": float(np.sqrt(np.mean(eb**2))),
    }
