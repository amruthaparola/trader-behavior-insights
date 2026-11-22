"""Risk and performance metrics.

Two families, kept separate on purpose:

* **Series metrics** take a *daily PnL series* (USD per calendar day, zeros on
  days with no activity): Sharpe, Sortino, max drawdown, VaR / CVaR, skew,
  kurtosis, and the statistical uncertainty of the Sharpe ratio itself.
* **Trade metrics** take *per-trade PnL*: hit rate, payoff ratio, profit factor,
  expectancy.

Why daily PnL in USD and not % returns?  The data has no account equity, so a
return on capital is unobservable. Sharpe is scale-invariant (mean / std), so for
a book run with roughly constant capital the USD-PnL Sharpe equals the return
Sharpe. Drawdowns and VaR are therefore reported in USD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config

ANN = config.PERIODS_PER_YEAR  # 365: crypto trades every calendar day


# --------------------------------------------------------------------------- #
# Series metrics (input: daily PnL)
# --------------------------------------------------------------------------- #
def sharpe(pnl, periods: int = ANN) -> float:
    """Annualised Sharpe = sqrt(periods) * mean / std. Risk-free rate taken as 0
    (daily PnL is already an excess quantity for a futures book)."""
    x = np.asarray(pnl, dtype=float)
    sd = x.std(ddof=1)
    return float(np.sqrt(periods) * x.mean() / sd) if sd > 0 else np.nan


def sortino(pnl, periods: int = ANN) -> float:
    """Like Sharpe, but the denominator only penalises *downside* moves:
    downside deviation = sqrt(mean(min(x, 0)^2)). Upside volatility is not risk."""
    x = np.asarray(pnl, dtype=float)
    dd = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    return float(np.sqrt(periods) * x.mean() / dd) if dd > 0 else np.inf


def max_drawdown(pnl) -> float:
    """Largest peak-to-trough fall of cumulative PnL (USD, reported as a positive
    number). Equity starts at 0, so an immediate loss counts as a drawdown."""
    equity = np.concatenate([[0.0], np.cumsum(np.asarray(pnl, dtype=float))])
    running_peak = np.maximum.accumulate(equity)
    return float(np.max(running_peak - equity))


def var_cvar(pnl, alpha: float = 0.95) -> tuple[float, float]:
    """Historical Value-at-Risk and Conditional VaR (Expected Shortfall).

    VaR_a  = the loss exceeded on only (1-a) of days         (a quantile)
    CVaR_a = the average loss on those worst (1-a) of days   (a tail mean)

    Both are reported as positive losses. CVaR is a *coherent* risk measure
    (in particular subadditive: diversification never increases it); VaR is not.
    """
    x = np.asarray(pnl, dtype=float)
    q = np.quantile(x, 1 - alpha)
    tail = x[x <= q]
    return float(-q), float(-tail.mean())


def sharpe_standard_error(pnl, periods: int = ANN) -> float:
    """Standard error of the annualised Sharpe estimate for i.i.d. non-normal
    returns (Lo 2002; Mertens 2002):

        Var(SR_hat) ~ (1 - skew*SR + (kurt - 1)/4 * SR^2) / (n - 1)

    with SR the per-period Sharpe and kurt the raw (non-excess) kurtosis.
    Negative skew and fat tails make the Sharpe *less* certain.
    """
    x = np.asarray(pnl, dtype=float)
    n = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    var = (1 - g3 * sr + (g4 - 1) / 4 * sr**2) / (n - 1)
    return float(np.sqrt(periods) * np.sqrt(max(var, 0.0)))


def probabilistic_sharpe(pnl, benchmark_sr_annual: float = 0.0, periods: int = ANN) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012):
    P(true Sharpe > benchmark | sample), using the standard error above.
    A PSR of 0.95 means we are ~95% confident the skill is real, not luck."""
    x = np.asarray(pnl, dtype=float)
    n = len(x)
    sr = x.mean() / x.std(ddof=1)
    sr_star = benchmark_sr_annual / np.sqrt(periods)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / denom))


def series_summary(pnl) -> dict:
    x = np.asarray(pnl, dtype=float)
    var95, cvar95 = var_cvar(x, 0.95)
    return {
        "n_days": len(x),
        "total_pnl": x.sum(),
        "mean_daily_pnl": x.mean(),
        "sharpe": sharpe(x),
        "sharpe_se": sharpe_standard_error(x),
        "psr_vs_0": probabilistic_sharpe(x),
        "sortino": sortino(x),
        "max_drawdown": max_drawdown(x),
        "var95": var95,
        "cvar95": cvar95,
        "skew": float(stats.skew(x)),
        "excess_kurtosis": float(stats.kurtosis(x)),
    }


# --------------------------------------------------------------------------- #
# Trade metrics (input: per-trade PnL of closing trades)
# --------------------------------------------------------------------------- #
def trade_summary(trade_pnl) -> dict:
    """Hit rate, payoff ratio, profit factor, expectancy.

    expectancy    = hit * avg_win - (1 - hit) * avg_loss   (= mean PnL per trade)
    profit factor = gross wins / gross losses   (> 1 means profitable overall)
    break-even hit rate for a given payoff b is 1 / (1 + b).
    """
    x = np.asarray(trade_pnl, dtype=float)
    wins, losses = x[x > 0], x[x < 0]
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = -losses.mean() if len(losses) else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else np.inf
    return {
        "n_trades": len(x),
        "hit_rate": len(wins) / len(x) if len(x) else np.nan,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "breakeven_hit_rate": 1 / (1 + payoff) if np.isfinite(payoff) else 0.0,
        "profit_factor": wins.sum() / -losses.sum() if len(losses) else np.inf,
        "expectancy": x.mean() if len(x) else np.nan,
    }


# --------------------------------------------------------------------------- #
# Calendar helpers + the two tables used in the report
# --------------------------------------------------------------------------- #
def calendar_daily_pnl(account_day: pd.DataFrame, col: str = "net_pnl", start=None, end=None) -> pd.Series:
    """Daily PnL on a complete calendar (zeros on days with no trades).

    Dropping inactive days would silently inflate Sharpe: a flat day is a real
    day of holding capital with zero return.
    """
    s = account_day.groupby("date")[col].sum()
    start = start or s.index.min()
    end = end or s.index.max()
    return s.reindex(pd.date_range(start, end, freq="D", name="date"), fill_value=0.0)


def regime_risk_table(account_day: pd.DataFrame, closes: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Risk profile of the whole trader book, split by same-day sentiment regime.

    Read each row as: "a book that only traded on <regime> days". Daily metrics
    use every calendar day of that regime in the sample window (zeros included).
    """
    daily = calendar_daily_pnl(account_day).to_frame("net_pnl")
    daily = daily.join(sentiment.set_index("date")["sentiment"])
    rows = {}
    for regime in config.SENTIMENT_ORDER:
        d = daily.loc[daily["sentiment"] == regime, "net_pnl"]
        t = closes.loc[closes["sentiment"] == regime, "net_pnl"]
        rows[regime] = {**series_summary(d), **trade_summary(t)}
    rows["All days"] = {**series_summary(daily["net_pnl"]), **trade_summary(closes["net_pnl"])}
    return pd.DataFrame(rows).T


def trader_leaderboard(account_day: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """One row per account, sorted by Sharpe. Calendar runs from the account's
    first to last active day. `fee_drag` = fees as a share of gross PnL."""
    rows = {}
    for acct, g in account_day.groupby("account"):
        daily = calendar_daily_pnl(g)
        trades = closes.loc[closes["account"] == acct, "net_pnl"]
        s = series_summary(daily)
        tr = trade_summary(trades) if len(trades) else {}
        rows[acct[:8] + "…"] = {
            "active_days": int((g["n_fills"] > 0).sum()),
            "calendar_days": s["n_days"],
            "net_pnl": s["total_pnl"],
            "sharpe": s["sharpe"],
            "sharpe_se": s["sharpe_se"],
            "psr_vs_0": s["psr_vs_0"],
            "max_drawdown": s["max_drawdown"],
            "hit_rate": tr.get("hit_rate", np.nan),
            "payoff_ratio": tr.get("payoff_ratio", np.nan),
            "profit_factor": tr.get("profit_factor", np.nan),
            "fee_drag": g["fees"].sum() / g["gross_pnl"].sum() if g["gross_pnl"].sum() > 0 else np.nan,
        }
    return pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)


def fee_impact(account_day: pd.DataFrame) -> pd.DataFrame:
    """Book-level Sharpe before and after fees."""
    gross = calendar_daily_pnl(account_day, "gross_pnl")
    net = calendar_daily_pnl(account_day, "net_pnl")
    return pd.DataFrame(
        {"gross": series_summary(gross), "net_of_fees": series_summary(net)}
    ).loc[["total_pnl", "sharpe", "sortino", "max_drawdown", "cvar95"]]
