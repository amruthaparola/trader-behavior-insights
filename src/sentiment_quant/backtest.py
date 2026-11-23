"""Backtest simple Fear & Greed strategies on daily BTC.

Timing convention (the most important line in any backtest)
-----------------------------------------------------------
    position held over day t  =  rule( sentiment of day t-1 )
    strategy return on day t  =  position_t * r_t  -  cost * |position_t - position_{t-1}|
    r_t                       =  close_t / close_{t-1} - 1      (UTC daily closes)

Day t-1's index value is public during day t-1, so the trade is placed at the
day t-1 close and earns day t's return. Using day t's sentiment would be
look-ahead: its value is published after day t has already started.

Overfitting guard
-----------------
Rules are designed/tuned on 2018-2022 only (in-sample, IS). 2023-01-01 onward is
held out (out-of-sample, OOS) and touched once. A circular-shift test asks
whether the strategy beats *random timing with the same amount and persistence
of exposure*.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config, data, prices
from . import metrics as mt

TEST_START = pd.Timestamp("2023-01-01")
COST_BPS = 10.0  # per unit of turnover (fee + spread + slippage); see sensitivity table


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_frame() -> pd.DataFrame:
    """Daily BTC returns joined to the PREVIOUS day's sentiment."""
    btc = prices.load_btc()[["date", "close"]]
    sent = data.add_lagged_sentiment(data.load_sentiment(), lag_days=1)
    df = btc.merge(sent[["date", "sentiment_score_lag1", "sentiment_lag1"]], on="date", how="inner")
    df["ret"] = df["close"].pct_change()
    return df.dropna(subset=["ret", "sentiment_score_lag1"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Rules: map yesterday's sentiment -> today's position in {-1, 0, 1}
# --------------------------------------------------------------------------- #
FEARFUL = {"Extreme Fear", "Fear"}
GREEDY = {"Greed", "Extreme Greed"}

RULES: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "Buy & hold": lambda d: np.ones(len(d)),
    "Contrarian: long in Fear": lambda d: d["sentiment_lag1"].isin(FEARFUL).astype(float).to_numpy(),
    "Extreme Fear only": lambda d: (d["sentiment_lag1"] == "Extreme Fear").astype(float).to_numpy(),
    "Momentum: long in Greed": lambda d: d["sentiment_lag1"].isin(GREEDY).astype(float).to_numpy(),
    "Long Fear / short Greed": lambda d: (
        d["sentiment_lag1"].isin(FEARFUL).astype(float) - d["sentiment_lag1"].isin(GREEDY).astype(float)
    ).to_numpy(),
}


def threshold_rule(k: float) -> Callable[[pd.DataFrame], np.ndarray]:
    """Long when yesterday's score < k, else flat."""
    return lambda d: (d["sentiment_score_lag1"] < k).astype(float).to_numpy()


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def run(positions: np.ndarray, returns: np.ndarray, cost_bps: float = COST_BPS) -> np.ndarray:
    """Net daily strategy returns. Starts flat, so the first entry pays a cost."""
    pos = np.asarray(positions, dtype=float)
    turnover = np.abs(np.diff(pos, prepend=0.0))
    return pos * np.asarray(returns, dtype=float) - turnover * cost_bps / 1e4


def max_drawdown_pct(strategy_returns) -> float:
    """Worst peak-to-trough fall of the compounded equity curve, as a fraction."""
    equity = np.cumprod(1 + np.asarray(strategy_returns, dtype=float))
    equity = np.concatenate([[1.0], equity])
    return float(np.max(1 - equity / np.maximum.accumulate(equity)))


def performance(strategy_returns, positions) -> dict:
    r = np.asarray(strategy_returns, dtype=float)
    pos = np.asarray(positions, dtype=float)
    years = len(r) / config.PERIODS_PER_YEAR
    total = float(np.prod(1 + r))
    return {
        "cagr": total ** (1 / years) - 1 if total > 0 else -1.0,
        "ann_vol": float(r.std(ddof=1) * np.sqrt(config.PERIODS_PER_YEAR)),
        "sharpe": mt.sharpe(r),
        "sharpe_se": mt.sharpe_standard_error(r),
        "sortino": mt.sortino(r),
        "max_dd": max_drawdown_pct(r),
        "exposure": float(np.mean(np.abs(pos) > 0)),
        "trades_per_year": float(np.abs(np.diff(pos, prepend=0.0)).sum() / years),
    }


def split_masks(d: pd.DataFrame) -> dict[str, np.ndarray]:
    test = (d["date"] >= TEST_START).to_numpy()
    return {"in-sample 2018-22": ~test, "out-of-sample 2023-25": test}


def evaluate_rules(d: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Every rule's IS and OOS performance. Positions are computed on the full
    series (they only use lagged data) and then split, so the OOS period starts
    with whatever position the rule already held - no artificial re-entry."""
    rows = []
    for name, rule in RULES.items():
        pos = rule(d)
        strat = run(pos, d["ret"], cost_bps)
        for split, mask in split_masks(d).items():
            rows.append({"rule": name, "period": split, **performance(strat[mask], pos[mask])})
    return pd.DataFrame(rows).set_index(["period", "rule"]).sort_index(level=0, sort_remaining=False)


def cost_sensitivity(d: pd.DataFrame, costs=(0, 5, 10, 25, 50)) -> pd.DataFrame:
    """OOS Sharpe of each rule as transaction costs rise."""
    mask = split_masks(d)["out-of-sample 2023-25"]
    out = {}
    for c in costs:
        out[f"{c} bps"] = {n: mt.sharpe(run(r(d), d["ret"], c)[mask]) for n, r in RULES.items()}
    return pd.DataFrame(out)


def threshold_search(d: pd.DataFrame, grid=range(10, 65, 5), cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Tune k in 'long when score_{t-1} < k' on IS only, then look at OOS.

    The point is the comparison: if IS Sharpe and OOS Sharpe are unrelated across
    k, picking the IS-best k is fitting noise (data snooping)."""
    masks = split_masks(d)
    rows = []
    for k in grid:
        pos = threshold_rule(k)(d)
        strat = run(pos, d["ret"], cost_bps)
        rows.append(
            {
                "k": k,
                "is_sharpe": mt.sharpe(strat[masks["in-sample 2018-22"]]),
                "oos_sharpe": mt.sharpe(strat[masks["out-of-sample 2023-25"]]),
                "exposure": float(pos.mean()),
            }
        )
    out = pd.DataFrame(rows).set_index("k")
    out.attrs["best_k_in_sample"] = int(out["is_sharpe"].idxmax())
    out.attrs["rank_corr_is_oos"] = float(out["is_sharpe"].corr(out["oos_sharpe"], method="spearman"))
    return out


def timing_test(positions, returns, cost_bps: float = COST_BPS, min_shift: int = 30) -> dict:
    """Does the rule's timing beat random timing with the same exposure profile?

    Rotate the position series against the return series by every shift k.
    Exposure, run lengths and turnover are preserved exactly; only the alignment
    with sentiment is destroyed. p = share of rotations with Sharpe >= actual.
    """
    pos = np.asarray(positions, dtype=float)
    ret = np.asarray(returns, dtype=float)
    actual = mt.sharpe(run(pos, ret, cost_bps))
    shifts = range(min_shift, len(pos) - min_shift + 1)
    null = np.array([mt.sharpe(run(np.roll(pos, k), ret, cost_bps)) for k in shifts])
    return {"sharpe": actual, "null_mean": float(np.nanmean(null)), "p": float((1 + np.sum(null >= actual)) / (1 + len(null)))}


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_equity(d: pd.DataFrame, path=config.FIGURES_DIR / "backtest_equity.png", cost_bps: float = COST_BPS):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference categorical palette, fixed order; benchmark in muted ink.
    style = {
        "Buy & hold": ("#8a8983", 2.0),
        "Contrarian: long in Fear": ("#2a78d6", 2.0),
        "Momentum: long in Greed": ("#eb6834", 2.0),
        "Long Fear / short Greed": ("#1baf7a", 2.0),
    }
    ink, muted, grid = "#2b2b28", "#6b6a63", "#e4e3dd"
    fig, (ax, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.08}
    )
    for name, (color, lw) in style.items():
        strat = run(RULES[name](d), d["ret"], cost_bps)
        eq = np.cumprod(1 + strat)
        dd = 1 - eq / np.maximum.accumulate(eq)
        ax.plot(d["date"], eq, color=color, lw=lw, label=name, solid_capstyle="round")
        ax.annotate(f" {name}", (d["date"].iloc[-1], eq[-1]), color=ink, fontsize=9, va="center")
        ax_dd.plot(d["date"], -100 * dd, color=color, lw=1.4)

    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)", color=muted)
    ax.set_title(
        f"Sentiment rules vs buy & hold on BTC, net of {cost_bps:.0f} bps costs (position from yesterday's index)",
        loc="left", color=ink, fontsize=12,
    )
    ax_dd.set_ylabel("Drawdown (%)", color=muted)
    for a in (ax, ax_dd):
        a.axvline(TEST_START, color=muted, lw=1, ls="--")
        a.grid(axis="y", color=grid, lw=0.8)
        a.tick_params(colors=muted, labelsize=9)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(grid)
    ax.text(TEST_START, ax.get_ylim()[1], "  out-of-sample →", color=muted, fontsize=9, va="top")
    ax.text(TEST_START, ax.get_ylim()[1], "← in-sample (rules designed here)  ", color=muted, fontsize=9, va="top", ha="right")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.subplots_adjust(right=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path
