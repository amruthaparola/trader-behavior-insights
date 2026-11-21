"""Load, clean and join Hyperliquid trades with the Bitcoin Fear & Greed index.

Design decisions (each one is a likely interview question):

1. **Unit of time = UTC calendar day.**
   Trades are stamped in IST (UTC+5:30); the Fear & Greed index is a daily UTC
   series. We convert trades to UTC before taking the date, otherwise ~23% of the
   day (00:00-05:30 IST) would be matched to the wrong sentiment value.

2. **Closing trades only for performance.**
   Hyperliquid reports `Closed PnL` only on fills that reduce/close a position.
   Opening fills always show 0, so including them counts ~half the sample as
   "not profitable" before the trade has even had a chance to win or lose.

3. **Two sentiment columns: same-day and lagged.**
   `sentiment_*`      = value for the trade's own day   -> describes co-movement.
   `sentiment_*_lag1` = value for the previous day      -> known before the day
                        starts, so it is safe for prediction / backtests
                        (no look-ahead bias).

4. **Fees.** `Closed PnL` is gross of fees; `Fee` is in USD (negative = maker
   rebate). Net PnL = closed PnL - fees, and fees are charged on *every* fill,
   opening ones included.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config

# Direction labels that reduce or close an existing position.
# Opening labels (OPEN LONG, OPEN SHORT, BUY) and SPOT DUST CONVERSION are excluded.
CLOSING_DIRECTIONS = frozenset(
    {
        "CLOSE LONG",
        "CLOSE SHORT",
        "SELL",  # spot sell = closing a spot holding
        "LONG > SHORT",  # position flip: closes the long leg
        "SHORT > LONG",  # position flip: closes the short leg
        "LIQUIDATED ISOLATED SHORT",
        "AUTO-DELEVERAGING",
        "SETTLEMENT",
    }
)

SENTIMENT_SCORE_COL = "sentiment_score"  # 0 (extreme fear) .. 100 (extreme greed)
SENTIMENT_LABEL_COL = "sentiment"


# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #
def load_sentiment(path=config.SENTIMENT_CSV) -> pd.DataFrame:
    """Daily Fear & Greed index on a *complete* daily calendar.

    The raw file is missing a few days. We reindex to every calendar day and
    forward-fill, i.e. use the most recent published value - the only value a
    trader could actually have known on a missing day.

    Returns columns: date (datetime64, midnight), sentiment_score (int), sentiment (label).
    """
    raw = pd.read_csv(path)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"]),
            SENTIMENT_SCORE_COL: raw["value"].astype(float),
            SENTIMENT_LABEL_COL: raw["classification"].astype(str).str.strip().str.title(),
        }
    )
    df = df.drop_duplicates("date").sort_values("date").set_index("date")

    full_calendar = pd.date_range(df.index.min(), df.index.max(), freq="D", name="date")
    df = df.reindex(full_calendar).ffill()
    df[SENTIMENT_SCORE_COL] = df[SENTIMENT_SCORE_COL].astype(int)
    df[SENTIMENT_LABEL_COL] = pd.Categorical(
        df[SENTIMENT_LABEL_COL], categories=config.SENTIMENT_ORDER, ordered=True
    )
    return df.reset_index()


def add_lagged_sentiment(sentiment: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """Add `<col>_lag{k}` = value from `lag_days` calendar days earlier.

    Uses the calendar (shift on a complete daily index), *not* the previous row of
    the trades table - "yesterday" must mean yesterday even if nobody traded.
    """
    out = sentiment.set_index("date").sort_index()
    for col in (SENTIMENT_SCORE_COL, SENTIMENT_LABEL_COL):
        out[f"{col}_lag{lag_days}"] = out[col].shift(lag_days, freq="D").reindex(out.index)
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Trades
# --------------------------------------------------------------------------- #
def load_trades(path=config.TRADES_CSV) -> pd.DataFrame:
    """Clean fill-level trade data.

    Returns one row per fill with: account, coin, timestamp_utc, date (UTC day),
    side, direction, execution_price, size_tokens, size_usd, start_position,
    closed_pnl, fee, net_pnl, crossed, is_close, is_win.
    """
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")

    df = pd.DataFrame(
        {
            "account": raw["account"].astype(str),
            "coin": raw["coin"].astype(str),
            "side": raw["side"].astype(str).str.strip().str.upper(),
            "direction": raw["direction"].astype(str).str.strip().str.upper(),
            "crossed": raw["crossed"].astype(str).str.strip().str.lower().eq("true"),
        }
    )
    for col in ["execution_price", "size_tokens", "size_usd", "start_position", "closed_pnl", "fee"]:
        df[col] = pd.to_numeric(raw[col], errors="coerce")

    # IST wall-clock -> timezone-aware -> UTC. (The numeric `timestamp` column is
    # stored in scientific notation and has lost precision, so we don't use it.)
    ts_ist = pd.to_datetime(raw["timestamp_ist"], format="%d-%m-%Y %H:%M")
    df["timestamp_utc"] = ts_ist.dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")
    df["date"] = df["timestamp_utc"].dt.tz_localize(None).dt.normalize()

    # A fill with zero notional is not a real trade.
    df = df[df["size_usd"] > 0].copy()

    df["is_close"] = df["direction"].isin(CLOSING_DIRECTIONS)
    df["net_pnl"] = df["closed_pnl"] - df["fee"]
    # A "win" is only defined for closing trades; NaN for opening fills.
    # Stored as float (1.0 / 0.0 / NaN) so means give win rates directly.
    df["is_win"] = (df["closed_pnl"] > 0).astype(float).where(df["is_close"])

    return df.sort_values("timestamp_utc").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Join + panels
# --------------------------------------------------------------------------- #
def attach_sentiment(trades: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Left-join same-day and lag-1 sentiment onto each trade by UTC date."""
    sent = add_lagged_sentiment(sentiment, lag_days=1)
    merged = trades.merge(sent, on="date", how="left", validate="many_to_one")
    missing = merged[SENTIMENT_LABEL_COL].isna().sum()
    if missing:
        raise ValueError(f"{missing} trades have no sentiment value - check date coverage")
    return merged


def build_account_day(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per (account, UTC day) with activity and PnL.

    This is the natural unit for inference: a trader's fills within a day are
    strongly dependent, so treating each fill as independent overstates n.
    """
    t = trades.assign(
        close_notional=trades["size_usd"].where(trades["is_close"], 0.0),
        gross_close_pnl=trades["closed_pnl"].where(trades["is_close"], 0.0),
        win=trades["is_win"].fillna(0.0),
    )
    g = t.groupby(["account", "date"], observed=True)
    panel = g.agg(
        n_fills=("size_usd", "size"),
        n_closes=("is_close", "sum"),
        n_wins=("win", "sum"),
        volume_usd=("size_usd", "sum"),
        close_notional_usd=("close_notional", "sum"),
        gross_pnl=("gross_close_pnl", "sum"),
        fees=("fee", "sum"),
    ).reset_index()
    panel["net_pnl"] = panel["gross_pnl"] - panel["fees"]
    sent_cols = [c for c in trades.columns if c.startswith("sentiment")]
    day_sent = trades.drop_duplicates("date").set_index("date")[sent_cols]
    return panel.join(day_sent, on="date")


def build_market_day(account_day: pd.DataFrame) -> pd.DataFrame:
    """One row per UTC day, aggregated across all accounts (a daily time series).

    `ret_on_volume` = net PnL per USD traded. We have no account equity, so a true
    % return on capital is not observable; PnL per dollar traded is the closest
    scale-free proxy (discussed further in Step 4).
    """
    g = account_day.groupby("date")
    daily = g.agg(
        n_accounts=("account", "nunique"),
        n_fills=("n_fills", "sum"),
        n_closes=("n_closes", "sum"),
        n_wins=("n_wins", "sum"),
        volume_usd=("volume_usd", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        fees=("fees", "sum"),
        net_pnl=("net_pnl", "sum"),
    )
    daily["ret_on_volume"] = daily["net_pnl"] / daily["volume_usd"]
    sent_cols = [c for c in account_day.columns if c.startswith("sentiment")]
    daily = daily.join(account_day.drop_duplicates("date").set_index("date")[sent_cols])
    return daily.sort_index().reset_index()


@dataclass
class Dataset:
    """Everything downstream steps need, built once."""

    sentiment: pd.DataFrame  # full daily index, 2018 -> 2025
    trades: pd.DataFrame  # every fill, with sentiment attached
    closes: pd.DataFrame  # closing fills only (performance sample)
    account_day: pd.DataFrame  # (account, day) panel
    market_day: pd.DataFrame  # daily time series


def load_dataset() -> Dataset:
    sentiment = load_sentiment()
    trades = attach_sentiment(load_trades(), sentiment)
    account_day = build_account_day(trades)
    return Dataset(
        sentiment=sentiment,
        trades=trades,
        closes=trades[trades["is_close"]].reset_index(drop=True),
        account_day=account_day,
        market_day=build_market_day(account_day),
    )


def win_rate_comparison(trades: pd.DataFrame) -> pd.DataFrame:
    """Win rate by regime: the notebook's definition (all fills) vs closing fills only."""
    naive = (trades["closed_pnl"] > 0).groupby(trades[SENTIMENT_LABEL_COL], observed=True).mean()
    closes = trades[trades["is_close"]]
    correct = closes["is_win"].groupby(closes[SENTIMENT_LABEL_COL], observed=True).mean()
    out = pd.DataFrame(
        {
            "n_fills": trades.groupby(SENTIMENT_LABEL_COL, observed=True).size(),
            "n_closes": closes.groupby(SENTIMENT_LABEL_COL, observed=True).size(),
            "win_rate_all_fills": naive,
            "win_rate_closes_only": correct,
        }
    )
    out["difference_pp"] = 100 * (out["win_rate_closes_only"] - out["win_rate_all_fills"])
    return out.reindex(config.SENTIMENT_ORDER)
