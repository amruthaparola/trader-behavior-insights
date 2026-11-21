"""Tests for the data pipeline: closing-trade flag, timezone handling, no look-ahead."""

import pandas as pd
import pytest

from sentiment_quant import config, data


# ---------------- small synthetic inputs (fast, fully controlled) ---------------- #
@pytest.fixture
def sentiment_csv(tmp_path):
    # Note: 2024-01-03 is deliberately missing.
    df = pd.DataFrame(
        {
            "timestamp": [0, 0, 0],
            "value": [10, 50, 90],
            "classification": ["Extreme Fear", "Neutral", "Extreme Greed"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-04"],
        }
    )
    path = tmp_path / "fg.csv"
    df.to_csv(path, index=False)
    return path


def test_missing_sentiment_day_is_forward_filled(sentiment_csv):
    s = data.load_sentiment(sentiment_csv).set_index("date")
    assert len(s) == 4  # complete calendar
    assert s.loc["2024-01-03", "sentiment_score"] == 50  # last known value


def test_lag_uses_previous_calendar_day(sentiment_csv):
    s = data.add_lagged_sentiment(data.load_sentiment(sentiment_csv)).set_index("date")
    assert pd.isna(s.loc["2024-01-01", "sentiment_score_lag1"])  # nothing known before day 1
    assert s.loc["2024-01-02", "sentiment_score_lag1"] == 10
    assert s.loc["2024-01-04", "sentiment_score_lag1"] == 50  # from the ffilled 01-03
    # The lag never equals information from the same or a later day:
    assert (s["sentiment_score_lag1"].dropna().values == s["sentiment_score"].shift(1).dropna().values).all()


# ---------------- real data (checks the actual pipeline end to end) ---------------- #
@pytest.fixture(scope="module")
def ds():
    return data.load_dataset()


def test_opening_fills_never_count_as_closes(ds):
    opens = ds.trades[ds.trades["direction"].isin(["OPEN LONG", "OPEN SHORT", "BUY"])]
    assert not opens["is_close"].any()
    assert opens["is_win"].isna().all()


def test_opening_fills_have_zero_closed_pnl(ds):
    # The reason opening fills must be excluded from win-rate calculations.
    opens = ds.trades[ds.trades["direction"].isin(["OPEN LONG", "OPEN SHORT"])]
    assert (opens["closed_pnl"] == 0).all()


def test_dates_are_utc_not_ist(ds):
    # 01:06 IST on 2023-05-01 is 19:36 UTC on 2023-04-30.
    first = ds.trades.iloc[0]
    assert first["timestamp_utc"] == pd.Timestamp("2023-04-30 19:36", tz="UTC")
    assert first["date"] == pd.Timestamp("2023-04-30")


def test_every_trade_has_sentiment(ds):
    assert ds.trades["sentiment"].notna().all()
    assert ds.trades["sentiment_lag1"].notna().all()


def test_panels_conserve_pnl(ds):
    # Aggregation must not create or lose money.
    total = ds.closes["closed_pnl"].sum()
    assert ds.account_day["gross_pnl"].sum() == pytest.approx(total)
    assert ds.market_day["gross_pnl"].sum() == pytest.approx(total)
    assert ds.market_day["fees"].sum() == pytest.approx(ds.trades["fee"].sum())


def test_sentiment_categories_are_ordered(ds):
    assert list(ds.trades["sentiment"].cat.categories) == config.SENTIMENT_ORDER
