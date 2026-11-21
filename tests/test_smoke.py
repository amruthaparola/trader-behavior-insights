"""Smoke test: the package imports and its config points at real data files."""

from sentiment_quant import config


def test_data_files_exist():
    assert config.TRADES_CSV.exists()
    assert config.SENTIMENT_CSV.exists()


def test_sentiment_order_is_complete():
    assert len(config.SENTIMENT_ORDER) == 5
