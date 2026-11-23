"""Daily BTC price series for the backtest.

The trade file only has BTC fills on 288 of 515 days, so we use an external,
public, complete daily series instead: Binance BTCUSDT 1-day candles (UTC days,
the same day convention as the Fear & Greed index), with Bitstamp BTC/USD as a
fallback if Binance is unreachable from your region.

`python -m sentiment_quant prices` downloads it once to csv_files/btc_usd_daily.csv;
everything else reads the saved file, so results don't change between runs.
"""

from __future__ import annotations

import json
import time
import urllib.request

import pandas as pd

from . import config

BTC_CSV = config.DATA_DIR / "btc_usd_daily.csv"
START = "2018-02-01"  # first day of the Fear & Greed index
END = "2025-05-02"  # last day of the Fear & Greed file


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "sentiment-quant-research/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)


def fetch_binance(start: str = START, end: str = END) -> pd.DataFrame:
    rows, cursor, stop = [], _ms(start), _ms(end) + 86_400_000
    while cursor < stop:
        url = (
            "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d"
            f"&startTime={cursor}&endTime={stop - 1}&limit=1000"
        )
        batch = _get_json(url)
        if not batch:
            break
        rows += batch
        cursor = batch[-1][0] + 86_400_000
        time.sleep(0.2)  # be polite to a free public API
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df["source"] = "binance BTCUSDT"
    return df[["date", "open", "high", "low", "close", "volume", "source"]]


def fetch_bitstamp(start: str = START, end: str = END) -> pd.DataFrame:
    rows, cursor, stop = [], _ms(start) // 1000, _ms(end) // 1000 + 86_400
    while cursor < stop:
        url = f"https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400&limit=1000&start={cursor}"
        batch = _get_json(url)["data"]["ohlc"]
        if not batch:
            break
        rows += batch
        cursor = int(batch[-1]["timestamp"]) + 86_400
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df["source"] = "bitstamp BTC/USD"
    return df[["date", "open", "high", "low", "close", "volume", "source"]]


def download_btc(path=BTC_CSV) -> pd.DataFrame:
    try:
        df = fetch_binance()
    except Exception as exc:  # e.g. HTTP 451 when Binance is geo-blocked
        print(f"Binance failed ({exc}); falling back to Bitstamp")
        df = fetch_bitstamp()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates("date").sort_values("date")
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    df.to_csv(path, index=False)
    return df


def load_btc(path=BTC_CSV) -> pd.DataFrame:
    """Saved daily BTC candles; raises with instructions if not downloaded yet."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m sentiment_quant prices` once")
    df = pd.read_csv(path, parse_dates=["date"])
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    if len(full) != len(df):
        raise ValueError(f"BTC series has {len(full) - len(df)} missing days")
    return df
