#!/usr/bin/env python3
"""
Fetch last 12 months of BTCUSDT perpetual futures 5-minute klines from Binance Futures API,
save raw CSV, compute indicators (EMA55-based Keltner-style bands at ±1 std and RSI-14),
handle pagination/rate limiting, fill any missing timestamps, and save final CSV.

Requirements:
- Python 3.8+
- requests, pandas, numpy

Usage:
    python fetch_btcusdt_futures_5m.py --months 12 --symbol BTCUSDT --interval 5m

Outputs (created under ./data):
    - btcusdt_perp_5m_raw.csv
    - btcusdt_perp_5m_with_indicators.csv
"""
from __future__ import annotations

import os
import time
import math
import argparse
import logging
from typing import List, Optional, Dict, Any

import requests
import pandas as pd
import numpy as np

BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"

# Binance REST limits are generous; we'll still be polite and backoff on errors.
MAX_LIMIT = 1500  # max klines per request
DEFAULT_INTERVAL = "5m"
DEFAULT_SYMBOL = "BTCUSDT"

# Map interval string to milliseconds
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def ms_to_ts(ms: int) -> pd.Timestamp:
    return pd.to_datetime(ms, unit="ms", utc=True)


def request_klines(
    symbol: str,
    interval: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = MAX_LIMIT,
    session: Optional[requests.Session] = None,
    max_retries: int = 5,
) -> List[List[Any]]:
    """Call Binance Futures klines endpoint with retry/backoff.

    Returns list of klines arrays. See https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
    """
    params: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, MAX_LIMIT),
    }
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    sess = session or requests.Session()

    backoff = 1.0
    for attempt in range(max_retries):
        try:
            resp = sess.get(BASE_URL + KLINES_ENDPOINT, params=params, timeout=30)
            if resp.status_code == 429:
                # Rate limit. Respect Retry-After if present; otherwise exponential backoff
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after is not None else backoff
                logging.warning("Rate limited (429). Sleeping %.2fs then retrying...", sleep_s)
                time.sleep(min(10.0, sleep_s))
                backoff = min(10.0, backoff * 2.0)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError(f"Unexpected response type: {type(data)} -> {data}")
            return data
        except (requests.RequestException, ValueError) as e:
            if attempt == max_retries - 1:
                logging.error("Failed to fetch klines after %d attempts: %s", attempt + 1, e)
                raise
            sleep_s = min(10.0, backoff)
            logging.warning("Request failed (attempt %d/%d): %s. Sleeping %.2fs...", attempt + 1, max_retries, e, sleep_s)
            time.sleep(sleep_s)
            backoff = min(10.0, backoff * 2.0)

    return []  # Unreachable, but for typing


def download_klines_last_months(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    months: int = 12,
) -> pd.DataFrame:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")

    now = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
    # Start X months ago (calendar months)
    start_ts = (now - pd.DateOffset(months=months)).floor("T")  # minute floor
    end_ts = now

    interval_ms = INTERVAL_MS[interval]
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    logging.info("Fetching %s %s klines from %s to %s", symbol, interval, start_ts.isoformat(), end_ts.isoformat())

    all_rows: List[List[Any]] = []
    sess = requests.Session()

    next_start = start_ms
    last_open_time = None
    requests_made = 0

    while True:
        batch = request_klines(
            symbol=symbol,
            interval=interval,
            start_ms=next_start,
            end_ms=end_ms,
            limit=MAX_LIMIT,
            session=sess,
        )
        requests_made += 1
        if not batch:
            logging.info("No more data returned. Breaking.")
            break

        # Prevent duplicates at boundaries
        if last_open_time is not None:
            batch = [row for row in batch if row[0] > last_open_time]

        if not batch:
            logging.info("Batch only contained duplicates. Breaking.")
            break

        all_rows.extend(batch)
        last_open_time = batch[-1][0]

        # Increment start to next candle
        next_start = last_open_time + interval_ms

        # Stop if we've reached or passed end time
        if last_open_time >= end_ms - interval_ms:
            break

        # Small friendly delay to avoid bursts
        time.sleep(0.05)

    logging.info("Total klines fetched: %d in %d requests", len(all_rows), requests_made)

    if not all_rows:
        raise RuntimeError("No data fetched from Binance.")

    # Build DataFrame
    cols = [
        "open_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time_ms",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)

    # Keep only required fields + times
    df = df[["open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume", "quote_volume"]]

    # Convert types
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = df["open_time_ms"].apply(ms_to_ts)
    df["close_time"] = df["close_time_ms"].apply(ms_to_ts)

    # Sort ascending by open_time
    df = df.sort_values("open_time").reset_index(drop=True)

    return df


def fill_missing_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Ensure a continuous time series by inserting missing 5-minute bars.

    Strategy:
    - Reindex to a complete 5-minute DateTimeIndex
    - For inserted candles: set O/H/L/C to previous close, volumes to 0
    - This preserves continuity for indicators while not inventing volume
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")

    if df.empty:
        return df

    df = df.copy()
    df = df.set_index("open_time")
    df.index = pd.to_datetime(df.index, utc=True)

    interval_ms = INTERVAL_MS[interval]
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=pd.to_timedelta(interval_ms, unit="ms"), tz="UTC")
    before_count = len(df)
    df = df.reindex(full_index)

    # Forward fill close first (for inserted rows)
    df["close"] = df["close"].ffill()

    # For rows that were missing (NaNs in open), set O/H/L to the filled close
    missing_mask = df["open"].isna()

    # The above line is a bit hacky to avoid SettingWithCopy warnings; do simple assigns:
    df.loc[missing_mask, "open"] = df.loc[missing_mask, "close"]
    df.loc[missing_mask, "high"] = df.loc[missing_mask, "close"]
    df.loc[missing_mask, "low"] = df.loc[missing_mask, "close"]

    # Volumes -> 0 for inserted rows
    df["volume"] = df["volume"].fillna(0.0)
    df["quote_volume"] = df["quote_volume"].fillna(0.0)

    # Carry over close_time_ms sensibly: set to open_time + interval
    interval_ms = INTERVAL_MS[interval]
    df["open_time_ms"] = (df.index.view("i8") // 10**6).astype("int64")
    df["close_time_ms"] = df["open_time_ms"] + interval_ms
    df["close_time"] = df["close_time_ms"].apply(ms_to_ts)

    after_count = len(df)
    missing_count = after_count - before_count
    if missing_count > 0:
        logging.info("Inserted %d missing candles to ensure continuity.", missing_count)

    df = df.reset_index().rename(columns={"index": "open_time"})

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA55, Keltner-style ±1 std bands (using rolling std of close), and RSI-14."""
    df = df.copy()

    # EMA 55 of close
    df["ema_55"] = df["close"].ewm(span=55, adjust=False).mean()

    # Rolling standard deviation over 55 periods
    df["std_55"] = df["close"].rolling(window=55, min_periods=55).std()

    df["kc_upper"] = df["ema_55"] + df["std_55"]
    df["kc_lower"] = df["ema_55"] - df["std_55"]

    # RSI-14 (Wilder's smoothing)
    df["rsi_14"] = rsi_wilder(df["close"], period=14)

    return df


def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing equivalent to EMA with alpha=1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle division by zero (avg_loss==0 -> RSI=100)
    rsi = rsi.where(avg_loss != 0, 100.0)

    return rsi


def ensure_data_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance Futures BTCUSDT 5m data and compute indicators.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading pair symbol (e.g., BTCUSDT)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Kline interval (e.g., 5m)")
    parser.add_argument("--months", type=int, default=12, help="Number of months back from now to fetch")
    parser.add_argument("--outdir", default="data", help="Directory to save CSV files")
    parser.add_argument("--log", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    ensure_data_dir(args.outdir)

    try:
        df_raw = download_klines_last_months(symbol=args.symbol, interval=args.interval, months=args.months)
    except Exception as e:
        logging.exception("Error while downloading klines: %s", e)
        raise SystemExit(1)

    # Save raw CSV
    raw_path = os.path.join(
        args.outdir, f"{args.symbol.lower()}_perp_{args.interval}_raw.csv"
    )

    # Reorder columns for readability
    raw_cols = [
        "open_time",
        "open_time_ms",
        "close_time",
        "close_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]
    df_raw[raw_cols].to_csv(raw_path, index=False)
    logging.info("Saved raw data to %s (%d rows)", raw_path, len(df_raw))

    # Fill missing candles if any
    df_full = fill_missing_candles(df_raw, args.interval)

    # Compute indicators
    df_final = compute_indicators(df_full)

    # Ensure chronological order
    df_final = df_final.sort_values("open_time").reset_index(drop=True)

    # Save final CSV
    final_path = os.path.join(
        args.outdir, f"{args.symbol.lower()}_perp_{args.interval}_with_indicators.csv"
    )

    df_final.to_csv(final_path, index=False)
    logging.info("Saved final dataset with indicators to %s (%d rows)", final_path, len(df_final))

    print("Completed: data download and processing finished successfully.")


if __name__ == "__main__":
    main()

