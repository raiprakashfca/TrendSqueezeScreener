"""
FYERS utilities (data + time normalization + market-day helpers)

Key features:
- init_fyers_session(): initializes a singleton FYERS client
- get_ohlc_15min_range(): fetches intraday data for a date range using chunking + retries
- get_ohlc_15min(): convenience wrapper (days_back)
- get_ohlc_daily_range(), get_ohlc_daily(): daily data fetch
- normalize_ohlc_index_to_ist(): converts candle index to IST-naive safely
- is_trading_day(), get_last_trading_day(): simple weekday-based trading calendar
"""

from __future__ import annotations

import time as _time
from datetime import datetime, date, timedelta
from typing import Optional, Union, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")

_fyers = None  # singleton session


# -----------------------------------------------------------------------------
# Session
# -----------------------------------------------------------------------------
def init_fyers_session(app_id: str, access_token: str) -> None:
    """
    Initialize FYERS session (singleton).
    """
    global _fyers
    from fyersapiv3 import fyersModel

    access_token = (access_token or "").strip()
    app_id = (app_id or "").strip()

    if not app_id or not access_token:
        raise ValueError("Missing FYERS app_id or access_token")

    _fyers = fyersModel.FyersModel(
        client_id=app_id,
        token=access_token,
        is_async=False,
        log_path="",
    )


def get_fyers_session():
    return _fyers


# -----------------------------------------------------------------------------
# Date helpers
# -----------------------------------------------------------------------------
def _to_date(d: Union[str, date, datetime]) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    # string
    return pd.to_datetime(d).date()


def _daterange_chunks(start: date, end: date, max_days: int) -> List[Tuple[date, date]]:
    """
    Inclusive chunking: [start..end], each chunk length <= max_days.
    """
    if end < start:
        return []
    out = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=max_days - 1), end)
        out.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return out


# -----------------------------------------------------------------------------
# Market-day helpers (simple weekday logic)
# -----------------------------------------------------------------------------
def is_trading_day(d: Union[date, datetime]) -> bool:
    """
    Simple NSE trading-day heuristic: Mon-Fri only.
    (Does not include exchange holidays.)
    """
    dd = d.date() if isinstance(d, datetime) else d
    return dd.weekday() < 5


def get_last_trading_day(d: Union[date, datetime]) -> date:
    """
    Returns the most recent Mon-Fri day on/ before d.
    """
    dd = d.date() if isinstance(d, datetime) else d
    while not is_trading_day(dd):
        dd = dd - timedelta(days=1)
    return dd


# -----------------------------------------------------------------------------
# Timezone normalization
# -----------------------------------------------------------------------------
def normalize_ohlc_index_to_ist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix timezone issues WITHOUT creating "future candles".

    Rules:
    - tz-aware index -> convert to IST and drop tz (IST-naive)
    - tz-naive index -> try UTC->IST; if that pushes last candle into the future,
      assume it was already IST-naive and keep it unchanged.
    """
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    out = df.copy()
    idx = out.index

    # tz-aware -> safe convert
    if getattr(idx, "tz", None) is not None:
        try:
            out.index = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        except Exception:
            pass
        return out

    # tz-naive -> attempt UTC->IST conversion
    try:
        idx_utc_to_ist = idx.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
    except Exception:
        return out

    # Future-check heuristic (3 min tolerance)
    try:
        now_local = datetime.now(IST).replace(tzinfo=None)
        last_conv = pd.Timestamp(idx_utc_to_ist[-1])
        if last_conv > now_local + timedelta(minutes=3):
            # likely already IST-naive
            out.index = idx
        else:
            out.index = idx_utc_to_ist
    except Exception:
        out.index = idx_utc_to_ist

    return out


# -----------------------------------------------------------------------------
# FYERS History fetch (core)
# -----------------------------------------------------------------------------
def _history_call(
    symbol: str,
    resolution: Union[int, str],
    range_from: date,
    range_to: date,
    cont_flag: int = 1,
    retries: int = 3,
    backoff_seconds: float = 1.0,
) -> dict:
    fy = get_fyers_session()
    if fy is None:
        raise RuntimeError("Fyers client not initialized. Call init_fyers_session")

    payload = {
        "symbol": f"NSE:{symbol}-EQ",
        "resolution": str(resolution),
        "date_format": "1",  # yyyy-mm-dd
        "range_from": range_from.strftime("%Y-%m-%d"),
        "range_to": range_to.strftime("%Y-%m-%d"),
        "cont_flag": str(cont_flag),
    }

    last_err = None
    for i in range(retries):
        try:
            resp = fy.history(payload)
            return resp
        except Exception as e:
            last_err = e
            _time.sleep(backoff_seconds * (2 ** i))

    raise RuntimeError(f"FYERS history failed after retries: {last_err}")


def _candles_to_df(candles: list) -> pd.DataFrame:
    """
    FYERS 'candles' format: [timestamp, open, high, low, close, volume]
    Timestamp is epoch seconds.
    """
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


# -----------------------------------------------------------------------------
# Public: Intraday 15M (range + days_back)
# -----------------------------------------------------------------------------
def get_ohlc_15min_range(
    symbol: str,
    start: Union[str, date, datetime],
    end: Union[str, date, datetime],
    *,
    chunk_days: int = 90,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Fetch 15-minute candles for a specific date range, chunked to respect
    FYERS intraday limits (~100 days/request).
    """
    start_d = _to_date(start)
    end_d = _to_date(end)
    if end_d < start_d:
        return None

    chunks = _daterange_chunks(start_d, end_d, max_days=int(chunk_days))
    frames: List[pd.DataFrame] = []

    for (c_from, c_to) in chunks:
        resp = _history_call(
            symbol=symbol,
            resolution=15,
            range_from=c_from,
            range_to=c_to,
            cont_flag=1,
            retries=retries,
        )

        status = str(resp.get("s", "")).lower()
        if status == "ok" and resp.get("candles"):
            frames.append(_candles_to_df(resp["candles"]))
            continue

        # no_data is allowed, just skip
        if status in ("no_data", "nodata") or not resp.get("candles"):
            continue

        # any other status -> fail fast with message
        raise RuntimeError(f"FYERS history error for {symbol} {c_from}..{c_to}: {resp}")

    if not frames:
        return None

    df = pd.concat(frames, axis=0).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = normalize_ohlc_index_to_ist(df)
    return df


def get_ohlc_15min(symbol: str, days_back: int = 75) -> Optional[pd.DataFrame]:
    """
    Convenience wrapper: fetch last N calendar days of 15-minute candles.
    """
    end = datetime.now(IST).date()
    start = end - timedelta(days=int(days_back))
    return get_ohlc_15min_range(symbol, start, end, chunk_days=min(90, max(5, int(days_back))), retries=3)


# -----------------------------------------------------------------------------
# Public: Daily (range + lookback_days)
# -----------------------------------------------------------------------------
def get_ohlc_daily_range(
    symbol: str,
    start: Union[str, date, datetime],
    end: Union[str, date, datetime],
    *,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily candles for a specific date range.
    """
    start_d = _to_date(start)
    end_d = _to_date(end)
    if end_d < start_d:
        return None

    resp = _history_call(
        symbol=symbol,
        resolution="D",
        range_from=start_d,
        range_to=end_d,
        cont_flag=1,
        retries=retries,
    )

    status = str(resp.get("s", "")).lower()
    if status != "ok" or not resp.get("candles"):
        return None

    df = _candles_to_df(resp["candles"])
    df = normalize_ohlc_index_to_ist(df)
    return df


def get_ohlc_daily(symbol: str, lookback_days: int = 252) -> Optional[pd.DataFrame]:
    """
    Convenience wrapper: fetch last N calendar days of daily candles.
    """
    end = datetime.now(IST).date()
    start = end - timedelta(days=int(lookback_days))
    return get_ohlc_daily_range(symbol, start, end, retries=3)
