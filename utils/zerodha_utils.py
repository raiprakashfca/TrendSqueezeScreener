# utils/zerodha_utils.py
"""
Fyers-powered OHLC helpers (drop-in replacement for old Zerodha helpers).

Key improvements:
- Candle timestamps are normalized to IST (tz-naive) at the source.
- Simple retry/backoff for FYERS history calls.
- De-dup + sorting safety.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional
from zoneinfo import ZoneInfo
import time as _time

import pandas as pd
from fyers_apiv3 import fyersModel

IST = ZoneInfo("Asia/Kolkata")

FYERS_APP_ID = None
FYERS_ACCESS_TOKEN = None
_fyers_client: Optional[fyersModel.FyersModel] = None


def init_fyers_session(app_id: str, access_token: str):
    global FYERS_APP_ID, FYERS_ACCESS_TOKEN, _fyers_client
    FYERS_APP_ID = (app_id or "").strip()
    FYERS_ACCESS_TOKEN = (access_token or "").strip()

    if not FYERS_APP_ID or not FYERS_ACCESS_TOKEN:
        raise ValueError("FYERS app_id or access_token missing/empty.")

    _fyers_client = fyersModel.FyersModel(
        client_id=FYERS_APP_ID,
        token=FYERS_ACCESS_TOKEN,
        log_path=""
    )


def _ensure_client() -> fyersModel.FyersModel:
    if _fyers_client is None:
        raise RuntimeError("Fyers client not initialized. Call init_fyers_session() first.")
    return _fyers_client


def to_fyers_symbol(symbol: str) -> str:
    """
    Convert simple symbol like 'RELIANCE' to Fyers form 'NSE:RELIANCE-EQ'.
    If already 'NSE:XYZ-EQ', return as-is.
    """
    s = (symbol or "").strip().upper()
    if s.startswith("NSE:"):
        return s
    return f"NSE:{s}-EQ"


def _epoch_to_ist_naive(epoch_series) -> pd.Series:
    # FYERS returns epoch seconds in UTC. Convert -> IST -> tz-naive.
    return (
        pd.to_datetime(epoch_series, unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )


def _history_with_retry(fyers, data: dict, retries: int = 2) -> dict:
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = fyers.history(data=data)
            return resp if isinstance(resp, dict) else {}
        except Exception as e:
            last_err = e
            # small exponential backoff
            _time.sleep(0.6 * (2 ** attempt))
    raise RuntimeError(f"FYERS history failed after retries: {last_err}")


def get_ohlc_15min(symbol: str, days_back: int = 30) -> pd.DataFrame:
    fyers = _ensure_client()

    now_ist = datetime.now(IST)
    to_date = now_ist.date()
    from_date = (now_ist - timedelta(days=days_back)).date()

    data = {
        "symbol": to_fyers_symbol(symbol),
        "resolution": "15",
        "date_format": "1",
        "range_from": from_date.strftime("%Y-%m-%d"),
        "range_to": to_date.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }

    resp = _history_with_retry(fyers, data=data, retries=2)
    if resp.get("s") != "ok" or not resp.get("candles"):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(resp["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = _epoch_to_ist_naive(df["timestamp"])
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    df = df.set_index("timestamp").sort_index()

    # de-dup safety (FYERS can occasionally return repeated epoch rows)
    df = df[~df.index.duplicated(keep="last")]
    return df


def get_ohlc_daily(symbol: str, lookback_days: int = 252) -> pd.DataFrame:
    fyers = _ensure_client()

    now_ist = datetime.now(IST)
    to_date = now_ist.date()
    from_date = (now_ist - timedelta(days=lookback_days)).date()

    data = {
        "symbol": to_fyers_symbol(symbol),
        "resolution": "D",
        "date_format": "1",
        "range_from": from_date.strftime("%Y-%m-%d"),
        "range_to": to_date.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }

    resp = _history_with_retry(fyers, data=data, retries=2)
    if resp.get("s") != "ok" or not resp.get("candles"):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(resp["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = _epoch_to_ist_naive(df["timestamp"])
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    df = df.set_index("timestamp").sort_index()

    df = df[~df.index.duplicated(keep="last")]
    return df


def is_trading_day(date_obj) -> bool:
    # Weekend-aware only (you can add NSE holidays later if you want)
    if isinstance(date_obj, str):
        date_obj = pd.to_datetime(date_obj).date()
    elif isinstance(date_obj, datetime):
        date_obj = date_obj.date()
    return date_obj.weekday() < 5


def get_last_trading_day(today) -> datetime.date:
    if isinstance(today, str):
        today = pd.to_datetime(today).date()
    elif isinstance(today, datetime):
        today = today.date()

    d = today
    while not is_trading_day(d):
        d = d - timedelta(days=1)
    return d


def build_instrument_token_map(kite, symbols) -> Dict[str, int]:
    # Legacy placeholder for compatibility
    return {sym: i + 1 for i, sym in enumerate(symbols)}
