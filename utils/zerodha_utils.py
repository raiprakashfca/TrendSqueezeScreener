# utils/zerodha_utils.py
"""
FYERS-powered OHLC helpers.

Keeps the same interface:
    get_ohlc_15min(...)
    get_ohlc_daily(...)
so app.py does not need big changes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo
import time as _time

import pandas as pd
from fyers_apiv3 import fyersModel

# ---------- FYERS SESSION ----------
FYERS_APP_ID: Optional[str] = None
FYERS_ACCESS_TOKEN: Optional[str] = None
_fyers_client: Optional[fyersModel.FyersModel] = None

IST = ZoneInfo("Asia/Kolkata")

# ---------- SIMPLE TTL CACHE (persists across reruns in same session) ----------
# key -> (expires_epoch, df)
_CACHE: Dict[Tuple[str, str, str, str], Tuple[float, pd.DataFrame]] = {}
TTL_15M = 300     # 5 min
TTL_DAILY = 1800  # 30 min


def init_fyers_session(app_id: str, access_token: str):
    """Initialize FYERS client once per app run."""
    global FYERS_APP_ID, FYERS_ACCESS_TOKEN, _fyers_client
    FYERS_APP_ID = (app_id or "").strip()
    FYERS_ACCESS_TOKEN = (access_token or "").strip()

    if not FYERS_APP_ID or not FYERS_ACCESS_TOKEN:
        raise ValueError("init_fyers_session(): missing app_id or access_token")

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
    """Convert 'RELIANCE' -> 'NSE:RELIANCE-EQ'. If already in FYERS form, return as-is."""
    s = (symbol or "").strip().upper()
    if s.startswith("NSE:"):
        return s
    return f"NSE:{s}-EQ"


def _today_ist() -> date:
    return datetime.now(IST).date()


def _dt_to_ist_naive_from_epoch_seconds(sec_series: pd.Series) -> pd.Series:
    """
    FYERS candles timestamps are epoch seconds.
    Epoch seconds are UTC. Convert: UTC -> IST -> tz-naive.
    """
    ts = pd.to_datetime(sec_series, unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return ts


def _fyers_history_with_retry(fyers: fyersModel.FyersModel, payload: dict, tries: int = 3, base_sleep: float = 0.6) -> dict:
    """
    Retry FYERS history on transient errors / throttling.
    """
    last_err = None
    for i in range(tries):
        try:
            resp = fyers.history(data=payload)
            # Typical good response: {"s":"ok","candles":[...]}
            if isinstance(resp, dict) and resp.get("s") == "ok":
                return resp

            # Soft-fail cases: rate limit / token expiry / intermittent failures
            last_err = resp
            msg = str(resp)[:200].lower()
            if any(k in msg for k in ["rate", "limit", "too many", "throttle", "temporarily", "timeout", "gateway"]):
                _time.sleep(base_sleep * (2 ** i))
                continue

            # Non-transient => don't keep retrying
            return resp

        except Exception as e:
            last_err = e
            _time.sleep(base_sleep * (2 ** i))

    # Final attempt return as dict-like error
    return {"s": "error", "message": f"history failed after retries: {last_err}"}


def _cache_get(cache_key: Tuple[str, str, str, str]) -> Optional[pd.DataFrame]:
    item = _CACHE.get(cache_key)
    if not item:
        return None
    exp, df = item
    if _time.time() > exp:
        _CACHE.pop(cache_key, None)
        return None
    return df


def _cache_set(cache_key: Tuple[str, str, str, str], df: pd.DataFrame, ttl_sec: int):
    _CACHE[cache_key] = (_time.time() + ttl_sec, df)


def _history_df(symbol: str, resolution: str, range_from: str, range_to: str, ttl_sec: int) -> pd.DataFrame:
    fyers = _ensure_client()
    fy_symbol = to_fyers_symbol(symbol)

    cache_key = (fy_symbol, resolution, range_from, range_to)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached.copy()

    payload = {
        "symbol": fy_symbol,
        "resolution": resolution,  # "15" or "D"
        "date_format": "1",        # yyyy-mm-dd
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1",
    }

    resp = _fyers_history_with_retry(fyers, payload)
    candles = (resp or {}).get("candles") if isinstance(resp, dict) else None

    if not candles:
        df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        _cache_set(cache_key, df_empty, ttl_sec)
        return df_empty.copy()

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])

    # ✅ Correct timezone handling
    df["timestamp"] = _dt_to_ist_naive_from_epoch_seconds(df["timestamp"])

    # Numeric types
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    df = df.set_index("timestamp").sort_index()

    _cache_set(cache_key, df, ttl_sec)
    return df.copy()


def get_ohlc_15min(symbol: str, days_back: int = 30) -> pd.DataFrame:
    """
    15-minute candles from FYERS.
    Returns indexed by tz-naive IST timestamps.
    """
    end = _today_ist()
    start = end - timedelta(days=int(days_back))
    return _history_df(symbol, "15", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), TTL_15M)


def get_ohlc_daily(symbol: str, lookback_days: int = 252) -> pd.DataFrame:
    """
    Daily candles from FYERS.
    Returns indexed by tz-naive IST timestamps (date boundary in IST).
    """
    end = _today_ist()
    start = end - timedelta(days=int(lookback_days))
    return _history_df(symbol, "D", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), TTL_DAILY)


# ---------- TRADING DAY HELPERS ----------
def is_trading_day(date_obj) -> bool:
    """Weekday-only trading day check (no NSE holidays)."""
    try:
        weekday = date_obj.weekday()
    except AttributeError:
        date_obj = pd.to_datetime(date_obj).date()
        weekday = date_obj.weekday()
    return weekday < 5


def get_last_trading_day(today) -> date:
    """Return the most recent weekday trading day on or before `today`."""
    if isinstance(today, str):
        today = pd.to_datetime(today).date()
    elif isinstance(today, datetime):
        today = today.date()

    d = today
    while not is_trading_day(d):
        d = d - timedelta(days=1)
    return d


# ---------- LEGACY PLACEHOLDER ----------
def build_instrument_token_map(kite, symbols) -> Dict[str, int]:
    """Legacy placeholder for compatibility."""
    return {sym: i + 1 for i, sym in enumerate(symbols)}
