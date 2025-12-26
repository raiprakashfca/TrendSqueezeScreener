# utils/zerodha_utils.py
"""
Fyers-powered OHLC helpers (replacing Zerodha-style functions).
Returns IST-naive timestamps so app/UI stays consistent.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
from fyers_apiv3 import fyersModel

FYERS_APP_ID = None
FYERS_ACCESS_TOKEN = None
_fyers_client: Optional[fyersModel.FyersModel] = None


def init_fyers_session(app_id: str, access_token: str):
    global FYERS_APP_ID, FYERS_ACCESS_TOKEN, _fyers_client
    FYERS_APP_ID = app_id
    FYERS_ACCESS_TOKEN = access_token

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
    if symbol.upper().startswith("NSE:"):
        return symbol.upper()
    return f"NSE:{symbol.upper()}-EQ"


def _candles_to_df(resp) -> pd.DataFrame:
    if resp.get("s") != "ok" or not resp.get("candles"):
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        resp["candles"],
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # FYERS gives epoch seconds => UTC
    t = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    t = t.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)  # IST-naive
    df["timestamp"] = t

    df = df.astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float
    })

    df = df.set_index("timestamp").sort_index()
    return df


def get_ohlc_15min(symbol: str, days_back: int = 30) -> pd.DataFrame:
    fyers = _ensure_client()
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)

    data = {
        "symbol": to_fyers_symbol(symbol),
        "resolution": "15",
        "date_format": "1",
        "range_from": from_date.strftime("%Y-%m-%d"),
        "range_to": to_date.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }

    resp = fyers.history(data=data)
    return _candles_to_df(resp)


def get_ohlc_daily(symbol: str, lookback_days: int = 252) -> pd.DataFrame:
    fyers = _ensure_client()
    to_date = datetime.now()
    from_date = to_date - timedelta(days=lookback_days)

    data = {
        "symbol": to_fyers_symbol(symbol),
        "resolution": "D",
        "date_format": "1",
        "range_from": from_date.strftime("%Y-%m-%d"),
        "range_to": to_date.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }

    resp = fyers.history(data=data)
    return _candles_to_df(resp)


def is_trading_day(date_obj) -> bool:
    try:
        weekday = date_obj.weekday()
    except AttributeError:
        date_obj = pd.to_datetime(date_obj).date()
        weekday = date_obj.weekday()
    return weekday < 5


def get_last_trading_day(today):
    if isinstance(today, str):
        today = pd.to_datetime(today).date()
    elif isinstance(today, datetime):
        today = today.date()

    d = today
    while not is_trading_day(d):
        d = d - timedelta(days=1)
    return d


def build_instrument_token_map(kite, symbols) -> Dict[str, int]:
    return {sym: i + 1 for i, sym in enumerate(symbols)}
