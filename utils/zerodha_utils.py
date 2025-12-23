# utils/zerodha_utils.py
"""
Replaced Zerodha-specific OHLC helpers with Fyers-powered functions.
All other code can keep importing:
    get_ohlc_15min(...)
    get_ohlc_daily(...)
so app.py does not need big changes.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from fyers_apiv3 import fyersModel

# ---------- FYERS SESSION ----------

FYERS_APP_ID = None      # set via init_fyers_session()
FYERS_ACCESS_TOKEN = None
_fyers_client: Optional[fyersModel.FyersModel] = None


def init_fyers_session(app_id: str, access_token: str):
    """
    Call this once from app.py after reading st.secrets.
    Example in app.py:
        from utils.zerodha_utils import init_fyers_session
        init_fyers_session(st.secrets["fyers_app_id"], st.secrets["fyers_access_token"])
    """
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

# ---------- SYMBOL HELPERS ----------

def to_fyers_symbol(symbol: str) -> str:
    """
    Convert simple symbol like 'RELIANCE' or 'TCS' to Fyers form 'NSE:RELIANCE-EQ'.
    If already looks like 'NSE:XYZ-EQ', return as-is.
    """
    if symbol.upper().startswith("NSE:"):
        return symbol.upper()
    return f"NSE:{symbol.upper()}-EQ"

# ---------- HISTORICAL OHLC HELPERS ----------

def get_ohlc_15min(symbol: str, days_back: int = 30) -> pd.DataFrame:
    """
    15‑minute candles from Fyers, returned in the same OHLC format
    your strategy / prepare_trend_squeeze already expects.
    """
    fyers = _ensure_client()

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)

    data = {
        "symbol": to_fyers_symbol(symbol),
        "resolution": "15",          # 15‑minute candles
        "date_format": "1",          # yyyy-mm-dd
        "range_from": from_date.strftime("%Y-%m-%d"),
        "range_to": to_date.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }

    resp = fyers.history(data=data)
    if resp.get("s") != "ok" or not resp.get("candles"):
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        resp["candles"],
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.astype({"open": float, "high": float, "low": float,
                    "close": float, "volume": float})
    df = df.set_index("timestamp").sort_index()
    return df


def get_ohlc_daily(symbol: str, lookback_days: int = 252) -> pd.DataFrame:
    """
    Daily candles from Fyers, same format as old Zerodha version.
    """
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
    if resp.get("s") != "ok" or not resp.get("candles"):
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        resp["candles"],
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.astype({"open": float, "high": float, "low": float,
                    "close": float, "volume": float})
    df = df.set_index("timestamp").sort_index()
    return df
# ---------- TRADING DAY HELPERS ----------

def is_trading_day(date_obj) -> bool:
    """
    Simple trading-day check:
    - Monday to Friday.
    - DOES NOT include NSE holiday list (your market-aware window already
      skips weekends; you can add holidays later if needed).
    """
    # date_obj can be datetime.date or datetime
    try:
        weekday = date_obj.weekday()
    except AttributeError:
        # If a string is passed, try to parse
        date_obj = pd.to_datetime(date_obj).date()
        weekday = date_obj.weekday()
    # 0=Monday ... 4=Friday
    return weekday < 5


def get_last_trading_day(today) -> datetime.date:
    """
    Return the most recent trading day on or before `today`.
    Weekend-aware only.
    """
    if isinstance(today, str):
        today = pd.to_datetime(today).date()
    elif isinstance(today, datetime):
        today = today.date()

    d = today
    # Walk backwards until a weekday (Mon–Fri)
    while not is_trading_day(d):
        d = d - timedelta(days=1)
    return d


# ---------- PLACEHOLDERS FOR OLD ZERODHA API ----------

def build_instrument_token_map(kite, symbols) -> Dict[str, int]:
    """
    Legacy placeholder kept for compatibility.

    Your new Fyers-based code does NOT need Zerodha tokens,
    but some parts of the app still call this. We simply return
    a dict mapping symbol → dummy token (index).

    Example:
        {"RELIANCE": 1, "TCS": 2, ...}
    """
    return {sym: i + 1 for i, sym in enumerate(symbols)}
