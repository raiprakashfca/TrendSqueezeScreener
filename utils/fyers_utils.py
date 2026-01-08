"""
FYERS Data Utilities
Handles data fetching, timezone normalization, and market day logic.
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta, time
from typing import Optional, Tuple, Dict
from zoneinfo import ZoneInfo

# FYERS session (singleton)
_fyers_session = None


def init_fyers_session(app_id: str, access_token: str) -> None:
    """
    Initialize FYERS session (called once per backtest).
    """
    global _fyers_session
    from fyersapiv3 import fyersModel

    session = fyersModel.SessionModel()
    session.set_token(access_token)
    _fyers_session = fyersModel.FyersModel(client_id=app_id, token=access_token, is_async=False)


def get_fyers_session() -> Optional["fyersModel.FyersModel"]:
    """
    Return the initialized FYERS session.
    """
    global _fyers_session
    return _fyers_session


def get_ohlc_15min(symbol: str, days_back: int = 75) -> Optional[pd.DataFrame]:
    """
    Fetch 15-minute OHLC data for symbol using FYERS API.
    days_back defaults to 75 to stay within FYERS limits.
    """
    session = get_fyers_session()
    if not session:
        return None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    try:
        data = {
            "symbol": f"NSE:{symbol}-EQ",
            "resolution": "15",
            "date_format": "1",
            "range_from": start_date.strftime("%Y-%m-%d"),
            "range_to": end_date.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }
        response = session.history(data)
        if response.get("s") == "ok" and response.get("candles"):
            df = pd.DataFrame(
                response["candles"],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"]
