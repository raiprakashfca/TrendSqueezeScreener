import pandas as pd
from datetime import datetime, time, timedelta
import streamlit as st
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# NSE holidays
indian_market_holidays = [
    "2025-01-26", "2025-03-29", "2025-04-10", "2025-04-14", "2025-05-01",
    "2025-08-15", "2025-10-02", "2025-11-03", "2025-12-25",
]

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

def is_trading_day(date_obj) -> bool:
    """Return True if given date is a regular NSE trading day."""
    return date_obj.weekday() < 5 and date_obj.isoformat() not in indian_market_holidays

def get_last_trading_day(reference_day):
    """Walk backward from reference_day until a valid trading day is found."""
    day = reference_day
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day

def _get_session_mode():
    """Determine whether we are in live-market hours or off-market."""
    now_ist = datetime.now(IST)
    today = now_ist.date()
    current_time = now_ist.time()
    
    if is_trading_day(today) and MARKET_OPEN <= current_time <= MARKET_CLOSE:
        return "live", now_ist, today
    else:
        return "eod", now_ist, today

def get_ohlc_15min(
    kite,
    instrument_token: int,
    lookback_days: int = 10,
    min_candles: int = 60,
    show_debug: bool = False,
):
    """Fetch 15-minute OHLCV data (existing function unchanged)."""
    mode, now_ist, today = _get_session_mode()
    
    if mode == "live":
        from_date = datetime.combine(today - timedelta(days=lookback_days), MARKET_OPEN)
        to_date = now_ist.replace(tzinfo=None)
        label = "LIVE intraday"
    else:
        last_trading = get_last_trading_day(today)
        from_date = datetime.combine(last_trading - timedelta(days=lookback_days), MARKET_OPEN)
        to_date = datetime.combine(last_trading, MARKET_CLOSE)
        label = f"EOD session ({last_trading.isoformat()})"
    
    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval="15minute",
        )
    except Exception as e:
        st.warning(f"{instrument_token}: Failed to fetch historical data ({label}): {e}")
        return None
    
    df = pd.DataFrame(data)
    if df.empty:
        st.warning(f"{instrument_token}: Empty historical data for {label}.")
        return None
    
    df["datetime"] = pd.to_datetime(df["date"])
    df.set_index("datetime", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    
    if df.shape[0] < min_candles:
        if show_debug:
            st.warning(
                f"{instrument_token}: Only {df.shape[0]} candles fetched "
                f"(min recommended {min_candles}) for {label}."
            )
        return df
    
    if show_debug:
        last_ts = df.index[-1]
        st.write(
            f"{instrument_token}: Using {df.shape[0]} candles up to "
            f"{last_ts.strftime('%d %b %Y %H:%M')} IST • {label}"
        )
    
    return df

def get_ohlc_daily(
    kite,
    instrument_token: int,
    lookback_days: int = 252,
    min_candles: int = 100,
    show_debug: bool = False,
):
    """Fetch daily OHLCV data for longer-term context."""
    mode, now_ist, today = _get_session_mode()
    
    if mode == "live":
        from_date = datetime.combine(today - timedelta(days=lookback_days), time(0, 0))
        to_date = now_ist.replace(tzinfo=None)
        label = "LIVE daily"
    else:
        last_trading = get_last_trading_day(today)
        from_date = datetime.combine(last_trading - timedelta(days=lookback_days), time(0, 0))
        to_date = datetime.combine(last_trading + timedelta(days=1), time(0, 0))
        label = f"EOD daily ({last_trading.isoformat()})"
    
    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval="day",
        )
    except Exception as e:
        if show_debug:
            st.warning(f"{instrument_token}: Daily fetch failed ({label}): {e}")
        return None
    
    df = pd.DataFrame(data)
    if df.empty:
        if show_debug:
            st.warning(f"{instrument_token}: Empty daily data for {label}.")
        return None
    
    df["datetime"] = pd.to_datetime(df["date"])
    df.set_index("datetime", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    
    if df.shape[0] < min_candles:
        if show_debug:
            st.warning(
                f"{instrument_token}: Only {df.shape[0]} daily candles "
                f"(min {min_candles}) for {label}."
            )
        return df
    
    if show_debug:
        last_ts = df.index[-1]
        st.write(
            f"{instrument_token}: {df.shape[0]} daily candles up to "
            f"{last_ts.strftime('%d %b %Y')} • {label}"
        )
    
    return df

def build_instrument_token_map(
    kite,
    symbols,
    exchange: str = "NSE",
    chunk_size: int = 50,
):
    """Resolve NSE symbols to instrument tokens (unchanged)."""
    token_map = {}
    instruments = [f"{exchange}:{s}" for s in symbols]
    
    for i in range(0, len(instruments), chunk_size):
        chunk = instruments[i : i + chunk_size]
        try:
            ltp_data = kite.ltp(chunk)
        except Exception as e:
            st.warning(f"Failed to fetch LTP for {chunk}: {e}")
            continue
        
        for inst, payload in ltp_data.items():
            symbol = inst.split(":")[1]
            token = payload.get("instrument_token")
            if token:
                token_map[symbol] = token
    
    missing = [s for s in symbols if s not in token_map]
    if missing:
        st.warning(f"No instrument token found for: {', '.join(missing)}")
    
    return token_map
