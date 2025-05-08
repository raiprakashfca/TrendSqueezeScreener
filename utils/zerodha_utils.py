import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

# NSE holidays – expand this for accuracy
indian_market_holidays = [
    "2025-01-26", "2025-03-29", "2025-04-10", "2025-04-14", "2025-05-01",
    "2025-08-15", "2025-10-02", "2025-11-03", "2025-12-25"
]

def is_trading_day(date):
    return date.weekday() < 5 and str(date) not in indian_market_holidays

def get_last_trading_day(reference_day, skip_today_if_trading=True):
    day = reference_day
    if skip_today_if_trading and is_trading_day(day):
        day -= timedelta(days=1)
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day

def fetch_ohlc_for_date(kite, symbol, date_obj):
    instrument = f"NSE:{symbol}"
    from_dt = datetime.combine(date_obj, datetime.strptime("09:15:00", "%H:%M:%S").time())
    to_dt = datetime.combine(date_obj, datetime.strptime("15:30:00", "%H:%M:%S").time())

    try:
        ltp_data = kite.ltp([instrument])
        instrument_token = ltp_data[instrument]['instrument_token']
    except Exception as e:
        st.warning(f"{symbol}: Failed to fetch LTP: {e}")
        return pd.DataFrame()

    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_dt,
            to_date=to_dt,
            interval='15minute'
        )
        df = pd.DataFrame(data)
        if not df.empty:
            df.rename(columns={
                'open': 'open', 'high': 'high', 'low': 'low',
                'close': 'close', 'volume': 'volume'
            }, inplace=True)
        return df
    except Exception as e:
        st.warning(f"{symbol}: Failed to fetch candles for {date_obj}: {e}")
        return pd.DataFrame()

def get_ohlc_15min(kite, symbol):
    today = datetime.now().date()

    # Try today's intraday data first
    df_today = fetch_ohlc_for_date(kite, symbol, today)
    if df_today is not None and df_today.shape[0] >= 60:
        st.info(f"{symbol}: ✅ Using today’s candles ({df_today.shape[0]})")
        return df_today
    else:
        msg = f"{symbol}: ⚠️ Only {df_today.shape[0] if df_today is not None else 0} candles today — falling back"
        st.warning(msg)

    # Fallback to last trading day
    last_day = get_last_trading_day(today, skip_today_if_trading=True)
    df_prev = fetch_ohlc_for_date(kite, symbol, last_day)

    if df_prev is not None and df_prev.shape[0] >= 60:
        st.info(f"{symbol}: 🔁 Fallback to {last_day} worked ({df_prev.shape[0]} candles)")
    else:
        st.warning(f"{symbol}: 🚫 Fallback to {last_day} also insufficient ({df_prev.shape[0] if df_prev is not None else 0})")

    return df_prev
