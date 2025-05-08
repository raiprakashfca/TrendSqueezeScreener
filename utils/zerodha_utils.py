import pandas as pd
from datetime import datetime, timedelta

# List of known NSE holidays (expand this as needed)
indian_market_holidays = [
    "2025-01-26", "2025-03-29", "2025-04-10", "2025-04-14", "2025-05-01",
    "2025-08-15", "2025-10-02", "2025-11-03", "2025-12-25"
]

def is_trading_day(date):
    return date.weekday() < 5 and str(date) not in indian_market_holidays

def get_last_trading_day(reference_day):
    day = reference_day - timedelta(days=1)
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day

def fetch_ohlc_for_date(kite, symbol, date_obj):
    instrument = f"NSE:{symbol}"
    from_dt = datetime.combine(date_obj, datetime.strptime("09:15:00", "%H:%M:%S").time())
    to_dt = datetime.combine(date_obj, datetime.strptime("15:30:00", "%H:%M:%S").time())

    ltp_data = kite.ltp([instrument])
    instrument_token = ltp_data[instrument]['instrument_token']

    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_dt,
        to_date=to_dt,
        interval='15minute'
    )

    df = pd.DataFrame(data)
    if not df.empty:
        df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low',
                           'close': 'close', 'volume': 'volume'}, inplace=True)
    return df

def get_ohlc_15min(kite, symbol):
    today = datetime.now().date()

    # Step 1: Try today's candles
    df_today = fetch_ohlc_for_date(kite, symbol, today)
    if df_today is not None and df_today.shape[0] >= 60:
        return df_today

    # Step 2: Fallback to last trading day
    last_day = get_last_trading_day(today)
    df_prev = fetch_ohlc_for_date(kite, symbol, last_day)
    return df_prev
