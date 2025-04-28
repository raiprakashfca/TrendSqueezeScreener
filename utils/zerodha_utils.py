import pandas as pd
from datetime import datetime, timedelta

# Add known major holidays manually (you can expand this list every year)
indian_market_holidays = [
    "2025-01-26", "2025-03-29", "2025-04-10", "2025-04-14", "2025-05-01",
    "2025-08-15", "2025-10-02", "2025-11-03", "2025-12-25"
]

def get_last_trading_day():
    today = datetime.now().date()
    last_trading_day = today - timedelta(days=1)
    while last_trading_day.weekday() >= 5 or str(last_trading_day) in indian_market_holidays:
        last_trading_day -= timedelta(days=1)
    return last_trading_day

def get_ohlc_15min(kite, symbol):
    instrument = f"NSE:{symbol}"
    
    # Determine correct last trading day
    last_trading_day = get_last_trading_day()
    from_datetime = datetime.combine(last_trading_day, datetime.strptime("09:15:00", "%H:%M:%S").time())
    to_datetime = datetime.combine(last_trading_day, datetime.strptime("15:30:00", "%H:%M:%S").time())

    # Fetch instrument token
    ltp_data = kite.ltp([instrument])
    instrument_token = ltp_data[instrument]['instrument_token']

    # Fetch historical intraday data
    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_datetime,
        to_date=to_datetime,
        interval='15minute'
    )

    # Convert to DataFrame
    df = pd.DataFrame(data)
    if not df.empty:
        df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
    return df
