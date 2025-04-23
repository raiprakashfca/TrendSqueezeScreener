import pandas as pd
from datetime import datetime, timedelta

def get_ohlc_15min(kite, symbol, days=3):
    instrument = f"NSE:{symbol}"
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    ltp_data = kite.ltp([instrument])
    instrument_token = ltp_data[instrument]['instrument_token']

    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval='15minute'
    )
    df = pd.DataFrame(data)
    df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
    return df
