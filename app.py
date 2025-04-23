import streamlit as st
import pandas as pd
import ta
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
from utils.token_utils import load_credentials_from_gsheet
from utils.zerodha_utils import get_ohlc_15min

st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")
st.title("📉 Trend Squeeze Screener (Low BBW after Trend)")

# Load credentials from Google Sheet
api_key, api_secret, access_token = load_credentials_from_gsheet("ZerodhaTokenStore")

# Initialize Kite Connect client
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Stock universe (can be expanded)
stock_list = [
    "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "AUROPHARMA", 
    "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BANDHANBNK", "BANKBARODA", "BERGEPAINT", 
    "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BPCL", "BRITANNIA", "CHOLAFIN", "CIPLA", "COALINDIA", "COLPAL", 
    "DABUR", "DIVISLAB", "DLF", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL", "GLAND", "GLAXO", 
    "GODREJCP", "GRASIM", "HAVELLS", "HCLTECH", "HDFC", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", 
    "HINDPETRO", "HINDUNILVR", "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IGL", "INDIGO", "INDUSINDBK", 
    "INDUSTOWER", "INFY", "IOC", "IRCTC", "ITC", "JINDALSTEL", "JSWSTEEL", "JUBLFOOD", "KOTAKBANK", "LICI", "LT", 
    "LTI", "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO", "MARUTI", "MOTHERSUMI", "MPHASIS", "MRF", "MUTHOOTFIN", 
    "NESTLEIND", "NTPC", "OBEROIRLTY", "ONGC", "PAGEIND", "PEL", "PETRONET", "PFC", "PIDILITIND", "PIIND", "PNB", "POWERGRID", 
    "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SIEMENS", "SRF", "SUNPHARMA", "TATACHEM", 
    "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM", "TRENT", "TVSMOTOR", 
    "UBL", "ULTRACEMCO", "UPL", "VEDL", "VOLTAS", "WIPRO", "ZEEL"
]

# User inputs
bbw_threshold = st.slider("Select BBW threshold", 0.01, 0.20, 0.05, step=0.005)
lookback_days = 3

# Helper function to calculate indicators
def calculate_indicators(df):
    df = df.copy()
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bbw'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    return df

# Main logic
screener_data = []
st.info("Fetching and analyzing data. This may take a minute...")

for symbol in stock_list:
    try:
        df = get_ohlc_15min(kite, symbol, days=lookback_days)
        df = calculate_indicators(df)
        latest = df.iloc[-1]

        trend = ""
        setup = ""

        if latest['bbw'] < bbw_threshold:
            if latest['close'] > latest['ema50'] > latest['ema200'] and latest['rsi'] > 60 and latest['adx'] > 20:
                trend = "Uptrend"
                setup = "Bullish Squeeze"
            elif latest['close'] < latest['ema50'] < latest['ema200'] and latest['rsi'] < 40 and latest['adx'] > 20:
                trend = "Downtrend"
                setup = "Bearish Squeeze"

            if setup:
                screener_data.append({
                    "Symbol": symbol,
                    "LTP": round(latest['close'], 2),
                    "BBW": round(latest['bbw'], 4),
                    "RSI": round(latest['rsi'], 1),
                    "ADX": round(latest['adx'], 1),
                    "Trend": trend,
                    "Setup": setup
                })
    except Exception as e:
        st.warning(f"{symbol}: Failed to fetch or compute — {str(e)}")

# Display results
if screener_data:
    df_out = pd.DataFrame(screener_data)
    st.success(f"{len(df_out)} stocks found matching criteria")
    st.dataframe(df_out, use_container_width=True)
else:
    st.info("No stocks currently match the squeeze criteria.")
