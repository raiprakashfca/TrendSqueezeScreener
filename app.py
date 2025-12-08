import streamlit as st
import pandas as pd
import ta
import requests
from kiteconnect import KiteConnect
from datetime import datetime, time
from streamlit_autorefresh import st_autorefresh
from zoneinfo import ZoneInfo

from utils.token_utils import load_credentials_from_gsheet
from utils.zerodha_utils import (
    get_ohlc_15min,
    build_instrument_token_map,
    is_trading_day,
)

# ✅ MUST BE FIRST Streamlit call
st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")

# 🔁 Auto-refresh every 5 minutes
st_autorefresh(interval=300000, key="auto_refresh")

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def market_open_now() -> bool:
    now = datetime.now(IST)
    return is_trading_day(now.date()) and MARKET_OPEN <= now.time() <= MARKET_CLOSE


st.title("📉 Trend Squeeze Screener (Low BBW after Trend)")
st.info("⏳ Auto-refresh every 5 minutes is enabled")

now_ist = datetime.now(IST)
mode_str = "🟢 LIVE MARKET" if market_open_now() else "🔵 EOD / OFF-MARKET (using last trading session data)"
st.caption(f"{mode_str} • Last refresh: {now_ist.strftime('%d %b %Y • %H:%M:%S')} IST")


# Telegram alert function
def send_telegram_alert(message: str):
    try:
        token = st.secrets["telegram_bot_token"]
        chat_id = st.secrets["telegram_chat_id"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning("⚠️ Telegram alert failed.")
    except Exception as e:
        st.warning(f"Telegram error: {e}")


# Load credentials
api_key, api_secret, access_token = load_credentials_from_gsheet("ZerodhaTokenStore")
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# NIFTY 100 stock list
stock_list = [
    "ADANIENT", "ADANIGREEN", "ADANIPORTS", "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJAJFINSV", "BAJFINANCE", "BAJAJHLDNG", "BANDHANBNK", "BANKBARODA", "BERGEPAINT", "BHARTIARTL", "BHEL",
    "BIOCON", "BPCL", "BRITANNIA", "CHOLAFIN", "CIPLA", "COALINDIA", "COLPAL", "DABUR", "DIVISLAB", "DLF",
    "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL", "GLAND", "GLAXO", "GODREJCP", "GRASIM",
    "HAVELLS", "HCLTECH", "HDFCBANK", "HDFCAMC", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDPETRO", "HINDUNILVR",
    "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDFCFIRSTB", "IGL", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY",
    "IOC", "IRCTC", "ITC", "JINDALSTEL", "JSWSTEEL", "JUBLFOOD", "KOTAKBANK", "LICI", "LT", "LTIM", "LTTS",
    "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO", "MARUTI", "MPHASIS", "MRF", "MUTHOOTFIN", "NESTLEIND",
    "NTPC", "OBEROIRLTY", "ONGC", "PAGEIND", "PEL", "PETRONET", "PFC", "PIDILITIND", "PIIND", "PNB", "POWERGRID",
    "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SIEMENS", "SRF", "SUNPHARMA",
    "TATACHEM", "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN", "TORNTPHARM",
    "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO", "UPL", "VEDL", "VOLTAS", "WIPRO", "ZEEL",
]

# BBW input
bbw_threshold = st.slider("Select BBW threshold", 0.01, 0.20, 0.05, step=0.005)


# Indicator calculator
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bbw"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    return df


# Build instrument token map once per run (2–3 API calls instead of 100)
instrument_token_map = build_instrument_token_map(kite, stock_list)

# Main logic
screener_data = []
st.info("📊 Fetching and analyzing NIFTY 100 stocks. Please wait...")

for symbol in stock_list:
    token = instrument_token_map.get(symbol)
    if not token:
        st.warning(f"{symbol}: No instrument token found — skipping.")
        continue

    try:
        df = get_ohlc_15min(kite, token)

        if df is None or df.shape[0] < 60:
            count = 0 if df is None else df.shape[0]
            st.warning(f"{symbol}: Only {count} candles available — skipping.")
            continue

        df = calculate_indicators(df)

        if df.isnull().values.any():
            st.warning(f"{symbol}: Indicators contain NaNs — skipping.")
            continue

        latest = df.iloc[-1]

        trend = ""
        setup = ""

        if latest["bbw"] < bbw_threshold:
            if latest["close"] > latest["ema50"] > latest["ema200"] and latest["rsi"] > 60 and latest["adx"] > 20:
                trend = "Uptrend"
                setup = "Bullish Squeeze"
            elif latest["close"] < latest["ema50"] < latest["ema200"] and latest["rsi"] < 40 and latest["adx"] > 20:
                trend = "Downtrend"
                setup = "Bearish Squeeze"

            if setup:
                message = f"""
📉 *Trend Squeeze Detected!*

*Symbol:* {symbol}
*Setup:* {setup}
*BBW:* {latest['bbw']:.4f}
*RSI:* {latest['rsi']:.1f}
*ADX:* {latest['adx']:.1f}
🕒 {datetime.now(IST).strftime('%d %b %Y • %H:%M:%S')} IST
"""
                # Only send alerts during live market
                if market_open_now():
                    send_telegram_alert(message)

                screener_data.append(
                    {
                        "Symbol": symbol,
                        "LTP": round(latest["close"], 2),
                        "BBW": round(latest["bbw"], 4),
                        "RSI": round(latest["rsi"], 1),
                        "ADX": round(latest["adx"], 1),
                        "Trend": trend,
                        "Setup": setup,
                    }
                )
    except Exception as e:
        st.warning(f"{symbol}: Failed to fetch or compute — {str(e)}")

# Display table
if screener_data:
    df_out = pd.DataFrame(screener_data)
    st.success(f"✅ {len(df_out)} stocks matched the squeeze criteria.")
    st.dataframe(df_out, use_container_width=True)
else:
    st.info("No stocks currently match the squeeze criteria.")
