import streamlit as st
import pandas as pd
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
from utils.strategy import prepare_trend_squeeze

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
mode_str = (
    "🟢 LIVE MARKET"
    if market_open_now()
    else "🔵 EOD / OFF-MARKET (using last trading session data)"
)
st.caption(
    f"{mode_str} • Last refresh: {now_ist.strftime('%d %b %Y • %H:%M:%S')} IST"
)

# Load Zerodha credentials
api_key, api_secret, access_token = load_credentials_from_gsheet(
    "ZerodhaTokenStore"
)
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# NIFTY 100 stock list (unchanged)
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

# --- Squeeze configuration ---

col1, col2 = st.columns(2)
with col1:
    bbw_abs_threshold = st.slider(
        "BBW absolute threshold (max BBW)",
        0.01,
        0.20,
        0.05,
        step=0.005,
        help="Maximum allowed BBW. Lower = tighter squeeze.",
    )
with col2:
    bbw_pct_threshold = st.slider(
        "BBW percentile threshold",
        0.10,
        0.80,
        0.35,
        step=0.05,
        help="BBW must be in the bottom X% of last 20 bars.",
    )

st.caption(
    "Squeeze = Bollinger inside Keltner AND BBW below both the absolute "
    "threshold and the rolling percentile threshold."
)

# Build instrument token map once per run (API efficient)
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

        df_prepped = prepare_trend_squeeze(
            df,
            bbw_abs_threshold=bbw_abs_threshold,
            bbw_pct_threshold=bbw_pct_threshold,
        )

        if df_prepped.isnull().values.any():
            st.warning(f"{symbol}: Indicators contain NaNs — skipping.")
            continue

        latest = df_prepped.iloc[-1]

        if latest["setup"]:
            screener_data.append(
                {
                    "Symbol": symbol,
                    "LTP": round(latest["close"], 2),
                    "BBW": round(latest["bbw"], 4),
                    "BBW %Rank (20)": round(latest["bbw_pct_rank"], 2)
                    if pd.notna(latest["bbw_pct_rank"])
                    else None,
                    "RSI": round(latest["rsi"], 1),
                    "ADX": round(latest["adx"], 1),
                    "Trend": latest["trend"],
                    "Setup": latest["setup"],
                }
            )

    except Exception as e:
        st.warning(f"{symbol}: Failed to fetch or compute — {str(e)}")

# Display table
if screener_data:
    df_out = pd.DataFrame(screener_data)
    st.success(f"✅ {len(df_out)} stocks matched the Trend Squeeze criteria.")
    st.dataframe(df_out, use_container_width=True)
else:
    st.info("No stocks currently match the Trend Squeeze criteria.")
