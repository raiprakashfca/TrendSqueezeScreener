import streamlit as st
import pandas as pd
import ta
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
from utils.token_utils import load_credentials_from_gsheet
from utils.zerodha_utils import fetch_historical_data, get_instrument_token

st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")
st.title("📉 Trend Squeeze Screener")

# Load Zerodha credentials
creds = load_credentials_from_gsheet("ZerodhaTokenStore")
kite = KiteConnect(api_key=creds["api_key"])
kite.set_access_token(creds["access_token"])

# Select stock universe
nifty_100 = [  # sample; replace with full list or fetch dynamically
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "LT",
    "AXISBANK", "ITC", "HINDUNILVR", "BAJFINANCE"
]

st.sidebar.header("🔍 Screener Settings")
bbw_threshold = st.sidebar.slider("Bollinger Bandwidth <", 0.01, 0.1, 0.05, 0.005)
lookback_days = st.sidebar.slider("Trend Detection Lookback (days)", 10, 60, 30)
universe = st.sidebar.multiselect("Select Stocks", options=nifty_100, default=nifty_100)

@st.cache_data(show_spinner=False)
def analyze_stock(symbol):
    try:
        token = get_instrument_token(kite, symbol)
        from_date = datetime.now() - timedelta(days=3)
        to_date = datetime.now()
        df = fetch_historical_data(kite, token, "15minute", from_date, to_date)

        df.dropna(inplace=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        adx = df.ta.adx(length=14)
        df = pd.concat([df, adx], axis=1)
        bbands = df.ta.bbands(length=20, append=True)
        df = pd.concat([df, bbands], axis=1)
        df["BBW"] = (df["BBU_20_2.0"] - df["BBL_20_2.0"]) / df["BBM_20_2.0"]

        last = df.iloc[-1]

        trend = None
        setup = None

        if last["EMA_50"] > last["EMA_200"] and last["RSI_14"] > 60 and last["ADX_14"] > 20:
            trend = "Uptrend"
            if last["BBW"] < bbw_threshold:
                setup = "Bullish Squeeze"
        elif last["EMA_50"] < last["EMA_200"] and last["RSI_14"] < 40 and last["ADX_14"] > 20:
            trend = "Downtrend"
            if last["BBW"] < bbw_threshold:
                setup = "Bearish Squeeze"

        return {
            "Symbol": symbol,
            "LTP": round(last["close"], 2),
            "BBW": round(last["BBW"], 4),
            "RSI": round(last["RSI_14"], 2),
            "ADX": round(last["ADX_14"], 2),
            "Trend": trend,
            "Setup": setup
        }
    except Exception as e:
        return {"Symbol": symbol, "Error": str(e)}

st.info("Running analysis across selected stocks...")
results = [analyze_stock(symbol) for symbol in universe]
df_out = pd.DataFrame(results)
df_out = df_out[df_out["Setup"].notnull()]

st.subheader("📊 Matching Stocks")
st.dataframe(df_out, use_container_width=True)

csv = df_out.to_csv(index=False)
st.download_button("📥 Download Results as CSV", csv, "squeeze_stocks.csv")
