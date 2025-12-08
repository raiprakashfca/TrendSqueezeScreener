import io
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh

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


def fetch_nifty50_symbols() -> list[str] | None:
    """
    Fetch the latest NIFTY 50 constituents from NSE's official CSV.

    Returns a list of SYMBOL strings (e.g. ['RELIANCE', 'HDFCBANK', ...])
    or None on failure.

    Uses a browser-like User-Agent and Referer to reduce chances of NSE blocking.
    """
    nifty50_csv_url = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Referer": "https://www.nseindia.com/",
        "Accept": "text/csv,application/vnd.ms-excel,application/octet-stream;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(nifty50_csv_url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        st.warning(f"Failed to download NIFTY 50 CSV from NSE: {e}")
        return None

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        st.warning(f"Failed to parse NIFTY 50 CSV: {e}")
        return None

    if "Symbol" not in df.columns:
        st.warning("NSE NIFTY 50 CSV does not contain 'Symbol' column. Format may have changed.")
        return None

    symbols = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )

    # Sort for stability
    symbols = sorted(symbols)

    if len(symbols) != 50:
        st.warning(
            f"Expected 50 NIFTY symbols, got {len(symbols)}. "
            "Proceeding, but please verify against NSE manually."
        )

    # 🔁 Post-demerger fix:
    # If NSE still shows TATAMOTORS but you trade TMPV, map it.
    if "TATAMOTORS" in symbols and "TMPV" not in symbols:
        symbols = ["TMPV" if s == "TATAMOTORS" else s for s in symbols]

    return symbols


st.title("📉 Trend Squeeze Screener (Low BBW after Trend)")
st.info("⏳ Auto-refresh every 5 minutes is enabled")

now_ist = datetime.now(IST)
mode_str = (
    "🟢 LIVE MARKET"
    if market_open_now()
    else "🔵 EOD / OFF-MARKET (using last trading session data)"
)
st.caption(f"{mode_str} • Last refresh: {now_ist.strftime('%d %b %Y • %H:%M:%S')} IST")

# -----------------------------
# 🔐 Zerodha credentials
# -----------------------------
api_key, api_secret, access_token = load_credentials_from_gsheet("ZerodhaTokenStore")
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# -----------------------------
# 📈 NIFTY 50 stock universe
# -----------------------------

# Fallback list in case NSE CSV fetch fails
fallback_nifty50 = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BHARTIARTL",
    "BPCL",
    "BRITANNIA",
    "CIPLA",
    "COALINDIA",
    "DIVISLAB",
    "DRREDDY",
    "EICHERMOT",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "LTIM",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBIN",
    "SBILIFE",
    "SUNPHARMA",
    "TATACONSUM",
    "TMPV",  # Tata Motors Passenger Vehicles (post-demerger symbol)
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "ULTRACEMCO",
    "UPL",
    "WIPRO",
    "HEROMOTOCO",
    "SHREECEM",
]

# Try to fetch live NIFTY50 list from NSE
live_nifty50 = fetch_nifty50_symbols()

INVALID_SYMBOLS = {"DUMMYHDLVR", "DUMMY1", "DUMMY2", "DUMMY"}

if live_nifty50:
    # Filter out NSE's occasional dummy / invalid placeholders
    clean_symbols = [
        s
        for s in live_nifty50
        if s.isalpha() and "DUMMY" not in s.upper() and s not in INVALID_SYMBOLS
    ]

    if len(clean_symbols) < len(live_nifty50):
        st.warning(
            "Some NSE symbols were invalid or temporary placeholders and were removed."
        )

    stock_list = clean_symbols
    st.caption(f"Universe: Latest NIFTY 50 from NSE (cleaned, {len(stock_list)} symbols).")
else:
    stock_list = fallback_nifty50
    st.caption("Universe: Fallback hard-coded NIFTY 50 list (NSE CSV unavailable).")

# -----------------------------
# 🎯 Squeeze configuration
# -----------------------------
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

# -----------------------------
# 🧩 Instrument tokens (API efficient)
# -----------------------------
instrument_token_map = build_instrument_token_map(kite, stock_list)

# -----------------------------
# 🔍 Main screening loop
# -----------------------------
screener_data: list[dict] = []
st.info("📊 Fetching and analyzing NIFTY 50 stocks. Please wait...")

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

# -----------------------------
# 📋 Output
# -----------------------------
if screener_data:
    df_out = pd.DataFrame(screener_data)
    st.success(f"✅ {len(df_out)} stocks matched the Trend Squeeze criteria.")
    st.dataframe(df_out, use_container_width=True)
else:
    st.info("No stocks currently match the Trend Squeeze criteria.")
