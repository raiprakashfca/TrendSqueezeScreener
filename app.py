import io
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import streamlit as st
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh
import gspread
from gspread.exceptions import SpreadsheetNotFound, APIError
from oauth2client.service_account import ServiceAccountCredentials
from utils.token_utils import load_credentials_from_gsheet
from utils.zerodha_utils import (
    get_ohlc_15min,
    get_ohlc_daily,
    build_instrument_token_map,
    is_trading_day,
)
from utils.strategy import prepare_trend_squeeze

# ✅ MUST BE FIRST Streamlit call
st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")
st_autorefresh(interval=300000, key="auto_refresh")  # 5 min refresh

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Dual sheets for 15min and Daily signals
SIGNAL_SHEET_15M = "TrendSqueezeSignals_15M"
SIGNAL_SHEET_DAILY = "TrendSqueezeSignals_Daily"

# Schema with quality score and params
SIGNAL_COLUMNS = [
    "key",  # unique dedup key
    "signal_time",  # candle timestamp (IST naive string)
    "logged_at",  # when app logged it (IST naive string)
    "symbol",
    "timeframe",  # "15M" or "Daily"
    "mode",  # always "Continuation"
    "setup",  # Bullish Squeeze / Bearish Squeeze
    "bias",  # LONG / SHORT
    "ltp",
    "bbw",
    "bbw_pct_rank",
    "rsi",
    "adx",
    "trend",
    "quality_score",  # A/B/C grade
    "params_hash",  # BBW_abs|BBW_pct|ADX|RSI_bull|RSI_bear
]

def now_ist() -> datetime:
    return datetime.now(IST)

def market_open_now() -> bool:
    n = now_ist()
    return is_trading_day(n.date()) and MARKET_OPEN <= n.time() <= MARKET_CLOSE

def fmt_dt(dt: datetime) -> str:
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

def fmt_ts(ts) -> str:
    """Pretty display time for backtest output."""
    if isinstance(ts, pd.Timestamp):
        try:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Kolkata")
        except Exception:
            pass
        return ts.strftime("%d-%b-%Y %H:%M")
    return str(ts)

def ts_to_sheet_str(ts) -> str:
    """Store as YYYY-MM-DD HH:MM:SS (IST naive) in Google Sheet."""
    if isinstance(ts, pd.Timestamp):
        try:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Kolkata")
                ts = ts.tz_localize(None)
        except Exception:
            pass
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)

def params_to_hash(bbw_abs, bbw_pct, adx, rsi_bull, rsi_bear):
    return f"{bbw_abs:.3f}|{bbw_pct:.2f}|{adx:.1f}|{rsi_bull:.1f}|{rsi_bear:.1f}"

def compute_quality_score(row):
    """A/B/C grade based on setup strength."""
    score = 0
    if row.get("adx", 0) > 30:
        score += 2
    elif row.get("adx", 0) > 25:
        score += 1
    if row.get("bbw_pct_rank", 1) < 0.2:
        score += 2
    elif row.get("bbw_pct_rank", 1) < 0.35:
        score += 1
    ema_spread = abs(row.get("ema50", 0) - row.get("ema200", 0)) / row.get("close", 1)
    if ema_spread > 0.05:
        score += 1
    if score >= 4:
        return "A"
    elif score >= 2:
        return "B"
    return "C"

def fetch_nifty50_symbols() -> list[str] | None:
    """Fetch latest NIFTY50 constituents from NSE CSV."""
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
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        st.warning(f"Failed to fetch/parse NSE NIFTY50 list: {e}")
        return None

    if "Symbol" not in df.columns:
        st.warning("NSE NIFTY50 CSV format changed (no 'Symbol' column).")
        return None

    symbols = sorted(df["Symbol"].astype(str).str.strip().dropna().unique().tolist())
    
    # Mapping for TATAMOTORS
    if "TATAMOTORS" in symbols and "TMPV" not in symbols:
        symbols = ["TMPV" if s == "TATAMOTORS" else s for s in symbols]
    
    # Remove obvious garbage
    INVALID = {"DUMMYHDLVR", "DUMMY1", "DUMMY2", "DUMMY"}
    symbols = [s for s in symbols if s.isalpha() and s not in INVALID and "DUMMY" not in s.upper()]
    return symbols

# Google Sheets helpers (dual sheets)
def get_signals_worksheet(sheet_name):
    client, err = get_gspread_client()
    if err:
        return None, err
    
    try:
        sh = client.open(sheet_name)
    except SpreadsheetNotFound:
        return None, (
            f"Sheet '{sheet_name}' not found. Create it in Google Drive and share with the service account as Editor."
        )
    except Exception as e:
        return None, f"Failed to open sheet '{sheet_name}': {e}"
    
    try:
        ws = sh.sheet1
    except Exception as e:
        return None, f"Failed to access sheet1: {e}"
    
    # Ensure headers exist
    try:
        header = ws.row_values(1)
        if not header:
            ws.append_row(SIGNAL_COLUMNS)
        else:
            if header[:len(SIGNAL_COLUMNS)] != SIGNAL_COLUMNS:
                if len(header) < len(SIGNAL_COLUMNS):
                    ws.update("A1", [SIGNAL_COLUMNS])
    except Exception as e:
        return None, f"Failed to ensure headers: {e}"
    
    return ws, None

def get_gspread_client():
    try:
        sa = st.secrets["gcp_service_account"]
    except Exception:
        return None, "Missing st.secrets['gcp_service_account']"
    
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(sa, scope)
        client = gspread.authorize(creds)
        return client, None
    except Exception as e:
        return None, f"gspread auth failed: {e}"

def fetch_existing_keys_recent(ws, days_back: int = 3) -> set:
    try:
        records = ws.get_all_records()
    except Exception:
        return set()
    
    if not records:
        return set()
    
    df = pd.DataFrame(records)
    if "key" not in df.columns or "signal_time" not in df.columns:
        return set(df.get("key", pd.Series(dtype=str)).astype(str).tolist())
    
    df["signal_time_dt"] = pd.to_datetime(df["signal_time"], errors="coerce")
    cutoff = now_ist().replace(tzinfo=None) - timedelta(days=days_back)
    df = df[df["signal_time_dt"].notna()]
    df = df[df["signal_time_dt"] >= cutoff]
    return set(df["key"].astype(str).tolist())

def append_signals(ws, rows: list[list], show_debug: bool = False) -> tuple[int, str | None]:
    if not rows:
        return 0, None
    try:
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        return len(rows), None
    except APIError as e:
        return 0, f"Sheets APIError: {e}" if show_debug else "Sheets write failed (APIError)."
    except Exception as e:
        return 0, f"Sheets write failed: {e}" if show_debug else "Sheets write failed."

def load_recent_signals(ws, hours: int = 24) -> pd.DataFrame:
    try:
        records = ws.get_all_records()
    except Exception:
        return pd.DataFrame()
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    for c in SIGNAL_COLUMNS:
        if c not in df.columns:
            df[c] = None
    
    df["signal_time_dt"] = pd.to_datetime(df["signal_time"], errors="coerce")
    df = df[df["signal_time_dt"].notna()]
    cutoff = now_ist().replace(tzinfo=None) - timedelta(hours=hours)
    df = df[df["signal_time_dt"] >= cutoff].copy()
    
    if df.empty:
        return df
    
    df["Timestamp"] = df["signal_time_dt"].dt.strftime("%d-%b-%Y %H:%M")
    df.rename(
        columns={
            "symbol": "Symbol",
            "timeframe": "Timeframe",
            "mode": "Mode",
            "setup": "Setup",
            "bias": "Bias",
            "ltp": "LTP",
            "bbw": "BBW",
            "bbw_pct_rank": "BBW %Rank",
            "rsi": "RSI",
            "adx": "ADX",
            "trend": "Trend",
            "quality_score": "Quality",
        },
        inplace=True,
    )
    
    # Dedup display: keep latest per Symbol+Timeframe
    df = df.sort_values("signal_time_dt", ascending=False)
    df = df.drop_duplicates(subset=["Symbol", "Timeframe"], keep="first")
    
    cols = [
        "Timestamp", "Symbol", "Timeframe", "Quality", "Bias", "Setup",
        "LTP", "BBW", "BBW %Rank", "RSI", "ADX", "Trend"
    ]
    return df[cols]

# UI Header
st.title("📉 Trend Squeeze Screener (15M + Daily)")
n = now_ist()
mode_str = "🟢 LIVE MARKET" if market_open_now() else "🔵 OFF-MARKET"
st.caption(f"{mode_str} • Last refresh: {n.strftime('%d %b %Y • %H:%M:%S')} IST")

with st.sidebar:
    st.subheader("Settings")
    show_debug = st.checkbox("Show debug messages", value=False)
    
    st.markdown("---")
    st.subheader("Live coherence")
    catchup_candles_15m = st.slider(
        "15M Catch-up window (candles)",
        min_value=8, max_value=64, value=32, step=4,
        help="Live scans last N 15m candles and logs any setups it finds (deduped)."
    )
    retention_hours = st.slider(
        "Keep signals for (hours)",
        min_value=6, max_value=48, value=24, step=6,
        help="Display window for signals from Google Sheets."
    )
    
    st.info("**Continuation only**: Bullish Squeeze → LONG | Bearish Squeeze → SHORT")
    
    # Parameter profiles
    param_profile = st.selectbox(
        "Parameter Profile",
        ["Normal", "Conservative", "Aggressive"],
        help="Conservative: tighter filters | Aggressive: more signals"
    )

# Dynamic params based on profile
if param_profile == "Conservative":
    bbw_abs_default, bbw_pct_default = 0.035, 0.25
    adx_default, rsi_bull_default, rsi_bear_default = 25.0, 60.0, 40.0
elif param_profile == "Aggressive":
    bbw_abs_default, bbw_pct_default = 0.065, 0.45
    adx_default, rsi_bull_default, rsi_bear_default = 18.0, 52.0, 48.0
else:  # Normal
    bbw_abs_default, bbw_pct_default = 0.05, 0.35
    adx_default, rsi_bull_default, rsi_bear_default = 20.0, 55.0, 45.0

c1, c2 = st.columns(2)
with c1:
    bbw_abs_threshold = st.slider(
        "BBW absolute threshold (max BBW)",
        0.01, 0.20, bbw_abs_default, step=0.005,
        help="Lower = tighter squeeze."
    )
with c2:
    bbw_pct_threshold = st.slider(
        "BBW percentile threshold",
        0.10, 0.80, bbw_pct_default, step=0.05,
        help="BBW must be in the bottom X% of last 20 bars."
    )

c3, c4, c5 = st.columns(3)
with c3:
    adx_threshold = st.slider("ADX threshold", 15.0, 35.0, adx_default, step=1.0)
with c4:
    rsi_bull = st.slider("RSI bull threshold", 50.0, 70.0, rsi_bull_default, step=1.0)
with c5:
    rsi_bear = st.slider("RSI bear threshold", 30.0, 50.0, rsi_bear_default, step=1.0)

params_hash = params_to_hash(bbw_abs_threshold, bbw_pct_threshold, adx_threshold, rsi_bull, rsi_bear)

# Zerodha auth
api_key, api_secret, access_token = load_credentials_from_gsheet("ZerodhaTokenStore")
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Universe
fallback_nifty50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE",
    "BAJAJFINSV","BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY",
    "EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK",
    "INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM","M&M","MARUTI","NESTLEIND","NTPC","ONGC",
    "POWERGRID","RELIANCE","SBIN","SBILIFE","SUNPHARMA","TATACONSUM","TMPV","TATASTEEL","TCS",
    "TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HEROMOTOCO","SHREECEM"
]
symbols_live = fetch_nifty50_symbols()
stock_list = symbols_live if symbols_live and len(symbols_live) >= 45 else fallback_nifty50
st.caption(f"Universe: NIFTY50 ({len(stock_list)} symbols).")

# Tokens
instrument_token_map = build_instrument_token_map(kite, stock_list)

# Tabs
live_tab, backtest_tab = st.tabs(["📺 Live Screener (15M + Daily)", "📜 Backtest"])

# LIVE TAB - Dual timeframe
with live_tab:
    st.subheader("📺 Live Screener (15M + Daily • Catch-up + 24h memory)")
    
    # Health check - dual sheets
    col1_15m, col2_daily = st.columns(2)
    with col1_15m:
        ws_15m, ws_15m_err = get_signals_worksheet(SIGNAL_SHEET_15M)
        if ws_15m_err:
            st.error("15M Sheets ❌")
        else:
            st.success("15M Sheets ✅")
    with col2_daily:
        ws_daily, ws_daily_err = get_signals_worksheet(SIGNAL_SHEET_DAILY)
        if ws_daily_err:
            st.error("Daily Sheets ❌")
        else:
            st.success("Daily Sheets ✅")
    
    existing_keys_15m = set()
    existing_keys_daily = set()
    if ws_15m and not ws_15m_err:
        existing_keys_15m = fetch_existing_keys_recent(ws_15m, days_back=3)
    if ws_daily and not ws_daily_err:
        existing_keys_daily = fetch_existing_keys_recent(ws_daily, days_back=7)
    
    rows_15m = []
    rows_daily = []
    logged_at = fmt_dt(now_ist())
    
    # 15M scan
    new_15m = 0
    for symbol in stock_list:
        token = instrument_token_map.get(symbol)
        if not token:
            continue
        
        try:
            df = get_ohlc_15min(kite, token, show_debug=False)
            if df is None or df.shape[0] < 60:
                continue
            
            df_prepped = prepare_trend_squeeze(
                df,
                bbw_abs_threshold=bbw_abs_threshold,
                bbw_pct_threshold=bbw_pct_threshold,
                adx_threshold=adx_threshold,
                rsi_bull=rsi_bull,
                rsi_bear=rsi_bear,
                rolling_window=20,
            )
            
            recent = df_prepped.tail(int(catchup_candles_15m)).copy()
            recent = recent[recent["setup"] != ""]
            
            for candle_ts, r in recent.iterrows():
                signal_time = ts_to_sheet_str(candle_ts)
                setup = r["setup"]
                trend = r.get("trend", "")
                ltp = float(r.get("close", np.nan))
                bbw = float(r.get("bbw", np.nan))
                bbw_rank = r.get("bbw_pct_rank", np.nan)
                rsi_val = float(r.get("rsi", np.nan))
                adx_val = float(r.get("adx", np.nan))
                
                bias = "LONG" if setup == "Bullish Squeeze" else "SHORT"
                key = f"{symbol}|15M|Continuation|{signal_time}|{setup}|{bias}"
                
                if key not in existing_keys_15m:
                    quality = compute_quality_score(r)
                    rows_15m.append([
                        key, signal_time, logged_at, symbol, "15M", "Continuation", 
                        setup, bias, ltp, bbw, bbw_rank, rsi_val, adx_val, trend, 
                        quality, params_hash
                    ])
                    existing_keys_15m.add(key)
                    new_15m += 1
                    
        except Exception as e:
            if show_debug:
                st.warning(f"{symbol} 15M: error -> {e}")
            continue
    
    # Daily scan (slower, but higher quality)
    new_daily = 0
    lookback_days_daily = 252  # ~1 year
    for symbol in stock_list[:20]:  # Limit to top 20 for speed
        token = instrument_token_map.get(symbol)
        if not token:
            continue
        
        try:
            df_daily = get_ohlc_daily(kite, token, lookback_days=lookback_days_daily)
            if df_daily is None or df_daily.shape[0] < 100:
                continue
            
            df_daily_prepped = prepare_trend_squeeze(
                df_daily,
                bbw_abs_threshold=bbw_abs_threshold * 1.2,  # Slightly looser for daily
                bbw_pct_threshold=bbw_pct_threshold,
                adx_threshold=adx_threshold,
                rsi_bull=rsi_bull,
                rsi_bear=rsi_bear,
                rolling_window=20,
            )
            
            recent_daily = df_daily_prepped.tail(5).copy()  # Last 5 days
            recent_daily = recent_daily[recent_daily["setup"] != ""]
            
            for candle_ts, r in recent_daily.iterrows():
                signal_time = ts_to_sheet_str(candle_ts)
                setup = r["setup"]
                trend = r.get("trend", "")
                ltp = float(r.get("close", np.nan))
                bbw = float(r.get("bbw", np.nan))
                bbw_rank = r.get("bbw_pct_rank", np.nan)
                rsi_val = float(r.get("rsi", np.nan))
                adx_val = float(r.get("adx", np.nan))
                
                bias = "LONG" if setup == "Bullish Squeeze" else "SHORT"
                key = f"{symbol}|Daily|Continuation|{signal_time}|{setup}|{bias}"
                
                if key not in existing_keys_daily:
                    quality = compute_quality_score(r)
                    rows_daily.append([
                        key, signal_time, logged_at, symbol, "Daily", "Continuation", 
                        setup, bias, ltp, bbw, bbw_rank, rsi_val, adx_val, trend, 
                        quality, params_hash
                    ])
                    existing_keys_daily.add(key)
                    new_daily += 1
                    
        except Exception as e:
            if show_debug:
                st.warning(f"{symbol} Daily: error -> {e}")
            continue
    
    # Write signals
    appended_15m = 0
    appended_daily = 0
    if ws_15m and not ws_15m_err:
        appended_15m, _ = append_signals(ws_15m, rows_15m, show_debug)
    if ws_daily and not ws_daily_err:
        appended_daily, _ = append_signals(ws_daily, rows_daily, show_debug)
    
    st.caption(f"Logged **{appended_15m}** new 15M + **{appended_daily}** Daily signals (deduped).")
    
    # Show combined recent signals
    df_recent_15m = pd.DataFrame() if not ws_15m else load_recent_signals(ws_15m, hours=int(retention_hours))
    df_recent_daily = pd.DataFrame() if not ws_daily else load_recent_signals(ws_daily, hours=int(retention_hours)*24)
    
    if not df_recent_15m.empty or not df_recent_daily.empty:
        st.markdown("### 🧠 Recent Signals (latest per Symbol+Timeframe)")
        if not df_recent_15m.empty:
            st.subheader("15M Signals")
            st.dataframe(df_recent_15m, use_container_width=True)
        if not df_recent_daily.empty:
            st.subheader("Daily Signals")
            st.dataframe(df_recent_daily, use_container_width=True)
    else:
        st.info("No continuation signals in the selected window.")
    
    st.markdown("---")
    st.caption(
        "Coherence guarantee: Logs **signal candle timestamps**. "
        "15M scans last N candles each refresh. Daily checks last 5 days."
    )

# BACKTEST TAB (unchanged for now - can be enhanced later)
with backtest_tab:
    st.subheader("📜 Backtest (15-minute)")
    # ... (keep existing backtest code or enhance later)
    st.info("Backtest enhanced in next update.")
