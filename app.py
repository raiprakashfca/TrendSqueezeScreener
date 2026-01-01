# =========================
# app.py — Trend Squeeze Screener
# =========================
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import json
import inspect

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import gspread
from gspread.exceptions import SpreadsheetNotFound, APIError
from oauth2client.service_account import ServiceAccountCredentials

from fyers_apiv3 import fyersModel

from utils.zerodha_utils import (
    init_fyers_session,
    get_ohlc_15min,
    get_ohlc_daily,
    is_trading_day,
    get_last_trading_day,
)
from utils.strategy import prepare_trend_squeeze

# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")
st_autorefresh(interval=300000, key="auto_refresh")

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

SIGNAL_SHEET_15M = "TrendSqueezeSignals_15M"
SIGNAL_SHEET_DAILY = "TrendSqueezeSignals_Daily"

SHEET_ID_15M = "1RP2tbh6WnMgEDAv5FWXD6GhePXno9bZ81ILFyoX6Fb8"
SHEET_ID_DAILY = "1u-W5vu3KM6XPRr78o8yyK8pPaAjRUIFEEYKHyYGUsM0"

SIGNAL_COLUMNS = [
    "key","signal_time","logged_at","symbol","timeframe","mode",
    "setup","bias","ltp","bbw","bbw_pct_rank","rsi","adx",
    "trend","quality_score","params_hash"
]

FYERS_TOKEN_HEADERS = [
    "fyers_app_id","fyers_secret_id","fyers_access_token","fyers_token_updated_at"
]

# =========================
# Time helpers (IST-safe)
# =========================
def now_ist():
    return datetime.now(IST)

def market_open_now():
    n = now_ist()
    return is_trading_day(n.date()) and MARKET_OPEN <= n.time() <= MARKET_CLOSE

def fmt_dt(dt):
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

def normalize_ohlc_index_to_ist(df: pd.DataFrame):
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    idx = df.index
    if idx.tz is None:
        df = df.copy()
        df.index = idx.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
    else:
        df = df.copy()
        df.index = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    return df

def ts_to_sheet_15m(ts):
    ts = pd.Timestamp(ts)
    return ts.floor("15min").strftime("%Y-%m-%d %H:%M")

def ts_to_sheet_daily(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")

# =========================
# Google Sheets
# =========================
def get_gspread_client():
    raw = st.secrets.get("gcp_service_account")
    if not raw:
        return None, "Missing gcp_service_account"
    sa = json.loads(raw) if isinstance(raw, str) else dict(raw)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa, scope)
    return gspread.authorize(creds), None

def get_signals_worksheet(sheet_name):
    client, err = get_gspread_client()
    if err:
        return None, err
    sh = client.open_by_key(SHEET_ID_15M if sheet_name == SIGNAL_SHEET_15M else SHEET_ID_DAILY)
    ws = sh.sheet1
    if not ws.row_values(1):
        ws.append_row(SIGNAL_COLUMNS)
    return ws, None

# =========================
# Quality score
# =========================
def compute_quality_score(row):
    score = 0
    if row.get("adx", 0) > 30: score += 2
    elif row.get("adx", 0) > 25: score += 1
    if row.get("bbw_pct_rank", 1) < 0.2: score += 2
    elif row.get("bbw_pct_rank", 1) < 0.35: score += 1
    return "A" if score >= 4 else "B" if score >= 2 else "C"

# =========================
# Runner Backtest Logic
# =========================
def simulate_trade(df, entry_i, side, max_hold=200):
    SL_PCT = 0.0025
    STEP = 0.005
    entry = float(df["close"].iloc[entry_i])
    qty_total = 100
    qty_rem = qty_total
    pnl = 0.0

    sl = entry * (1 - SL_PCT) if side == "LONG" else entry * (1 + SL_PCT)
    k = 1

    def target(k):
        return entry * (1 + STEP*k) if side=="LONG" else entry * (1 - STEP*k)

    for i in range(entry_i+1, min(entry_i+max_hold, len(df))):
        hi, lo = df["high"].iloc[i], df["low"].iloc[i]

        if (side=="LONG" and lo<=sl) or (side=="SHORT" and hi>=sl):
            pnl += (qty_rem/qty_total) * ((sl-entry)/entry if side=="LONG" else (entry-sl)/entry)
            return pnl*100

        while qty_rem>0:
            t = target(k)
            hit = hi>=t if side=="LONG" else lo<=t
            if not hit: break
            book = max(1, qty_rem//2)
            pnl += (book/qty_total)*((t-entry)/entry if side=="LONG" else (entry-t)/entry)
            qty_rem -= book
            sl = entry if k==1 else target(k-1)
            k += 1

    pnl += (qty_rem/qty_total)*((df["close"].iloc[-1]-entry)/entry if side=="LONG" else (entry-df["close"].iloc[-1])/entry)
    return pnl*100

# =========================
# UI Header
# =========================
st.title("📉 Trend Squeeze Screener")
st.caption(f"{'🟢 LIVE' if market_open_now() else '🔵 OFF MARKET'} • {now_ist().strftime('%d %b %Y %H:%M:%S')} IST")

# =========================
# Universe
# =========================
NIFTY50 = [
 "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV",
 "BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH",
 "HDFCBANK","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM",
 "M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBIN","SBILIFE","SUNPHARMA","TATACONSUM",
 "TATAMOTORS","TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HEROMOTOCO","SHREECEM"
]

# =========================
# Recent Signals
# =========================
ws15, e1 = get_signals_worksheet(SIGNAL_SHEET_15M)
wsd, e2 = get_signals_worksheet(SIGNAL_SHEET_DAILY)

st.subheader("🧠 Recent Signals (Market-Aware)")

if not e1:
    df15 = pd.DataFrame(ws15.get_all_records())
    if not df15.empty:
        df15["signal_time"] = pd.to_datetime(df15["signal_time"])
        df15 = df15[df15["signal_time"] <= now_ist().replace(tzinfo=None)]
        df15 = df15.sort_values("signal_time", ascending=False).head(20)
        st.dataframe(df15[["signal_time","symbol","timeframe","quality_score","bias","setup","ltp"]], use_container_width=True)
    else:
        st.info("No 15M signals")

if not e2:
    dfd = pd.DataFrame(wsd.get_all_records())
    if not dfd.empty:
        dfd["signal_time"] = pd.to_datetime(dfd["signal_time"])
        dfd = dfd[dfd["signal_time"] <= now_ist().replace(tzinfo=None)]
        dfd = dfd.sort_values("signal_time", ascending=False).head(20)
        st.dataframe(dfd[["signal_time","symbol","quality_score","bias","setup","ltp"]], use_container_width=True)
    else:
        st.info("No Daily signals")

st.caption("✔ Clean UI • ✔ No future candles • ✔ Runner backtest enabled")
