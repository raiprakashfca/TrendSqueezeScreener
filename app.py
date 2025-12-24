from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import json

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

# ✅ MUST BE FIRST Streamlit call
st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")
st_autorefresh(interval=300000, key="auto_refresh")  # 5 min refresh

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Dual sheets for 15min and Daily signals
SIGNAL_SHEET_15M = "TrendSqueezeSignals_15M"
SIGNAL_SHEET_DAILY = "TrendSqueezeSignals_Daily"

# Direct Sheet IDs (from your URLs)
SHEET_ID_15M = "1RP2tbh6WnMgEDAv5FWXD6GhePXno9bZ81ILFyoX6Fb8"
SHEET_ID_DAILY = "1u-W5vu3KM6XPRr78o8yyK8pPaAjRUIFEEYKHyYGUsM0"

# Schema with quality score and params
SIGNAL_COLUMNS = [
    "key",
    "signal_time",
    "logged_at",
    "symbol",
    "timeframe",
    "mode",
    "setup",
    "bias",
    "ltp",
    "bbw",
    "bbw_pct_rank",
    "rsi",
    "adx",
    "trend",
    "quality_score",
    "params_hash",
]

# ---------- GOOGLE SHEETS AUTH (used for FYERS token sheet + signals logging) ----------
def get_gspread_client():
    try:
        raw = st.secrets["gcp_service_account"]
    except Exception:
        return None, "Missing st.secrets['gcp_service_account']"

    try:
        sa = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception as e:
        return None, f"Invalid gcp_service_account format: {e}"

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


# ---------- FYERS TOKEN LOADER (REUSE SAME GSHEET LOGIC) ----------
def load_fyers_credentials() -> tuple[str | None, str | None, str]:
    """
    Returns (app_id, access_token, source_label)
    Priority:
      1) Google Sheet (if FYERS_TOKEN_SHEET_KEY or FYERS_TOKEN_SHEET_NAME is set)
      2) Streamlit secrets: fyers_app_id + fyers_secret (access token)

    Google Sheet supported formats:
      A) Header row + first data row with columns:
         app_id / fyers_app_id / client_id  AND  access_token / fyers_access_token / token
      B) Simple cells:
         A1 = app_id, B1 = access_token
    """
    app_id = None
    token = None

    # 1) Try Google Sheet (preferred if configured)
    sheet_key = None
    sheet_name = None
    try:
        sheet_key = st.secrets.get("FYERS_TOKEN_SHEET_KEY")
        sheet_name = st.secrets.get("FYERS_TOKEN_SHEET_NAME")
    except Exception:
        sheet_key = None
        sheet_name = None

    if sheet_key or sheet_name:
        client, err = get_gspread_client()
        if err:
            raise RuntimeError(err)

        try:
            sh = client.open_by_key(sheet_key) if sheet_key else client.open(sheet_name)
            ws = sh.sheet1

            # Try header+first row
            records = ws.get_all_records()
            if records:
                row = records[0]
                app_id = (
                    row.get("fyers_app_id")
                    or row.get("app_id")
                    or row.get("client_id")
                )
                token = (
                    row.get("fyers_access_token")
                    or row.get("access_token")
                    or row.get("token")
                )

            # Fallback: simple cells A1/B1 if needed
            if not app_id or not token:
                a1 = ws.acell("A1").value
                b1 = ws.acell("B1").value
                if a1 and b1:
                    # If A1 contains a header text, ignore
                    if str(a1).strip().lower() not in {"app_id", "fyers_app_id", "client_id"}:
                        app_id = a1
                    if str(b1).strip().lower() not in {"access_token", "fyers_access_token", "token"}:
                        token = b1

            if app_id and token:
                return str(app_id).strip(), str(token).strip(), "Google Sheet"
        except Exception as e:
            raise RuntimeError(f"Failed to read FYERS token sheet: {e}")

    # 2) Fallback to Streamlit secrets
    try:
        app_id = st.secrets.get("fyers_app_id")
        token = st.secrets.get("fyers_secret")
    except Exception:
        app_id = None
        token = None

    if app_id and token:
        return str(app_id).strip(), str(token).strip(), "Streamlit secrets"

    return None, None, "Not configured"


@st.cache_data(ttl=120, show_spinner=False)
def fyers_smoke_test() -> tuple[bool, str]:
    """
    Minimal data call to verify token is not expired.
    We try a small OHLC fetch for a liquid symbol.
    """
    try:
        df = get_ohlc_15min("RELIANCE", days_back=2)
        if df is None or len(df) < 10:
            return False, "OHLC returned empty/too short"
        # sanity: last candle timestamp should be recent-ish on trading days
        return True, "OK"
    except Exception as e:
        return False, str(e)[:180]


# ---------- FYERS SESSION INIT (MATCHES YOUR SECRETS) ----------
fyers_ok = False
fyers_source = "Unknown"
fyers_health_ok = False
fyers_health_msg = "Not checked"

try:
    fyappid, fyaccesstoken, fyers_source = load_fyers_credentials()
    if not fyappid or not fyaccesstoken:
        raise RuntimeError(
            "Missing FYERS credentials. Provide fyers_app_id + fyers_secret in Streamlit secrets, "
            "or configure FYERS_TOKEN_SHEET_KEY / FYERS_TOKEN_SHEET_NAME to load from Google Sheet."
        )

    init_fyers_session(str(fyappid), str(fyaccesstoken))
    fyers_ok = True

    # Token validity / API health check (cached)
    fyers_health_ok, fyers_health_msg = fyers_smoke_test()

    if fyers_health_ok:
        st.success(f"✅ Fyers API Connected ({fyers_source})")
    else:
        st.warning(f"⚠️ Fyers connected but data fetch failed ({fyers_source}): {fyers_health_msg}")

except Exception as e:
    st.error(f"🔴 Fyers init failed: {str(e)[:200]}")
    st.info(
        "Expected configuration:\n"
        "- gcp_service_account (for Google Sheet access)\n"
        "Either:\n"
        "  • fyers_app_id + fyers_secret (access token) in Streamlit secrets, OR\n"
        "  • FYERS_TOKEN_SHEET_KEY / FYERS_TOKEN_SHEET_NAME (token stored in a Google Sheet)\n"
    )
    st.stop()


def now_ist() -> datetime:
    return datetime.now(IST)


def market_open_now() -> bool:
    n = now_ist()
    return is_trading_day(n.date()) and MARKET_OPEN <= n.time() <= MARKET_CLOSE


def fmt_dt(dt: datetime) -> str:
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def fmt_ts(ts) -> str:
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
        return ts.strftime("%d-%b-%Y %H:%M")
    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(IST).replace(tzinfo=None)
        return ts.strftime("%d-%b-%Y %H:%M")
    return str(ts)


def ts_to_sheet_str(ts) -> str:
    """Return a stable candle timestamp string for de-dupe + sheet storage."""
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
        # minute precision avoids duplicate spam due to seconds jitter
        return ts.strftime("%Y-%m-%d %H:%M")

    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(IST).replace(tzinfo=None)
        return ts.strftime("%Y-%m-%d %H:%M")

    # Fallback
    return str(ts)


def params_to_hash(bbw_abs, bbw_pct, adx, rsi_bull, rsi_bear):
    return f"{bbw_abs:.3f}|{bbw_pct:.2f}|{adx:.1f}|{rsi_bull:.1f}|{rsi_bear:.1f}"


def compute_quality_score(row):
    score = 0

    adx_val = row.get("adx", np.nan)
    if pd.notna(adx_val):
        if adx_val > 30:
            score += 2
        elif adx_val > 25:
            score += 1

    bbw_rank = row.get("bbw_pct_rank", np.nan)
    if pd.notna(bbw_rank):
        if bbw_rank < 0.2:
            score += 2
        elif bbw_rank < 0.35:
            score += 1

    # EMA spread bonus only if columns exist
    ema50 = row.get("ema50", np.nan)
    ema200 = row.get("ema200", np.nan)
    close = row.get("close", np.nan)
    if pd.notna(ema50) and pd.notna(ema200) and pd.notna(close) and close != 0:
        ema_spread = abs(float(ema50) - float(ema200)) / float(close)
        if ema_spread > 0.05:
            score += 1

    if score >= 4:
        return "A"
    elif score >= 2:
        return "B"
    return "C"


def get_market_aware_window(base_hours: int, timeframe: str = "15M") -> int:
    now = now_ist()
    target_date = now.date()
    trading_days_back = 0
    calendar_days_back = 0
    while trading_days_back * 6 < base_hours:
        calendar_days_back += 1
        check_date = target_date - timedelta(days=calendar_days_back)
        if is_trading_day(check_date):
            trading_days_back += 1
    smart_hours = trading_days_back * 6
    if timeframe == "Daily":
        smart_hours *= 24
    return min(smart_hours, base_hours * 2)


def get_signals_worksheet(sheet_name: str):
    client, err = get_gspread_client()
    if err:
        return None, err
    try:
        if sheet_name == SIGNAL_SHEET_15M and SHEET_ID_15M:
            sh = client.open_by_key(SHEET_ID_15M)
        elif sheet_name == SIGNAL_SHEET_DAILY and SHEET_ID_DAILY:
            sh = client.open_by_key(SHEET_ID_DAILY)
        else:
            sh = client.open(sheet_name)
    except SpreadsheetNotFound:
        return None, f"Sheet '{sheet_name}' not found. Check file name or Sheet ID in app.py."
    except Exception as e:
        return None, f"Failed to open sheet '{sheet_name}': {e}"

    try:
        ws = sh.sheet1
    except Exception as e:
        return None, f"Failed to access sheet1: {e}"

    try:
        header = ws.row_values(1)
        if not header:
            ws.append_row(SIGNAL_COLUMNS)
        else:
            if header[: len(SIGNAL_COLUMNS)] != SIGNAL_COLUMNS:
                if len(header) < len(SIGNAL_COLUMNS):
                    ws.update("A1", [SIGNAL_COLUMNS])
    except Exception as e:
        return None, f"Failed to ensure headers: {e}"

    return ws, None


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
        return (0, f"Sheets APIError: {e}") if show_debug else (0, "Sheets write failed (APIError).")
    except Exception as e:
        return (0, f"Sheets write failed: {e}") if show_debug else (0, "Sheets write failed.")


def load_recent_signals(ws, base_hours: int = 24, timeframe: str = "15M") -> pd.DataFrame:
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
    smart_hours = get_market_aware_window(base_hours, timeframe)
    cutoff = now_ist().replace(tzinfo=None) - timedelta(hours=smart_hours)
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
    df = df.sort_values("signal_time_dt", ascending=False)
    df = df.drop_duplicates(subset=["Symbol", "Timeframe"], keep="first")
    cols = [
        "Timestamp",
        "Symbol",
        "Timeframe",
        "Quality",
        "Bias",
        "Setup",
        "LTP",
        "BBW",
        "BBW %Rank",
        "RSI",
        "ADX",
        "Trend",
    ]
    return df[cols]


# UI Header
st.title("📉 Trend Squeeze Screener (15M + Daily) - Market Aware")
n = now_ist()
mode_str = "🟢 LIVE MARKET" if market_open_now() else "🔵 OFF-MARKET"
st.caption(f"{mode_str} • Last refresh: {n.strftime('%d %b %Y • %H:%M:%S')} IST")

# Market status display
last_trading_day = get_last_trading_day(n.date())
st.caption(f"🗓️ Last trading day: {last_trading_day.strftime('%d %b %Y')}")

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
    "TMPV",
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

stock_list = fallback_nifty50
st.caption(f"Universe: NIFTY50 ({len(stock_list)} symbols).")

live_tab, backtest_tab = st.tabs(["📺 Live Screener (15M + Daily)", "📜 Backtest"])

with live_tab:
    st.subheader("📺 Live Screener (15M + Daily • Market-Aware)")

    with st.sidebar:
        st.subheader("⚙️ Settings")
        show_debug = st.checkbox("Show debug messages", value=False)

        st.markdown("---")
        st.subheader("🔐 FYERS Token Status")
        if fyers_health_ok:
            st.success(f"Token OK ({fyers_source})")
        else:
            st.error(f"Token/Fetch issue ({fyers_source})")
            st.caption(str(fyers_health_msg))

        st.markdown("---")
        st.subheader("📊 Live coherence")
        catchup_candles_15m = st.slider(
            "15M Catch-up window (candles)",
            min_value=8,
            max_value=64,
            value=32,
            step=4,
            help="Live scans last N 15m candles and logs any setups it finds (deduped).",
        )
        retention_hours = st.slider(
            "Base retention (calendar hours)",
            min_value=6,
            max_value=48,
            value=24,
            step=6,
            help="App auto-adjusts for weekends/holidays to show relevant trading signals.",
        )

        st.info("**Continuation only**: Bullish Squeeze → LONG | Bearish Squeeze → SHORT")

        param_profile = st.selectbox(
            "Parameter Profile",
            ["Normal", "Conservative", "Aggressive"],
            help="Conservative: tighter filters | Aggressive: more signals",
        )

    if param_profile == "Conservative":
        bbw_abs_default, bbw_pct_default = 0.035, 0.25
        adx_default, rsi_bull_default, rsi_bear_default = 25.0, 60.0, 40.0
    elif param_profile == "Aggressive":
        bbw_abs_default, bbw_pct_default = 0.065, 0.45
        adx_default, rsi_bull_default, rsi_bear_default = 18.0, 52.0, 48.0
    else:
        bbw_abs_default, bbw_pct_default = 0.05, 0.35
        adx_default, rsi_bull_default, rsi_bear_default = 20.0, 55.0, 45.0

    c1, c2 = st.columns(2)
    with c1:
        bbw_abs_threshold = st.slider(
            "BBW absolute threshold (max BBW)",
            0.01,
            0.20,
            bbw_abs_default,
            step=0.005,
            help="Lower = tighter squeeze.",
        )
    with c2:
        bbw_pct_threshold = st.slider(
            "BBW percentile threshold",
            0.10,
            0.80,
            bbw_pct_default,
            step=0.05,
            help="BBW must be in the bottom X% of last 20 bars.",
        )

    c3, c4, c5 = st.columns(3)
    with c3:
        adx_threshold = st.slider("ADX threshold", 15.0, 35.0, adx_default, step=1.0)
    with c4:
        rsi_bull = st.slider("RSI bull threshold", 50.0, 70.0, rsi_bull_default, step=1.0)
    with c5:
        rsi_bear = st.slider("RSI bear threshold", 30.0, 50.0, rsi_bear_default, step=1.0)

    params_hash = params_to_hash(bbw_abs_threshold, bbw_pct_threshold, adx_threshold, rsi_bull, rsi_bear)

    smart_retention_15m = get_market_aware_window(retention_hours)
    smart_retention_daily = get_market_aware_window(retention_hours, "Daily")

    st.caption(
        f"🧠 Showing last ~{smart_retention_15m}h trading time (15M) | "
        f"~{smart_retention_daily/24:.0f} trading days (Daily)"
    )

    col1_15m, col2_daily = st.columns(2)
    with col1_15m:
        ws_15m, ws_15m_err = get_signals_worksheet(SIGNAL_SHEET_15M)
        if ws_15m_err:
            st.error(f"15M Sheets ❌: {ws_15m_err}")
        else:
            st.success("15M Sheets ✅")
    with col2_daily:
        ws_daily, ws_daily_err = get_signals_worksheet(SIGNAL_SHEET_DAILY)
        if ws_daily_err:
            st.error(f"Daily Sheets ❌: {ws_daily_err}")
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

    new_15m = 0
    for symbol in stock_list:
        try:
            df = get_ohlc_15min(symbol, days_back=7)
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
                # normalize to stable 15m slot string
                signal_time = pd.Timestamp(candle_ts).floor("15min").strftime("%Y-%m-%d %H:%M")

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
                    rows_15m.append(
                        [
                            key,
                            signal_time,
                            logged_at,
                            symbol,
                            "15M",
                            "Continuation",
                            setup,
                            bias,
                            ltp,
                            bbw,
                            bbw_rank,
                            rsi_val,
                            adx_val,
                            trend,
                            quality,
                            params_hash,
                        ]
                    )
                    existing_keys_15m.add(key)
                    new_15m += 1

        except Exception as e:
            if show_debug:
                st.warning(f"{symbol} 15M error: {e}")
            continue

    new_daily = 0
    lookback_days_daily = 252
    for symbol in stock_list[:20]:
        try:
            df_daily = get_ohlc_daily(symbol, lookback_days=lookback_days_daily)
            if df_daily is None or df_daily.shape[0] < 100:
                continue

            df_daily_prepped = prepare_trend_squeeze(
                df_daily,
                bbw_abs_threshold=bbw_abs_threshold * 1.2,
                bbw_pct_threshold=bbw_pct_threshold,
                adx_threshold=adx_threshold,
                rsi_bull=rsi_bull,
                rsi_bear=rsi_bear,
                rolling_window=20,
            )

            recent_daily = df_daily_prepped.tail(5).copy()
            recent_daily = recent_daily[recent_daily["setup"] != ""]
            for candle_ts, r in recent_daily.iterrows():
                signal_time = pd.Timestamp(candle_ts).floor("1D").strftime("%Y-%m-%d")

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
                    rows_daily.append(
                        [
                            key,
                            signal_time,
                            logged_at,
                            symbol,
                            "Daily",
                            "Continuation",
                            setup,
                            bias,
                            ltp,
                            bbw,
                            bbw_rank,
                            rsi_val,
                            adx_val,
                            trend,
                            quality,
                            params_hash,
                        ]
                    )
                    existing_keys_daily.add(key)
                    new_daily += 1

        except Exception as e:
            if show_debug:
                st.warning(f"{symbol} Daily error: {e}")
            continue

    appended_15m = 0
    appended_daily = 0
    if ws_15m and not ws_15m_err:
        appended_15m, _ = append_signals(ws_15m, rows_15m, show_debug)
    if ws_daily and not ws_daily_err:
        appended_daily, _ = append_signals(ws_daily, rows_daily, show_debug)

    st.caption(f"Logged **{appended_15m}** new 15M + **{appended_daily}** Daily signals.")

    df_recent_15m = pd.DataFrame() if not ws_15m else load_recent_signals(ws_15m, retention_hours, "15M")
    df_recent_daily = pd.DataFrame() if not ws_daily else load_recent_signals(ws_daily, retention_hours, "Daily")

    if not df_recent_15m.empty or not df_recent_daily.empty:
        st.markdown("### 🧠 Recent Signals (Market-Aware)")
        if not df_recent_15m.empty:
            st.subheader("15M Signals")
            st.dataframe(df_recent_15m, use_container_width=True)
        if not df_recent_daily.empty:
            st.subheader("Daily Signals")
            st.dataframe(df_recent_daily, use_container_width=True)
    else:
        st.info("No continuation signals in recent trading sessions.")

    st.markdown("---")
    st.caption(
        "✅ **Market-aware**: Skips weekends/holidays automatically. "
        "Always shows relevant trading signals, not calendar time."
    )

with backtest_tab:
    st.markdown("## 📜 Trend Squeeze Backtest (15M)")

    def run_backtest_15m(symbols, days_back=30, params=None):
        if params is None:
            params = {
                "bbw_abs_threshold": 0.05,
                "bbw_pct_threshold": 0.35,
                "adx_threshold": 20.0,
                "rsi_bull": 55.0,
                "rsi_bear": 45.0,
            }
        trades = []
        total_bars = 0

        for symbol in symbols[:10]:
            try:
                df = get_ohlc_15min(symbol, days_back=days_back)
                if df is None or len(df) < 100:
                    continue

                df_prepped = prepare_trend_squeeze(
                    df,
                    bbw_abs_threshold=params["bbw_abs_threshold"],
                    bbw_pct_threshold=params["bbw_pct_threshold"],
                    adx_threshold=params["adx_threshold"],
                    rsi_bull=params["rsi_bull"],
                    rsi_bear=params["rsi_bear"],
                    rolling_window=20,
                )

                signals = df_prepped[df_prepped["setup"] != ""]
                total_bars += len(df_prepped)

                for ts, row in signals.iterrows():
                    bias = "LONG" if row["setup"] == "Bullish Squeeze" else "SHORT"
                    entry = float(row["close"])
                    trades.append(
                        {
                            "Symbol": symbol,
                            "Time": fmt_ts(ts),
                            "Bias": bias,
                            "Entry": entry,
                            "BBW": float(row.get("bbw", np.nan)),
                            "RSI": float(row.get("rsi", np.nan)),
                            "ADX": float(row.get("adx", np.nan)),
                            "Quality": compute_quality_score(row),
                        }
                    )
            except Exception:
                continue

        return pd.DataFrame(trades), total_bars

    col_bt1, col_bt2, col_bt3 = st.columns(3)
    with col_bt1:
        bt_days = st.slider("Days back", 10, 90, 30, step=5)
    with col_bt2:
        bt_symbols = st.slider("Symbols", 5, 20, 10, step=1)
    with col_bt3:
        bt_profile = st.selectbox("Profile", ["Normal", "Conservative", "Aggressive"])

    if bt_profile == "Conservative":
        bt_params = {
            "bbw_abs_threshold": 0.035,
            "bbw_pct_threshold": 0.25,
            "adx_threshold": 25.0,
            "rsi_bull": 60.0,
            "rsi_bear": 40.0,
        }
    elif bt_profile == "Aggressive":
        bt_params = {
            "bbw_abs_threshold": 0.065,
            "bbw_pct_threshold": 0.45,
            "adx_threshold": 18.0,
            "rsi_bull": 52.0,
            "rsi_bear": 48.0,
        }
    else:
        bt_params = {
            "bbw_abs_threshold": 0.05,
            "bbw_pct_threshold": 0.35,
            "adx_threshold": 20.0,
            "rsi_bull": 55.0,
            "rsi_bear": 45.0,
        }

    if st.button("🚀 Run Backtest", type="primary"):
        if not fyers_ok:
            st.error("Fyers is not initialized. Check fyers_app_id / fyers_secret in secrets.")
        else:
            with st.spinner("Running backtest on Fyers data..."):
                bt_results, total_bars = run_backtest_15m(stock_list[:bt_symbols], bt_days, bt_params)

            if not bt_results.empty:
                st.success(f"✅ Backtest complete: {len(bt_results)} signals across {total_bars:,} bars")
                q_counts = bt_results["Quality"].value_counts()
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("A Signals", int(q_counts.get("A", 0)))
                with c2:
                    st.metric("B Signals", int(q_counts.get("B", 0)))
                with c3:
                    st.metric("C Signals", int(q_counts.get("C", 0)))

                st.dataframe(bt_results, use_container_width=True)

                bias_pct = bt_results["Bias"].value_counts(normalize=True) * 100
                st.subheader("Bias Distribution (%)")
                st.bar_chart(bias_pct)
            else:
                st.warning("No signals found in the selected backtest period.")

    st.caption("Backtest uses the same Fyers OHLC + Trend Squeeze logic as the live screener.")

st.markdown("---")
st.caption("✅ Fyers-powered: real-time data + dual timeframe + market-aware signals + full backtest.")
