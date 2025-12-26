# app.py
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import json
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# -------------------- Helpers: Time --------------------
def now_ist() -> datetime:
    return datetime.now(IST)

def market_open_now() -> bool:
    n = now_ist()
    return is_trading_day(n.date()) and MARKET_OPEN <= n.time() <= MARKET_CLOSE

def fmt_dt(dt: datetime) -> str:
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

def to_ist_naive(ts) -> pd.Timestamp:
    """
    After the updated utils/zerodha_utils.py, OHLC index is already IST-naive.
    This function keeps you safe if something slips in tz-aware.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Kolkata").tz_localize(None)
    return t

def fmt_ts(ts) -> str:
    t = to_ist_naive(ts)
    return t.strftime("%d-%b-%Y %H:%M")

def human_age(dt: pd.Timestamp, now: pd.Timestamp) -> str:
    delta = now - dt
    mins = int(delta.total_seconds() // 60)
    if mins < 60:
        return f"{mins}m ago"
    hrs = mins // 60
    if hrs < 24:
        return f"{hrs}h ago"
    days = hrs // 24
    return f"{days}d ago"


# -------------------- Google Sheets Auth --------------------
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
        return gspread.authorize(creds), None
    except Exception as e:
        return None, f"gspread auth failed: {e}"


# -------------------- FYERS Token Sheet Manager --------------------
FYERS_TOKEN_HEADERS = ["fyers_app_id", "fyers_secret_id", "fyers_access_token", "fyers_token_updated_at"]

def get_fyers_token_worksheet():
    sheet_key = st.secrets.get("FYERS_TOKEN_SHEET_KEY", None)
    sheet_name = st.secrets.get("FYERS_TOKEN_SHEET_NAME", None)
    if not sheet_key and not sheet_name:
        return None, "Missing FYERS_TOKEN_SHEET_KEY (or FYERS_TOKEN_SHEET_NAME) in Streamlit secrets."

    client, err = get_gspread_client()
    if err:
        return None, err

    try:
        sh = client.open_by_key(sheet_key) if sheet_key else client.open(sheet_name)
        ws = sh.sheet1
    except SpreadsheetNotFound:
        return None, "FYERS token sheet not found (check key/name and sharing permissions)."
    except Exception as e:
        return None, f"Failed to open FYERS token sheet: {e}"

    # Ensure headers exist/contain required columns
    try:
        header = ws.row_values(1)
        if not header:
            ws.update("A1", [FYERS_TOKEN_HEADERS])
        else:
            header_l = [h.strip() for h in header]
            changed = False
            for h in FYERS_TOKEN_HEADERS:
                if h not in header_l:
                    header_l.append(h)
                    changed = True
            if changed:
                ws.update("A1", [header_l])
    except Exception as e:
        return None, f"Failed to ensure FYERS token sheet headers: {e}"

    return ws, None

def read_fyers_row(ws):
    try:
        records = ws.get_all_records()
    except Exception:
        records = []

    if records:
        row = records[0]
        app_id = (row.get("fyers_app_id") or row.get("app_id") or row.get("client_id") or "").strip()
        secret_id = (row.get("fyers_secret_id") or row.get("secret_id") or row.get("secret_key") or "").strip()
        token = (row.get("fyers_access_token") or row.get("access_token") or row.get("token") or "").strip()
        updated_at = (row.get("fyers_token_updated_at") or row.get("token_updated_at") or "").strip()
        return app_id, secret_id, token, updated_at

    # Fallback raw cell layout
    try:
        app_id = (ws.acell("A2").value or "").strip()
        secret_id = (ws.acell("B2").value or "").strip()
        token = (ws.acell("C2").value or "").strip()
        updated_at = (ws.acell("D2").value or "").strip()
        return app_id, secret_id, token, updated_at
    except Exception:
        return "", "", "", ""

def ensure_fyers_row_exists(ws):
    try:
        vals = ws.get_all_values()
        if len(vals) < 2:
            ws.append_row(["", "", "", ""], value_input_option="USER_ENTERED")
    except Exception:
        pass

def update_fyers_token(ws, token: str, timestamp: str):
    header = [h.strip() for h in ws.row_values(1)]

    if "fyers_access_token" not in header:
        header.append("fyers_access_token")
        ws.update("A1", [header])
    if "fyers_token_updated_at" not in header:
        header.append("fyers_token_updated_at")
        ws.update("A1", [header])

    col_token = header.index("fyers_access_token") + 1
    col_ts = header.index("fyers_token_updated_at") + 1

    ensure_fyers_row_exists(ws)
    ws.update_cell(2, col_token, token)
    ws.update_cell(2, col_ts, timestamp)

def build_fyers_login_url(app_id: str, secret_id: str, redirect_uri: str) -> str:
    session = fyersModel.SessionModel(
        client_id=app_id,
        secret_key=secret_id,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )
    return session.generate_authcode()

def exchange_auth_code_for_token(app_id: str, secret_id: str, redirect_uri: str, auth_code: str) -> str:
    session = fyersModel.SessionModel(
        client_id=app_id,
        secret_key=secret_id,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )
    session.set_token(auth_code)
    resp = session.generate_token()
    if not isinstance(resp, dict) or "access_token" not in resp:
        raise RuntimeError(f"Token exchange failed: {resp}")
    return str(resp["access_token"]).strip()

@st.cache_data(ttl=120, show_spinner=False)
def fyers_smoke_test() -> tuple[bool, str]:
    try:
        df = get_ohlc_15min("RELIANCE", days_back=2)
        if df is None or len(df) < 10:
            return False, "OHLC returned empty/too short"
        return True, "OK"
    except Exception as e:
        return False, str(e)[:200]


# -------------------- Sheets for signals logging --------------------
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
        return None, f"Sheet '{sheet_name}' not found. Check file name or Sheet ID."
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
            if header[: len(SIGNAL_COLUMNS)] != SIGNAL_COLUMNS and len(header) < len(SIGNAL_COLUMNS):
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
    df = df[df["signal_time_dt"].notna()]
    df["signal_time_dt"] = df["signal_time_dt"].apply(to_ist_naive)

    cutoff = now_ist().replace(tzinfo=None) - timedelta(days=days_back)
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
    df["signal_time_dt"] = df["signal_time_dt"].apply(to_ist_naive)

    smart_hours = get_market_aware_window(base_hours, timeframe)
    cutoff = now_ist().replace(tzinfo=None) - timedelta(hours=smart_hours)
    df = df[df["signal_time_dt"] >= cutoff].copy()
    if df.empty:
        return df

    now_local = pd.Timestamp(now_ist().replace(tzinfo=None))
    df["Timestamp"] = df["signal_time_dt"].dt.strftime("%d-%b-%Y %H:%M")
    df["Age"] = df["signal_time_dt"].apply(lambda x: human_age(pd.Timestamp(x), now_local))

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

    # Sort: latest first, but we'll also allow Quality-first sorting later
    df = df.sort_values("signal_time_dt", ascending=False)

    cols = ["Timestamp", "Age", "Symbol", "Timeframe", "Quality", "Bias", "Setup", "LTP", "BBW", "BBW %Rank", "RSI", "ADX", "Trend"]
    return df[cols]


# -------------------- Universe --------------------
fallback_nifty50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV",
    "BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH",
    "HDFCBANK","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM",
    "M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBIN","SBILIFE","SUNPHARMA","TATACONSUM",
    "TMPV","TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HEROMOTOCO","SHREECEM"
]
stock_list = fallback_nifty50


# ---------- Signature detection (compat safety) ----------
_STRAT_SIG = inspect.signature(prepare_trend_squeeze)
_HAS_ENGINE = "engine" in _STRAT_SIG.parameters
_HAS_BOX_WIDTH = "box_width_pct_max" in _STRAT_SIG.parameters
_HAS_RSI_FLOOR = "rsi_floor_short" in _STRAT_SIG.parameters
_HAS_DI = "require_di_confirmation" in _STRAT_SIG.parameters
_HAS_ADX_USE_FILTER = "adx_use_as_filter" in _STRAT_SIG.parameters


# -------------------- Sidebar: Token Manager UI --------------------
with st.sidebar:
    st.subheader("🔐 FYERS Token")
    ws_fyers, ws_fyers_err = get_fyers_token_worksheet()

    if ws_fyers_err:
        st.error("FYERS token sheet ❌")
        st.caption(ws_fyers_err)
        app_id_sheet = secret_id_sheet = token_sheet = updated_at_sheet = ""
    else:
        app_id_sheet, secret_id_sheet, token_sheet, updated_at_sheet = read_fyers_row(ws_fyers)

        if token_sheet:
            st.success("Token found ✅")
        else:
            st.warning("No token yet ❗")

        if updated_at_sheet:
            st.caption(f"Last updated: {updated_at_sheet}")

    st.markdown("---")
    with st.expander("🔑 Generate / Update Token", expanded=False):
        redirect_uri = st.secrets.get("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri/index.html")

        if not app_id_sheet or not secret_id_sheet:
            st.error("Missing fyers_app_id / fyers_secret_id in the FYERS token sheet (row 2).")
        else:
            try:
                login_url = build_fyers_login_url(app_id_sheet, secret_id_sheet, redirect_uri)
                st.link_button("1) Open FYERS Login", login_url)
                st.caption("After login, copy `auth_code` from redirected URL and paste below.")
            except Exception as e:
                st.error(f"Could not generate login URL: {e}")
                login_url = None

            auth_code = st.text_input("2) Paste auth_code", value="", help="From redirected URL after FYERS login.")

            if st.button("3) Exchange & Save Token", type="primary"):
                if not auth_code.strip():
                    st.error("Please paste auth_code first.")
                else:
                    try:
                        token = exchange_auth_code_for_token(app_id_sheet, secret_id_sheet, redirect_uri, auth_code.strip())
                        ts = now_ist().strftime("%Y-%m-%d %H:%M:%S")
                        update_fyers_token(ws_fyers, token, ts)
                        st.success("Saved access token ✅")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Token exchange failed: {e}")


# -------------------- FYERS INIT (non-blocking) --------------------
fyers_ok = False
fyers_health_ok = False
fyers_health_msg = "Not checked"

ws_fyers, ws_fyers_err = get_fyers_token_worksheet()
app_id_sheet, secret_id_sheet, token_sheet, updated_at_sheet = ("", "", "", "")
if not ws_fyers_err and ws_fyers:
    app_id_sheet, secret_id_sheet, token_sheet, updated_at_sheet = read_fyers_row(ws_fyers)

if token_sheet and app_id_sheet:
    try:
        init_fyers_session(app_id_sheet, token_sheet)
        fyers_ok = True
        fyers_health_ok, fyers_health_msg = fyers_smoke_test()
    except Exception as e:
        fyers_ok = False
        fyers_health_ok = False
        fyers_health_msg = str(e)[:200]
else:
    fyers_ok = False
    fyers_health_ok = False
    fyers_health_msg = "No token (scan disabled)."


# -------------------- Header --------------------
st.title("📉 Trend Squeeze Screener (15M + Daily) — Market Aware")

n = now_ist()
mode_str = "🟢 LIVE MARKET" if market_open_now() else "🔵 OFF-MARKET"
st.caption(f"{mode_str} • Last refresh: {n.strftime('%d %b %Y • %H:%M:%S')} IST")

last_trading_day = get_last_trading_day(n.date())
st.caption(f"🗓️ Last trading day: {last_trading_day.strftime('%d %b %Y')}")

if fyers_ok and fyers_health_ok:
    st.success("✅ FYERS OK")
elif fyers_ok and not fyers_health_ok:
    st.warning(f"⚠️ FYERS session set, but fetch failing: {fyers_health_msg}")
else:
    st.warning("⚠️ FYERS not ready (token missing/invalid). You can still view previously logged signals.")

st.caption(f"Universe: NIFTY50 ({len(stock_list)} symbols).")


# -------------------- Tabs: Signals, Settings, Backtest --------------------
tab_signals, tab_settings, tab_backtest = st.tabs(["📌 Signals (Now)", "🛠 Settings", "📜 Backtest"])


# -------------------- Settings (stored in session state) --------------------
def ss_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

# Defaults
ss_init("show_debug", False)
ss_init("audit_mode", False)
ss_init("retention_hours", 24)
ss_init("catchup_candles_15m", 32)

ss_init("auto_scan_market_hours", True)
ss_init("scan_parallel", True)
ss_init("parallel_workers", 6)

ss_init("signal_mode", "✅ Breakout Confirmed (trade)")
ss_init("breakout_lookback", 20)
ss_init("require_bbw_expansion", True)
ss_init("require_volume_spike", False)
ss_init("volume_spike_mult", 1.5)

ss_init("param_profile", "Normal")

# Engine defaults
ss_init("engine_ui", "Hybrid (recommended)")
ss_init("box_width_pct_max", 1.2)  # percent
ss_init("require_di_confirmation", True)
ss_init("adx_use_as_filter", False)

ss_init("rsi_floor_short", 30.0)
ss_init("rsi_ceiling_long", 70.0)

# Filter defaults
ss_init("quality_main", ["A"])
ss_init("bias_filter", "All")
ss_init("symbol_search", "")


with tab_settings:
    st.subheader("🛠 Settings (hidden from Signals tab)")

    cA, cB, cC = st.columns(3)
    with cA:
        st.session_state.show_debug = st.checkbox("Show debug messages", value=st.session_state.show_debug)
        st.session_state.audit_mode = st.checkbox(
            "Audit mode (show all signals, no de-dup in UI)",
            value=st.session_state.audit_mode,
            help="Shows all signals instead of keeping only latest per symbol/timeframe.",
        )
    with cB:
        st.session_state.auto_scan_market_hours = st.checkbox(
            "Auto-scan during market hours",
            value=st.session_state.auto_scan_market_hours,
            help="If OFF, app will only show logged signals and scan only when you click Scan Now.",
        )
        st.session_state.retention_hours = st.slider("Retention (base calendar hours)", 6, 72, st.session_state.retention_hours, 6)
    with cC:
        st.session_state.catchup_candles_15m = st.slider("15M Catch-up window (candles)", 8, 96, st.session_state.catchup_candles_15m, 4)

    st.markdown("---")
    st.subheader("🧨 Signal type + confirmation")
    st.session_state.signal_mode = st.radio(
        "Signal type",
        ["✅ Breakout Confirmed (trade)", "🟡 Setup Forming (watchlist)"],
        index=0 if st.session_state.signal_mode.startswith("✅") else 1,
        horizontal=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state.breakout_lookback = st.slider("Range lookback (candles)", 10, 60, st.session_state.breakout_lookback, 5)
    with c2:
        st.session_state.require_bbw_expansion = st.checkbox("Require BBW expansion", value=st.session_state.require_bbw_expansion)
    with c3:
        st.session_state.require_volume_spike = st.checkbox("Require volume spike", value=st.session_state.require_volume_spike)
    with c4:
        st.session_state.volume_spike_mult = st.slider("Volume spike mult (vs 20 SMA)", 1.0, 3.0, st.session_state.volume_spike_mult, 0.1)

    st.markdown("---")
    st.subheader("🧩 Engine")
    if _HAS_ENGINE:
        st.session_state.engine_ui = st.selectbox(
            "Detection engine",
            ["Hybrid (recommended)", "Box Breakout (range)", "Squeeze Breakout (TTM)"],
            index=["Hybrid (recommended)", "Box Breakout (range)", "Squeeze Breakout (TTM)"].index(st.session_state.engine_ui)
        )
    else:
        st.info("Your strategy.py does not support engine parameter. (App still works.)")

    if _HAS_BOX_WIDTH:
        st.session_state.box_width_pct_max = st.slider("Max box width (%)", 0.3, 3.0, st.session_state.box_width_pct_max, 0.1)
    if _HAS_DI:
        st.session_state.require_di_confirmation = st.checkbox("Require DI confirmation (+DI/-DI)", value=st.session_state.require_di_confirmation)
    if _HAS_ADX_USE_FILTER:
        st.session_state.adx_use_as_filter = st.checkbox("Use ADX as filter in Box engine", value=st.session_state.adx_use_as_filter)

    if _HAS_RSI_FLOOR:
        st.markdown("---")
        st.subheader("🚫 Anti-chase filters")
        st.session_state.rsi_floor_short = st.slider("Block SHORT if RSI below", 10.0, 45.0, st.session_state.rsi_floor_short, 1.0)
        st.session_state.rsi_ceiling_long = st.slider("Block LONG if RSI above", 55.0, 90.0, st.session_state.rsi_ceiling_long, 1.0)

    st.markdown("---")
    st.subheader("⚡ Performance")
    st.session_state.scan_parallel = st.checkbox("Parallel scan (faster)", value=st.session_state.scan_parallel)
    if st.session_state.scan_parallel:
        st.session_state.parallel_workers = st.slider("Parallel workers", 2, 10, st.session_state.parallel_workers, 1)
    st.caption("Tip: FYERS may rate-limit if you go too high. 6 is a sane default.")

    st.markdown("---")
    st.subheader("🧪 Parameter Profile (quick presets)")
    st.session_state.param_profile = st.selectbox("Profile", ["Normal", "Conservative", "Aggressive"],
                                                  index=["Normal", "Conservative", "Aggressive"].index(st.session_state.param_profile))


# -------------------- Parameter profile -> thresholds --------------------
def get_profile_thresholds(profile: str):
    if profile == "Conservative":
        return 0.035, 0.25, 25.0, 60.0, 40.0
    if profile == "Aggressive":
        return 0.065, 0.45, 18.0, 52.0, 48.0
    return 0.05, 0.35, 20.0, 55.0, 45.0


# -------------------- Scan core --------------------
def build_engine_arg(engine_ui: str) -> str | None:
    if not _HAS_ENGINE:
        return None
    if engine_ui.startswith("Box"):
        return "box"
    if engine_ui.startswith("Squeeze"):
        return "squeeze"
    return "hybrid"

def make_kwargs(bbw_abs_threshold, bbw_pct_threshold, adx_threshold, rsi_bull, rsi_bear):
    kwargs = dict(
        bbw_abs_threshold=bbw_abs_threshold,
        bbw_pct_threshold=bbw_pct_threshold,
        adx_threshold=adx_threshold,
        rsi_bull=rsi_bull,
        rsi_bear=rsi_bear,
        rolling_window=20,
        breakout_lookback=int(st.session_state.breakout_lookback),
        require_bbw_expansion=bool(st.session_state.require_bbw_expansion),
        require_volume_spike=bool(st.session_state.require_volume_spike),
        volume_spike_mult=float(st.session_state.volume_spike_mult),
    )
    eng = build_engine_arg(st.session_state.engine_ui)
    if eng is not None:
        kwargs["engine"] = eng
    if _HAS_BOX_WIDTH:
        kwargs["box_width_pct_max"] = float(st.session_state.box_width_pct_max) / 100.0
    if _HAS_DI:
        kwargs["require_di_confirmation"] = bool(st.session_state.require_di_confirmation)
    if _HAS_ADX_USE_FILTER:
        kwargs["adx_use_as_filter"] = bool(st.session_state.adx_use_as_filter)
    if _HAS_RSI_FLOOR:
        kwargs["rsi_floor_short"] = float(st.session_state.rsi_floor_short)
        kwargs["rsi_ceiling_long"] = float(st.session_state.rsi_ceiling_long)
    return kwargs

def scan_symbol_15m(symbol: str, use_setup_col: str, catchup_candles: int, existing_keys: set, params_hash: str):
    rows = []
    try:
        df = get_ohlc_15min(symbol, days_back=15)
        if df is None or df.shape[0] < 220:
            return rows, None

        bbw_abs, bbw_pct, adx_thr, rsi_bull, rsi_bear = get_profile_thresholds(st.session_state.param_profile)
        kwargs = make_kwargs(bbw_abs, bbw_pct, adx_thr, rsi_bull, rsi_bear)

        df_prepped = prepare_trend_squeeze(df, **kwargs)

        recent = df_prepped.tail(int(catchup_candles)).copy()
        recent = recent[recent[use_setup_col] != ""]

        logged_at = fmt_dt(now_ist())

        for candle_ts, r in recent.iterrows():
            ts_ist = to_ist_naive(candle_ts).floor("15min")
            signal_time = ts_ist.strftime("%Y-%m-%d %H:%M")
            setup = r[use_setup_col]
            trend = r.get("trend", "")
            ltp = float(r.get("close", np.nan))
            bbw = float(r.get("bbw", np.nan))
            bbw_rank = r.get("bbw_pct_rank", np.nan)
            rsi_val = float(r.get("rsi", np.nan))
            adx_val = float(r.get("adx", np.nan))

            bias = "LONG" if str(setup).startswith("Bullish") else "SHORT"
            key = f"{symbol}|15M|Continuation|{signal_time}|{setup}|{bias}"

            if key not in existing_keys:
                quality = compute_quality_score(r)
                rows.append([
                    key, signal_time, logged_at, symbol, "15M", "Continuation",
                    setup, bias, ltp, bbw, bbw_rank, rsi_val, adx_val, trend,
                    quality, params_hash
                ])
                existing_keys.add(key)

        return rows, None
    except Exception as e:
        return rows, str(e)

def scan_symbol_daily(symbol: str, use_setup_col: str, existing_keys: set, params_hash: str):
    rows = []
    try:
        df_daily = get_ohlc_daily(symbol, lookback_days=252)
        if df_daily is None or df_daily.shape[0] < 100:
            return rows, None

        bbw_abs, bbw_pct, adx_thr, rsi_bull, rsi_bear = get_profile_thresholds(st.session_state.param_profile)
        # Slightly relaxed BBW abs for daily
        kwargs = make_kwargs(bbw_abs * 1.2, bbw_pct, adx_thr, rsi_bull, rsi_bear)

        df_prepped = prepare_trend_squeeze(df_daily, **kwargs)

        recent = df_prepped.tail(7).copy()
        recent = recent[recent[use_setup_col] != ""]

        logged_at = fmt_dt(now_ist())

        for candle_ts, r in recent.iterrows():
            ts_ist = to_ist_naive(candle_ts).floor("1D")
            signal_time = ts_ist.strftime("%Y-%m-%d")
            setup = r[use_setup_col]
            trend = r.get("trend", "")
            ltp = float(r.get("close", np.nan))
            bbw = float(r.get("bbw", np.nan))
            bbw_rank = r.get("bbw_pct_rank", np.nan)
            rsi_val = float(r.get("rsi", np.nan))
            adx_val = float(r.get("adx", np.nan))

            bias = "LONG" if str(setup).startswith("Bullish") else "SHORT"
            key = f"{symbol}|Daily|Continuation|{signal_time}|{setup}|{bias}"

            if key not in existing_keys:
                quality = compute_quality_score(r)
                rows.append([
                    key, signal_time, logged_at, symbol, "Daily", "Continuation",
                    setup, bias, ltp, bbw, bbw_rank, rsi_val, adx_val, trend,
                    quality, params_hash
                ])
                existing_keys.add(key)

        return rows, None
    except Exception as e:
        return rows, str(e)


# -------------------- SIGNALS TAB: show signals FIRST --------------------
with tab_signals:
    st.subheader("🧠 Recent Signals (Market-Aware)")

    # Minimal filters row (trader-friendly)
    f1, f2, f3, f4 = st.columns([1.2, 1.0, 1.2, 1.6])
    with f1:
        quality_main = st.multiselect("Quality", ["A", "B", "C"], default=st.session_state.quality_main)
        st.session_state.quality_main = quality_main or ["A"]
    with f2:
        bias_filter = st.selectbox("Bias", ["All", "LONG", "SHORT"], index=["All", "LONG", "SHORT"].index(st.session_state.bias_filter))
        st.session_state.bias_filter = bias_filter
    with f3:
        mode_filter = st.selectbox("Mode", ["All", "Breakouts", "Forming"], index=0)
    with f4:
        symbol_search = st.text_input("Search symbol", value=st.session_state.symbol_search, placeholder="e.g. SBIN")
        st.session_state.symbol_search = symbol_search

    # Sheet load
    ws_15m, ws_15m_err = get_signals_worksheet(SIGNAL_SHEET_15M)
    ws_daily, ws_daily_err = get_signals_worksheet(SIGNAL_SHEET_DAILY)

    # FIXED: no inline ternary that returns DeltaGenerator (your “weird output” bug)
    if ws_15m_err:
        st.error(f"15M Sheet ❌: {ws_15m_err}")
    else:
        st.success("15M Sheet ✅")

    if ws_daily_err:
        st.error(f"Daily Sheet ❌: {ws_daily_err}")
    else:
        st.success("Daily Sheet ✅")

    retention_hours = int(st.session_state.retention_hours)
    smart_retention_15m = get_market_aware_window(retention_hours, "15M")
    smart_retention_daily = get_market_aware_window(retention_hours, "Daily")
    st.caption(f"Showing ~{smart_retention_15m}h trading-time (15M) | ~{smart_retention_daily/24:.0f} trading-days (Daily)")

    df_15m = pd.DataFrame() if (not ws_15m or ws_15m_err) else load_recent_signals(ws_15m, retention_hours, "15M")
    df_daily = pd.DataFrame() if (not ws_daily or ws_daily_err) else load_recent_signals(ws_daily, retention_hours, "Daily")

    # Apply UI filters
    def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        out = df.copy()

        # Quality filter
        out = out[out["Quality"].isin(st.session_state.quality_main)]

        # Bias filter
        if st.session_state.bias_filter != "All":
            out = out[out["Bias"] == st.session_state.bias_filter]

        # Mode filter (rough): based on Setup label
        if mode_filter == "Breakouts":
            out = out[out["Setup"].str.contains("Breakout", na=False)]
        elif mode_filter == "Forming":
            out = out[~out["Setup"].str.contains("Breakout", na=False)]

        # Symbol search
        q = (st.session_state.symbol_search or "").strip().upper()
        if q:
            out = out[out["Symbol"].astype(str).str.upper().str.contains(q, na=False)]

        # Quality-first sorting
        order = {"A": 0, "B": 1, "C": 2}
        out["_q"] = out["Quality"].map(order).fillna(9)
        out = out.sort_values(["_q", "Timestamp"], ascending=[True, False]).drop(columns=["_q"])
        return out

    df_15m_f = apply_filters(df_15m)
    df_daily_f = apply_filters(df_daily)

    # Summary metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("15M Signals", 0 if df_15m_f.empty else len(df_15m_f))
    m2.metric("Daily Signals", 0 if df_daily_f.empty else len(df_daily_f))

    newest = ""
    if not df_15m_f.empty:
        newest = df_15m_f.iloc[0]["Timestamp"]
    if not newest and not df_daily_f.empty:
        newest = df_daily_f.iloc[0]["Timestamp"]
    m3.metric("Newest", newest or "—")
    m4.metric("Market", "LIVE" if market_open_now() else "OFF")

    # Tables (15M left, Daily right)
    left, right = st.columns(2)

    with left:
        st.markdown("### 15M Signals")
        if df_15m_f.empty:
            st.info("No matching 15M signals in this window.")
        else:
            st.dataframe(df_15m_f, use_container_width=True)
            st.download_button(
                "⬇️ Download 15M CSV",
                df_15m_f.to_csv(index=False).encode("utf-8"),
                file_name="trend_squeeze_15m_signals.csv",
                mime="text/csv"
            )

    with right:
        st.markdown("### Daily Signals")
        if df_daily_f.empty:
            st.info("No matching Daily signals in this window.")
        else:
            st.dataframe(df_daily_f, use_container_width=True)
            st.download_button(
                "⬇️ Download Daily CSV",
                df_daily_f.to_csv(index=False).encode("utf-8"),
                file_name="trend_squeeze_daily_signals.csv",
                mime="text/csv"
            )

    st.markdown("---")

    # Scan controls kept minimal here
    scan_now = st.button("🔄 Scan Now (log new signals)", type="primary", disabled=(not fyers_ok or not fyers_health_ok))

    auto_scan = st.session_state.auto_scan_market_hours and market_open_now()
    do_scan = scan_now or auto_scan

    if not (fyers_ok and fyers_health_ok) and (scan_now):
        st.warning("Scan disabled: FYERS not healthy. Fix token first.")

    if do_scan and fyers_ok and fyers_health_ok and ws_15m and ws_daily and not ws_15m_err and not ws_daily_err:
        with st.spinner("Scanning NIFTY50 (15M + Daily) and logging new signals..."):
            show_debug = bool(st.session_state.show_debug)

            use_setup_col = "setup" if st.session_state.signal_mode.startswith("✅") else "setup_forming"

            bbw_abs, bbw_pct, adx_thr, rsi_bull, rsi_bear = get_profile_thresholds(st.session_state.param_profile)
            params_hash = params_to_hash(bbw_abs, bbw_pct, adx_thr, rsi_bull, rsi_bear)

            existing_keys_15m = fetch_existing_keys_recent(ws_15m, days_back=3)
            existing_keys_daily = fetch_existing_keys_recent(ws_daily, days_back=10)

            rows_15m, rows_daily = [], []

            # 15M scan
            if st.session_state.scan_parallel:
                errs = []
                with ThreadPoolExecutor(max_workers=int(st.session_state.parallel_workers)) as ex:
                    futs = [
                        ex.submit(
                            scan_symbol_15m,
                            sym,
                            use_setup_col,
                            int(st.session_state.catchup_candles_15m),
                            existing_keys_15m,
                            params_hash,
                        )
                        for sym in stock_list
                    ]
                    for fut in as_completed(futs):
                        rws, err = fut.result()
                        if rws:
                            rows_15m.extend(rws)
                        if err:
                            errs.append(err)

                if show_debug and errs:
                    st.warning(f"15M scan had {len(errs)} errors (showing first 3): {errs[:3]}")
            else:
                for sym in stock_list:
                    rws, err = scan_symbol_15m(
                        sym,
                        use_setup_col,
                        int(st.session_state.catchup_candles_15m),
                        existing_keys_15m,
                        params_hash,
                    )
                    rows_15m.extend(rws)
                    if show_debug and err:
                        st.warning(f"{sym} 15M error: {err}")

            # Daily scan (ALL 50)
            if st.session_state.scan_parallel:
                errs = []
                with ThreadPoolExecutor(max_workers=int(st.session_state.parallel_workers)) as ex:
                    futs = [
                        ex.submit(
                            scan_symbol_daily,
                            sym,
                            use_setup_col,
                            existing_keys_daily,
                            params_hash,
                        )
                        for sym in stock_list
                    ]
                    for fut in as_completed(futs):
                        rws, err = fut.result()
                        if rws:
                            rows_daily.extend(rws)
                        if err:
                            errs.append(err)
                if show_debug and errs:
                    st.warning(f"Daily scan had {len(errs)} errors (showing first 3): {errs[:3]}")
            else:
                for sym in stock_list:
                    rws, err = scan_symbol_daily(sym, use_setup_col, existing_keys_daily, params_hash)
                    rows_daily.extend(rws)
                    if show_debug and err:
                        st.warning(f"{sym} Daily error: {err}")

            appended_15m, _ = append_signals(ws_15m, rows_15m, show_debug)
            appended_daily, _ = append_signals(ws_daily, rows_daily, show_debug)

        st.success(f"Logged {appended_15m} new 15M + {appended_daily} new Daily signals.")
        st.rerun()


# -------------------- Backtest (kept simple) --------------------
with tab_backtest:
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

        for symbol in symbols:
            try:
                df = get_ohlc_15min(symbol, days_back=days_back)
                if df is None or len(df) < 100:
                    continue

                kwargs = dict(
                    bbw_abs_threshold=params["bbw_abs_threshold"],
                    bbw_pct_threshold=params["bbw_pct_threshold"],
                    adx_threshold=params["adx_threshold"],
                    rsi_bull=params["rsi_bull"],
                    rsi_bear=params["rsi_bear"],
                    rolling_window=20,
                    breakout_lookback=20,
                    require_bbw_expansion=True,
                    require_volume_spike=False,
                    volume_spike_mult=1.5,
                )

                df_prepped = prepare_trend_squeeze(df, **kwargs)
                signals = df_prepped[df_prepped["setup"] != ""]
                total_bars += len(df_prepped)

                for ts, row in signals.iterrows():
                    setup = str(row.get("setup", ""))
                    bias = "LONG" if setup.startswith("Bullish") else "SHORT"
                    entry = float(row.get("close", np.nan))

                    trades.append({
                        "Symbol": symbol,
                        "Time": fmt_ts(ts),
                        "Setup": setup,
                        "Bias": bias,
                        "Entry": entry,
                        "BBW": float(row.get("bbw", np.nan)),
                        "RSI": float(row.get("rsi", np.nan)),
                        "ADX": float(row.get("adx", np.nan)),
                        "Quality": compute_quality_score(row),
                    })
            except Exception:
                continue

        return pd.DataFrame(trades), total_bars

    col_bt1, col_bt2, col_bt3 = st.columns(3)
    with col_bt1:
        bt_days = st.slider("Days back", 10, 90, 30, step=5)
    with col_bt2:
        bt_symbols = st.slider("Symbols", 5, 50, 10, step=5)
    with col_bt3:
        bt_profile = st.selectbox("Profile", ["Normal", "Conservative", "Aggressive"], index=0)

    bbw_abs, bbw_pct, adx_thr, rsi_bull, rsi_bear = get_profile_thresholds(bt_profile)
    bt_params = {"bbw_abs_threshold": bbw_abs, "bbw_pct_threshold": bbw_pct, "adx_threshold": adx_thr, "rsi_bull": rsi_bull, "rsi_bear": rsi_bear}

    if st.button("🚀 Run Backtest", type="primary"):
        if not (fyers_ok and fyers_health_ok):
            st.warning("FYERS not healthy: backtest may fail. Fix token first.")
        with st.spinner("Running backtest on FYERS data..."):
            bt_results, total_bars = run_backtest_15m(stock_list[:bt_symbols], bt_days, bt_params)

        if not bt_results.empty:
            st.success(f"✅ Backtest complete: {len(bt_results)} signals across {total_bars:,} bars")
            q_counts = bt_results["Quality"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("A Signals", int(q_counts.get("A", 0)))
            c2.metric("B Signals", int(q_counts.get("B", 0)))
            c3.metric("C Signals", int(q_counts.get("C", 0)))
            st.dataframe(bt_results, use_container_width=True)
            st.download_button(
                "⬇️ Download Backtest CSV",
                bt_results.to_csv(index=False).encode("utf-8"),
                file_name="trend_squeeze_backtest.csv",
                mime="text/csv"
            )
        else:
            st.warning("No signals found in the selected backtest period.")


st.markdown("---")
st.caption("✅ FYERS-powered: signals-first UI • dual timeframe • market-aware window • CSV export • backtest.")
