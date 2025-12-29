# app.py
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
# Streamlit setup (MUST be first)
# =========================
st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")
st_autorefresh(interval=300000, key="auto_refresh")  # 5 min refresh

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Signal log sheets (IDs you already use)
SIGNAL_SHEET_15M = "TrendSqueezeSignals_15M"
SIGNAL_SHEET_DAILY = "TrendSqueezeSignals_Daily"
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

FYERS_TOKEN_HEADERS = ["fyers_app_id", "fyers_secret_id", "fyers_access_token", "fyers_token_updated_at"]


# =========================
# Time helpers
# =========================
def now_ist() -> datetime:
    return datetime.now(IST)


def market_open_now() -> bool:
    n = now_ist()
    return is_trading_day(n.date()) and MARKET_OPEN <= n.time() <= MARKET_CLOSE


def fmt_dt(dt: datetime) -> str:
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def to_ist(ts) -> pd.Timestamp:
    """
    IMPORTANT: In your FYERS history wrapper (utils/zerodha_utils.py),
    timestamps are derived from epoch seconds -> pandas datetime (tz-naive).
    Those are effectively UTC. So: treat tz-naive as UTC, convert to IST.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
    else:
        t = t.tz_convert("Asia/Kolkata").tz_localize(None)
    return t


def ts_to_sheet_str(ts, timeframe: str) -> str:
    t = to_ist(ts)
    if timeframe == "15M":
        t = t.floor("15min")
        return t.strftime("%Y-%m-%d %H:%M")
    # Daily
    t = t.floor("1D")
    return t.strftime("%Y-%m-%d")


# =========================
# Google Sheets auth
# =========================
def get_gspread_client():
    try:
        raw = st.secrets["gcp_service_account"]
    except Exception:
        return None, "Missing Streamlit secret: gcp_service_account"

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


# =========================
# FYERS token sheet manager
# =========================
def get_fyers_token_worksheet():
    """
    Uses st.secrets:
      - FYERS_TOKEN_SHEET_KEY (preferred) OR FYERS_TOKEN_SHEET_NAME
    """
    sheet_key = st.secrets.get("FYERS_TOKEN_SHEET_KEY", None)
    sheet_name = st.secrets.get("FYERS_TOKEN_SHEET_NAME", None)
    if not sheet_key and not sheet_name:
        return None, "Missing Streamlit secret: FYERS_TOKEN_SHEET_KEY (or FYERS_TOKEN_SHEET_NAME)"

    client, err = get_gspread_client()
    if err:
        return None, err

    try:
        sh = client.open_by_key(sheet_key) if sheet_key else client.open(sheet_name)
        ws = sh.sheet1
    except SpreadsheetNotFound:
        return None, "FYERS token sheet not found (check key/name + sharing permissions)."
    except Exception as e:
        return None, f"Failed to open FYERS token sheet: {e}"

    # Ensure required headers exist
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


def ensure_fyers_row_exists(ws):
    try:
        vals = ws.get_all_values()
        if len(vals) < 2:
            ws.append_row(["", "", "", ""], value_input_option="USER_ENTERED")
    except Exception:
        pass


def read_fyers_row(ws):
    """
    Reads first data row (row 2) as dict using get_all_records().
    Returns (app_id, secret_id, access_token, updated_at_str)
    """
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

    # Fallback raw cell layout A2/B2/C2/D2
    try:
        app_id = (ws.acell("A2").value or "").strip()
        secret_id = (ws.acell("B2").value or "").strip()
        token = (ws.acell("C2").value or "").strip()
        updated_at = (ws.acell("D2").value or "").strip()
        return app_id, secret_id, token, updated_at
    except Exception:
        return "", "", "", ""


def update_fyers_token(ws, token: str, timestamp: str):
    header = [h.strip() for h in ws.row_values(1)]

    if "fyers_access_token" not in header:
        header.append("fyers_access_token")
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


# =========================
# Signal sheets helpers
# =========================
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
        return None, f"Sheet '{sheet_name}' not found. Check Sheet ID / sharing."
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

    # signal_time is stored as IST string, keep it as naive local time
    df["signal_time_dt"] = pd.to_datetime(df["signal_time"], errors="coerce")
    df = df[df["signal_time_dt"].notna()]
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


def load_recent_signals(ws, base_hours: int, timeframe: str, audit_mode: bool) -> pd.DataFrame:
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
    if not audit_mode:
        df = df.drop_duplicates(subset=["Symbol", "Timeframe"], keep="first")

    cols = ["Timestamp", "Symbol", "Timeframe", "Quality", "Bias", "Setup", "LTP", "BBW", "BBW %Rank", "RSI", "ADX", "Trend"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


# =========================
# Strategy signature detection (keeps compatibility if strategy.py changes)
# =========================
_STRAT_SIG = inspect.signature(prepare_trend_squeeze)
_HAS_ENGINE = "engine" in _STRAT_SIG.parameters
_HAS_BOX_WIDTH = "box_width_pct_max" in _STRAT_SIG.parameters
_HAS_RSI_FLOOR = "rsi_floor_short" in _STRAT_SIG.parameters
_HAS_DI = "require_di_confirmation" in _STRAT_SIG.parameters
_HAS_ADX_FILTER = "adx_use_as_filter" in _STRAT_SIG.parameters


# =========================
# Sidebar: FYERS login + settings
# =========================
with st.sidebar:
    st.subheader("🔐 FYERS Token (Google Sheet)")

    ws_fyers, ws_fyers_err = get_fyers_token_worksheet()
    app_id_sheet = secret_id_sheet = token_sheet = updated_at_sheet = ""

    if ws_fyers_err:
        st.error("FYERS token sheet ❌")
        st.caption(ws_fyers_err)
    else:
        app_id_sheet, secret_id_sheet, token_sheet, updated_at_sheet = read_fyers_row(ws_fyers)
        if token_sheet:
            st.success("Token found ✅")
        else:
            st.warning("No token yet ❗")
        if updated_at_sheet:
            st.caption(f"Last updated: {updated_at_sheet}")

    st.markdown("---")
    st.subheader("🔑 Daily Login (Generate Token)")

    redirect_uri = st.secrets.get("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri/index.html")

    if not ws_fyers_err and ws_fyers:
        if not app_id_sheet or not secret_id_sheet:
            st.error("Fill row 2 in FYERS sheet: fyers_app_id and fyers_secret_id.")
        else:
            try:
                login_url = build_fyers_login_url(app_id_sheet, secret_id_sheet, redirect_uri)
                # Avoid st.link_button() compatibility issues across Streamlit versions
                st.markdown(f"**1) Open FYERS Login:**\n\n➡️ {login_url}")
                st.caption("Open the URL above, login, then copy the `auth_code` from redirected URL.")
            except Exception as e:
                st.error(f"Login URL failed: {e}")

            auth_code = st.text_input("2) Paste auth_code", value="", key="auth_code_input")

            if st.button("3) Exchange & Save Token", type="primary", key="exchange_save_token_btn"):
                if not auth_code.strip():
                    st.error("Paste the auth_code first.")
                else:
                    try:
                        token = exchange_auth_code_for_token(app_id_sheet, secret_id_sheet, redirect_uri, auth_code.strip())
                        ts = now_ist().strftime("%Y-%m-%d %H:%M:%S")
                        update_fyers_token(ws_fyers, token, ts)
                        st.success("Saved access token + timestamp ✅")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Token exchange failed: {e}")

    st.markdown("---")
    st.subheader("⚙️ Display")
    show_debug = st.checkbox("Show debug logs", value=False, key="show_debug_cb")
    audit_mode = st.checkbox(
        "Audit mode (no de-dup in table)",
        value=False,
        help="Use this if you suspect signals were logged but UI hid them.",
        key="audit_mode_cb",
    )

    st.subheader("⭐ Quality filter")
    show_quality_a = st.checkbox("Show A", True, key="show_q_a")
    show_quality_b = st.checkbox("Show B", False, key="show_q_b")
    show_quality_c = st.checkbox("Show C", False, key="show_q_c")

    st.markdown("---")
    with st.expander("⚙️ Screener Settings (advanced)", expanded=False):
        catchup_candles_15m = st.slider(
            "15M catch-up candles",
            min_value=8, max_value=80, value=32, step=4,
            help="Scans last N candles and logs any setups found (deduped).",
            key="catchup_15m",
        )
        retention_hours = st.slider(
            "Retention window (calendar hours)",
            min_value=6, max_value=72, value=24, step=6,
            help="App auto-adjusts around weekends.",
            key="retention_hours",
        )

        st.markdown("**Signal mode**")
        signal_mode = st.radio(
            "Type",
            ["✅ Breakout Confirmed (trade)", "🟡 Setup Forming (watchlist)"],
            index=0,
            key="signal_mode_radio",
        )
        use_setup_col = "setup" if signal_mode.startswith("✅") else "setup_forming"

        st.markdown("**Breakout confirmation**")
        breakout_lookback = st.slider(
            "Consolidation lookback (candles)",
            10, 60, 20, 5,
            key="breakout_lookback",
        )
        require_bbw_expansion = st.checkbox("Require BBW expansion", value=True, key="bbw_expand_cb")
        require_volume_spike = st.checkbox("Require volume spike (if volume exists)", value=False, key="vol_spike_cb")
        volume_spike_mult = st.slider("Volume spike mult (vs 20 SMA)", 1.0, 3.0, 1.5, 0.1, key="vol_spike_mult")

        st.markdown("**Parameter profile**")
        param_profile = st.selectbox(
            "Profile",
            ["Normal", "Conservative", "Aggressive"],
            index=0,
            key="param_profile_select",
        )

        # Optional engine controls (only if strategy supports them)
        engine_arg = None
        box_width_pct_max = None
        require_di_confirmation = None
        rsi_floor_short = None
        rsi_ceiling_long = None
        adx_use_as_filter = None

        if _HAS_ENGINE:
            engine_ui = st.selectbox(
                "Detection engine",
                ["Hybrid (recommended)", "Box Breakout (range)", "Squeeze Breakout (TTM)"],
                index=0,
                key="engine_select",
            )
            engine_arg = "hybrid" if engine_ui.startswith("Hybrid") else ("box" if engine_ui.startswith("Box") else "squeeze")

        if _HAS_BOX_WIDTH:
            box_width_pct_max = st.slider("Max box width (%)", 0.3, 3.0, 1.2, 0.1, key="box_width") / 100.0

        if _HAS_DI:
            require_di_confirmation = st.checkbox("Require DI confirmation (+DI/-DI)", value=True, key="di_confirm_cb")

        if _HAS_ADX_FILTER:
            adx_use_as_filter = st.checkbox("Use ADX as filter for Box engine", value=False, key="adx_filter_cb")

        if _HAS_RSI_FLOOR:
            st.markdown("**Anti-chase guardrails**")
            rsi_floor_short = st.slider("Block SHORT if RSI below", 10.0, 45.0, 30.0, 1.0, key="rsi_floor_short")
            rsi_ceiling_long = st.slider("Block LONG if RSI above", 55.0, 90.0, 70.0, 1.0, key="rsi_ceiling_long")
    # end expander


# Provide defaults if advanced expander not opened (Streamlit still defines them if in expander).
# But to be safe in case future edits remove expander: set sensible fallback.
catchup_candles_15m = locals().get("catchup_candles_15m", 32)
retention_hours = locals().get("retention_hours", 24)
use_setup_col = locals().get("use_setup_col", "setup")
param_profile = locals().get("param_profile", "Normal")
breakout_lookback = locals().get("breakout_lookback", 20)
require_bbw_expansion = locals().get("require_bbw_expansion", True)
require_volume_spike = locals().get("require_volume_spike", False)
volume_spike_mult = locals().get("volume_spike_mult", 1.5)

engine_arg = locals().get("engine_arg", None)
box_width_pct_max = locals().get("box_width_pct_max", None)
require_di_confirmation = locals().get("require_di_confirmation", None)
rsi_floor_short = locals().get("rsi_floor_short", None)
rsi_ceiling_long = locals().get("rsi_ceiling_long", None)
adx_use_as_filter = locals().get("adx_use_as_filter", None)


# =========================
# FYERS init (sheet-driven only)
# =========================
fyers_ok = False
fyers_health_ok = False
fyers_health_msg = "Not checked"

ws_fyers, ws_fyers_err = get_fyers_token_worksheet()
app_id_sheet = secret_id_sheet = token_sheet = updated_at_sheet = ""
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
        fyers_health_msg = str(e)[:250]
else:
    fyers_ok = False
    fyers_health_ok = False
    fyers_health_msg = "No token yet. Login via sidebar and save token to the FYERS sheet."


# =========================
# Main UI
# =========================
st.title("📉 Trend Squeeze Screener (15M + Daily)")
n = now_ist()
mode_str = "🟢 LIVE MARKET" if market_open_now() else "🔵 OFF-MARKET"
st.caption(f"{mode_str} • Last refresh: {n.strftime('%d %b %Y • %H:%M:%S')} IST")
last_trading_day = get_last_trading_day(n.date())
st.caption(f"🗓️ Last trading day: {last_trading_day.strftime('%d %b %Y')}")

if fyers_ok and fyers_health_ok:
    st.success("✅ FYERS data fetch OK")
elif fyers_ok and not fyers_health_ok:
    st.warning(f"⚠️ FYERS session set but data fetch failing: {fyers_health_msg}")
else:
    st.error("🔴 FYERS not initialized (no valid token). Use the sidebar login flow.")
    st.stop()

fallback_nifty50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV",
    "BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH",
    "HDFCBANK","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM",
    "M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBIN","SBILIFE","SUNPHARMA","TATACONSUM",
    "TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HEROMOTOCO","SHREECEM"
]
stock_list = fallback_nifty50
st.caption(f"Universe: NIFTY50 ({len(stock_list)} symbols).")

# Tabs
live_tab, backtest_tab = st.tabs(["📺 Live Screener", "📜 Backtest"])


# =========================
# Params (profile -> thresholds)
# =========================
if param_profile == "Conservative":
    bbw_abs_default, bbw_pct_default = 0.035, 0.25
    adx_default, rsi_bull_default, rsi_bear_default = 25.0, 60.0, 40.0
elif param_profile == "Aggressive":
    bbw_abs_default, bbw_pct_default = 0.065, 0.45
    adx_default, rsi_bull_default, rsi_bear_default = 18.0, 52.0, 48.0
else:
    bbw_abs_default, bbw_pct_default = 0.05, 0.35
    adx_default, rsi_bull_default, rsi_bear_default = 20.0, 55.0, 45.0


# =========================
# Live tab
# =========================
with live_tab:
    # --- Top: Recent Signals section FIRST (as requested) ---
    st.subheader("🧠 Recent Signals (Market-Aware)")

    # Worksheet handles
    ws_15m, ws_15m_err = get_signals_worksheet(SIGNAL_SHEET_15M)
    ws_daily, ws_daily_err = get_signals_worksheet(SIGNAL_SHEET_DAILY)

    cols_status = st.columns(2)
    with cols_status[0]:
        st.success("15M Sheets ✅") if not ws_15m_err else st.error(f"15M Sheets ❌: {ws_15m_err}")
    with cols_status[1]:
        st.success("Daily Sheets ✅") if not ws_daily_err else st.error(f"Daily Sheets ❌: {ws_daily_err}")

    smart_retention_15m = get_market_aware_window(retention_hours, "15M")
    smart_retention_daily = get_market_aware_window(retention_hours, "Daily")
    st.caption(
        f"Showing ~{smart_retention_15m}h trading time (15M) • ~{smart_retention_daily/24:.0f} trading days (Daily)"
    )

    df_recent_15m = pd.DataFrame() if (not ws_15m or ws_15m_err) else load_recent_signals(ws_15m, retention_hours, "15M", audit_mode)
    df_recent_daily = pd.DataFrame() if (not ws_daily or ws_daily_err) else load_recent_signals(ws_daily, retention_hours, "Daily", audit_mode)

    # Quality filter
    allowed_q = set()
    if show_quality_a:
        allowed_q.add("A")
    if show_quality_b:
        allowed_q.add("B")
    if show_quality_c:
        allowed_q.add("C")

    def _apply_quality_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df is None or df.empty:
            return df, pd.DataFrame()
        if "Quality" not in df.columns:
            return df, pd.DataFrame()
        df_a = df[df["Quality"] == "A"].copy()
        df_other = df[df["Quality"].isin(["B", "C"])].copy()
        # apply sidebar visibility
        df_main = df[df["Quality"].isin(list(allowed_q))].copy() if allowed_q else df_a
        # Also provide B/C separately
        return df_main, df_other

    # Show A signals first (top)
    if df_recent_15m.empty and df_recent_daily.empty:
        st.info("No recent signals in the log window.")
    else:
        # 15M display
        if not df_recent_15m.empty:
            st.markdown("### 15M Signals")
            df_15m_main, df_15m_bc = _apply_quality_filter(df_recent_15m)
            # Sort Quality A first regardless
            if "Quality" in df_15m_main.columns:
                df_15m_main = df_15m_main.sort_values(["Quality", "Timestamp"], ascending=[True, False])
            st.dataframe(df_15m_main, use_container_width=True)

            with st.expander("Show hidden B/C (15M)", expanded=False):
                if df_15m_bc.empty:
                    st.caption("No B/C signals in window.")
                else:
                    st.dataframe(df_15m_bc, use_container_width=True)

        # Daily display
        if not df_recent_daily.empty:
            st.markdown("### Daily Signals")
            df_daily_main, df_daily_bc = _apply_quality_filter(df_recent_daily)
            if "Quality" in df_daily_main.columns:
                df_daily_main = df_daily_main.sort_values(["Quality", "Timestamp"], ascending=[True, False])
            st.dataframe(df_daily_main, use_container_width=True)

            with st.expander("Show hidden B/C (Daily)", expanded=False):
                if df_daily_bc.empty:
                    st.caption("No B/C signals in window.")
                else:
                    st.dataframe(df_daily_bc, use_container_width=True)

    st.markdown("---")

    # --- Below: scanning + logging ---
    st.subheader("📡 Live Scan & Logging")

    # Parameter sliders placed in main area (minimal, presentable)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        bbw_abs_threshold = st.slider("BBW max", 0.01, 0.20, bbw_abs_default, step=0.005, key="bbw_abs")
    with c2:
        bbw_pct_threshold = st.slider("BBW %Rank max", 0.10, 0.80, bbw_pct_default, step=0.05, key="bbw_pct")
    with c3:
        adx_threshold = st.slider("ADX", 15.0, 35.0, adx_default, step=1.0, key="adx_th")
    with c4:
        rsi_bull = st.slider("RSI Bull", 50.0, 70.0, rsi_bull_default, step=1.0, key="rsi_bull")
    with c5:
        rsi_bear = st.slider("RSI Bear", 30.0, 50.0, rsi_bear_default, step=1.0, key="rsi_bear")

    params_hash = params_to_hash(bbw_abs_threshold, bbw_pct_threshold, adx_threshold, rsi_bull, rsi_bear)

    existing_keys_15m = fetch_existing_keys_recent(ws_15m, days_back=3) if ws_15m and not ws_15m_err else set()
    existing_keys_daily = fetch_existing_keys_recent(ws_daily, days_back=14) if ws_daily and not ws_daily_err else set()

    rows_15m, rows_daily = [], []
    logged_at = fmt_dt(now_ist())

    def build_strategy_kwargs(is_daily: bool):
        kwargs = dict(
            bbw_abs_threshold=(bbw_abs_threshold * 1.2 if is_daily else bbw_abs_threshold),
            bbw_pct_threshold=bbw_pct_threshold,
            adx_threshold=adx_threshold,
            rsi_bull=rsi_bull,
            rsi_bear=rsi_bear,
            rolling_window=20,
            breakout_lookback=breakout_lookback,
            require_bbw_expansion=require_bbw_expansion,
            require_volume_spike=require_volume_spike,
            volume_spike_mult=volume_spike_mult,
        )

        # optional upgrades only if your strategy.py supports them
        if _HAS_ENGINE and engine_arg:
            kwargs["engine"] = engine_arg
        if _HAS_BOX_WIDTH and box_width_pct_max is not None:
            kwargs["box_width_pct_max"] = box_width_pct_max
        if _HAS_DI and require_di_confirmation is not None:
            kwargs["require_di_confirmation"] = require_di_confirmation
        if _HAS_ADX_FILTER and adx_use_as_filter is not None:
            kwargs["adx_use_as_filter"] = adx_use_as_filter
        if _HAS_RSI_FLOOR and rsi_floor_short is not None and rsi_ceiling_long is not None:
            kwargs["rsi_floor_short"] = rsi_floor_short
            kwargs["rsi_ceiling_long"] = rsi_ceiling_long

        return kwargs

    # --- 15M scan ---
    for symbol in stock_list:
        try:
            df = get_ohlc_15min(symbol, days_back=15)
            # Need enough bars for stable EMA200 on 15m
            if df is None or df.shape[0] < 220:
                continue

            df_prepped = prepare_trend_squeeze(df, **build_strategy_kwargs(is_daily=False))

            recent = df_prepped.tail(int(catchup_candles_15m)).copy()
            if use_setup_col not in recent.columns:
                continue

            recent = recent[recent[use_setup_col] != ""]
            for candle_ts, r in recent.iterrows():
                signal_time = ts_to_sheet_str(candle_ts, "15M")
                setup = r.get(use_setup_col, "")
                if not setup:
                    continue

                trend = r.get("trend", "")
                ltp = float(r.get("close", np.nan))
                bbw = float(r.get("bbw", np.nan))
                bbw_rank = r.get("bbw_pct_rank", np.nan)
                rsi_val = float(r.get("rsi", np.nan))
                adx_val = float(r.get("adx", np.nan))

                bias = "LONG" if str(setup).startswith("Bullish") else "SHORT"
                key = f"{symbol}|15M|Continuation|{signal_time}|{setup}|{bias}"

                if key not in existing_keys_15m:
                    quality = compute_quality_score(r)
                    rows_15m.append([
                        key, signal_time, logged_at, symbol, "15M", "Continuation",
                        setup, bias, ltp, bbw, bbw_rank, rsi_val, adx_val, trend,
                        quality, params_hash
                    ])
                    existing_keys_15m.add(key)

        except Exception as e:
            if show_debug:
                st.warning(f"{symbol} 15M error: {e}")
            continue

    # --- Daily scan: INCLUDE ALL 50 (your request) ---
    for symbol in stock_list:
        try:
            df_daily = get_ohlc_daily(symbol, lookback_days=400)
            if df_daily is None or df_daily.shape[0] < 180:
                continue

            df_daily_prepped = prepare_trend_squeeze(df_daily, **build_strategy_kwargs(is_daily=True))

            recent_daily = df_daily_prepped.tail(7).copy()
            if use_setup_col not in recent_daily.columns:
                continue

            recent_daily = recent_daily[recent_daily[use_setup_col] != ""]
            for candle_ts, r in recent_daily.iterrows():
                signal_time = ts_to_sheet_str(candle_ts, "Daily")
                setup = r.get(use_setup_col, "")
                if not setup:
                    continue

                trend = r.get("trend", "")
                ltp = float(r.get("close", np.nan))
                bbw = float(r.get("bbw", np.nan))
                bbw_rank = r.get("bbw_pct_rank", np.nan)
                rsi_val = float(r.get("rsi", np.nan))
                adx_val = float(r.get("adx", np.nan))

                bias = "LONG" if str(setup).startswith("Bullish") else "SHORT"
                key = f"{symbol}|Daily|Continuation|{signal_time}|{setup}|{bias}"

                if key not in existing_keys_daily:
                    quality = compute_quality_score(r)
                    rows_daily.append([
                        key, signal_time, logged_at, symbol, "Daily", "Continuation",
                        setup, bias, ltp, bbw, bbw_rank, rsi_val, adx_val, trend,
                        quality, params_hash
                    ])
                    existing_keys_daily.add(key)

        except Exception as e:
            if show_debug:
                st.warning(f"{symbol} Daily error: {e}")
            continue

    appended_15m = appended_daily = 0
    err_15m = err_daily = None
    if ws_15m and not ws_15m_err:
        appended_15m, err_15m = append_signals(ws_15m, rows_15m, show_debug)
    if ws_daily and not ws_daily_err:
        appended_daily, err_daily = append_signals(ws_daily, rows_daily, show_debug)

    if err_15m:
        st.warning(f"15M log: {err_15m}")
    if err_daily:
        st.warning(f"Daily log: {err_daily}")

    st.caption(f"Logged **{appended_15m}** new 15M + **{appended_daily}** new Daily signals.")


# =========================
# Backtest tab (fix duplicate IDs via keys)
# =========================
with backtest_tab:
    st.subheader("📜 Trend Squeeze Backtest (15M)")

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

        # keep it limited so Streamlit stays snappy
        for symbol in symbols[:10]:
            try:
                df = get_ohlc_15min(symbol, days_back=days_back)
                if df is None or len(df) < 150:
                    continue

                kwargs = dict(
                    bbw_abs_threshold=params["bbw_abs_threshold"],
                    bbw_pct_threshold=params["bbw_pct_threshold"],
                    adx_threshold=params["adx_threshold"],
                    rsi_bull=params["rsi_bull"],
                    rsi_bear=params["rsi_bear"],
                    rolling_window=20,
                    breakout_lookback=breakout_lookback,
                    require_bbw_expansion=require_bbw_expansion,
                    require_volume_spike=require_volume_spike,
                    volume_spike_mult=volume_spike_mult,
                )
                if _HAS_ENGINE and engine_arg:
                    kwargs["engine"] = engine_arg
                if _HAS_BOX_WIDTH and box_width_pct_max is not None:
                    kwargs["box_width_pct_max"] = box_width_pct_max
                if _HAS_DI and require_di_confirmation is not None:
                    kwargs["require_di_confirmation"] = require_di_confirmation
                if _HAS_ADX_FILTER and adx_use_as_filter is not None:
                    kwargs["adx_use_as_filter"] = adx_use_as_filter
                if _HAS_RSI_FLOOR and rsi_floor_short is not None and rsi_ceiling_long is not None:
                    kwargs["rsi_floor_short"] = rsi_floor_short
                    kwargs["rsi_ceiling_long"] = rsi_ceiling_long

                df_prepped = prepare_trend_squeeze(df, **kwargs)

                if "setup" not in df_prepped.columns:
                    continue

                signals = df_prepped[df_prepped["setup"] != ""]
                total_bars += len(df_prepped)

                for ts, row in signals.iterrows():
                    setup = str(row.get("setup", ""))
                    bias = "LONG" if setup.startswith("Bullish") else "SHORT"
                    entry = float(row.get("close", np.nan))

                    trades.append({
                        "Symbol": symbol,
                        "Time": to_ist(ts).strftime("%d-%b-%Y %H:%M"),
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
        bt_days = st.slider("Days back", 10, 120, 30, step=5, key="bt_days")
    with col_bt2:
        bt_symbols = st.slider("Symbols", 5, 20, 10, step=1, key="bt_symbols")
    with col_bt3:
        bt_profile = st.selectbox("Profile", ["Normal", "Conservative", "Aggressive"], index=0, key="bt_profile")

    if bt_profile == "Conservative":
        bt_params = {"bbw_abs_threshold": 0.035, "bbw_pct_threshold": 0.25, "adx_threshold": 25.0, "rsi_bull": 60.0, "rsi_bear": 40.0}
    elif bt_profile == "Aggressive":
        bt_params = {"bbw_abs_threshold": 0.065, "bbw_pct_threshold": 0.45, "adx_threshold": 18.0, "rsi_bull": 52.0, "rsi_bear": 48.0}
    else:
        bt_params = {"bbw_abs_threshold": 0.05, "bbw_pct_threshold": 0.35, "adx_threshold": 20.0, "rsi_bull": 55.0, "rsi_bear": 45.0}

    if st.button("🚀 Run Backtest", type="primary", key="run_backtest_btn"):
        with st.spinner("Running backtest on FYERS data..."):
            bt_results, total_bars = run_backtest_15m(stock_list[:bt_symbols], bt_days, bt_params)

        if not bt_results.empty:
            st.success(f"✅ Backtest complete: {len(bt_results)} signals across {total_bars:,} bars")
            q_counts = bt_results["Quality"].value_counts()
            m1, m2, m3 = st.columns(3)
            m1.metric("A Signals", int(q_counts.get("A", 0)))
            m2.metric("B Signals", int(q_counts.get("B", 0)))
            m3.metric("C Signals", int(q_counts.get("C", 0)))
            st.dataframe(bt_results, use_container_width=True)
        else:
            st.warning("No signals found in the selected backtest period.")

st.markdown("---")
st.caption("✅ FYERS-powered • IST-correct timestamps • Daily scans all NIFTY50 • A-first UI • Sidebar login preserved.")
