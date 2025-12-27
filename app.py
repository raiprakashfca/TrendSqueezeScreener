# app.py
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import json
import inspect
from typing import Tuple, Optional

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

FYERS_TOKEN_HEADERS = ["fyers_app_id", "fyers_secret_id", "fyers_access_token", "fyers_token_updated_at"]


# -------------------- Helpers: Time --------------------
def now_ist() -> datetime:
    return datetime.now(IST)


def market_open_now() -> bool:
    n = now_ist()
    return is_trading_day(n.date()) and MARKET_OPEN <= n.time() <= MARKET_CLOSE


def fmt_dt(dt: datetime) -> str:
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def to_ist(ts):
    """
    Normalize timestamp to IST and return tz-naive.

    FYERS candles are epoch seconds -> we convert to naive datetime; that is effectively UTC-naive in many pipelines.
    Some sources may give IST-naive. We apply a heuristic:
      If tz-naive hour < 8 => treat as UTC.
      Else treat as IST.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        if t.hour < 8:
            t = t.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
        else:
            t = t.tz_localize("Asia/Kolkata").tz_localize(None)
    else:
        t = t.tz_convert("Asia/Kolkata").tz_localize(None)
    return t


def fmt_ts(ts) -> str:
    t = to_ist(pd.Timestamp(ts))
    return t.strftime("%d-%b-%Y %H:%M")


def ts_to_sheet_str(ts, tf: str = "15M") -> str:
    t = to_ist(pd.Timestamp(ts))
    if tf == "Daily":
        t = t.floor("1D")
        return t.strftime("%Y-%m-%d")
    t = t.floor("15min")
    return t.strftime("%Y-%m-%d %H:%M")


# -------------------- Google Sheets Auth --------------------
def get_gspread_client() -> Tuple[Optional[gspread.Client], Optional[str]]:
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

    # Fallback raw cell layout (A2/B2/C2/D2)
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
def fyers_smoke_test() -> Tuple[bool, str]:
    try:
        df = get_ohlc_15min("RELIANCE", days_back=2)
        if df is None or len(df) < 10:
            return False, "OHLC returned empty/too short"
        return True, "OK"
    except Exception as e:
        return False, str(e)[:200]


def token_age_badge(updated_at: str) -> Tuple[str, str]:
    """Returns (emoji+label, color-ish message)."""
    if not updated_at:
        return "⚠️ Unknown", "No timestamp found for token."
    try:
        t = pd.to_datetime(updated_at, errors="coerce")
        if pd.isna(t):
            return "⚠️ Unknown", "Timestamp parse failed."
        # assume IST
        t = to_ist(t)
        age_h = (now_ist().replace(tzinfo=None) - t).total_seconds() / 3600.0
        if age_h < 12:
            return "✅ Fresh", f"Token age ≈ {age_h:.1f}h"
        if age_h < 24:
            return "🟡 Aging", f"Token age ≈ {age_h:.1f}h"
        return "🔴 Likely expired", f"Token age ≈ {age_h:.1f}h (regen recommended)"
    except Exception:
        return "⚠️ Unknown", "Timestamp parse failed."


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
    df["signal_time_dt"] = df["signal_time_dt"].apply(to_ist)

    cutoff = now_ist().replace(tzinfo=None) - timedelta(days=days_back)
    df = df[df["signal_time_dt"] >= cutoff]
    return set(df["key"].astype(str).tolist())


def append_signals(ws, rows: list[list], show_debug: bool = False) -> Tuple[int, Optional[str]]:
    if not rows:
        return 0, None
    try:
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        return len(rows), None
    except APIError as e:
        return (0, f"Sheets APIError: {e}") if show_debug else (0, "Sheets write failed (APIError).")
    except Exception as e:
        return (0, f"Sheets write failed: {e}") if show_debug else (0, "Sheets write failed.")


def params_to_hash(bbw_abs, bbw_pct, adx, rsi_bull, rsi_bear) -> str:
    return f"{bbw_abs:.3f}|{bbw_pct:.2f}|{adx:.1f}|{rsi_bull:.1f}|{rsi_bear:.1f}"


def compute_quality_score(row) -> str:
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
    if score >= 2:
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


def load_recent_signals(ws, base_hours: int = 24, timeframe: str = "15M", audit_mode: bool = False) -> pd.DataFrame:
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
    df["signal_time_dt"] = df["signal_time_dt"].apply(to_ist)

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
    return df[cols]


# -------------------- Cached OHLC wrappers --------------------
@st.cache_data(ttl=120, show_spinner=False)
def cached_ohlc_15m(symbol: str, days_back: int) -> pd.DataFrame:
    df = get_ohlc_15min(symbol, days_back=days_back)
    if df is None:
        return pd.DataFrame()
    # ensure index is datetime and sorted
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    except Exception:
        pass
    return df


@st.cache_data(ttl=300, show_spinner=False)
def cached_ohlc_daily(symbol: str, lookback_days: int) -> pd.DataFrame:
    df = get_ohlc_daily(symbol, lookback_days=lookback_days)
    if df is None:
        return pd.DataFrame()
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    except Exception:
        pass
    return df


# -------------------- Universe --------------------
fallback_nifty50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV",
    "BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH",
    "HDFCBANK","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM",
    "M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBIN","SBILIFE","SUNPHARMA","TATACONSUM",
    "TMPV","TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HEROMOTOCO","SHREECEM"
]
stock_list = fallback_nifty50


# -------------------- Strategy signature detection --------------------
_STRAT_SIG = inspect.signature(prepare_trend_squeeze)
_HAS_ENGINE = "engine" in _STRAT_SIG.parameters
_HAS_BOX_WIDTH = "box_width_pct_max" in _STRAT_SIG.parameters
_HAS_RSI_FLOOR = "rsi_floor_short" in _STRAT_SIG.parameters
_HAS_DI = "require_di_confirmation" in _STRAT_SIG.parameters
_HAS_ADX_USE_AS_FILTER = "adx_use_as_filter" in _STRAT_SIG.parameters


# -------------------- Sidebar: Token + Controls --------------------
with st.sidebar:
    st.subheader("🔐 FYERS Login (Daily)")

    ws_fyers, ws_fyers_err = get_fyers_token_worksheet()
    app_id_sheet = secret_id_sheet = token_sheet = updated_at_sheet = ""

    if ws_fyers_err:
        st.error("FYERS token sheet ❌")
        st.caption(ws_fyers_err)
    else:
        app_id_sheet, secret_id_sheet, token_sheet, updated_at_sheet = read_fyers_row(ws_fyers)

        badge, age_msg = token_age_badge(updated_at_sheet)
        if token_sheet:
            st.success(f"Token found ✅  ({badge})")
            st.caption(age_msg)
        else:
            st.warning("No access token in sheet yet ❗")

        if updated_at_sheet:
            st.caption(f"Last updated: {updated_at_sheet}")

    st.markdown("---")
    st.subheader("🔑 Generate / Update Token")
    redirect_uri = st.secrets.get("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri/index.html")

    # IMPORTANT: Do NOT st.stop() in sidebar (it hides UI and makes you think the button is gone)
    if not app_id_sheet or not secret_id_sheet:
        st.error("Fill `fyers_app_id` + `fyers_secret_id` in FYERS sheet (row 2). Then refresh.")
    else:
        try:
            login_url = build_fyers_login_url(app_id_sheet, secret_id_sheet, redirect_uri)
            st.markdown(
    f"""
    <a href="{login_url}" target="_blank">
        <button style="width:100%;padding:8px;font-size:15px;">
            👉 1) Open FYERS Login
        </button>
    </a>
    """,
    unsafe_allow_html=True), key="btn_fyers_login")
            st.caption("After login, copy `auth_code` from redirected URL and paste below.")
        except Exception as e:
            st.error(f"Login URL failed: {e}")

        auth_code = st.text_input("2) Paste auth_code", value="", key="inp_auth_code")
        if st.button("3) Exchange & Save Token", type="primary", key="btn_exchange_save"):
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
    st.subheader("⚙️ Screener Controls")

    show_debug = st.checkbox("Show debug", value=False, key="cb_show_debug")
    audit_mode = st.checkbox(
        "Audit mode (show all; no de-dup)",
        value=False,
        help="Use this to verify signals were logged but UI de-dup hid them.",
        key="cb_audit_mode"
    )

    retention_hours = st.slider(
        "Retention window (hours)",
        min_value=6, max_value=72, value=24, step=6,
        help="Market-aware (weekend adjusted).",
        key="sl_retention"
    )

    st.markdown("**Quality filter**")
    show_A_only = st.checkbox("Show only A by default", value=True, key="cb_A_only")

    with st.expander("Show B/C (optional)", expanded=False):
        show_B = st.checkbox("Show B", value=False, key="cb_show_B")
        show_C = st.checkbox("Show C", value=False, key="cb_show_C")

    st.markdown("---")
    st.subheader("🧨 Signal Type")
    signal_mode = st.radio(
        "Signal type",
        ["✅ Breakout Confirmed (trade)", "🟡 Setup Forming (watchlist)"],
        index=0,
        key="rd_signal_type",
        help="Trade mode waits for range-break confirmation; watchlist is early."
    )
    use_setup_col = "setup" if signal_mode.startswith("✅") else "setup_forming"

    breakout_lookback = st.slider("Range lookback (candles)", 10, 60, 20, 5, key="sl_breakout_lb")
    require_bbw_expansion = st.checkbox("Require BBW expansion", value=True, key="cb_bbw_expand")
    require_volume_spike = st.checkbox("Require volume spike (if available)", value=False, key="cb_vol_spike")
    volume_spike_mult = st.slider("Vol spike multiplier", 1.0, 3.0, 1.5, 0.1, key="sl_vol_mult")

    st.markdown("---")
    st.subheader("🧩 Engine / Guardrails")

    if _HAS_ENGINE:
        engine_ui = st.selectbox(
            "Detection engine",
            ["Hybrid (recommended)", "Box Breakout (range)", "Squeeze Breakout (TTM)"],
            index=0,
            key="sb_engine"
        )
        engine_arg = "hybrid"
        if engine_ui.startswith("Box"):
            engine_arg = "box"
        elif engine_ui.startswith("Squeeze"):
            engine_arg = "squeeze"
    else:
        engine_arg = None
        st.info("Engine controls not supported by your current utils/strategy.py")

    box_width_pct_max = None
    if _HAS_BOX_WIDTH:
        box_width_pct_max = st.slider("Max box width (%)", 0.3, 3.0, 1.2, 0.1, key="sl_box_width") / 100.0

    require_di_confirmation = None
    if _HAS_DI:
        require_di_confirmation = st.checkbox("Require DI confirmation (+DI/-DI)", value=True, key="cb_di")

    rsi_floor_short = rsi_ceiling_long = None
    if _HAS_RSI_FLOOR:
        rsi_floor_short = st.slider("Block SHORT if RSI below", 10.0, 45.0, 30.0, 1.0, key="sl_rsi_floor")
        rsi_ceiling_long = st.slider("Block LONG if RSI above", 55.0, 90.0, 70.0, 1.0, key="sl_rsi_ceil")

    adx_use_as_filter = None
    if _HAS_ADX_USE_AS_FILTER:
        adx_use_as_filter = st.checkbox("Use ADX as filter for Box engine", value=False, key="cb_adx_box")

    st.markdown("---")
    st.subheader("🎛️ Threshold Profile")
    param_profile = st.selectbox(
        "Profile",
        ["Normal", "Conservative", "Aggressive"],
        index=0,
        key="sb_param_profile"
    )


# -------------------- FYERS INIT (main flow) --------------------
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
        fyers_health_msg = str(e)[:200]
else:
    fyers_ok = False
    fyers_health_ok = False
    fyers_health_msg = "No token yet. Use the sidebar login flow."

# -------------------- Main header (keep it tight) --------------------
st.title("📉 Trend Squeeze Screener")
n = now_ist()
mode_str = "🟢 LIVE" if market_open_now() else "🔵 OFF-MARKET"
st.caption(f"{mode_str} • Refreshed: {n.strftime('%d %b %Y %H:%M:%S')} IST • Universe: NIFTY50 ({len(stock_list)})")

last_trading_day = get_last_trading_day(n.date())
st.caption(f"🗓️ Last trading day: {last_trading_day.strftime('%d %b %Y')}")

if fyers_ok and fyers_health_ok:
    st.success("✅ FYERS data fetch OK")
elif fyers_ok and not fyers_health_ok:
    st.warning(f"⚠️ FYERS session set but data fetch failing: {fyers_health_msg}")
else:
    st.error("🔴 FYERS not initialized (no valid token). Use the sidebar login flow.")
    st.stop()


# -------------------- Threshold defaults --------------------
if param_profile == "Conservative":
    bbw_abs_threshold, bbw_pct_threshold = 0.035, 0.25
    adx_threshold, rsi_bull, rsi_bear = 25.0, 60.0, 40.0
elif param_profile == "Aggressive":
    bbw_abs_threshold, bbw_pct_threshold = 0.065, 0.45
    adx_threshold, rsi_bull, rsi_bear = 18.0, 52.0, 48.0
else:
    bbw_abs_threshold, bbw_pct_threshold = 0.05, 0.35
    adx_threshold, rsi_bull, rsi_bear = 20.0, 55.0, 45.0

# Compact top controls (main page)
with st.expander("⚙️ Advanced thresholds (optional)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        bbw_abs_threshold = st.slider(
            "BBW absolute threshold (max BBW)",
            0.01, 0.20, float(bbw_abs_threshold), 0.005,
            key="main_bbw_abs"
        )
    with c2:
        bbw_pct_threshold = st.slider(
            "BBW percentile threshold",
            0.10, 0.80, float(bbw_pct_threshold), 0.05,
            key="main_bbw_pct"
        )

    c3, c4, c5 = st.columns(3)
    with c3:
        adx_threshold = st.slider("ADX threshold", 15.0, 35.0, float(adx_threshold), 1.0, key="main_adx")
    with c4:
        rsi_bull = st.slider("RSI bull threshold", 50.0, 70.0, float(rsi_bull), 1.0, key="main_rsi_bull")
    with c5:
        rsi_bear = st.slider("RSI bear threshold", 30.0, 50.0, float(rsi_bear), 1.0, key="main_rsi_bear")

params_hash = params_to_hash(bbw_abs_threshold, bbw_pct_threshold, adx_threshold, rsi_bull, rsi_bear)


# -------------------- Read + display recent signals FIRST (top priority UI) --------------------
ws_15m, ws_15m_err = get_signals_worksheet(SIGNAL_SHEET_15M)
ws_daily, ws_daily_err = get_signals_worksheet(SIGNAL_SHEET_DAILY)

df_recent_15m = pd.DataFrame()
df_recent_daily = pd.DataFrame()

if ws_15m and not ws_15m_err:
    df_recent_15m = load_recent_signals(ws_15m, retention_hours, "15M", audit_mode=audit_mode)
if ws_daily and not ws_daily_err:
    df_recent_daily = load_recent_signals(ws_daily, retention_hours, "Daily", audit_mode=audit_mode)

# Apply quality filter UI
allowed_qualities = {"A"}
if not show_A_only:
    allowed_qualities = {"A", "B", "C"}
else:
    if show_B:
        allowed_qualities.add("B")
    if show_C:
        allowed_qualities.add("C")

def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Quality" not in df.columns:
        return df
    return df[df["Quality"].astype(str).isin(allowed_qualities)].copy()

df_recent_15m = filter_quality(df_recent_15m)
df_recent_daily = filter_quality(df_recent_daily)

st.markdown("## 🧠 Recent Signals (Market-Aware)")
top_left, top_right = st.columns(2, gap="large")

with top_left:
    st.markdown("### 15M")
    if ws_15m_err:
        st.error(f"15M sheet error: {ws_15m_err}")
    elif df_recent_15m.empty:
        st.info("No recent 15M signals in the window.")
    else:
        # Quality A first: keep as-is; already sorted by time desc
        st.dataframe(df_recent_15m, use_container_width=True, hide_index=True)

with top_right:
    st.markdown("### Daily")
    if ws_daily_err:
        st.error(f"Daily sheet error: {ws_daily_err}")
    elif df_recent_daily.empty:
        st.info("No recent Daily signals in the window.")
    else:
        st.dataframe(df_recent_daily, use_container_width=True, hide_index=True)

st.markdown("---")


# -------------------- Live scan (kept below; can be collapsed) --------------------
with st.expander("📺 Live Scan (15M + Daily) — run & log signals", expanded=True):
    c_run1, c_run2, c_run3, c_run4 = st.columns([1.2, 1.2, 1.2, 2.4])
    with c_run1:
        catchup_candles_15m = st.slider(
            "15M catch-up candles",
            8, 64, 32, 4,
            help="How many recent 15m candles to scan/log (deduped).",
            key="sl_catchup"
        )
    with c_run2:
        days_back_15m = st.slider(
            "15M history days",
            10, 30, 15, 1,
            help="Needs enough bars for EMA200 stability.",
            key="sl_days_back_15m"
        )
    with c_run3:
        daily_symbols = st.slider(
            "Daily scan symbols",
            10, 50, 50, 5,
            help="Requested: scan all 50 by default.",
            key="sl_daily_symbols"
        )
    with c_run4:
        run_now = st.button("🚀 Run Live Scan Now", type="primary", key="btn_run_scan")

    if not run_now:
        st.caption("Auto-refresh runs every 5 minutes, but **logging** happens when scan is executed in this session.")
    else:
        existing_keys_15m = fetch_existing_keys_recent(ws_15m, days_back=3) if ws_15m and not ws_15m_err else set()
        existing_keys_daily = fetch_existing_keys_recent(ws_daily, days_back=7) if ws_daily and not ws_daily_err else set()

        rows_15m, rows_daily = [], []
        logged_at = fmt_dt(now_ist())

        # Build kwargs for strategy (only pass supported args)
        kwargs_base = dict(
            bbw_abs_threshold=bbw_abs_threshold,
            bbw_pct_threshold=bbw_pct_threshold,
            adx_threshold=adx_threshold,
            rsi_bull=rsi_bull,
            rsi_bear=rsi_bear,
            rolling_window=20,
            breakout_lookback=int(breakout_lookback),
            require_bbw_expansion=bool(require_bbw_expansion),
            require_volume_spike=bool(require_volume_spike),
            volume_spike_mult=float(volume_spike_mult),
        )
        if _HAS_ENGINE and engine_arg is not None:
            kwargs_base["engine"] = engine_arg
        if _HAS_BOX_WIDTH and box_width_pct_max is not None:
            kwargs_base["box_width_pct_max"] = float(box_width_pct_max)
        if _HAS_DI and require_di_confirmation is not None:
            kwargs_base["require_di_confirmation"] = bool(require_di_confirmation)
        if _HAS_RSI_FLOOR and rsi_floor_short is not None and rsi_ceiling_long is not None:
            kwargs_base["rsi_floor_short"] = float(rsi_floor_short)
            kwargs_base["rsi_ceiling_long"] = float(rsi_ceiling_long)
        if _HAS_ADX_USE_AS_FILTER and adx_use_as_filter is not None:
            kwargs_base["adx_use_as_filter"] = bool(adx_use_as_filter)

        # 15M scan
        scanned_15m = 0
        for symbol in stock_list:
            try:
                df = cached_ohlc_15m(symbol, days_back=int(days_back_15m))
                if df.empty or df.shape[0] < 220:
                    continue

                df_prepped = prepare_trend_squeeze(df, **kwargs_base)
                recent = df_prepped.tail(int(catchup_candles_15m)).copy()
                if use_setup_col not in recent.columns:
                    continue

                recent = recent[recent[use_setup_col].astype(str) != ""]
                if recent.empty:
                    scanned_15m += 1
                    continue

                for candle_ts, r in recent.iterrows():
                    signal_time = ts_to_sheet_str(candle_ts, tf="15M")
                    setup = str(r.get(use_setup_col, "")).strip()
                    if not setup:
                        continue

                    trend = str(r.get("trend", "")).strip()
                    ltp = float(r.get("close", np.nan))
                    bbw = float(r.get("bbw", np.nan))
                    bbw_rank = r.get("bbw_pct_rank", np.nan)
                    rsi_val = float(r.get("rsi", np.nan))
                    adx_val = float(r.get("adx", np.nan))

                    bias = "LONG" if setup.startswith("Bullish") else "SHORT"
                    key = f"{symbol}|15M|Continuation|{signal_time}|{setup}|{bias}"

                    if key not in existing_keys_15m:
                        quality = compute_quality_score(r)
                        rows_15m.append([
                            key, signal_time, logged_at, symbol, "15M", "Continuation",
                            setup, bias, ltp, bbw, bbw_rank, rsi_val, adx_val, trend,
                            quality, params_hash
                        ])
                        existing_keys_15m.add(key)

                scanned_15m += 1
            except Exception as e:
                if show_debug:
                    st.warning(f"{symbol} 15M error: {e}")
                continue

        # Daily scan (ALL 50 by default as requested)
        scanned_daily = 0
        lookback_days_daily = 252
        for symbol in stock_list[: int(daily_symbols)]:
            try:
                df_daily = cached_ohlc_daily(symbol, lookback_days=int(lookback_days_daily))
                if df_daily.empty or df_daily.shape[0] < 100:
                    continue

                kwargs_daily = dict(kwargs_base)
                kwargs_daily["bbw_abs_threshold"] = float(bbw_abs_threshold) * 1.2  # daily more tolerant

                df_daily_prepped = prepare_trend_squeeze(df_daily, **kwargs_daily)

                recent_daily = df_daily_prepped.tail(5).copy()
                if use_setup_col not in recent_daily.columns:
                    continue

                recent_daily = recent_daily[recent_daily[use_setup_col].astype(str) != ""]
                for candle_ts, r in recent_daily.iterrows():
                    signal_time = ts_to_sheet_str(candle_ts, tf="Daily")
                    setup = str(r.get(use_setup_col, "")).strip()
                    if not setup:
                        continue

                    trend = str(r.get("trend", "")).strip()
                    ltp = float(r.get("close", np.nan))
                    bbw = float(r.get("bbw", np.nan))
                    bbw_rank = r.get("bbw_pct_rank", np.nan)
                    rsi_val = float(r.get("rsi", np.nan))
                    adx_val = float(r.get("adx", np.nan))

                    bias = "LONG" if setup.startswith("Bullish") else "SHORT"
                    key = f"{symbol}|Daily|Continuation|{signal_time}|{setup}|{bias}"

                    if key not in existing_keys_daily:
                        quality = compute_quality_score(r)
                        rows_daily.append([
                            key, signal_time, logged_at, symbol, "Daily", "Continuation",
                            setup, bias, ltp, bbw, bbw_rank, rsi_val, adx_val, trend,
                            quality, params_hash
                        ])
                        existing_keys_daily.add(key)

                scanned_daily += 1
            except Exception as e:
                if show_debug:
                    st.warning(f"{symbol} Daily error: {e}")
                continue

        appended_15m = appended_daily = 0
        err15 = errD = None
        if ws_15m and not ws_15m_err:
            appended_15m, err15 = append_signals(ws_15m, rows_15m, show_debug)
        if ws_daily and not ws_daily_err:
            appended_daily, errD = append_signals(ws_daily, rows_daily, show_debug)

        csum1, csum2, csum3 = st.columns(3)
        csum1.metric("15M new logs", appended_15m)
        csum2.metric("Daily new logs", appended_daily)
        csum3.metric("Scan symbols", f"15M:{scanned_15m} / D:{scanned_daily}")

        if err15:
            st.warning(f"15M sheet write: {err15}")
        if errD:
            st.warning(f"Daily sheet write: {errD}")

        st.success("Scan complete. Refreshing Recent Signals at top…")
        st.cache_data.clear()
        st.rerun()


# -------------------- Backtest (kept tidy; fixed duplicate element IDs) --------------------
with st.expander("📜 Backtest (15M) — quick sanity check", expanded=False):
    st.caption("This is a signal-frequency backtest (not P&L). It helps validate whether signals exist under your settings.")

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

        bt_kwargs = dict(
            bbw_abs_threshold=params["bbw_abs_threshold"],
            bbw_pct_threshold=params["bbw_pct_threshold"],
            adx_threshold=params["adx_threshold"],
            rsi_bull=params["rsi_bull"],
            rsi_bear=params["rsi_bear"],
            rolling_window=20,
            breakout_lookback=int(breakout_lookback),
            require_bbw_expansion=bool(require_bbw_expansion),
            require_volume_spike=bool(require_volume_spike),
            volume_spike_mult=float(volume_spike_mult),
        )
        if _HAS_ENGINE and engine_arg is not None:
            bt_kwargs["engine"] = engine_arg
        if _HAS_BOX_WIDTH and box_width_pct_max is not None:
            bt_kwargs["box_width_pct_max"] = float(box_width_pct_max)
        if _HAS_DI and require_di_confirmation is not None:
            bt_kwargs["require_di_confirmation"] = bool(require_di_confirmation)
        if _HAS_RSI_FLOOR and rsi_floor_short is not None and rsi_ceiling_long is not None:
            bt_kwargs["rsi_floor_short"] = float(rsi_floor_short)
            bt_kwargs["rsi_ceiling_long"] = float(rsi_ceiling_long)
        if _HAS_ADX_USE_AS_FILTER and adx_use_as_filter is not None:
            bt_kwargs["adx_use_as_filter"] = bool(adx_use_as_filter)

        for symbol in symbols:
            try:
                df = cached_ohlc_15m(symbol, days_back=int(days_back))
                if df.empty or len(df) < 150:
                    continue

                df_prepped = prepare_trend_squeeze(df, **bt_kwargs)
                if "setup" not in df_prepped.columns:
                    continue

                signals = df_prepped[df_prepped["setup"].astype(str) != ""]
                total_bars += len(df_prepped)

                for ts, row in signals.iterrows():
                    setup = str(row.get("setup", ""))
                    bias = "LONG" if setup.startswith("Bullish") else "SHORT"
                    entry = float(row.get("close", np.nan))

                    trades.append({
                        "Symbol": symbol,
                        "Time (IST)": fmt_ts(ts),
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

    bt_c1, bt_c2, bt_c3 = st.columns(3)
    with bt_c1:
        bt_days = st.slider("Backtest days", 10, 90, 30, step=5, key="bt_days")
    with bt_c2:
        bt_symbols = st.slider("Symbols", 5, 50, 15, step=5, key="bt_syms")
    with bt_c3:
        bt_profile = st.selectbox(
            "Backtest Profile",
            ["Normal", "Conservative", "Aggressive"],
            index=0,
            key="bt_profile"
        )

    if bt_profile == "Conservative":
        bt_params = {"bbw_abs_threshold": 0.035, "bbw_pct_threshold": 0.25, "adx_threshold": 25.0, "rsi_bull": 60.0, "rsi_bear": 40.0}
    elif bt_profile == "Aggressive":
        bt_params = {"bbw_abs_threshold": 0.065, "bbw_pct_threshold": 0.45, "adx_threshold": 18.0, "rsi_bull": 52.0, "rsi_bear": 48.0}
    else:
        bt_params = {"bbw_abs_threshold": 0.05, "bbw_pct_threshold": 0.35, "adx_threshold": 20.0, "rsi_bull": 55.0, "rsi_bear": 45.0}

    if st.button("🚀 Run Backtest", type="primary", key="btn_run_bt"):
        with st.spinner("Running backtest on FYERS data…"):
            bt_results, total_bars = run_backtest_15m(stock_list[: int(bt_symbols)], bt_days, bt_params)

        if not bt_results.empty:
            st.success(f"✅ Backtest complete: {len(bt_results)} signals across {total_bars:,} bars")
            q_counts = bt_results["Quality"].value_counts()
            m1, m2, m3 = st.columns(3)
            m1.metric("A Signals", int(q_counts.get("A", 0)))
            m2.metric("B Signals", int(q_counts.get("B", 0)))
            m3.metric("C Signals", int(q_counts.get("C", 0)))

            # show A first
            bt_sorted = bt_results.copy()
            bt_sorted["QRank"] = bt_sorted["Quality"].map({"A": 0, "B": 1, "C": 2}).fillna(9)
            bt_sorted = bt_sorted.sort_values(["QRank", "Time (IST)"], ascending=[True, False]).drop(columns=["QRank"])
            st.dataframe(bt_sorted, use_container_width=True, hide_index=True)
        else:
            st.warning("No signals found in selected backtest period.")


# -------------------- Footer (minimal) --------------------
st.caption("✅ FYERS-powered • IST timestamps • Quality-first display • Daily scan supports all 50 symbols.")


