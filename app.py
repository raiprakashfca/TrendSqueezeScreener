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


# ✅ MUST BE FIRST Streamlit call
st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")
st_autorefresh(interval=300000, key="auto_refresh_5m")  # 5 min

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

FYERS_TOKEN_HEADERS = [
    "fyers_app_id",
    "fyers_secret_id",
    "fyers_access_token",
    "fyers_token_updated_at",
]


# -------------------- Helpers: Time --------------------
def now_ist() -> datetime:
    return datetime.now(IST)


def market_open_now() -> bool:
    n = now_ist()
    return is_trading_day(n.date()) and MARKET_OPEN <= n.time() <= MARKET_CLOSE


def fmt_dt(dt: datetime) -> str:
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def to_ist(ts) -> pd.Timestamp:
    """
    Convert ANY timestamp to IST and return tz-naive.
    IMPORTANT FIX:
    - FYERS candles come from epoch seconds -> pandas Timestamp is tz-naive but represents UTC time.
    - So: treat ALL tz-naive inputs as UTC and convert to IST.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
    else:
        t = t.tz_convert("Asia/Kolkata").tz_localize(None)
    return t


def fmt_ts(ts) -> str:
    return to_ist(ts).strftime("%d-%b-%Y %H:%M")


def ts_to_sheet_str(ts, tf: str = "15M") -> str:
    t = to_ist(ts)
    if tf.upper() in {"15M", "15"}:
        t = t.floor("15min")
        return t.strftime("%Y-%m-%d %H:%M")
    # Daily
    t = t.floor("1D")
    return t.strftime("%Y-%m-%d")


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

    # Ensure headers exist / contain required columns
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
            # if sheet is shorter, force correct header
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
    df = df[df["signal_time_dt"].notna()]
    df["signal_time_dt"] = df["signal_time_dt"].apply(to_ist)

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

    # A first, then time
    quality_order = {"A": 0, "B": 1, "C": 2}
    df["q_rank"] = df["Quality"].map(quality_order).fillna(9).astype(int)
    df = df.sort_values(["q_rank", "signal_time_dt"], ascending=[True, False]).drop(columns=["q_rank"])

    if not audit_mode:
        # IMPORTANT: This can hide earlier signals. Keep default, but user can enable audit_mode.
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


def apply_signal_filters(df: pd.DataFrame, qualities: list[str], bias: str, symbol_q: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if qualities:
        out = out[out["Quality"].isin(qualities)]
    if bias in {"LONG", "SHORT"}:
        out = out[out["Bias"] == bias]
    if symbol_q.strip():
        q = symbol_q.strip().upper()
        out = out[out["Symbol"].astype(str).str.upper().str.contains(q, na=False)]
    return out


# -------------------- Session defaults --------------------
def _ss_default(k, v):
    if k not in st.session_state:
        st.session_state[k] = v


_ss_default("param_profile", "Normal")
_ss_default("signal_mode", "✅ Breakout Confirmed (trade)")
_ss_default("breakout_lookback", 20)
_ss_default("require_bbw_expansion", True)
_ss_default("require_volume_spike", False)
_ss_default("volume_spike_mult", 1.5)
_ss_default("retention_hours", 24)
_ss_default("catchup_candles_15m", 32)
_ss_default("audit_mode", False)
_ss_default("auto_scan", True)
_ss_default("engine_ui", "Hybrid (recommended)")
_ss_default("box_width_pct_max", 1.2)  # percent
_ss_default("require_di_confirmation", True)
_ss_default("rsi_floor_short", 30.0)
_ss_default("rsi_ceiling_long", 70.0)


# -------------------- Sidebar: Token Manager UI --------------------
with st.sidebar:
    st.markdown("## 🔐 FYERS Login")

    ws_fyers, ws_fyers_err = None, None
    try:
        ws_fyers, ws_fyers_err = get_fyers_token_worksheet()
    except Exception as e:
        ws_fyers, ws_fyers_err = None, str(e)

    if ws_fyers_err:
        st.error("Token sheet error")
        st.caption(ws_fyers_err)
        app_id_sheet = secret_id_sheet = token_sheet = updated_at_sheet = ""
    else:
        app_id_sheet, secret_id_sheet, token_sheet, updated_at_sheet = read_fyers_row(ws_fyers)
        if token_sheet:
            st.success("Token found ✅")
        else:
            st.warning("No token yet")

        if updated_at_sheet:
            st.caption(f"Updated: {updated_at_sheet}")

    st.markdown("---")
    st.caption("Generate token (only when expired / first time).")

    redirect_uri = st.secrets.get("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri/index.html")

    if not app_id_sheet or not secret_id_sheet:
        st.error("Fill fyers_app_id + fyers_secret_id in the FYERS token sheet (row 2).")
        st.stop()

    try:
        login_url = build_fyers_login_url(app_id_sheet, secret_id_sheet, redirect_uri)
        st.link_button("1) Open FYERS Login", login_url, key="btn_fyers_login")
        st.caption("After login, copy `auth_code` from the redirected URL.")
    except Exception as e:
        st.error(f"Login URL failed: {e}")
        login_url = None

    auth_code = st.text_input("2) Paste auth_code", value="", key="auth_code_input")

    if st.button("3) Exchange & Save Token", type="primary", key="btn_exchange_token"):
        if not auth_code.strip():
            st.error("Paste auth_code first.")
        else:
            try:
                token = exchange_auth_code_for_token(app_id_sheet, secret_id_sheet, redirect_uri, auth_code.strip())
                ts = now_ist().strftime("%Y-%m-%d %H:%M:%S")
                update_fyers_token(ws_fyers, token, ts)
                st.success("Saved token + timestamp ✅")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Token exchange failed: {e}")

    st.markdown("---")
    st.markdown("## ⚙️ Quick Settings")
    st.session_state.auto_scan = st.toggle("Auto-run scan every refresh", value=st.session_state.auto_scan, key="tog_auto_scan")
    st.session_state.audit_mode = st.toggle("Audit mode (show all, no de-dup)", value=st.session_state.audit_mode, key="tog_audit_mode")


# -------------------- FYERS INIT --------------------
fyers_ok = False
fyers_health_ok = False
fyers_health_msg = "Not checked"

ws_fyers, ws_fyers_err = get_fyers_token_worksheet()
app_id_sheet, secret_id_sheet, token_sheet, updated_at_sheet = ("", "", "", "")
if not ws_fyers_err and ws_fyers:
    app_id_sheet, secret_id_sheet, token_sheet, updated_at_sheet = read_fyers_row(ws_fyers)

if token_sheet:
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


# -------------------- Main Header (compact) --------------------
n = now_ist()
last_trading_day = get_last_trading_day(n.date())

h1, h2, h3, h4 = st.columns([1.6, 0.9, 0.9, 1.1])
with h1:
    st.title("📉 Trend Squeeze Screener")
with h2:
    st.metric("Market", "LIVE 🟢" if market_open_now() else "OFF 🔵")
with h3:
    st.metric("Refresh", n.strftime("%H:%M:%S"))
with h4:
    st.metric("Last trading day", last_trading_day.strftime("%d-%b"))

# Health guard
if fyers_ok and fyers_health_ok:
    pass
elif fyers_ok and not fyers_health_ok:
    st.warning(f"FYERS session set but data fetch failing: {fyers_health_msg}")
else:
    st.error("FYERS not initialized (no valid token). Use the sidebar login flow.")
    st.stop()

# Universe
fallback_nifty50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV",
    "BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH",
    "HDFCBANK","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM",
    "M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBIN","SBILIFE","SUNPHARMA","TATACONSUM",
    "TMPV","TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HEROMOTOCO","SHREECEM"
]
stock_list = fallback_nifty50

# Strategy signature detection (keeps app compatible if strategy.py is older)
_STRAT_SIG = inspect.signature(prepare_trend_squeeze)
_HAS_ENGINE = "engine" in _STRAT_SIG.parameters
_HAS_BOX_WIDTH = "box_width_pct_max" in _STRAT_SIG.parameters
_HAS_RSI_FLOOR = "rsi_floor_short" in _STRAT_SIG.parameters
_HAS_DI = "require_di_confirmation" in _STRAT_SIG.parameters

# Sheets
ws_15m, ws_15m_err = get_signals_worksheet(SIGNAL_SHEET_15M)
ws_daily, ws_daily_err = get_signals_worksheet(SIGNAL_SHEET_DAILY)

# -------------------- Tabs --------------------
tab_signals, tab_settings, tab_backtest = st.tabs(["🧠 Signals", "⚙️ Settings", "📜 Backtest"])


# ==================== SETTINGS TAB ====================
with tab_settings:
    st.subheader("⚙️ Screener Settings (controls how signals are generated)")

    cA, cB, cC = st.columns(3)
    with cA:
        st.session_state.param_profile = st.selectbox(
            "Profile",
            ["Normal", "Conservative", "Aggressive"],
            index=["Normal", "Conservative", "Aggressive"].index(st.session_state.param_profile),
            key="settings_profile_select",
        )
    with cB:
        st.session_state.retention_hours = st.slider(
            "Retention (hours)",
            6, 72, int(st.session_state.retention_hours), 6,
            key="settings_retention_hours",
            help="Display window for recent signals (market-aware)."
        )
    with cC:
        st.session_state.catchup_candles_15m = st.slider(
            "15M catch-up candles (scan window)",
            8, 96, int(st.session_state.catchup_candles_15m), 4,
            key="settings_catchup_15m"
        )

    st.markdown("---")

    left, right = st.columns([1.1, 1.0])

    with left:
        st.session_state.signal_mode = st.radio(
            "Signal type",
            ["✅ Breakout Confirmed (trade)", "🟡 Setup Forming (watchlist)"],
            index=0 if st.session_state.signal_mode.startswith("✅") else 1,
            key="settings_signal_mode",
        )

        st.session_state.breakout_lookback = st.slider(
            "Consolidation range lookback (candles)",
            10, 60, int(st.session_state.breakout_lookback), 5,
            key="settings_breakout_lookback",
        )

        st.session_state.require_bbw_expansion = st.checkbox(
            "Require BBW expansion on breakout",
            value=bool(st.session_state.require_bbw_expansion),
            key="settings_require_bbw_expansion",
        )

        st.session_state.require_volume_spike = st.checkbox(
            "Require volume spike (if volume available)",
            value=bool(st.session_state.require_volume_spike),
            key="settings_require_volume_spike",
        )

        st.session_state.volume_spike_mult = st.slider(
            "Volume spike multiplier (vs 20 SMA)",
            1.0, 3.0, float(st.session_state.volume_spike_mult), 0.1,
            key="settings_volume_spike_mult",
        )

    with right:
        st.markdown("#### 🧩 Engine (optional)")
        if _HAS_ENGINE:
            st.session_state.engine_ui = st.selectbox(
                "Detection engine",
                ["Hybrid (recommended)", "Box Breakout (range)", "Squeeze Breakout (TTM)"],
                index=["Hybrid (recommended)", "Box Breakout (range)", "Squeeze Breakout (TTM)"].index(st.session_state.engine_ui),
                key="settings_engine_ui",
            )
        else:
            st.info("Your utils/strategy.py does not support Engine controls (still works).")

        if _HAS_BOX_WIDTH:
            st.session_state.box_width_pct_max = st.slider(
                "Max box width (%)",
                0.3, 3.0, float(st.session_state.box_width_pct_max), 0.1,
                key="settings_box_width_pct_max",
                help="Smaller = tighter consolidation box."
            )
        if _HAS_DI:
            st.session_state.require_di_confirmation = st.checkbox(
                "Require DI confirmation (+DI/-DI)",
                value=bool(st.session_state.require_di_confirmation),
                key="settings_require_di",
            )
        if _HAS_RSI_FLOOR:
            st.markdown("#### 🚫 Anti-chase guardrails")
            st.session_state.rsi_floor_short = st.slider(
                "Block SHORT if RSI below",
                10.0, 45.0, float(st.session_state.rsi_floor_short), 1.0,
                key="settings_rsi_floor_short",
            )
            st.session_state.rsi_ceiling_long = st.slider(
                "Block LONG if RSI above",
                55.0, 90.0, float(st.session_state.rsi_ceiling_long), 1.0,
                key="settings_rsi_ceiling_long",
            )

    st.markdown("---")
    st.caption("Tip: If you feel signals come late, try **Hybrid engine**, a slightly larger box width (e.g., 1.5%), and keep DI confirmation ON.")


# ==================== SIGNALS TAB ====================
with tab_signals:
    st.subheader("🧠 Recent Signals (Market-Aware)")

    # Compact health (hidden by default)
    with st.expander("🩺 System Health (FYERS / Sheets)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("FYERS", "OK ✅" if (fyers_ok and fyers_health_ok) else "NOT OK ❌")
            if not (fyers_ok and fyers_health_ok):
                st.caption(fyers_health_msg)
        with c2:
            st.metric("15M Sheet", "OK ✅" if not ws_15m_err else "ERROR ❌")
            if ws_15m_err:
                st.caption(ws_15m_err)
        with c3:
            st.metric("Daily Sheet", "OK ✅" if not ws_daily_err else "ERROR ❌")
            if ws_daily_err:
                st.caption(ws_daily_err)

    # Filters row (A is default; B/C hidden)
    f1, f2, f3, f4 = st.columns([1.2, 1.0, 1.0, 1.5])
    with f1:
        qualities = ["A"]  # default
        st.caption("Quality shown: **A** (default)")
    with f2:
        bias_filter = st.selectbox("Bias", ["All", "LONG", "SHORT"], index=0, key="sig_bias_filter")
    with f3:
        symbol_q = st.text_input("Search", value="", placeholder="SBIN", key="sig_symbol_search")
    with f4:
        retention_hours = st.slider(
            "Retention (hours)",
            6, 72, int(st.session_state.retention_hours), 6,
            key="sig_retention_hours",
        )

    with st.expander("Show B/C signals (optional)", expanded=False):
        show_b = st.checkbox("Include B", value=False, key="show_b_quality")
        show_c = st.checkbox("Include C", value=False, key="show_c_quality")
        if show_b:
            qualities.append("B")
        if show_c:
            qualities.append("C")

    # Load recent signals FIRST (so UI is instantly useful even if scan is slow)
    df_recent_15m = pd.DataFrame() if not ws_15m else load_recent_signals(ws_15m, retention_hours, "15M", audit_mode=st.session_state.audit_mode)
    df_recent_daily = pd.DataFrame() if not ws_daily else load_recent_signals(ws_daily, retention_hours, "Daily", audit_mode=st.session_state.audit_mode)

    bias_arg = bias_filter if bias_filter in {"LONG", "SHORT"} else "All"
    df_15m_show = apply_signal_filters(df_recent_15m, qualities, bias_arg, symbol_q)
    df_daily_show = apply_signal_filters(df_recent_daily, qualities, bias_arg, symbol_q)

    # Display tables (signals-first)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### 15M")
        if df_15m_show is None or df_15m_show.empty:
            st.info("No matching 15M signals in the window.")
        else:
            st.dataframe(df_15m_show, use_container_width=True, hide_index=True)
    with t2:
        st.markdown("### Daily (All 50)")
        if df_daily_show is None or df_daily_show.empty:
            st.info("No matching Daily signals in the window.")
        else:
            st.dataframe(df_daily_show, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Scan controls in an expander (so it doesn't hog space)
    with st.expander("🔎 Live Scan & Log to Sheets", expanded=bool(st.session_state.auto_scan)):
        st.caption(
            "This scans **NIFTY50** and logs new signals to Google Sheets (deduped). "
            "If Auto-scan feels heavy, turn it off in the sidebar."
        )

        # show_debug only here
        show_debug = st.checkbox("Show debug", value=False, key="scan_show_debug")

        # Parameter profile → thresholds
        if st.session_state.param_profile == "Conservative":
            bbw_abs_default, bbw_pct_default = 0.035, 0.25
            adx_default, rsi_bull_default, rsi_bear_default = 25.0, 60.0, 40.0
        elif st.session_state.param_profile == "Aggressive":
            bbw_abs_default, bbw_pct_default = 0.065, 0.45
            adx_default, rsi_bull_default, rsi_bear_default = 18.0, 52.0, 48.0
        else:
            bbw_abs_default, bbw_pct_default = 0.05, 0.35
            adx_default, rsi_bull_default, rsi_bear_default = 20.0, 55.0, 45.0

        p1, p2, p3, p4, p5 = st.columns([1, 1, 1, 1, 1])
        with p1:
            bbw_abs_threshold = st.slider("BBW max", 0.01, 0.20, float(bbw_abs_default), 0.005, key="scan_bbw_abs")
        with p2:
            bbw_pct_threshold = st.slider("BBW %Rank max", 0.10, 0.80, float(bbw_pct_default), 0.05, key="scan_bbw_pct")
        with p3:
            adx_threshold = st.slider("ADX min", 15.0, 35.0, float(adx_default), 1.0, key="scan_adx")
        with p4:
            rsi_bull = st.slider("RSI bull min", 50.0, 70.0, float(rsi_bull_default), 1.0, key="scan_rsi_bull")
        with p5:
            rsi_bear = st.slider("RSI bear max", 30.0, 50.0, float(rsi_bear_default), 1.0, key="scan_rsi_bear")

        params_hash = params_to_hash(bbw_abs_threshold, bbw_pct_threshold, adx_threshold, rsi_bull, rsi_bear)
        use_setup_col = "setup" if st.session_state.signal_mode.startswith("✅") else "setup_forming"

        # Engine args
        if _HAS_ENGINE:
            if st.session_state.engine_ui.startswith("Box"):
                engine_arg = "box"
            elif st.session_state.engine_ui.startswith("Squeeze"):
                engine_arg = "squeeze"
            else:
                engine_arg = "hybrid"
        else:
            engine_arg = None

        def run_scan_once() -> tuple[int, int, str]:
            if ws_15m_err or ws_daily_err:
                return 0, 0, "Sheets not ready."

            existing_keys_15m = fetch_existing_keys_recent(ws_15m, days_back=3) if ws_15m else set()
            existing_keys_daily = fetch_existing_keys_recent(ws_daily, days_back=14) if ws_daily else set()

            rows_15m, rows_daily = [], []
            logged_at = fmt_dt(now_ist())

            # Build kwargs supported by strategy
            base_kwargs = dict(
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

            if _HAS_ENGINE and engine_arg is not None:
                base_kwargs["engine"] = engine_arg
            if _HAS_BOX_WIDTH:
                base_kwargs["box_width_pct_max"] = float(st.session_state.box_width_pct_max) / 100.0
            if _HAS_DI:
                base_kwargs["require_di_confirmation"] = bool(st.session_state.require_di_confirmation)
            if _HAS_RSI_FLOOR:
                base_kwargs["rsi_floor_short"] = float(st.session_state.rsi_floor_short)
                base_kwargs["rsi_ceiling_long"] = float(st.session_state.rsi_ceiling_long)

            progress = st.progress(0, text="Scanning NIFTY50…")
            total = len(stock_list)

            # 15M scan
            for i, symbol in enumerate(stock_list, start=1):
                try:
                    df = get_ohlc_15min(symbol, days_back=15)
                    if df is None or df.shape[0] < 220:
                        progress.progress(i / total, text=f"{symbol}: insufficient 15M candles")
                        continue

                    df_prepped = prepare_trend_squeeze(df, **base_kwargs)

                    recent = df_prepped.tail(int(st.session_state.catchup_candles_15m)).copy()
                    recent = recent[recent[use_setup_col] != ""]

                    for candle_ts, r in recent.iterrows():
                        signal_time = ts_to_sheet_str(candle_ts, "15M")  # IST-correct
                        setup = r[use_setup_col]
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

                    progress.progress(i / total, text=f"{symbol}: scanned")
                except Exception as e:
                    if show_debug:
                        st.warning(f"{symbol} 15M error: {e}")
                    progress.progress(i / total, text=f"{symbol}: error")
                    continue

            # Daily scan (ALL 50)
            lookback_days_daily = 252
            for symbol in stock_list:
                try:
                    df_daily = get_ohlc_daily(symbol, lookback_days=lookback_days_daily)
                    if df_daily is None or df_daily.shape[0] < 100:
                        continue

                    daily_kwargs = dict(base_kwargs)
                    daily_kwargs["bbw_abs_threshold"] = float(bbw_abs_threshold) * 1.2  # slightly looser for daily

                    df_daily_prepped = prepare_trend_squeeze(df_daily, **daily_kwargs)

                    recent_daily = df_daily_prepped.tail(5).copy()
                    recent_daily = recent_daily[recent_daily[use_setup_col] != ""]

                    for candle_ts, r in recent_daily.iterrows():
                        signal_time = ts_to_sheet_str(candle_ts, "D")
                        setup = r[use_setup_col]
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

            appended_15m, _ = append_signals(ws_15m, rows_15m, show_debug) if ws_15m else (0, None)
            appended_daily, _ = append_signals(ws_daily, rows_daily, show_debug) if ws_daily else (0, None)

            progress.empty()
            return appended_15m, appended_daily, "OK"

        run_now = st.button("Run Scan Now", type="primary", key="btn_run_scan_now")
        if st.session_state.auto_scan or run_now:
            a15, ad, msg = run_scan_once()
            st.success(f"Logged {a15} new 15M + {ad} new Daily signals.")
            st.cache_data.clear()
            st.rerun()


# ==================== BACKTEST TAB ====================
with tab_backtest:
    st.subheader("📜 Backtest (15M) — signals only")
    st.caption("This backtest is a quick signal audit, not a full P&L engine (yet).")

    # Avoid duplicate element IDs by assigning keys everywhere
    bt1, bt2, bt3 = st.columns(3)
    with bt1:
        bt_days = st.slider("Days back", 10, 90, 30, 5, key="bt_days")
    with bt2:
        bt_symbols = st.slider("Symbols", 5, 50, 10, 1, key="bt_symbols")
    with bt3:
        bt_profile = st.selectbox(
            "Profile",
            ["Normal", "Conservative", "Aggressive"],
            index=0,
            key="bt_profile",  # ✅ FIXED duplicate id problem
        )

    if bt_profile == "Conservative":
        bt_params = {"bbw_abs_threshold": 0.035, "bbw_pct_threshold": 0.25, "adx_threshold": 25.0, "rsi_bull": 60.0, "rsi_bear": 40.0}
    elif bt_profile == "Aggressive":
        bt_params = {"bbw_abs_threshold": 0.065, "bbw_pct_threshold": 0.45, "adx_threshold": 18.0, "rsi_bull": 52.0, "rsi_bear": 48.0}
    else:
        bt_params = {"bbw_abs_threshold": 0.05, "bbw_pct_threshold": 0.35, "adx_threshold": 20.0, "rsi_bull": 55.0, "rsi_bear": 45.0}

    def run_backtest_15m(symbols, days_back=30, params=None):
        if params is None:
            params = bt_params

        trades = []
        total_bars = 0

        # Build kwargs (use current settings for breakout layer)
        kwargs = dict(
            bbw_abs_threshold=params["bbw_abs_threshold"],
            bbw_pct_threshold=params["bbw_pct_threshold"],
            adx_threshold=params["adx_threshold"],
            rsi_bull=params["rsi_bull"],
            rsi_bear=params["rsi_bear"],
            rolling_window=20,
            breakout_lookback=int(st.session_state.breakout_lookback),
            require_bbw_expansion=bool(st.session_state.require_bbw_expansion),
            require_volume_spike=bool(st.session_state.require_volume_spike),
            volume_spike_mult=float(st.session_state.volume_spike_mult),
        )
        if _HAS_ENGINE:
            if st.session_state.engine_ui.startswith("Box"):
                kwargs["engine"] = "box"
            elif st.session_state.engine_ui.startswith("Squeeze"):
                kwargs["engine"] = "squeeze"
            else:
                kwargs["engine"] = "hybrid"
        if _HAS_BOX_WIDTH:
            kwargs["box_width_pct_max"] = float(st.session_state.box_width_pct_max) / 100.0
        if _HAS_DI:
            kwargs["require_di_confirmation"] = bool(st.session_state.require_di_confirmation)
        if _HAS_RSI_FLOOR:
            kwargs["rsi_floor_short"] = float(st.session_state.rsi_floor_short)
            kwargs["rsi_ceiling_long"] = float(st.session_state.rsi_ceiling_long)

        for symbol in symbols[:bt_symbols]:
            try:
                df = get_ohlc_15min(symbol, days_back=days_back)
                if df is None or len(df) < 100:
                    continue

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

    if st.button("🚀 Run Backtest", type="primary", key="btn_run_backtest"):
        with st.spinner("Running backtest on FYERS data..."):
            bt_results, total_bars = run_backtest_15m(stock_list[:bt_symbols], bt_days, bt_params)

        if not bt_results.empty:
            st.success(f"Backtest complete: {len(bt_results)} signals across {total_bars:,} bars")
            q_counts = bt_results["Quality"].value_counts()
            m1, m2, m3 = st.columns(3)
            m1.metric("A", int(q_counts.get("A", 0)))
            m2.metric("B", int(q_counts.get("B", 0)))
            m3.metric("C", int(q_counts.get("C", 0)))
            st.dataframe(bt_results, use_container_width=True, hide_index=True)
        else:
            st.warning("No signals found for the selected period.")


st.caption("✅ FYERS-powered • IST-correct timestamps • Signals-first UI • Daily scans all 50 • A-first display (B/C optional)")
