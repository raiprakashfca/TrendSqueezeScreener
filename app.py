from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import json
import inspect
import string

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

# Direct Sheet IDs
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
def now_ist_aware() -> datetime:
    return datetime.now(IST)

def now_ist_naive() -> datetime:
    return now_ist_aware().replace(tzinfo=None)

def market_open_now() -> bool:
    n = now_ist_aware()
    return is_trading_day(n.date()) and MARKET_OPEN <= n.time() <= MARKET_CLOSE

def fmt_dt(dt: datetime) -> str:
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

def fmt_ts(ts) -> str:
    t = pd.Timestamp(ts)
    # candles now come IST-naive from utils; keep consistent
    return t.strftime("%d-%b-%Y %H:%M")

def _col_letter(n: int) -> str:
    # 1->A, 2->B ...
    letters = ""
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters

# -------------------- Google Sheets Auth --------------------
@st.cache_resource(show_spinner=False)
def _gspread_client_cached(sa_json_str: str):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    sa = json.loads(sa_json_str)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa, scope)
    return gspread.authorize(creds)

def get_gspread_client():
    try:
        raw = st.secrets["gcp_service_account"]
    except Exception:
        return None, "Missing st.secrets['gcp_service_account']"

    try:
        sa = json.loads(raw) if isinstance(raw, str) else dict(raw)
        sa_str = json.dumps(sa, sort_keys=True)
    except Exception as e:
        return None, f"Invalid gcp_service_account format: {e}"

    try:
        return _gspread_client_cached(sa_str), None
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
    changed = False

    if "fyers_access_token" not in header:
        header.append("fyers_access_token")
        changed = True
    if "fyers_token_updated_at" not in header:
        header.append("fyers_token_updated_at")
        changed = True
    if changed:
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

# -------------------- OHLC caching (biggest performance win) --------------------
@st.cache_data(ttl=290, show_spinner=False)  # ~matches 5-min refresh
def ohlc_15m_cached(symbol: str, days_back: int):
    return get_ohlc_15min(symbol, days_back=days_back)

@st.cache_data(ttl=6 * 3600, show_spinner=False)  # daily doesn’t change often
def ohlc_daily_cached(symbol: str, lookback_days: int):
    return get_ohlc_daily(symbol, lookback_days=lookback_days)

@st.cache_data(ttl=120, show_spinner=False)
def fyers_smoke_test() -> tuple[bool, str]:
    try:
        df = ohlc_15m_cached("RELIANCE", days_back=2)
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

    # Ensure header EXACT match (avoid partial mismatch bugs)
    try:
        header = ws.row_values(1)
        if header != SIGNAL_COLUMNS:
            ws.update("A1", [SIGNAL_COLUMNS])
    except Exception as e:
        return None, f"Failed to ensure headers: {e}"

    return ws, None

def fetch_existing_keys_recent(ws, days_back: int = 3) -> set[str]:
    """
    Fast de-dupe: reads only columns A:B (key, signal_time).
    """
    try:
        rows = ws.get("A:B")
    except Exception:
        return set()

    if not rows or len(rows) < 2:
        return set()

    data = rows[1:]  # skip header
    keys = []
    times = []
    for r in data:
        if not r:
            continue
        k = (r[0] if len(r) >= 1 else "") or ""
        t = (r[1] if len(r) >= 2 else "") or ""
        if k:
            keys.append(str(k))
            times.append(str(t))

    if not keys:
        return set()

    df = pd.DataFrame({"key": keys, "signal_time": times})
    # parse both "YYYY-MM-DD HH:MM" and "YYYY-MM-DD"
    df["dt"] = pd.to_datetime(df["signal_time"], errors="coerce")
    df = df[df["dt"].notna()].copy()

    cutoff = now_ist_naive() - timedelta(days=days_back)
    df = df[df["dt"] >= cutoff]
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
    now = now_ist_aware()
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

def load_recent_signals(ws, base_hours: int = 24, timeframe: str = "15M", audit_mode: bool = False, max_rows: int = 1500) -> pd.DataFrame:
    """
    Faster loader: pulls A:P (used range) and only keeps the last `max_rows`.
    """
    last_col = _col_letter(len(SIGNAL_COLUMNS))  # A..P for 16
    rng = f"A:{last_col}"

    try:
        values = ws.get(rng)
    except Exception:
        return pd.DataFrame()

    if not values or len(values) < 2:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]
    if len(rows) > max_rows:
        rows = rows[-max_rows:]

    df = pd.DataFrame(rows, columns=header[: len(rows[0])] if rows else header)
    # ensure expected cols exist
    for c in SIGNAL_COLUMNS:
        if c not in df.columns:
            df[c] = None

    df["signal_time_dt"] = pd.to_datetime(df["signal_time"], errors="coerce")
    df = df[df["signal_time_dt"].notna()].copy()

    smart_hours = get_market_aware_window(base_hours, timeframe)
    cutoff = now_ist_naive() - timedelta(hours=smart_hours)
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
    if not audit_mode:
        df = df.drop_duplicates(subset=["Symbol", "Timeframe"], keep="first")

    cols = ["Timestamp", "Symbol", "Timeframe", "Quality", "Bias", "Setup", "LTP", "BBW", "BBW %Rank", "RSI", "ADX", "Trend"]
    return df[cols]

# -------------------- Sidebar: Token Manager UI --------------------
with st.sidebar:
    st.subheader("🔐 FYERS Token Status")

    ws_fyers, ws_fyers_err = None, None
    try:
        ws_fyers, ws_fyers_err = get_fyers_token_worksheet()
    except Exception as e:
        ws_fyers, ws_fyers_err = None, str(e)

    app_id_sheet = secret_id_sheet = token_sheet = updated_at_sheet = ""

    if ws_fyers_err:
        st.error("FYERS token sheet ❌")
        st.caption(ws_fyers_err)
    else:
        app_id_sheet, secret_id_sheet, token_sheet, updated_at_sheet = read_fyers_row(ws_fyers)
        if token_sheet:
            st.success("Token found in Google Sheet ✅")
        else:
            st.warning("No access token in sheet yet ❗")

        if updated_at_sheet:
            st.caption(f"Last updated: {updated_at_sheet}")

    st.markdown("---")
    st.subheader("🔑 Generate / Update FYERS Access Token")
    redirect_uri = st.secrets.get("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri/index.html")

    if not ws_fyers_err and ws_fyers and app_id_sheet and secret_id_sheet:
        try:
            login_url = build_fyers_login_url(app_id_sheet, secret_id_sheet, redirect_uri)
            st.link_button("1) Open FYERS Login", login_url)
            st.caption("After login, copy `auth_code` from redirected URL and paste below.")
        except Exception as e:
            st.error(f"Could not generate login URL: {e}")
            login_url = None

        auth_code = st.text_input("2) Paste auth_code here", value="", help="From redirected URL after FYERS login.")

        if st.button("3) Exchange & Save Token", type="primary"):
            if not auth_code.strip():
                st.error("Please paste the auth_code first.")
            else:
                try:
                    token = exchange_auth_code_for_token(app_id_sheet, secret_id_sheet, redirect_uri, auth_code.strip())
                    ts = now_ist_naive().strftime("%Y-%m-%d %H:%M:%S")
                    update_fyers_token(ws_fyers, token, ts)
                    st.success("Saved access token + timestamp to the FYERS sheet ✅")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Token exchange failed: {e}")
    else:
        st.info("Add fyers_app_id + fyers_secret_id in FYERS token sheet row 2 to enable login flow.")

# -------------------- FYERS INIT --------------------
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
    fyers_health_msg = "No token yet. Use the login panel above."

# -------------------- Main UI Header --------------------
st.title("📉 Trend Squeeze Screener (15M + Daily) - Market Aware")
n = now_ist_aware()
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
    "TMPV","TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HEROMOTOCO","SHREECEM"
]
stock_list = fallback_nifty50
st.caption(f"Universe: NIFTY50 ({len(stock_list)} symbols).")

live_tab, backtest_tab = st.tabs(["📺 Live Screener (15M + Daily)", "📜 Backtest"])

# ---------- Signature detection ----------
_STRAT_SIG = inspect.signature(prepare_trend_squeeze)
_HAS_ENGINE = "engine" in _STRAT_SIG.parameters
_HAS_BOX_WIDTH = "box_width_pct_max" in _STRAT_SIG.parameters
_HAS_RSI_FLOOR = "rsi_floor_short" in _STRAT_SIG.parameters
_HAS_DI = "require_di_confirmation" in _STRAT_SIG.parameters

with live_tab:
    st.subheader("📺 Live Screener (15M + Daily • Market-Aware)")

    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ Screener Settings")

        show_debug = st.checkbox("Show debug messages", value=False)

        audit_mode = st.checkbox(
            "Audit mode (show all signals, no de-dup)",
            value=False,
            help="Useful to verify whether the app logged something but UI hid it due to de-dup."
        )

        st.markdown("---")
        st.subheader("📊 Live coherence")
        catchup_candles_15m = st.slider("15M Catch-up window (candles)", 8, 64, 32, 4)
        retention_hours = st.slider("Base retention (calendar hours)", 6, 48, 24, 6)

        st.markdown("---")
        st.subheader("🧨 Breakout confirmation")
        signal_mode = st.radio(
            "Signal type",
            ["✅ Breakout Confirmed (trade)", "🟡 Setup Forming (watchlist)"],
            index=0
        )
        breakout_lookback = st.slider("Consolidation range lookback (candles)", 10, 60, 20, 5)
        require_bbw_expansion = st.checkbox("Require BBW expansion on breakout", value=True)
        require_volume_spike = st.checkbox("Require volume spike (if volume is available)", value=False)
        volume_spike_mult = st.slider("Volume spike multiplier (vs 20 SMA)", 1.0, 3.0, 1.5, 0.1)

        st.markdown("---")
        st.subheader("🧩 Engine (optional upgrade)")
        if _HAS_ENGINE:
            engine_ui = st.selectbox(
                "Detection engine",
                ["Hybrid (recommended)", "Box Breakout (range)", "Squeeze Breakout (TTM)"],
                index=0
            )
            if engine_ui.startswith("Box"):
                engine_arg = "box"
            elif engine_ui.startswith("Squeeze"):
                engine_arg = "squeeze"
            else:
                engine_arg = "hybrid"
        else:
            engine_arg = None
            st.info("Your strategy.py does not support Engine controls. (App will still work.)")

        if _HAS_BOX_WIDTH:
            box_width_pct_max = st.slider("Max box width (%)", 0.3, 3.0, 1.2, 0.1) / 100.0
        else:
            box_width_pct_max = None

        if _HAS_DI:
            require_di_confirmation = st.checkbox("Require DI confirmation (+DI/-DI)", value=True)
        else:
            require_di_confirmation = None

        if _HAS_RSI_FLOOR:
            st.markdown("---")
            st.subheader("🚫 Anti-chase filters")
            rsi_floor_short = st.slider("Block SHORT if RSI below", 10.0, 45.0, 30.0, 1.0)
            rsi_ceiling_long = st.slider("Block LONG if RSI above", 55.0, 90.0, 70.0, 1.0)
        else:
            rsi_floor_short = None
            rsi_ceiling_long = None

        st.markdown("---")
        param_profile = st.selectbox("Parameter Profile", ["Normal", "Conservative", "Aggressive"])

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
        bbw_abs_threshold = st.slider("BBW absolute threshold (max BBW)", 0.01, 0.20, bbw_abs_default, 0.005)
    with c2:
        bbw_pct_threshold = st.slider("BBW percentile threshold", 0.10, 0.80, bbw_pct_default, 0.05)

    c3, c4, c5 = st.columns(3)
    with c3:
        adx_threshold = st.slider("ADX threshold", 15.0, 35.0, adx_default, 1.0)
    with c4:
        rsi_bull = st.slider("RSI bull threshold", 50.0, 70.0, rsi_bull_default, 1.0)
    with c5:
        rsi_bear = st.slider("RSI bear threshold", 30.0, 50.0, rsi_bear_default, 1.0)

    params_hash = params_to_hash(bbw_abs_threshold, bbw_pct_threshold, adx_threshold, rsi_bull, rsi_bear)
    use_setup_col = "setup" if signal_mode.startswith("✅") else "setup_forming"

    smart_retention_15m = get_market_aware_window(retention_hours)
    smart_retention_daily = get_market_aware_window(retention_hours, "Daily")
    st.caption(f"🧠 Showing last ~{smart_retention_15m}h trading time (15M) | ~{smart_retention_daily/24:.0f} trading days (Daily)")

    # --- Sheets status (FIXED: no ternary Streamlit calls) ---
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

    existing_keys_15m = fetch_existing_keys_recent(ws_15m, days_back=3) if ws_15m and not ws_15m_err else set()
    existing_keys_daily = fetch_existing_keys_recent(ws_daily, days_back=14) if ws_daily and not ws_daily_err else set()

    rows_15m, rows_daily = [], []
    logged_at = fmt_dt(now_ist_naive())

    # ----------- scan stats -----------
    stats = {
        "15m_scanned": 0,
        "15m_skipped_short": 0,
        "15m_errors": 0,
        "daily_scanned": 0,
        "daily_skipped_short": 0,
        "daily_errors": 0,
        "daily_throttled": False,
    }

    # -------------------- 15M scan (always) --------------------
    prog = st.progress(0, text="Scanning 15M…")
    for i, symbol in enumerate(stock_list):
        stats["15m_scanned"] += 1
        prog.progress(int((i + 1) / len(stock_list) * 100), text=f"Scanning 15M… {symbol}")

        try:
            df = ohlc_15m_cached(symbol, days_back=15)
            if df is None or df.shape[0] < 220:
                stats["15m_skipped_short"] += 1
                continue

            kwargs = dict(
                bbw_abs_threshold=bbw_abs_threshold,
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

            if _HAS_ENGINE and engine_arg is not None:
                kwargs["engine"] = engine_arg
            if _HAS_BOX_WIDTH and box_width_pct_max is not None:
                kwargs["box_width_pct_max"] = box_width_pct_max
            if _HAS_DI and require_di_confirmation is not None:
                kwargs["require_di_confirmation"] = require_di_confirmation
            if _HAS_RSI_FLOOR and rsi_floor_short is not None and rsi_ceiling_long is not None:
                kwargs["rsi_floor_short"] = rsi_floor_short
                kwargs["rsi_ceiling_long"] = rsi_ceiling_long

            df_prepped = prepare_trend_squeeze(df, **kwargs)

            recent = df_prepped.tail(int(catchup_candles_15m)).copy()
            recent = recent[recent[use_setup_col] != ""]

            for candle_ts, r in recent.iterrows():
                # candles already IST-naive
                signal_time = pd.Timestamp(candle_ts).floor("15min").strftime("%Y-%m-%d %H:%M")
                setup = r[use_setup_col]
                trend = r.get("trend", "")
                ltp = float(r.get("close", np.nan))
                bbw = float(r.get("bbw", np.nan))
                bbw_rank = r.get("bbw_pct_rank", np.nan)
                rsi_val = float(r.get("rsi", np.nan))
                adx_val = float(r.get("adx", np.nan))

                bias = "LONG" if str(setup).startswith("Bullish") else "SHORT"

                # keep key stable per candle+setup+bias (parameter changes go into params_hash column)
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
            stats["15m_errors"] += 1
            if show_debug:
                st.warning(f"{symbol} 15M error: {e}")
            continue

    prog.empty()

    # -------------------- Daily scan (ALL 50) with throttle --------------------
    # Improvement: avoid re-running heavy Daily compute every 5 minutes.
    # We still INCLUDE all 50, but we compute at most once per hour (unless first run).
    throttle_minutes = 60
    last_daily_run = st.session_state.get("last_daily_scan_ts", None)
    now_naive = now_ist_naive()

    should_run_daily = True
    if last_daily_run is not None:
        if isinstance(last_daily_run, str):
            last_daily_run = pd.to_datetime(last_daily_run)
        if now_naive - last_daily_run < timedelta(minutes=throttle_minutes):
            should_run_daily = False

    if not should_run_daily:
        stats["daily_throttled"] = True
    else:
        st.session_state["last_daily_scan_ts"] = now_naive

        lookback_days_daily = 252
        prog2 = st.progress(0, text="Scanning Daily…")
        for i, symbol in enumerate(stock_list):  # ✅ all 50
            stats["daily_scanned"] += 1
            prog2.progress(int((i + 1) / len(stock_list) * 100), text=f"Scanning Daily… {symbol}")

            try:
                df_daily = ohlc_daily_cached(symbol, lookback_days=lookback_days_daily)
                if df_daily is None or df_daily.shape[0] < 100:
                    stats["daily_skipped_short"] += 1
                    continue

                kwargs = dict(
                    bbw_abs_threshold=bbw_abs_threshold * 1.2,
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

                if _HAS_ENGINE and engine_arg is not None:
                    kwargs["engine"] = engine_arg
                if _HAS_BOX_WIDTH and box_width_pct_max is not None:
                    kwargs["box_width_pct_max"] = box_width_pct_max
                if _HAS_DI and require_di_confirmation is not None:
                    kwargs["require_di_confirmation"] = require_di_confirmation
                if _HAS_RSI_FLOOR and rsi_floor_short is not None and rsi_ceiling_long is not None:
                    kwargs["rsi_floor_short"] = rsi_floor_short
                    kwargs["rsi_ceiling_long"] = rsi_ceiling_long

                df_daily_prepped = prepare_trend_squeeze(df_daily, **kwargs)

                recent_daily = df_daily_prepped.tail(5).copy()
                recent_daily = recent_daily[recent_daily[use_setup_col] != ""]

                for candle_ts, r in recent_daily.iterrows():
                    signal_time = pd.Timestamp(candle_ts).floor("1D").strftime("%Y-%m-%d")
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
                stats["daily_errors"] += 1
                if show_debug:
                    st.warning(f"{symbol} Daily error: {e}")
                continue

        prog2.empty()

    # -------------------- Write to sheets --------------------
    appended_15m = appended_daily = 0
    if ws_15m and not ws_15m_err:
        appended_15m, _ = append_signals(ws_15m, rows_15m, show_debug)
    if ws_daily and not ws_daily_err and not stats["daily_throttled"]:
        appended_daily, _ = append_signals(ws_daily, rows_daily, show_debug)

    # -------------------- Scan stats panel --------------------
    with st.expander("🧾 Scan Health (what actually happened)", expanded=False):
        cA, cB, cC = st.columns(3)
        cA.metric("15M scanned", stats["15m_scanned"])
        cB.metric("15M skipped (short data)", stats["15m_skipped_short"])
        cC.metric("15M errors", stats["15m_errors"])

        cD, cE, cF = st.columns(3)
        cD.metric("Daily scanned", stats["daily_scanned"])
        cE.metric("Daily skipped (short data)", stats["daily_skipped_short"])
        cF.metric("Daily errors", stats["daily_errors"])

        if stats["daily_throttled"]:
            st.info("Daily scan throttled (runs at most once per hour). This is intentional to avoid heavy re-compute every 5 minutes.")

    st.caption(f"Logged **{appended_15m}** new 15M + **{appended_daily}** Daily signals.")

    # -------------------- Display recent signals --------------------
    df_recent_15m = pd.DataFrame() if not ws_15m else load_recent_signals(ws_15m, retention_hours, "15M", audit_mode=audit_mode)
    df_recent_daily = pd.DataFrame() if not ws_daily else load_recent_signals(ws_daily, retention_hours, "Daily", audit_mode=audit_mode)

def _quality_filter_ui(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Default: show only Quality A.
    B/C are hidden behind an expander + multiselect.
    """
    if df.empty or "Quality" not in df.columns:
        return df

    # Default view
    default_qualities = ["A"]

    with st.expander(f"🔎 {label}: show/hide Quality levels", expanded=False):
        qualities = ["A", "B", "C"]
        show_q = st.multiselect(
            "Show qualities",
            options=qualities,
            default=default_qualities,
            help="Default is only A. Add B/C if you want more (and noisier) signals."
        )

        # Optional: quick toggle
        show_all = st.checkbox("Show all qualities (A+B+C)", value=False)
        if show_all:
            show_q = qualities

    # Apply filter (if user unselects everything, show nothing)
    return df[df["Quality"].isin(show_q)].copy()


if not df_recent_15m.empty or not df_recent_daily.empty:
    st.markdown("### 🧠 Recent Signals (Market-Aware)")

    # ---- 15M ----
    if not df_recent_15m.empty:
        st.subheader("15M Signals")
        df_15m_filtered = _quality_filter_ui(df_recent_15m, "15M Signals")
        if not df_15m_filtered.empty:
            # A-first ordering (and then newest first)
            df_15m_filtered["__q_rank"] = df_15m_filtered["Quality"].map({"A": 0, "B": 1, "C": 2}).fillna(9)
            df_15m_filtered = df_15m_filtered.sort_values(["__q_rank", "Timestamp"], ascending=[True, False]).drop(columns="__q_rank")
            st.dataframe(df_15m_filtered, use_container_width=True)
        else:
            st.info("No signals match the selected Quality filter for 15M.")

    # ---- Daily ----
    if not df_recent_daily.empty:
        st.subheader("Daily Signals")
        df_daily_filtered = _quality_filter_ui(df_recent_daily, "Daily Signals")
        if not df_daily_filtered.empty:
            df_daily_filtered["__q_rank"] = df_daily_filtered["Quality"].map({"A": 0, "B": 1, "C": 2}).fillna(9)
            df_daily_filtered = df_daily_filtered.sort_values(["__q_rank", "Timestamp"], ascending=[True, False]).drop(columns="__q_rank")
            st.dataframe(df_daily_filtered, use_container_width=True)
        else:
            st.info("No signals match the selected Quality filter for Daily.")
else:
    st.info("No signals in recent trading sessions.")


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
                df = ohlc_15m_cached(symbol, days_back=days_back)
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
        bt_symbols = st.slider("Symbols", 5, 20, 10, step=1)
    with col_bt3:
        bt_profile = st.selectbox("Profile", ["Normal", "Conservative", "Aggressive"])

    if bt_profile == "Conservative":
        bt_params = {"bbw_abs_threshold": 0.035, "bbw_pct_threshold": 0.25, "adx_threshold": 25.0, "rsi_bull": 60.0, "rsi_bear": 40.0}
    elif bt_profile == "Aggressive":
        bt_params = {"bbw_abs_threshold": 0.065, "bbw_pct_threshold": 0.45, "adx_threshold": 18.0, "rsi_bull": 52.0, "rsi_bear": 48.0}
    else:
        bt_params = {"bbw_abs_threshold": 0.05, "bbw_pct_threshold": 0.35, "adx_threshold": 20.0, "rsi_bull": 55.0, "rsi_bear": 45.0}

    if st.button("🚀 Run Backtest", type="primary"):
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
        else:
            st.warning("No signals found in the selected backtest period.")

st.markdown("---")
st.caption("✅ FYERS-powered: real-time data + dual timeframe + market-aware signals + backtest.")

