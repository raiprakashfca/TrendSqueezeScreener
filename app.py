
# app.py
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import json
import inspect
from typing import Any, Dict, List, Set
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
from utils.index_utils import fetch_nifty50_symbols


# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")
st_autorefresh(interval=300000, key="auto_refresh")  # 5 min refresh

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

SIGNAL_SHEET_15M = "TrendSqueezeSignals_15M"
SIGNAL_SHEET_DAILY = "TrendSqueezeSignals_Daily"

# Direct Sheet IDs (signals logs)
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


# =========================
# Time helpers (IST)
# =========================
def now_ist() -> datetime:
    return datetime.now(IST)


def market_open_now() -> bool:
    n = now_ist()
    return is_trading_day(n.date()) and MARKET_OPEN <= n.time() <= MARKET_CLOSE


def fmt_dt(dt: datetime) -> str:
    # store as tz-naive string, but it is IST context
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def to_ist_sheet_naive(ts) -> pd.Timestamp:
    """
    Sheet timestamps must be IST-naive and must NOT be shifted if tz-naive.
    - tz-aware -> convert to IST-naive
    - tz-naive -> assume already IST-naive
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(None)
    return t.tz_convert("Asia/Kolkata").tz_localize(None)


def ts_to_sheet_str(ts, freq: str = "15min") -> str:
    t = to_ist_sheet_naive(pd.Timestamp(ts)).floor(freq)
    return t.strftime("%Y-%m-%d %H:%M")


def normalize_ohlc_index_to_ist(df: pd.DataFrame) -> pd.DataFrame:
    """
    FYERS candles arrive as epoch seconds -> your utils often makes tz-naive timestamps
    that represent UTC. Convert OHLC index to IST-naive consistently.
    """
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    idx = df.index
    if getattr(idx, "tz", None) is None:
        # tz-naive -> treat as UTC and convert to IST
        try:
            new_idx = idx.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
            out = df.copy()
            out.index = new_idx
            return out
        except Exception:
            return df
    else:
        # tz-aware -> convert to IST-naive
        try:
            out = df.copy()
            out.index = idx.tz_convert("Asia/Kolkata").tz_localize(None)
            return out
        except Exception:
            return df


def last_closed_15m_candle_ist() -> pd.Timestamp:
    """
    Prevent 'signals from the future':
    Only allow signals on FULLY CLOSED 15m candles.
    """
    return pd.Timestamp(now_ist()).floor("15min") - pd.Timedelta(minutes=15)


# =========================
# Google Sheets Auth
# =========================
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


# =========================
# FYERS Token Sheet Manager
# =========================
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

    # Ensure headers exist
    try:
        header = ws.row_values(1)
        if not header:
            ws.update(values=[FYERS_TOKEN_HEADERS], range_name="A1")
        else:
            header_l = [h.strip() for h in header]
            changed = False
            for h in FYERS_TOKEN_HEADERS:
                if h not in header_l:
                    header_l.append(h)
                    changed = True
            if changed:
                ws.update(values=[header_l], range_name="A1")
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
        app_id = (row.get("fyers_app_id") or "").strip()
        secret_id = (row.get("fyers_secret_id") or "").strip()
        token = (row.get("fyers_access_token") or "").strip()
        updated_at = (row.get("fyers_token_updated_at") or "").strip()
        return app_id, secret_id, token, updated_at

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
        ws.update(values=[header], range_name="A1")
    if "fyers_token_updated_at" not in header:
        header.append("fyers_token_updated_at")
        ws.update(values=[header], range_name="A1")

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
        df = normalize_ohlc_index_to_ist(df)
        if df is None or len(df) < 10:
            return False, "OHLC returned empty/too short"
        return True, "OK"
    except Exception as e:
        return False, str(e)[:200]


# =========================
# Signals sheets (logging)
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
            if len(header) < len(SIGNAL_COLUMNS):
                ws.update(values=[SIGNAL_COLUMNS], range_name="A1")
    except Exception as e:
        return None, f"Failed to ensure headers: {e}"

    return ws, None


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
    df["signal_time_dt"] = df["signal_time_dt"].apply(to_ist_sheet_naive)

    cutoff = now_ist().replace(tzinfo=None) - timedelta(days=days_back)
    df = df[df["signal_time_dt"] >= cutoff]
    return set(df["key"].astype(str).tolist())


def append_signals_batch(ws, rows: list[list], show_debug: bool = False) -> tuple[int, str | None]:
    """
    Optimized: Appends all rows in a single API call.
    """
    if not rows:
        return 0, None
    try:
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        return len(rows), None
    except APIError as e:
        return (0, f"Sheets APIError: {e}") if show_debug else (0, "Sheets write failed (APIError).")
    except Exception as e:
        return (0, f"Sheets write failed: {e}") if show_debug else (0, "Sheets write failed.")


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
    df["signal_time_dt"] = df["signal_time_dt"].apply(to_ist_sheet_naive)

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
    if not audit_mode:
        df = df.drop_duplicates(subset=["Symbol", "Timeframe"], keep="first")

    cols = ["Timestamp", "Symbol", "Timeframe", "Quality", "Bias", "Setup", "LTP", "BBW", "BBW %Rank", "RSI", "ADX", "Trend"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


# =========================
# One-time repair helpers (safe + optional)
# =========================
def _col_to_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _looks_non_ist_time(t: pd.Timestamp, timeframe: str) -> bool:
    if pd.isna(t):
        return False
    hh = int(t.hour)
    # Out of normal India market window => likely UTC-naive
    if timeframe.upper().startswith("15"):
        return hh < 8 or hh > 17
    return hh < 8 or hh > 17


def _assume_utc_to_ist_naive(t: pd.Timestamp) -> pd.Timestamp:
    return t.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)


def repair_sheet_timestamps(ws, timeframe: str = "15M", dry_run: bool = True, max_rows: int = 5000) -> Dict[str, Any]:
    summary = {"timeframe": timeframe, "rows_scanned": 0, "rows_changed": 0, "cells_changed": 0, "dry_run": dry_run}

    values = ws.get_all_values()
    if not values or len(values) < 2:
        return summary

    header = [h.strip() for h in values[0]]
    if "signal_time" not in header:
        return {**summary, "error": "signal_time column not found"}

    col_signal = header.index("signal_time") + 1
    col_logged = header.index("logged_at") + 1 if "logged_at" in header else None

    updates = []
    data_rows = values[1 : 1 + max_rows]

    for i, row in enumerate(data_rows, start=2):
        summary["rows_scanned"] += 1

        old_signal = row[col_signal - 1].strip() if col_signal - 1 < len(row) else ""
        new_signal = None
        if old_signal:
            ts = pd.to_datetime(old_signal, errors="coerce")
            if pd.notna(ts):
                ts = pd.Timestamp(ts).tz_localize(None)
                if _looks_non_ist_time(ts, timeframe):
                    fixed = _assume_utc_to_ist_naive(ts)
                    new_signal = fixed.strftime("%Y-%m-%d %H:%M") if timeframe.upper().startswith("15") else fixed.strftime("%Y-%m-%d")

        new_logged = None
        if col_logged:
            old_logged = row[col_logged - 1].strip() if col_logged - 1 < len(row) else ""
            if old_logged:
                ts2 = pd.to_datetime(old_logged, errors="coerce")
                if pd.notna(ts2):
                    ts2 = pd.Timestamp(ts2).tz_localize(None)
                    if _looks_non_ist_time(ts2, timeframe):
                        fixed2 = _assume_utc_to_ist_naive(ts2)
                        new_logged = fixed2.strftime("%Y-%m-%d %H:%M:%S")

        changed = False
        if new_signal and new_signal != old_signal:
            changed = True
            summary["cells_changed"] += 1
            if not dry_run:
                updates.append({"range": f"{_col_to_letter(col_signal)}{i}", "values": [[new_signal]]})

        if col_logged and new_logged:
            old_logged = row[col_logged - 1].strip() if col_logged - 1 < len(row) else ""
            if new_logged != old_logged:
                changed = True
                summary["cells_changed"] += 1
                if not dry_run:
                    updates.append({"range": f"{_col_to_letter(col_logged)}{i}", "values": [[new_logged]]})

        if changed:
            summary["rows_changed"] += 1

    if (not dry_run) and updates:
        chunk_size = 300
        for j in range(0, len(updates), chunk_size):
            ws.batch_update(updates[j : j + chunk_size], value_input_option="USER_ENTERED")

    return summary


# =========================
# Strategy signature detection
# =========================
_STRAT_SIG = inspect.signature(prepare_trend_squeeze)
_HAS_ENGINE = "engine" in _STRAT_SIG.parameters
_HAS_BOX_WIDTH = "box_width_pct_max" in _STRAT_SIG.parameters
_HAS_RSI_FLOOR = "rsi_floor_short" in _STRAT_SIG.parameters
_HAS_DI = "require_di_confirmation" in _STRAT_SIG.parameters


# =========================
# Sidebar: Token Manager
# =========================
with st.sidebar:
    st.header("🔐 FYERS Login")

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
            st.warning("No access token yet ❗")
        if updated_at_sheet:
            st.caption(f"Last updated: {updated_at_sheet}")

    redirect_uri = st.secrets.get("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri/index.html")

    if ws_fyers_err:
        st.stop()

    if not app_id_sheet or not secret_id_sheet:
        st.error("Fill fyers_app_id and fyers_secret_id in FYERS token sheet row 2.")
        st.stop()

    try:
        login_url = build_fyers_login_url(app_id_sheet, secret_id_sheet, redirect_uri)
        st.link_button("1) Open FYERS Login", login_url)
        st.caption("Login → copy `auth_code` from redirected URL → paste below.")
    except Exception as e:
        st.error(f"Login URL failed: {e}")
        login_url = None

    auth_code = st.text_input("2) Paste auth_code", value="", key="auth_code_input")

    if st.button("3) Exchange & Save Token", type="primary", key="exchange_save_btn"):
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

    st.divider()
    st.subheader("⚙️ View settings")

    show_debug = st.checkbox("Show debug", value=False, key="show_debug")
    audit_mode = st.checkbox("Audit mode (no de-dup)", value=False, key="audit_mode")
    retention_hours = st.slider("Retention (hours)", 6, 48, 24, 6, key="retention_hours")

    with st.expander("🧠 Quality filter", expanded=False):
        quality_pref = st.radio(
            "Show quality",
            ["A only", "A + B", "A + B + C"],
            index=0,
            key="quality_pref",
            horizontal=True,
        )

    with st.expander("🧨 Breakout confirmation", expanded=False):
        signal_mode = st.radio(
            "Signal type",
            ["✅ Breakout Confirmed (trade)", "🟡 Setup Forming (watchlist)"],
            index=0,
            key="signal_mode",
        )
        breakout_lookback = st.slider("Consolidation lookback (candles)", 10, 60, 20, 5, key="breakout_lookback")
        require_bbw_expansion = st.checkbox("Require BBW expansion", value=True, key="require_bbw_expansion")
        require_volume_spike = st.checkbox("Require volume spike", value=False, key="require_volume_spike")
        volume_spike_mult = st.slider("Volume spike mult", 1.0, 3.0, 1.5, 0.1, key="volume_spike_mult")

    with st.expander("🧩 Engine controls", expanded=False):
        if _HAS_ENGINE:
            engine_ui = st.selectbox(
                "Engine",
                ["Hybrid (recommended)", "Box Breakout (range)", "Squeeze Breakout (TTM)"],
                index=0,
                key="engine_ui",
            )
        else:
            engine_ui = "Hybrid (recommended)"
            st.info("Engine controls not available in your strategy.py.")

        if _HAS_BOX_WIDTH:
            box_width_pct_max = st.slider("Max box width (%)", 0.3, 3.0, 1.2, 0.1, key="box_width") / 100.0

        if _HAS_DI:
            require_di_confirmation = st.checkbox("Require DI confirmation", value=True, key="di_confirm")

        if _HAS_RSI_FLOOR:
            rsi_floor_short = st.slider("Block SHORT if RSI below", 10.0, 45.0, 30.0, 1.0, key="rsi_floor_short")
            rsi_ceiling_long = st.slider("Block LONG if RSI above", 55.0, 90.0, 70.0, 1.0, key="rsi_ceiling_long")

    with st.expander("📈 Threshold profiles", expanded=False):
        param_profile = st.selectbox("Profile", ["Normal", "Conservative", "Aggressive"], index=0, key="param_profile")

        if param_profile == "Conservative":
            bbw_abs_default, bbw_pct_default = 0.035, 0.25
            adx_default, rsi_bull_default, rsi_bear_default = 25.0, 60.0, 40.0
        elif param_profile == "Aggressive":
            bbw_abs_default, bbw_pct_default = 0.065, 0.45
            adx_default, rsi_bull_default, rsi_bear_default = 18.0, 52.0, 48.0
        else:
            bbw_abs_default, bbw_pct_default = 0.05, 0.35
            adx_default, rsi_bull_default, rsi_bear_default = 20.0, 55.0, 45.0

        bbw_abs_threshold = st.slider("BBW abs max", 0.01, 0.20, bbw_abs_default, 0.005, key="bbw_abs_threshold")
        bbw_pct_threshold = st.slider("BBW pct rank max", 0.10, 0.80, bbw_pct_default, 0.05, key="bbw_pct_threshold")

        adx_threshold = st.slider("ADX min", 15.0, 35.0, adx_default, 1.0, key="adx_threshold")
        rsi_bull = st.slider("RSI bull min", 50.0, 70.0, rsi_bull_default, 1.0, key="rsi_bull")
        rsi_bear = st.slider("RSI bear max", 30.0, 50.0, rsi_bear_default, 1.0, key="rsi_bear")

    catchup_candles_15m = st.slider("15M catch-up candles", 8, 64, 32, 4, key="catchup_candles_15m")


# =========================
# FYERS INIT (from token sheet)
# =========================
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
    fyers_health_msg = "No token yet. Use sidebar login flow."


# =========================
# Main header
# =========================
st.title("📉 Trend Squeeze Screener")
n = now_ist()
mode_str = "🟢 LIVE MARKET" if market_open_now() else "🔵 OFF-MARKET"
st.caption(f"{mode_str} • Last refresh: {n.strftime('%d %b %Y • %H:%M:%S')} IST")

last_trading_day = get_last_trading_day(n.date())
st.caption(f"🗓️ Last trading day: {last_trading_day.strftime('%d %b %Y')}")

lc15 = last_closed_15m_candle_ist()
st.caption(f"⏱ Closed-candle guard: last closed 15M candle = {lc15.strftime('%d %b %Y %H:%M')} IST")

if fyers_ok and fyers_health_ok:
    st.success("✅ FYERS data fetch OK")
elif fyers_ok and not fyers_health_ok:
    st.warning(f"⚠️ FYERS session set but data fetch failing: {fyers_health_msg}")
else:
    st.error("🔴 FYERS not initialized (no valid token). Use the sidebar login flow.")
    st.stop()


# =========================
# Universe
# =========================
# Try to fetch fresh Nifty 50, otherwise fallback
fetched_symbols = fetch_nifty50_symbols()
stock_list = fetched_symbols if fetched_symbols else [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV",
    "BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH",
    "HDFCBANK","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM",
    "M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBIN","SBILIFE","SUNPHARMA","TATACONSUM",
    "TATAMOTORS","TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HEROMOTOCO","SHREECEM"
]
st.caption(f"Universe: NIFTY50 ({len(stock_list)} symbols).")

use_setup_col = "setup" if st.session_state.get("signal_mode", "✅").startswith("✅") else "setup_forming"


# =========================
# ✅ IMPORTANT: Define sheets BEFORE any UI uses them
# =========================
ws_15m, ws_15m_err = get_signals_worksheet(SIGNAL_SHEET_15M)
ws_daily, ws_daily_err = get_signals_worksheet(SIGNAL_SHEET_DAILY)

# Safe defaults to avoid NameError even if you later shuffle blocks
ws_15m = ws_15m if ws_15m is not None else None
ws_daily = ws_daily if ws_daily is not None else None
ws_15m_err = ws_15m_err if ws_15m_err is not None else None
ws_daily_err = ws_daily_err if ws_daily_err is not None else None


# =========================
# Top: recent signals
# =========================
top_row = st.columns([2, 2, 3])
with top_row[0]:
    st.write("**15M Sheet**")
    if ws_15m_err:
        st.error(ws_15m_err)
    else:
        st.success("Connected ✅")
with top_row[1]:
    st.write("**Daily Sheet**")
    if ws_daily_err:
        st.error(ws_daily_err)
    else:
        st.success("Connected ✅")
with top_row[2]:
    st.write("**Display**")
    st.caption("Quality filter + de-dup are in the sidebar.")

df_recent_15m = pd.DataFrame() if ws_15m_err or not ws_15m else load_recent_signals(
    ws_15m, st.session_state["retention_hours"], "15M", audit_mode=st.session_state["audit_mode"]
)
df_recent_daily = pd.DataFrame() if ws_daily_err or not ws_daily else load_recent_signals(
    ws_daily, st.session_state["retention_hours"], "Daily", audit_mode=st.session_state["audit_mode"]
)

st.subheader("🧠 Recent Signals (Market-Aware)")
q_pref = st.session_state.get("quality_pref", "A only")

def _filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Quality" not in df.columns:
        return df
    if q_pref == "A only":
        return df[df["Quality"] == "A"].copy()
    if q_pref == "A + B":
        return df[df["Quality"].isin(["A", "B"])].copy()
    return df

c1, c2 = st.columns(2)
with c1:
    st.markdown("### 15M")
    d = _filter_quality(df_recent_15m)
    st.dataframe(d, use_container_width=True) if d is not None and not d.empty else st.info("No signals in window.")
with c2:
    st.markdown("### Daily")
    d = _filter_quality(df_recent_daily)
    st.dataframe(d, use_container_width=True) if d is not None and not d.empty else st.info("No signals in window.")

st.divider()


# =========================
# One-time repair UI (NOW SAFE)
# =========================
with st.expander("🧹 One-time repair: convert non-IST timestamps in Sheets", expanded=False):
    st.warning("This edits existing rows in Google Sheets. Copy sheet first if you’re nervous 🙂")
    dry = st.checkbox("Dry run (report only, no writing)", value=True, key="repair_dry_run")

    can_run = (ws_15m is not None and not ws_15m_err) and (ws_daily is not None and not ws_daily_err)
    if not can_run:
        st.error("Sheets not ready. Fix sheet connectivity first (see the status boxes above).")
    else:
        if st.button("Run repair on 15M + Daily sheets", type="primary", key="run_repair"):
            with st.spinner("Repairing 15M sheet..."):
                r15 = repair_sheet_timestamps(ws_15m, timeframe="15M", dry_run=dry)
            with st.spinner("Repairing Daily sheet..."):
                rd = repair_sheet_timestamps(ws_daily, timeframe="Daily", dry_run=dry)

            st.success("Repair complete ✅")
            st.json({"15M": r15, "Daily": rd})

            if not dry:
                st.cache_data.clear()
                st.rerun()


# =========================
# Scan + logging (PARALLEL & BATCH OPTIMIZED)
# =========================

def process_scan_task(symbol: str, timeframe: str, params: dict, last_closed_ts: pd.Timestamp):
    """
    Worker function to fetch data and run strategy for a single symbol.
    Returns: None or a dict with result data.
    """
    try:
        # 1. Fetch Data
        if timeframe == "15M":
            # 15M needs catchup candles + lookback
            df = get_ohlc_15min(symbol, days_back=15)
        else:
            # Daily needs longer lookback (EMA200 etc)
            df = get_ohlc_daily(symbol, lookback_days=252)

        df = normalize_ohlc_index_to_ist(df)
        min_len = 220 if timeframe == "15M" else 100
        if df is None or df.shape[0] < min_len:
            return None

        # 2. Run Strategy
        df_prepped = prepare_trend_squeeze(df, **params)

        # 3. Filter for valid signals
        # Only care about the *latest* relevant candle(s)
        # 15M: Filter by closed-candle guard
        if timeframe == "15M":
            catchup = int(st.session_state.get("catchup_candles_15m", 32))
            recent = df_prepped.tail(catchup).copy()
            recent = recent[recent.index <= last_closed_ts]  # 🚫 no future/forming candle
        else:
            # Daily: last 7 days check
            recent = df_prepped.tail(7).copy()

        # Check for setups
        use_col = params.get("_use_setup_col", "setup")
        recent = recent[recent[use_col] != ""]
        
        if recent.empty:
            return None

        # Return list of signals found for this symbol
        signals = []
        for candle_ts, r in recent.iterrows():
            signals.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "candle_ts": candle_ts,
                "row_data": r
            })
        return signals

    except Exception:
        return None


with st.expander("🔎 Scan & Log (runs every refresh)", expanded=True):
    # Only run if Fyers is healthy
    if not fyers_ok or not fyers_health_ok:
        st.error("Cannot scan: FYERS not connected.")
    else:
        existing_keys_15m = fetch_existing_keys_recent(ws_15m, days_back=3) if ws_15m and not ws_15m_err else set()
        existing_keys_daily = fetch_existing_keys_recent(ws_daily, days_back=7) if ws_daily and not ws_daily_err else set()

        params_hash = params_to_hash(
            st.session_state["bbw_abs_threshold"],
            st.session_state["bbw_pct_threshold"],
            st.session_state["adx_threshold"],
            st.session_state["rsi_bull"],
            st.session_state["rsi_bear"],
        )

        logged_at = fmt_dt(now_ist())
        rows_15m_to_append = []
        rows_daily_to_append = []

        # Build Strategy Params
        def build_kwargs(is_daily: bool):
            kwargs = dict(
                bbw_abs_threshold=st.session_state["bbw_abs_threshold"] * (1.2 if is_daily else 1.0),
                bbw_pct_threshold=st.session_state["bbw_pct_threshold"],
                adx_threshold=st.session_state["adx_threshold"],
                rsi_bull=st.session_state["rsi_bull"],
                rsi_bear=st.session_state["rsi_bear"],
                rolling_window=20,
                breakout_lookback=st.session_state["breakout_lookback"],
                require_bbw_expansion=st.session_state["require_bbw_expansion"],
                require_volume_spike=st.session_state["require_volume_spike"],
                volume_spike_mult=st.session_state["volume_spike_mult"],
                _use_setup_col=use_setup_col # passed for worker to know which col to check
            )

            if _HAS_ENGINE:
                engine_val = st.session_state.get("engine_ui", "Hybrid (recommended)")
                kwargs["engine"] = "hybrid" if engine_val.startswith("Hybrid") else ("box" if engine_val.startswith("Box") else "squeeze")

            if _HAS_BOX_WIDTH and "box_width" in st.session_state:
                kwargs["box_width_pct_max"] = float(st.session_state["box_width"])

            if _HAS_DI and "di_confirm" in st.session_state:
                kwargs["require_di_confirmation"] = bool(st.session_state["di_confirm"])

            if _HAS_RSI_FLOOR:
                kwargs["rsi_floor_short"] = float(st.session_state.get("rsi_floor_short", 30.0))
                kwargs["rsi_ceiling_long"] = float(st.session_state.get("rsi_ceiling_long", 70.0))

            return kwargs

        last_closed_ts = last_closed_15m_candle_ist()
        st.caption(f"✅ Closed-candle guard active. Ignoring any 15M candle after {last_closed_ts.strftime('%d %b %Y %H:%M')} IST.")

        # --- Parallel Execution ---
        
        # Prepare params
        params_15m = build_kwargs(is_daily=False)
        params_daily = build_kwargs(is_daily=True)
        
        # We'll run scanning for both timeframes in parallel threads
        futures = []
        total_tasks = len(stock_list) * 2 # 15m + daily for each stock
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        completed_tasks = 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            for symbol in stock_list:
                # Submit 15M Task
                futures.append(executor.submit(process_scan_task, symbol, "15M", params_15m, last_closed_ts))
                # Submit Daily Task
                futures.append(executor.submit(process_scan_task, symbol, "Daily", params_daily, last_closed_ts))
            
            for future in as_completed(futures):
                completed_tasks += 1
                progress_bar.progress(completed_tasks / total_tasks)
                
                result_list = future.result() # returns list of dicts or None
                if result_list:
                    for item in result_list:
                        sym = item["symbol"]
                        tf = item["timeframe"]
                        r = item["row_data"]
                        candle_ts = item["candle_ts"]
                        
                        setup = r[use_setup_col]
                        trend = r.get("trend", "")
                        ltp = float(r.get("close", np.nan))
                        bbw = float(r.get("bbw", np.nan))
                        bbw_rank = r.get("bbw_pct_rank", np.nan)
                        rsi_val = float(r.get("rsi", np.nan))
                        adx_val = float(r.get("adx", np.nan))
                        bias = "LONG" if str(setup).startswith("Bullish") else "SHORT"
                        quality = compute_quality_score(r)
                        
                        if tf == "15M":
                            signal_time = ts_to_sheet_str(candle_ts, "15min")
                            key = f"{sym}|15M|Continuation|{signal_time}|{setup}|{bias}"
                            if key not in existing_keys_15m:
                                rows_15m_to_append.append([
                                    key, signal_time, logged_at, sym, "15M", "Continuation",
                                    setup, bias, ltp, bbw, bbw_rank, rsi_val, adx_val, trend,
                                    quality, params_hash
                                ])
                                existing_keys_15m.add(key)
                                status_text.text(f"Found 15M signal: {sym}")
                        
                        else: # Daily
                            signal_time = pd.Timestamp(candle_ts).strftime("%Y-%m-%d")
                            key = f"{sym}|Daily|Continuation|{signal_time}|{setup}|{bias}"
                            if key not in existing_keys_daily:
                                rows_daily_to_append.append([
                                    key, signal_time, logged_at, sym, "Daily", "Continuation",
                                    setup, bias, ltp, bbw, bbw_rank, rsi_val, adx_val, trend,
                                    quality, params_hash
                                ])
                                existing_keys_daily.add(key)
                                status_text.text(f"Found Daily signal: {sym}")

        status_text.text("Scan complete. Writing to sheets...")
        progress_bar.progress(1.0)
        
        # --- Batch Write ---
        appended_15m = 0
        appended_daily = 0
        
        if rows_15m_to_append:
            if ws_15m and not ws_15m_err:
                appended_15m, _ = append_signals_batch(ws_15m, rows_15m_to_append, st.session_state.get("show_debug", False))
            else:
                st.error("Found 15M signals but Sheet is not connected!")

        if rows_daily_to_append:
            if ws_daily and not ws_daily_err:
                appended_daily, _ = append_signals_batch(ws_daily, rows_daily_to_append, st.session_state.get("show_debug", False))
            else:
                st.error("Found Daily signals but Sheet is not connected!")

        st.caption(f"Logged **{appended_15m}** new 15M + **{appended_daily}** Daily signals (IST-only).")
        if appended_15m > 0 or appended_daily > 0:
            st.toast(f"✅ Logged {appended_15m + appended_daily} new signals!")

st.caption("✅ FYERS-powered: token-sheet login + IST-correct OHLC + closed-candle guard + IST-only sheet logging + safe repair UI + Parallel Scan.")
