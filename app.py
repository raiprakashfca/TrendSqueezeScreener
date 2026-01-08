# app.py
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta
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

from utils.risk_calculator import add_risk_metrics_to_signal
from utils.zerodha_utils import (
    init_fyers_session,
    get_ohlc_15min,
    get_ohlc_daily,
    is_trading_day,
    get_last_trading_day,
)
from utils.strategy import prepare_trend_squeeze


# =========================
# Streamlit config (ONLY ONCE)
# =========================
st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")
st_autorefresh(interval=300000, key="auto_refresh")  # 5 min refresh

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

# "non-IST detector" tolerance window
MARKET_OPEN_TOL = dtime(9, 0)
MARKET_CLOSE_TOL = dtime(15, 45)

SIGNAL_SHEET_15M = "TrendSqueezeSignals_15M"
SIGNAL_SHEET_DAILY = "TrendSqueezeSignals_Daily"

# Direct Sheet IDs (signals logs)
SHEET_ID_15M = st.secrets.get("SHEET_ID_15M", "1RP2tbh6WnMgEDAv5FWXD6GhePXno9bZ81ILFyoX6Fb8")
SHEET_ID_DAILY = st.secrets.get("SHEET_ID_DAILY", "1u-W5vu3KM6XPRr78o8yyK8pPaAjRUIFEEYKHyYGUsM0")

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
    # Risk management columns
    "atr",
    "stop_loss",
    "stop_method",
    "shares",
    "position_value",
    "rupee_risk",
    "risk_pct",
    "risk_per_share",
    "target_1.5R",
    "target_2R",
    "target_3R",
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
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def _safe_parse_dt(val) -> pd.Timestamp | None:
    try:
        t = pd.to_datetime(val, errors="coerce")
        if pd.isna(t):
            return None
        return pd.Timestamp(t)
    except Exception:
        return None


def to_ist_naive_assume_ist_if_naive(ts) -> pd.Timestamp:
    """
    tz-aware -> IST-naive
    tz-naive -> assume already IST-naive (Sheet timestamps)
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(None)
    return t.tz_convert("Asia/Kolkata").tz_localize(None)


def ts_to_sheet_str(ts, freq: str = "15min") -> str:
    t = to_ist_naive_assume_ist_if_naive(pd.Timestamp(ts)).floor(freq)
    return t.strftime("%Y-%m-%d %H:%M")


def normalize_ohlc_index_to_ist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix timezone issues WITHOUT creating 'future candles'.

    - tz-aware index -> convert to IST-naive
    - tz-naive index -> try UTC->IST; if that creates future candles, treat as already IST-naive
    """
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    out = df.copy()
    idx = out.index

    if getattr(idx, "tz", None) is not None:
        try:
            out.index = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        except Exception:
            pass
        return out

    try:
        idx_utc_to_ist = idx.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
    except Exception:
        return out

    try:
        last_conv = pd.Timestamp(idx_utc_to_ist[-1])
        now_local = now_ist().replace(tzinfo=None)
        if last_conv > now_local + timedelta(minutes=3):
            out.index = idx.tz_localize(None)  # already IST-naive
        else:
            out.index = idx_utc_to_ist
    except Exception:
        out.index = idx_utc_to_ist

    return out


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

    # Fallback cells
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
    for col in ["fyers_access_token", "fyers_token_updated_at"]:
        if col not in header:
            header.append(col)
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
        df = get_ohlc_15min("RELIANCE", days_back=7)
        df = normalize_ohlc_index_to_ist(df)
        if df is not None and len(df) >= 10:
            return True, "OK"
        return False, "OHLC returned empty/too short"
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

    # Enforce headers (order matters because we append rows in this order)
    try:
        header = ws.row_values(1)
        if header != SIGNAL_COLUMNS:
            ws.update(values=[SIGNAL_COLUMNS], range_name="A1")
    except Exception as e:
        return None, f"Failed to ensure headers: {e}"

    return ws, None


def fetch_existing_keys_recent(ws, days_back: int = 3) -> set[str]:
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
    df["signal_time_dt"] = df["signal_time_dt"].apply(to_ist_naive_assume_ist_if_naive)

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


def compute_quality_score(row: pd.Series) -> str:
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
    df["signal_time_dt"] = df["signal_time_dt"].apply(to_ist_naive_assume_ist_if_naive)

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
            "stop_loss": "Stop",
            "shares": "Qty",
            "position_value": "Position ₹",
            "target_2R": "Target 2R",
        },
        inplace=True,
    )

    df = df.sort_values("signal_time_dt", ascending=False)
    if not audit_mode:
        df = df.drop_duplicates(subset=["Symbol", "Timeframe"], keep="first")

    cols = [
        "Timestamp", "Symbol", "Timeframe", "Quality", "Bias", "Setup",
        "LTP", "Stop", "Qty", "Position ₹", "Target 2R",
        "BBW", "BBW %Rank", "RSI", "ADX", "Trend",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


# =========================
# One-time repair: convert non-IST timestamps in Sheets
# =========================
def _looks_non_ist_15m(dt_naive: pd.Timestamp) -> bool:
    if dt_naive is None or pd.isna(dt_naive):
        return False
    t = dt_naive.time()
    if t < MARKET_OPEN_TOL or t > MARKET_CLOSE_TOL:
        return True
    if dt_naive > now_ist().replace(tzinfo=None) + timedelta(minutes=3):
        return True
    return False


def repair_sheet_timestamps(ws, timeframe: str = "15M", dry_run: bool = True, max_rows: int = 5000) -> dict:
    res = {"rows_total": 0, "rows_scanned": 0, "candidates": 0, "updated": 0, "skipped": 0, "errors": 0, "notes": []}

    try:
        values = ws.get_all_values()
    except Exception as e:
        res["errors"] += 1
        res["notes"].append(f"read failed: {e}")
        return res

    if not values or len(values) < 2:
        res["notes"].append("no data rows")
        return res

    header = values[0]
    try:
        col_signal = header.index("signal_time")
        col_key = header.index("key")
    except ValueError:
        res["notes"].append("missing required columns: key/signal_time")
        return res

    res["rows_total"] = len(values) - 1
    rows = values[1 : 1 + max_rows]
    res["rows_scanned"] = len(rows)

    updates = []
    for i, row in enumerate(rows, start=2):
        try:
            if len(row) <= max(col_signal, col_key):
                res["skipped"] += 1
                continue
            sig_raw = (row[col_signal] or "").strip()
            key_raw = (row[col_key] or "").strip()
            if not sig_raw or not key_raw:
                res["skipped"] += 1
                continue

            dt = _safe_parse_dt(sig_raw)
            if dt is None:
                res["skipped"] += 1
                continue

            needs_fix = _looks_non_ist_15m(dt) if timeframe.upper().startswith("15") else (pd.Timestamp(dt).date() > now_ist().date())
            if not needs_fix:
                continue

            res["candidates"] += 1
            fixed_dt = (pd.Timestamp(dt) + pd.Timedelta(hours=5, minutes=30)).to_pydatetime()
            new_sig = fixed_dt.strftime("%Y-%m-%d %H:%M") if timeframe.upper().startswith("15") else fixed_dt.strftime("%Y-%m-%d")

            new_key = key_raw
            parts = key_raw.split("|")
            if len(parts) >= 6:
                parts[3] = new_sig
                new_key = "|".join(parts)

            updates.append((i, new_sig, new_key))
        except Exception as e:
            res["errors"] += 1
            res["notes"].append(f"row {i}: {e}")

    if not updates:
        res["notes"].append("no candidates found")
        return res

    if dry_run:
        res["notes"].append(f"dry run: would update {len(updates)} rows")
        return res

    try:
        for (row_idx, new_sig, new_key) in updates:
            ws.update_cell(row_idx, col_signal + 1, new_sig)
            ws.update_cell(row_idx, col_key + 1, new_key)
            res["updated"] += 1
    except Exception as e:
        res["errors"] += 1
        res["notes"].append(f"write failed: {e}")

    return res


# =========================
# Strategy signature detection
# =========================
_STRAT_SIG = inspect.signature(prepare_trend_squeeze)
_HAS_ENGINE = "engine" in _STRAT_SIG.parameters
_HAS_BOX_WIDTH = "box_width_pct_max" in _STRAT_SIG.parameters
_HAS_RSI_FLOOR = "rsi_floor_short" in _STRAT_SIG.parameters
_HAS_DI = "require_di_confirmation" in _STRAT_SIG.parameters


# =========================
# Sidebar: FYERS login + controls
# =========================
with st.sidebar:
    st.header("🔐 FYERS Login")

    ws_fyers, ws_fyers_err = get_fyers_token_worksheet()
    if ws_fyers_err:
        st.error("FYERS token sheet ❌")
        st.caption(ws_fyers_err)
        st.stop()

    app_id_sheet, secret_id_sheet, token_sheet, updated_at_sheet = read_fyers_row(ws_fyers)

    if token_sheet:
        st.success("Token found ✅")
    else:
        st.warning("No access token yet ❗")
    if updated_at_sheet:
        st.caption(f"Last updated: {updated_at_sheet}")

    redirect_uri = st.secrets.get("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri/index.html")

    if not app_id_sheet or not secret_id_sheet:
        st.error("Fill fyers_app_id and fyers_secret_id in FYERS token sheet row 2.")
        st.stop()

    try:
        login_url = build_fyers_login_url(app_id_sheet, secret_id_sheet, redirect_uri)
        st.link_button("1) Open FYERS Login", login_url)
        st.caption("Login → copy `auth_code` from redirected URL → paste below.")
    except Exception as e:
        st.error(f"Login URL failed: {e}")

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
    st.checkbox("Show debug", value=False, key="show_debug")
    st.checkbox("Audit mode (no de-dup)", value=False, key="audit_mode")
    st.slider("Retention (hours)", 6, 48, 24, 6, key="retention_hours")

    st.radio("Show quality", ["A only", "A + B", "A + B + C"], index=0, key="quality_pref", horizontal=True)

    with st.expander("🧨 Breakout confirmation", expanded=False):
        st.radio("Signal type", ["✅ Breakout Confirmed (trade)", "🟡 Setup Forming (watchlist)"], index=0, key="signal_mode")
        st.slider("Consolidation lookback (candles)", 10, 60, 20, 5, key="breakout_lookback")
        st.checkbox("Require BBW expansion", value=True, key="require_bbw_expansion")
        st.checkbox("Require volume spike", value=False, key="require_volume_spike")
        st.slider("Volume spike mult", 1.0, 3.0, 1.5, 0.1, key="volume_spike_mult")

    with st.expander("🧩 Engine controls", expanded=False):
        if _HAS_ENGINE:
            st.selectbox("Engine", ["Hybrid (recommended)", "Box Breakout (range)", "Squeeze Breakout (TTM)"], index=0, key="engine_ui")
        else:
            st.info("Engine controls not available in your strategy.py.")

        if _HAS_BOX_WIDTH:
            st.slider("Max box width (%)", 0.3, 3.0, 1.2, 0.1, key="box_width")

        if _HAS_DI:
            st.checkbox("Require DI confirmation", value=True, key="di_confirm")

        if _HAS_RSI_FLOOR:
            st.slider("Block SHORT if RSI below", 10.0, 45.0, 30.0, 1.0, key="rsi_floor_short")
            st.slider("Block LONG if RSI above", 55.0, 90.0, 70.0, 1.0, key="rsi_ceiling_long")

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

        st.slider("BBW abs max", 0.01, 0.20, bbw_abs_default, 0.005, key="bbw_abs_threshold")
        st.slider("BBW pct rank max", 0.10, 0.80, bbw_pct_default, 0.05, key="bbw_pct_threshold")
        st.slider("ADX min", 15.0, 35.0, adx_default, 1.0, key="adx_threshold")
        st.slider("RSI bull min", 50.0, 70.0, rsi_bull_default, 1.0, key="rsi_bull")
        st.slider("RSI bear max", 30.0, 50.0, rsi_bear_default, 1.0, key="rsi_bear")

    st.slider("15M catch-up candles", 8, 64, 32, 4, key="catchup_candles_15m")

    st.divider()
    st.subheader("💰 Risk Management")
    st.number_input("Account Capital (₹)", min_value=10000.0, max_value=100000000.0, value=1000000.0, step=50000.0, key="account_capital")
    st.slider("Risk Per Trade (%)", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="risk_per_trade")
    st.slider("ATR Multiplier for Stop", min_value=1.5, max_value=3.0, value=2.0, step=0.5, key="atr_multiplier")


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
        fyers_health_msg = str(e)[:220]
else:
    fyers_ok = False
    fyers_health_ok = False
    fyers_health_msg = "No token yet. Use the sidebar login flow."


# =========================
# Main header
# =========================
st.title("📉 Trend Squeeze Screener")
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


# =========================
# Universe
# =========================
stock_list = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV",
    "BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH",
    "HDFCBANK","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM",
    "M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBIN","SBILIFE","SUNPHARMA","TATACONSUM",
    "TATAMOTORS","TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HEROMOTOCO","SHREECEM"
]
st.caption(f"Universe: NIFTY50 ({len(stock_list)} symbols).")

use_setup_col_local = "setup" if st.session_state.get("signal_mode", "✅").startswith("✅") else "setup_forming"


# =========================
# Sheets connect
# =========================
ws_15m, ws_15m_err = get_signals_worksheet(SIGNAL_SHEET_15M)
ws_daily, ws_daily_err = get_signals_worksheet(SIGNAL_SHEET_DAILY)


# =========================
# Repair UI
# =========================
with st.expander("🧹 One-time repair: convert non-IST timestamps in Sheets", expanded=False):
    st.warning("This will EDIT existing rows. Take a backup first 📦")
    dry = st.checkbox("Dry run (report only, no writing)", value=True, key="repair_dry_run")

    if st.button("Run repair now", type="primary", key="run_repair"):
        if ws_15m_err or ws_daily_err:
            st.error("Fix sheet connectivity first.")
        else:
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
# Recent Signals UI
# =========================
top_row = st.columns([2, 2, 3])
with top_row[0]:
    st.write("**15M Sheet**")
    st.success("Connected ✅" if not ws_15m_err else "Not connected ❌")
    if ws_15m_err:
        st.caption(ws_15m_err)
with top_row[1]:
    st.write("**Daily Sheet**")
    st.success("Connected ✅" if not ws_daily_err else "Not connected ❌")
    if ws_daily_err:
        st.caption(ws_daily_err)
with top_row[2]:
    st.write("**Display**")
    st.caption("Quality filter + de-dup are in the sidebar.")

df_recent_15m = load_recent_signals(ws_15m, st.session_state["retention_hours"], "15M", audit_mode=st.session_state["audit_mode"]) if ws_15m and not ws_15m_err else pd.DataFrame()
df_recent_daily = load_recent_signals(ws_daily, st.session_state["retention_hours"], "Daily", audit_mode=st.session_state["audit_mode"]) if ws_daily and not ws_daily_err else pd.DataFrame()

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
    dfA_15m = _filter_quality(df_recent_15m)
    st.dataframe(dfA_15m, use_container_width=True) if not dfA_15m.empty else st.info("No signals in window.")
with c2:
    st.markdown("### Daily")
    dfA_daily = _filter_quality(df_recent_daily)
    st.dataframe(dfA_daily, use_container_width=True) if not dfA_daily.empty else st.info("No signals in window.")

st.divider()


# =========================
# Scan + logging
# =========================
with st.expander("🔎 Scan & Log (runs every refresh)", expanded=False):
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
    rows_15m, rows_daily = [], []

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
        )

        if _HAS_ENGINE:
            engine_ui = st.session_state.get("engine_ui", "Hybrid (recommended)")
            kwargs["engine"] = "hybrid" if engine_ui.startswith("Hybrid") else ("box" if engine_ui.startswith("Box") else "squeeze")

        if _HAS_BOX_WIDTH and "box_width" in st.session_state:
            kwargs["box_width_pct_max"] = float(st.session_state["box_width"]) / 100.0

        if _HAS_DI and "di_confirm" in st.session_state:
            kwargs["require_di_confirmation"] = bool(st.session_state["di_confirm"])

        if _HAS_RSI_FLOOR:
            kwargs["rsi_floor_short"] = float(st.session_state.get("rsi_floor_short", 30.0))
            kwargs["rsi_ceiling_long"] = float(st.session_state.get("rsi_ceiling_long", 70.0))

        return kwargs

    # 15M scan
    for symbol in stock_list:
        try:
            df = normalize_ohlc_index_to_ist(get_ohlc_15min(symbol, days_back=15))
            if df is None or df.shape[0] < 220:
                continue

            if pd.Timestamp(df.index[-1]) > now_ist().replace(tzinfo=None) + timedelta(minutes=3):
                if st.session_state.get("show_debug", False):
                    st.warning(f"{symbol}: OHLC last candle looks future, skipping.")
                continue

            df_prepped = prepare_trend_squeeze(df, **build_kwargs(is_daily=False))

            recent = df_prepped.tail(int(st.session_state["catchup_candles_15m"])).copy()
            recent = recent[recent[use_setup_col_local] != ""]

            for candle_ts, r in recent.iterrows():
                signal_time = ts_to_sheet_str(candle_ts, "15min")
                dt_sig = _safe_parse_dt(signal_time)
                if dt_sig is not None and _looks_non_ist_15m(dt_sig):
                    continue

                setup = r[use_setup_col_local]
                trend = r.get("trend", "")
                ltp = float(r.get("close", np.nan))
                bbw = float(r.get("bbw", np.nan))
                bbw_rank = r.get("bbw_pct_rank", np.nan)
                rsi_val = float(r.get("rsi", np.nan))
                adx_val = float(r.get("adx", np.nan))

                bias = "LONG" if str(setup).startswith("Bullish") else "SHORT"
                key = f"{symbol}|15M|Continuation|{signal_time}|{setup}|{bias}"

                if key in existing_keys_15m:
                    continue

                quality = compute_quality_score(r)

                signal_dict = {
                    "key": key,
                    "signal_time": signal_time,
                    "logged_at": logged_at,
                    "symbol": symbol,
                    "timeframe": "15M",
                    "mode": "Continuation",
                    "setup": setup,
                    "bias": bias,
                    "ltp": ltp,
                    "bbw": bbw,
                    "bbw_pct_rank": bbw_rank,
                    "rsi": rsi_val,
                    "adx": adx_val,
                    "trend": trend,
                    "quality_score": quality,
                    "params_hash": params_hash,
                }

                enhanced_signal = add_risk_metrics_to_signal(
                    signal_row=signal_dict,
                    df_ohlc=df,
                    account_capital=st.session_state.get("account_capital", 1000000.0),
                    risk_per_trade_pct=st.session_state.get("risk_per_trade", 1.0),
                    atr_multiplier=st.session_state.get("atr_multiplier", 2.0),
                    lookback_swing=20,
                )

                rows_15m.append([enhanced_signal.get(c) for c in SIGNAL_COLUMNS])
                existing_keys_15m.add(key)

        except Exception as e:
            if st.session_state.get("show_debug", False):
                st.warning(f"{symbol} 15M error: {e}")
            continue

    # Daily scan
    for symbol in stock_list:
        try:
            df_daily = normalize_ohlc_index_to_ist(get_ohlc_daily(symbol, lookback_days=252))
            if df_daily is None or df_daily.shape[0] < 100:
                continue

            df_daily_prepped = prepare_trend_squeeze(df_daily, **build_kwargs(is_daily=True))

            recent_daily = df_daily_prepped.tail(7).copy()
            recent_daily = recent_daily[recent_daily[use_setup_col_local] != ""]

            for candle_ts, r in recent_daily.iterrows():
                signal_time = pd.Timestamp(candle_ts).strftime("%Y-%m-%d")
                if pd.to_datetime(signal_time).date() > now_ist().date():
                    continue

                setup = r[use_setup_col_local]
                trend = r.get("trend", "")
                ltp = float(r.get("close", np.nan))
                bbw = float(r.get("bbw", np.nan))
                bbw_rank = r.get("bbw_pct_rank", np.nan)
                rsi_val = float(r.get("rsi", np.nan))
                adx_val = float(r.get("adx", np.nan))

                bias = "LONG" if str(setup).startswith("Bullish") else "SHORT"
                key = f"{symbol}|Daily|Continuation|{signal_time}|{setup}|{bias}"

                if key in existing_keys_daily:
                    continue

                quality = compute_quality_score(r)

                signal_dict = {
                    "key": key,
                    "signal_time": signal_time,
                    "logged_at": logged_at,
                    "symbol": symbol,
                    "timeframe": "Daily",
                    "mode": "Continuation",
                    "setup": setup,
                    "bias": bias,
                    "ltp": ltp,
                    "bbw": bbw,
                    "bbw_pct_rank": bbw_rank,
                    "rsi": rsi_val,
                    "adx": adx_val,
                    "trend": trend,
                    "quality_score": quality,
                    "params_hash": params_hash,
                }

                enhanced_signal = add_risk_metrics_to_signal(
                    signal_row=signal_dict,
                    df_ohlc=df_daily,
                    account_capital=st.session_state.get("account_capital", 1000000.0),
                    risk_per_trade_pct=st.session_state.get("risk_per_trade", 1.0),
                    atr_multiplier=st.session_state.get("atr_multiplier", 2.0),
                    lookback_swing=20,
                )

                rows_daily.append([enhanced_signal.get(c) for c in SIGNAL_COLUMNS])
                existing_keys_daily.add(key)

        except Exception as e:
            if st.session_state.get("show_debug", False):
                st.warning(f"{symbol} Daily error: {e}")
            continue

    appended_15m = appended_daily = 0
    err15 = errd = None
    if ws_15m and not ws_15m_err:
        appended_15m, err15 = append_signals(ws_15m, rows_15m, st.session_state.get("show_debug", False))
    if ws_daily and not ws_daily_err:
        appended_daily, errd = append_signals(ws_daily, rows_daily, st.session_state.get("show_debug", False))

    if err15 or errd:
        st.error(f"Logging errors: 15M={err15} | Daily={errd}")

    st.caption(f"Logged **{appended_15m}** new 15M + **{appended_daily}** Daily signals.")
