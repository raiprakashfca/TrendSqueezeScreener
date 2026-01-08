# app.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional autorefresh (won't crash if missing)
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None  # type: ignore

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import SpreadsheetNotFound, APIError

# FYERS (v3)
FYERS_IMPORT_OK = True
FYERS_IMPORT_ERR = ""
try:
    from fyers_apiv3 import fyersModel
except Exception as e:
    FYERS_IMPORT_OK = False
    FYERS_IMPORT_ERR = str(e)


# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")

if st_autorefresh is not None:
    st_autorefresh(interval=300000, key="auto_refresh")  # 5 minutes


# =========================
# Constants
# =========================
IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

SIGNAL_SHEET_15M_NAME = "TrendSqueezeSignals_15M"
SIGNAL_SHEET_DAILY_NAME = "TrendSqueezeSignals_Daily"

# Default signal sheet IDs (you can override via st.secrets)
DEFAULT_SHEET_ID_15M = "1RP2tbh6WnMgEDAv5FWXD6GhePXno9bZ81ILFyoX6Fb8"
DEFAULT_SHEET_ID_DAILY = "1u-W5vu3KM6XPRr78o8yyK8pPaAjRUIFEEYKHyYGUsM0"

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
    # Risk columns
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

NIFTY50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV",
    "BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH",
    "HDFCBANK","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM",
    "M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBIN","SBILIFE","SUNPHARMA","TATACONSUM",
    "TATAMOTORS","TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HEROMOTOCO","SHREECEM"
]


# =========================
# Time helpers
# =========================
def now_ist() -> datetime:
    return datetime.now(IST)

def is_trading_day(d) -> bool:
    # Minimal fallback: Mon–Fri. (Holiday calendar not included.)
    return pd.Timestamp(d).dayofweek < 5

def market_open_now() -> bool:
    n = now_ist()
    return is_trading_day(n.date()) and (MARKET_OPEN <= n.time() <= MARKET_CLOSE)

def fmt_dt(dt: datetime) -> str:
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

def ts_to_sheet_str(ts: pd.Timestamp, freq: str = "15min") -> str:
    t = pd.Timestamp(ts).tz_localize(None).floor(freq)
    return t.strftime("%Y-%m-%d %H:%M")

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# =========================
# Google Sheets auth
# =========================
@st.cache_resource(show_spinner=False)
def get_gspread_client() -> Tuple[Optional[gspread.Client], Optional[str]]:
    try:
        raw = st.secrets["gcp_service_account"]
    except Exception:
        return None, "Missing st.secrets['gcp_service_account']"

    try:
        sa = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception as e:
        return None, f"Invalid gcp_service_account format: {e}"

    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(sa, scopes=scopes)
        return gspread.authorize(creds), None
    except Exception as e:
        return None, f"Google auth failed: {e}"


def open_or_create_sheet_by_key(client: gspread.Client, sheet_key: str):
    return client.open_by_key(sheet_key)

def ensure_header(ws, header: List[str]) -> None:
    existing = ws.row_values(1)
    if not existing:
        ws.update("A1", [header])
        return
    # If shorter or missing columns, overwrite row1 with canonical header
    if len(existing) < len(header):
        ws.update("A1", [header])


def get_signals_worksheet(sheet_kind: str) -> Tuple[Optional[gspread.Worksheet], Optional[str]]:
    """
    sheet_kind: "15M" or "DAILY"
    """
    client, err = get_gspread_client()
    if err:
        return None, err

    sheet_id_15m = st.secrets.get("SHEET_ID_15M", DEFAULT_SHEET_ID_15M)
    sheet_id_daily = st.secrets.get("SHEET_ID_DAILY", DEFAULT_SHEET_ID_DAILY)

    try:
        if sheet_kind.upper() == "15M":
            sh = open_or_create_sheet_by_key(client, sheet_id_15m)
        else:
            sh = open_or_create_sheet_by_key(client, sheet_id_daily)
        ws = sh.sheet1
        ensure_header(ws, SIGNAL_COLUMNS)
        return ws, None
    except SpreadsheetNotFound:
        return None, f"Signals sheet not found for {sheet_kind}. Check Sheet ID and sharing."
    except Exception as e:
        return None, f"Failed to open signals sheet ({sheet_kind}): {e}"


# =========================
# FYERS token sheet manager
# =========================
def get_fyers_token_worksheet() -> Tuple[Optional[gspread.Worksheet], Optional[str]]:
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
        # Ensure headers
        existing = ws.row_values(1)
        if not existing:
            ws.update("A1", [FYERS_TOKEN_HEADERS])
        else:
            header_l = [h.strip() for h in existing]
            changed = False
            for h in FYERS_TOKEN_HEADERS:
                if h not in header_l:
                    header_l.append(h)
                    changed = True
            if changed:
                ws.update("A1", [header_l])
        return ws, None
    except SpreadsheetNotFound:
        return None, "FYERS token sheet not found (check key/name and sharing)."
    except Exception as e:
        return None, f"Failed to open FYERS token sheet: {e}"


def read_fyers_row(ws: gspread.Worksheet) -> Tuple[str, str, str, str]:
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

    # fallback row2 A-D
    try:
        app_id = (ws.acell("A2").value or "").strip()
        secret_id = (ws.acell("B2").value or "").strip()
        token = (ws.acell("C2").value or "").strip()
        updated_at = (ws.acell("D2").value or "").strip()
        return app_id, secret_id, token, updated_at
    except Exception:
        return "", "", "", ""


def ensure_row2(ws: gspread.Worksheet) -> None:
    vals = ws.get_all_values()
    if len(vals) < 2:
        ws.append_row(["", "", "", ""], value_input_option="USER_ENTERED")


def update_fyers_token(ws: gspread.Worksheet, token: str, timestamp: str) -> None:
    header = [h.strip() for h in ws.row_values(1)]
    for h in FYERS_TOKEN_HEADERS:
        if h not in header:
            header.append(h)
    ws.update("A1", [header])

    ensure_row2(ws)
    col_token = header.index("fyers_access_token") + 1
    col_ts = header.index("fyers_token_updated_at") + 1
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


def init_fyers(app_id: str, token: str):
    # FYERS v3: token should usually be "appId:access_token" OR plain depending on broker config.
    # Most setups work with plain token passed as `token=...`.
    return fyersModel.FyersModel(client_id=app_id, is_async=False, token=token, log_path="")


# =========================
# Indicators (self-contained)
# =========================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1/length, adjust=False).mean()

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1/length, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr_.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr_.replace(0, np.nan))

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1/length, adjust=False).mean()

def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(length).mean()
    std = close.rolling(length).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    return lower, mid, upper

def keltner(df: pd.DataFrame, length: int = 20, mult: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # Typical KC: EMA(close) +/- mult*ATR
    mid = ema(df["close"], length)
    rng = atr(df, length)
    upper = mid + mult * rng
    lower = mid - mult * rng
    return lower, mid, upper

def bbw(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.Series:
    lower, mid, upper = bollinger(close, length, mult)
    width = (upper - lower) / mid.replace(0, np.nan)
    return width

def pct_rank(series: pd.Series, lookback: int = 252) -> pd.Series:
    # Percentile rank of current value vs rolling window
    def _rank(window: np.ndarray) -> float:
        if len(window) < 5 or np.isnan(window[-1]):
            return np.nan
        x = window[-1]
        w = window[~np.isnan(window)]
        if len(w) < 5:
            return np.nan
        return float((w <= x).mean())
    return series.rolling(lookback).apply(_rank, raw=True)

def trend_label(close: pd.Series) -> pd.Series:
    e50 = ema(close, 50)
    e200 = ema(close, 200)
    out = pd.Series(index=close.index, dtype=object)
    out[(e50 > e200)] = "Bull"
    out[(e50 < e200)] = "Bear"
    out = out.fillna("Sideways")
    return out


# =========================
# Strategy: prepare signals
# =========================
@dataclass
class StrategyParams:
    bbw_abs_max: float
    bbw_pct_max: float
    adx_min: float
    rsi_bull_min: float
    rsi_bear_max: float

    breakout_lookback: int
    require_bbw_expansion: bool
    require_volume_spike: bool
    volume_spike_mult: float

def prepare_trend_squeeze_df(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    out = df.copy()
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["rsi"] = rsi(out["close"], 14)
    out["adx"] = adx(out, 14)
    out["trend"] = trend_label(out["close"])
    out["bbw"] = bbw(out["close"], 20, 2.0)
    out["bbw_pct_rank"] = pct_rank(out["bbw"], lookback=252)

    # Consolidation box
    look = int(params.breakout_lookback)
    box_high = out["high"].rolling(look).max()
    box_low = out["low"].rolling(look).min()
    out["box_high"] = box_high
    out["box_low"] = box_low

    # Breakout direction
    out["break_up"] = out["close"] > out["box_high"].shift(1)
    out["break_dn"] = out["close"] < out["box_low"].shift(1)

    # Optional BBW expansion: current bbw > median of last look candles
    bbw_med = out["bbw"].rolling(look).median()
    out["bbw_expand"] = out["bbw"] > bbw_med.shift(1)

    # Optional volume spike
    vol_avg = out["volume"].rolling(look).mean()
    out["vol_spike"] = out["volume"] > (params.volume_spike_mult * vol_avg.shift(1))

    # Core squeeze filter
    squeeze_ok = (
        (out["bbw"] <= params.bbw_abs_max) &
        (out["bbw_pct_rank"] <= params.bbw_pct_max) &
        (out["adx"] >= params.adx_min)
    )

    # Momentum bias via RSI thresholds
    bull_mom = out["rsi"] >= params.rsi_bull_min
    bear_mom = out["rsi"] <= params.rsi_bear_max

    # Build setup strings
    out["setup_forming"] = ""
    out["setup"] = ""

    # Setup forming (watchlist): squeeze + (bull or bear momentum) even without breakout
    out.loc[squeeze_ok & bull_mom, "setup_forming"] = "Bullish Squeeze (forming)"
    out.loc[squeeze_ok & bear_mom, "setup_forming"] = "Bearish Squeeze (forming)"

    # Breakout confirmed: squeeze + breakout + optional confirmations
    confirm_mask = pd.Series(True, index=out.index)
    if params.require_bbw_expansion:
        confirm_mask &= out["bbw_expand"]
    if params.require_volume_spike:
        confirm_mask &= out["vol_spike"]

    long_break = squeeze_ok & bull_mom & out["break_up"] & confirm_mask
    short_break = squeeze_ok & bear_mom & out["break_dn"] & confirm_mask

    out.loc[long_break, "setup"] = "Bullish Breakout"
    out.loc[short_break, "setup"] = "Bearish Breakdown"

    return out


# =========================
# Risk metrics
# =========================
def add_risk_metrics(
    row: Dict[str, Any],
    df_ohlc: pd.DataFrame,
    account_capital: float,
    risk_per_trade_pct: float,
    atr_mult: float,
) -> Dict[str, Any]:
    out = dict(row)

    if df_ohlc is None or df_ohlc.empty:
        return out

    a = atr(df_ohlc, 14).iloc[-1]
    ltp = safe_float(out.get("ltp"))

    if not np.isfinite(a) or not np.isfinite(ltp) or ltp <= 0:
        return out

    stop = ltp - atr_mult * a if out.get("bias") == "LONG" else ltp + atr_mult * a
    risk_per_share = abs(ltp - stop)

    rupee_risk = account_capital * (risk_per_trade_pct / 100.0)
    shares = int(max(1, np.floor(rupee_risk / max(risk_per_share, 1e-6))))
    position_value = shares * ltp
    risk_pct = (rupee_risk / account_capital) * 100.0

    out.update({
        "atr": float(a),
        "stop_loss": float(stop),
        "stop_method": f"{atr_mult:.1f}xATR",
        "shares": shares,
        "position_value": float(position_value),
        "rupee_risk": float(rupee_risk),
        "risk_pct": float(risk_pct),
        "risk_per_share": float(risk_per_share),
        "target_1.5R": float(ltp + (1.5 * risk_per_share) if out.get("bias") == "LONG" else ltp - (1.5 * risk_per_share)),
        "target_2R": float(ltp + (2.0 * risk_per_share) if out.get("bias") == "LONG" else ltp - (2.0 * risk_per_share)),
        "target_3R": float(ltp + (3.0 * risk_per_share) if out.get("bias") == "LONG" else ltp - (3.0 * risk_per_share)),
    })
    return out


# =========================
# FYERS OHLC fetch
# =========================
def symbol_to_fyers(symbol: str) -> str:
    return f"NSE:{symbol}-EQ"

@st.cache_data(ttl=300, show_spinner=False)
def fyers_history_df(
    token: str,
    app_id: str,
    fy_symbol: str,
    resolution: str,
    range_from: str,
    range_to: str,
) -> pd.DataFrame:
    fy = init_fyers(app_id, token)
    data = {
        "symbol": fy_symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1",
    }
    resp = fy.history(data=data)
    if not isinstance(resp, dict) or resp.get("s") not in ("ok", "no_data"):
        raise RuntimeError(f"FYERS history error: {resp}")

    candles = resp.get("candles", []) or []
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.set_index("ts")
    return df


def get_ohlc_15m(token: str, app_id: str, symbol: str, days_back: int = 15) -> pd.DataFrame:
    to_dt = now_ist().date()
    from_dt = (now_ist() - timedelta(days=days_back)).date()
    return fyers_history_df(
        token=token,
        app_id=app_id,
        fy_symbol=symbol_to_fyers(symbol),
        resolution="15",
        range_from=str(from_dt),
        range_to=str(to_dt),
    )

def get_ohlc_daily(token: str, app_id: str, symbol: str, lookback_days: int = 400) -> pd.DataFrame:
    to_dt = now_ist().date()
    from_dt = (now_ist() - timedelta(days=lookback_days)).date()
    return fyers_history_df(
        token=token,
        app_id=app_id,
        fy_symbol=symbol_to_fyers(symbol),
        resolution="D",
        range_from=str(from_dt),
        range_to=str(to_dt),
    )


# =========================
# Sheet read/write helpers
# =========================
def fetch_existing_keys_recent(ws: gspread.Worksheet, days_back: int = 3) -> set:
    try:
        records = ws.get_all_records()
    except Exception:
        return set()
    if not records:
        return set()

    df = pd.DataFrame(records)
    if "key" not in df.columns:
        return set()
    if "signal_time" not in df.columns:
        return set(df["key"].astype(str).tolist())

    df["signal_time_dt"] = pd.to_datetime(df["signal_time"], errors="coerce")
    df = df[df["signal_time_dt"].notna()].copy()
    cutoff = now_ist().replace(tzinfo=None) - timedelta(days=days_back)
    df = df[df["signal_time_dt"] >= cutoff]
    return set(df["key"].astype(str).tolist())

def append_signals(ws: gspread.Worksheet, rows: List[List[Any]], debug: bool = False) -> Tuple[int, Optional[str]]:
    if not rows:
        return 0, None
    try:
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        return len(rows), None
    except APIError as e:
        return 0, (f"Sheets APIError: {e}" if debug else "Sheets write failed (APIError).")
    except Exception as e:
        return 0, (f"Sheets write failed: {e}" if debug else "Sheets write failed.")

def load_recent_signals(ws: gspread.Worksheet, hours: int, dedup: bool) -> pd.DataFrame:
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
    df = df[df["signal_time_dt"].notna()].copy()

    cutoff = now_ist().replace(tzinfo=None) - timedelta(hours=hours)
    df = df[df["signal_time_dt"] >= cutoff].copy()
    if df.empty:
        return df

    df["Timestamp"] = df["signal_time_dt"].dt.strftime("%d-%b-%Y %H:%M")

    df.rename(columns={
        "symbol": "Symbol",
        "timeframe": "Timeframe",
        "setup": "Setup",
        "bias": "Bias",
        "ltp": "LTP",
        "quality_score": "Quality",
        "stop_loss": "Stop",
        "shares": "Qty",
        "position_value": "Position ₹",
        "target_2R": "Target 2R",
        "bbw": "BBW",
        "bbw_pct_rank": "BBW %Rank",
        "rsi": "RSI",
        "adx": "ADX",
        "trend": "Trend",
    }, inplace=True)

    df = df.sort_values("signal_time_dt", ascending=False)

    if dedup and {"Symbol", "Timeframe"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["Symbol", "Timeframe"], keep="first")

    cols = ["Timestamp","Symbol","Timeframe","Quality","Bias","Setup","LTP","Stop","Qty","Position ₹","Target 2R","BBW","BBW %Rank","RSI","ADX","Trend"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def compute_quality_score(r: pd.Series) -> str:
    score = 0
    adx_val = r.get("adx", np.nan)
    if pd.notna(adx_val):
        if adx_val > 30:
            score += 2
        elif adx_val > 25:
            score += 1

    bbw_rank = r.get("bbw_pct_rank", np.nan)
    if pd.notna(bbw_rank):
        if bbw_rank < 0.2:
            score += 2
        elif bbw_rank < 0.35:
            score += 1

    ema50 = r.get("ema50", np.nan)
    ema200 = r.get("ema200", np.nan)
    close = r.get("close", np.nan)
    if pd.notna(ema50) and pd.notna(ema200) and pd.notna(close) and float(close) != 0:
        ema_spread = abs(float(ema50) - float(ema200)) / float(close)
        if ema_spread > 0.05:
            score += 1

    if score >= 4:
        return "A"
    if score >= 2:
        return "B"
    return "C"


def params_to_hash(p: StrategyParams) -> str:
    return f"{p.bbw_abs_max:.3f}|{p.bbw_pct_max:.2f}|{p.adx_min:.1f}|{p.rsi_bull_min:.1f}|{p.rsi_bear_max:.1f}|L{p.breakout_lookback}"


# =========================
# UI: Sidebar
# =========================
with st.sidebar:
    st.header("🔐 FYERS Login")

    if not FYERS_IMPORT_OK:
        st.error("FYERS client library missing ❌")
        st.caption(f"Import error: {FYERS_IMPORT_ERR}")
        st.caption("Fix: add `fyers-apiv3==3.1.7` to requirements.txt and redeploy.")
        st.stop()

    ws_token, ws_token_err = get_fyers_token_worksheet()
    if ws_token_err:
        st.error("FYERS token sheet ❌")
        st.caption(ws_token_err)
        st.stop()

    app_id, secret_id, access_token, updated_at = read_fyers_row(ws_token)

    if updated_at:
        st.caption(f"Last updated: {updated_at}")
    if access_token:
        st.success("Token found ✅")
    else:
        st.warning("No access token yet ❗")

    redirect_uri = st.secrets.get(
        "FYERS_REDIRECT_URI",
        "https://trade.fyers.in/api-login/redirect-uri/index.html"
    )

    if not app_id or not secret_id:
        st.error("Fill fyers_app_id and fyers_secret_id in token sheet (row 2).")
        st.stop()

    try:
        login_url = build_fyers_login_url(app_id, secret_id, redirect_uri)
        if hasattr(st, "link_button"):
            st.link_button("1) Open FYERS Login", login_url)
        else:
            st.markdown(f"[1) Open FYERS Login]({login_url})")
        st.caption("Login → copy `auth_code` from redirected URL → paste below.")
    except Exception as e:
        st.error(f"Login URL failed: {e}")

    auth_code = st.text_input("2) Paste auth_code", value="", key="auth_code_input")
    if st.button("3) Exchange & Save Token", type="primary"):
        if not auth_code.strip():
            st.error("Paste auth_code first.")
        else:
            try:
                new_token = exchange_auth_code_for_token(app_id, secret_id, redirect_uri, auth_code.strip())
                ts = now_ist().strftime("%Y-%m-%d %H:%M:%S")
                update_fyers_token(ws_token, new_token, ts)
                st.success("Saved token ✅")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Token exchange failed: {e}")

    st.divider()
    st.subheader("⚙️ Settings")

    st.checkbox("Show debug", value=False, key="debug")
    st.checkbox("Audit mode (no de-dup)", value=False, key="audit_mode")
    st.slider("Retention (hours)", 6, 72, 24, 6, key="retention_hours")

    st.radio("Quality filter", ["A only", "A + B", "A + B + C"], index=0, key="quality_pref", horizontal=True)
    st.radio("Signal type", ["✅ Breakout Confirmed (trade)", "🟡 Setup Forming (watchlist)"], index=0, key="signal_mode")

    st.slider("Consolidation lookback (candles)", 10, 60, 20, 5, key="breakout_lookback")
    st.checkbox("Require BBW expansion", value=True, key="require_bbw_expansion")
    st.checkbox("Require volume spike", value=False, key="require_volume_spike")
    st.slider("Volume spike mult", 1.0, 3.0, 1.5, 0.1, key="volume_spike_mult")

    st.divider()
    st.subheader("📈 Thresholds")
    st.slider("BBW abs max", 0.01, 0.20, 0.05, 0.005, key="bbw_abs_max")
    st.slider("BBW pct rank max", 0.10, 0.80, 0.35, 0.05, key="bbw_pct_max")
    st.slider("ADX min", 15.0, 35.0, 20.0, 1.0, key="adx_min")
    st.slider("RSI bull min", 50.0, 70.0, 55.0, 1.0, key="rsi_bull_min")
    st.slider("RSI bear max", 30.0, 50.0, 45.0, 1.0, key="rsi_bear_max")

    st.slider("15M catch-up candles", 8, 64, 32, 4, key="catchup_15m")

    st.divider()
    st.subheader("💰 Risk Management")
    st.number_input("Account Capital (₹)", min_value=10_000.0, max_value=100_000_000.0, value=1_000_000.0, step=50_000.0, key="capital")
    st.slider("Risk Per Trade (%)", 0.5, 2.0, 1.0, 0.1, key="risk_pct")
    st.slider("ATR Multiplier for Stop", 1.5, 3.0, 2.0, 0.5, key="atr_mult")


# =========================
# Main
# =========================
st.title("📉 Trend Squeeze Screener")

n = now_ist()
st.caption(f"{'🟢 LIVE MARKET' if market_open_now() else '🔵 OFF-MARKET'} • Last refresh: {n.strftime('%d %b %Y • %H:%M:%S')} IST")
st.caption(f"Universe: NIFTY50 ({len(NIFTY50)} symbols)")

if not access_token:
    st.error("🔴 FYERS not initialized (no valid token). Use the sidebar login flow.")
    st.stop()

# Init FYERS + smoke test on a single symbol
try:
    _ = init_fyers(app_id, access_token)
    # quick test call
    test_df = get_ohlc_15m(access_token, app_id, "RELIANCE", days_back=7)
    if test_df is None or test_df.empty:
        st.warning("⚠️ FYERS session OK but history returned empty for RELIANCE.")
    else:
        st.success("✅ FYERS data fetch OK")
except Exception as e:
    st.error(f"🔴 FYERS not initialized: {e}")
    st.stop()


# =========================
# Sheets connect
# =========================
ws_15m, err_15m = get_signals_worksheet("15M")
ws_daily, err_daily = get_signals_worksheet("DAILY")

cA, cB = st.columns(2)
with cA:
    st.write("**15M Sheet**")
    st.success("Connected ✅" if not err_15m else "Not connected ❌")
    if err_15m:
        st.caption(err_15m)
with cB:
    st.write("**Daily Sheet**")
    st.success("Connected ✅" if not err_daily else "Not connected ❌")
    if err_daily:
        st.caption(err_daily)

if err_15m or err_daily:
    st.stop()


# =========================
# Recent signals display
# =========================
def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Quality" not in df.columns:
        return df
    pref = st.session_state.get("quality_pref", "A only")
    if pref == "A only":
        return df[df["Quality"] == "A"].copy()
    if pref == "A + B":
        return df[df["Quality"].isin(["A", "B"])].copy()
    return df

st.subheader("🧠 Recent Signals")
dedup = not st.session_state.get("audit_mode", False)

df_recent_15m = load_recent_signals(ws_15m, st.session_state["retention_hours"], dedup=dedup)
df_recent_daily = load_recent_signals(ws_daily, st.session_state["retention_hours"], dedup=dedup)

df_recent_15m = filter_quality(df_recent_15m)
df_recent_daily = filter_quality(df_recent_daily)

c1, c2 = st.columns(2)
with c1:
    st.markdown("### 15M")
    st.dataframe(df_recent_15m, use_container_width=True) if not df_recent_15m.empty else st.info("No signals in window.")
with c2:
    st.markdown("### Daily")
    st.dataframe(df_recent_daily, use_container_width=True) if not df_recent_daily.empty else st.info("No signals in window.")

st.divider()


# =========================
# Scan + log
# =========================
with st.expander("🔎 Scan & Log (runs every refresh)", expanded=False):
    debug = st.session_state.get("debug", False)

    params = StrategyParams(
        bbw_abs_max=float(st.session_state["bbw_abs_max"]),
        bbw_pct_max=float(st.session_state["bbw_pct_max"]),
        adx_min=float(st.session_state["adx_min"]),
        rsi_bull_min=float(st.session_state["rsi_bull_min"]),
        rsi_bear_max=float(st.session_state["rsi_bear_max"]),
        breakout_lookback=int(st.session_state["breakout_lookback"]),
        require_bbw_expansion=bool(st.session_state["require_bbw_expansion"]),
        require_volume_spike=bool(st.session_state["require_volume_spike"]),
        volume_spike_mult=float(st.session_state["volume_spike_mult"]),
    )
    params_hash = params_to_hash(params)
    logged_at = fmt_dt(now_ist())

    use_confirmed = st.session_state.get("signal_mode", "").startswith("✅")
    setup_col = "setup" if use_confirmed else "setup_forming"

    existing_15m = fetch_existing_keys_recent(ws_15m, days_back=3)
    existing_daily = fetch_existing_keys_recent(ws_daily, days_back=14)

    rows_15m: List[List[Any]] = []
    rows_daily: List[List[Any]] = []

    prog = st.progress(0)
    total = len(NIFTY50)

    # Daily scan only once per date unless user is in audit/debug mood
    today = now_ist().date()
    last_daily_scan = st.session_state.get("last_daily_scan_date", None)
    run_daily = (last_daily_scan != str(today))

    for i, sym in enumerate(NIFTY50, start=1):
        prog.progress(i / total)

        # --- 15M ---
        try:
            df15 = get_ohlc_15m(access_token, app_id, sym, days_back=15)
            if df15 is not None and len(df15) >= 200:
                prep = prepare_trend_squeeze_df(df15, params)
                recent = prep.tail(int(st.session_state["catchup_15m"])).copy()
                recent = recent[recent[setup_col] != ""]
                for candle_ts, r in recent.iterrows():
                    sig_time = ts_to_sheet_str(pd.Timestamp(candle_ts), "15min")
                    setup = str(r[setup_col])
                    bias = "LONG" if setup.startswith("Bull") else "SHORT"
                    key = f"{sym}|15M|Continuation|{sig_time}|{setup}|{bias}"
                    if key in existing_15m:
                        continue

                    base = {
                        "key": key,
                        "signal_time": sig_time,
                        "logged_at": logged_at,
                        "symbol": sym,
                        "timeframe": "15M",
                        "mode": "Continuation",
                        "setup": setup,
                        "bias": bias,
                        "ltp": safe_float(r.get("close")),
                        "bbw": safe_float(r.get("bbw")),
                        "bbw_pct_rank": safe_float(r.get("bbw_pct_rank")),
                        "rsi": safe_float(r.get("rsi")),
                        "adx": safe_float(r.get("adx")),
                        "trend": str(r.get("trend", "")),
                        "quality_score": compute_quality_score(r),
                        "params_hash": params_hash,
                    }

                    enriched = add_risk_metrics(
                        base,
                        df15,
                        account_capital=float(st.session_state["capital"]),
                        risk_per_trade_pct=float(st.session_state["risk_pct"]),
                        atr_mult=float(st.session_state["atr_mult"]),
                    )

                    rows_15m.append([enriched.get(c) for c in SIGNAL_COLUMNS])
                    existing_15m.add(key)
        except Exception as e:
            if debug:
                st.warning(f"{sym} 15M error: {e}")

        # --- Daily (once per day) ---
        if run_daily:
            try:
                dfd = get_ohlc_daily(access_token, app_id, sym, lookback_days=450)
                if dfd is not None and len(dfd) >= 150:
                    prep_d = prepare_trend_squeeze_df(dfd, params)
                    recent_d = prep_d.tail(10).copy()
                    recent_d = recent_d[recent_d[setup_col] != ""]
                    for candle_ts, r in recent_d.iterrows():
                        sig_time = pd.Timestamp(candle_ts).strftime("%Y-%m-%d")
                        setup = str(r[setup_col])
                        bias = "LONG" if setup.startswith("Bull") else "SHORT"
                        key = f"{sym}|Daily|Continuation|{sig_time}|{setup}|{bias}"
                        if key in existing_daily:
                            continue

                        base = {
                            "key": key,
                            "signal_time": sig_time,
                            "logged_at": logged_at,
                            "symbol": sym,
                            "timeframe": "Daily",
                            "mode": "Continuation",
                            "setup": setup,
                            "bias": bias,
                            "ltp": safe_float(r.get("close")),
                            "bbw": safe_float(r.get("bbw")),
                            "bbw_pct_rank": safe_float(r.get("bbw_pct_rank")),
                            "rsi": safe_float(r.get("rsi")),
                            "adx": safe_float(r.get("adx")),
                            "trend": str(r.get("trend", "")),
                            "quality_score": compute_quality_score(r),
                            "params_hash": params_hash,
                        }

                        enriched = add_risk_metrics(
                            base,
                            dfd,
                            account_capital=float(st.session_state["capital"]),
                            risk_per_trade_pct=float(st.session_state["risk_pct"]),
                            atr_mult=float(st.session_state["atr_mult"]),
                        )

                        rows_daily.append([enriched.get(c) for c in SIGNAL_COLUMNS])
                        existing_daily.add(key)
            except Exception as e:
                if debug:
                    st.warning(f"{sym} Daily error: {e}")

    prog.empty()

    a15, e15 = append_signals(ws_15m, rows_15m, debug=debug)
    ad, ed = append_signals(ws_daily, rows_daily, debug=debug) if run_daily else (0, None)

    if run_daily:
        st.session_state["last_daily_scan_date"] = str(today)

    if e15:
        st.error(e15)
    if ed:
        st.error(ed)

    st.caption(f"Logged **{a15}** new 15M signals" + (f" + **{ad}** Daily signals." if run_daily else ". (Daily scan skipped — already done today.)"))

st.caption("If this feels slower than your broker’s app at 9:15, that’s because we’re doing 50×2 history calls. 😄 Reduce refresh frequency or scan fewer symbols if you hit FYERS rate limits.")
