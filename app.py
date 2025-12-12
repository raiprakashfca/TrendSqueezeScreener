import io
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh

import gspread
from gspread.exceptions import SpreadsheetNotFound, APIError
from oauth2client.service_account import ServiceAccountCredentials

from utils.token_utils import load_credentials_from_gsheet
from utils.zerodha_utils import (
    get_ohlc_15min,
    build_instrument_token_map,
    is_trading_day,
)
from utils.strategy import prepare_trend_squeeze

# ✅ MUST BE FIRST Streamlit call
st.set_page_config(page_title="📉 Trend Squeeze Screener", layout="wide")
st_autorefresh(interval=300000, key="auto_refresh")  # 5 min refresh

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

SIGNAL_SHEET_NAME = "TrendSqueezeSignals"

# ✅ new schema (includes KEY + candle-time)
SIGNAL_COLUMNS = [
    "key",              # unique dedup key
    "signal_time",      # candle timestamp (IST naive string)
    "logged_at",        # when app logged it (IST naive string)
    "symbol",
    "mode",             # Continuation / Reversal
    "setup",            # Bullish Squeeze / Bearish Squeeze
    "bias",             # LONG / SHORT
    "ltp",
    "bbw",
    "bbw_pct_rank",
    "rsi",
    "adx",
    "trend",
]


def now_ist() -> datetime:
    return datetime.now(IST)


def market_open_now() -> bool:
    n = now_ist()
    return is_trading_day(n.date()) and MARKET_OPEN <= n.time() <= MARKET_CLOSE


def fmt_dt(dt: datetime) -> str:
    # store sheet timestamps without tz noise
    return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def fmt_ts(ts) -> str:
    # convert pandas timestamp to nice display string
    if isinstance(ts, pd.Timestamp):
        try:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Kolkata")
        except Exception:
            pass
        return ts.strftime("%d-%b-%Y %H:%M")
    return str(ts)


def ts_to_sheet_str(ts) -> str:
    # store as YYYY-MM-DD HH:MM:SS (IST naive)
    if isinstance(ts, pd.Timestamp):
        try:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Kolkata")
        except Exception:
            pass
        return ts.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S") if ts.tzinfo is None else ts.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


def fetch_nifty50_symbols() -> list[str] | None:
    """Fetch latest NIFTY50 constituents from NSE CSV."""
    nifty50_csv_url = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Referer": "https://www.nseindia.com/",
        "Accept": "text/csv,application/vnd.ms-excel,application/octet-stream;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(nifty50_csv_url, headers=headers, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        st.warning(f"Failed to fetch/parse NSE NIFTY50 list: {e}")
        return None

    if "Symbol" not in df.columns:
        st.warning("NSE NIFTY50 CSV format changed (no 'Symbol' column).")
        return None

    symbols = sorted(df["Symbol"].astype(str).str.strip().dropna().unique().tolist())

    # Mapping you requested
    if "TATAMOTORS" in symbols and "TMPV" not in symbols:
        symbols = ["TMPV" if s == "TATAMOTORS" else s for s in symbols]

    # Remove obvious garbage
    INVALID = {"DUMMYHDLVR", "DUMMY1", "DUMMY2", "DUMMY"}
    symbols = [s for s in symbols if s.isalpha() and s not in INVALID and "DUMMY" not in s.upper()]

    return symbols


# ======================
#  Indicators for backtest exits
# ======================

def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    hl2 = (high + low) / 2.0
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean()

    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    for i in range(1, len(df)):
        if basic_ub.iloc[i] < final_ub.iloc[i - 1] or close.iloc[i - 1] > final_ub.iloc[i - 1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i - 1]

        if basic_lb.iloc[i] > final_lb.iloc[i - 1] or close.iloc[i - 1] < final_lb.iloc[i - 1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i - 1]

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = final_ub.iloc[i]
            direction.iloc[i] = -1
        else:
            if supertrend.iloc[i - 1] == final_ub.iloc[i - 1]:
                if close.iloc[i] <= final_ub.iloc[i]:
                    supertrend.iloc[i] = final_ub.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = final_lb.iloc[i]
                    direction.iloc[i] = 1
            else:
                if close.iloc[i] >= final_lb.iloc[i]:
                    supertrend.iloc[i] = final_lb.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = final_ub.iloc[i]
                    direction.iloc[i] = -1

    df["supertrend"] = supertrend
    df["st_dir"] = direction
    return df


def ensure_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if "rsi" in df.columns:
        return df

    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.rolling(period).mean()
    return df


# ======================
#  Google Sheets helpers (robust)
# ======================

def get_gspread_client():
    try:
        sa = st.secrets["gcp_service_account"]
    except Exception:
        return None, "Missing st.secrets['gcp_service_account']"

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


def get_signals_worksheet():
    client, err = get_gspread_client()
    if err:
        return None, err

    try:
        sh = client.open(SIGNAL_SHEET_NAME)
    except SpreadsheetNotFound:
        return None, (
            f"Sheet '{SIGNAL_SHEET_NAME}' not found. Create it in Google Drive and share with the service account as Editor."
        )
    except Exception as e:
        return None, f"Failed to open sheet '{SIGNAL_SHEET_NAME}': {e}"

    try:
        ws = sh.sheet1
    except Exception as e:
        return None, f"Failed to access sheet1: {e}"

    # Ensure headers
    try:
        header = ws.row_values(1)
        if not header:
            ws.append_row(SIGNAL_COLUMNS)
        else:
            # If existing header differs, we won't destroy it; we just warn.
            if header[: len(SIGNAL_COLUMNS)] != SIGNAL_COLUMNS:
                # Attempt safe fix only if header is empty-ish
                if len(header) < len(SIGNAL_COLUMNS):
                    ws.update("A1", [SIGNAL_COLUMNS])
    except Exception as e:
        return None, f"Failed to ensure headers: {e}"

    return ws, None


def fetch_existing_keys_recent(ws, days_back: int = 3) -> set:
    """
    Pull recent keys to dedup without reading entire sheet forever.
    If sheet is small, get_all_records is fine; else we filter by time.
    """
    try:
        records = ws.get_all_records()
    except Exception:
        return set()

    if not records:
        return set()

    df = pd.DataFrame(records)
    if "key" not in df.columns or "signal_time" not in df.columns:
        # legacy data; no dedup keys
        return set(df.get("key", pd.Series(dtype=str)).astype(str).tolist())

    # parse signal_time
    try:
        df["signal_time_dt"] = pd.to_datetime(df["signal_time"], errors="coerce")
    except Exception:
        df["signal_time_dt"] = pd.NaT

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
        if show_debug:
            return 0, f"Sheets APIError: {e}"
        return 0, "Sheets write failed (APIError)."
    except Exception as e:
        if show_debug:
            return 0, f"Sheets write failed: {e}"
        return 0, "Sheets write failed."


def load_recent_signals(ws, hours: int = 24) -> pd.DataFrame:
    try:
        records = ws.get_all_records()
    except Exception:
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Ensure expected columns exist
    for c in SIGNAL_COLUMNS:
        if c not in df.columns:
            df[c] = None

    # Parse signal_time
    df["signal_time_dt"] = pd.to_datetime(df["signal_time"], errors="coerce")
    df = df[df["signal_time_dt"].notna()]

    cutoff = now_ist().replace(tzinfo=None) - timedelta(hours=hours)
    df = df[df["signal_time_dt"] >= cutoff].copy()
    if df.empty:
        return df

    # display-friendly
    df["Timestamp"] = df["signal_time_dt"].dt.strftime("%d-%b-%Y %H:%M")
    df.rename(
        columns={
            "symbol": "Symbol",
            "mode": "Mode",
            "setup": "Setup",
            "bias": "Bias",
            "ltp": "LTP",
            "bbw": "BBW",
            "bbw_pct_rank": "BBW %Rank (20)",
            "rsi": "RSI",
            "adx": "ADX",
            "trend": "Trend",
        },
        inplace=True,
    )

    # Dedup display: keep latest signal per Symbol+Mode
    df = df.sort_values("signal_time_dt", ascending=False)
    df = df.drop_duplicates(subset=["Symbol", "Mode"], keep="first")

    cols = [
        "Timestamp",
        "Symbol",
        "Mode",
        "Bias",
        "LTP",
        "BBW",
        "BBW %Rank (20)",
        "RSI",
        "ADX",
        "Trend",
        "Setup",
    ]
    return df[cols]


# ======================
#  Backtest (coherent signal timestamps)
# ======================

def backtest_trend_squeeze(
    df: pd.DataFrame,
    symbol: str,
    hold_bars: int,
    bbw_abs_threshold: float,
    bbw_pct_threshold: float,
    st_period: int,
    st_mult: float,
    rsi_long_target: float,
    rsi_short_target: float,
    trade_mode: str = "Continuation",
    atr_period: int = 14,
    atr_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Coherent backtest:
      - Signals are taken from df_prepped where setup != "" (same as Live logic)
      - Entry: next candle open
      - Exits:
          1) RSI target
          2) Supertrend+ATR hybrid trailing stop
          3) Time exit
    """

    df_prepped = prepare_trend_squeeze(
        df,
        bbw_abs_threshold=bbw_abs_threshold,
        bbw_pct_threshold=bbw_pct_threshold,
    )
    df_prepped = ensure_rsi(df_prepped)
    df_prepped = compute_supertrend(df_prepped, period=st_period, multiplier=st_mult)
    df_prepped = compute_atr(df_prepped, period=atr_period)

    signals = df_prepped[df_prepped["setup"] != ""].copy()
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "symbol", "signal_time", "entry_time", "exit_time",
                "setup", "direction", "exit_reason",
                "entry_price", "exit_price", "return_pct",
            ]
        )

    trades = []
    indexed_df = df_prepped.copy()

    for ts in signals.index:
        try:
            idx = indexed_df.index.get_loc(ts)
        except KeyError:
            continue

        entry_idx = idx + 1
        if entry_idx >= len(indexed_df):
            continue

        max_exit_idx = min(entry_idx + hold_bars - 1, len(indexed_df) - 1)
        setup = signals.loc[ts, "setup"]

        # Direction mapping must match Live exactly
        if trade_mode == "Reversal":
            is_long = (setup == "Bearish Squeeze")
        else:
            is_long = (setup == "Bullish Squeeze")

        direction = 1 if is_long else -1

        entry_time = indexed_df.index[entry_idx]
        entry_price = indexed_df["open"].iloc[entry_idx]

        entry_atr = indexed_df["atr"].iloc[entry_idx]
        if np.isnan(entry_atr) or entry_atr <= 0:
            continue

        # initial hybrid stop using ATR
        trail_stop = (
            entry_price - atr_mult * entry_atr
            if is_long
            else entry_price + atr_mult * entry_atr
        )

        exit_idx = max_exit_idx
        exit_reason = "TIME_EXIT"

        for j in range(entry_idx, max_exit_idx + 1):
            row = indexed_df.iloc[j]
            c = row["close"]
            rsi = row["rsi"]
            st_line = row["supertrend"]
            atr_val = row["atr"]

            # update hybrid stop
            if pd.notna(st_line) and pd.notna(atr_val) and atr_val > 0:
                if is_long:
                    proposed = min(st_line, c - atr_mult * atr_val)
                    trail_stop = max(trail_stop, proposed)  # only tighter upward
                else:
                    proposed = max(st_line, c + atr_mult * atr_val)
                    trail_stop = min(trail_stop, proposed)  # only tighter downward

            # exit priority
            if is_long:
                if rsi >= rsi_long_target:
                    exit_idx, exit_reason = j, "TARGET_RSI"
                    break
                if c <= trail_stop:
                    exit_idx, exit_reason = j, "TRAIL_STOP_HYBRID"
                    break
            else:
                if rsi <= rsi_short_target:
                    exit_idx, exit_reason = j, "TARGET_RSI"
                    break
                if c >= trail_stop:
                    exit_idx, exit_reason = j, "TRAIL_STOP_HYBRID"
                    break

        exit_time = indexed_df.index[exit_idx]
        exit_price = indexed_df["close"].iloc[exit_idx]
        ret_pct = direction * (exit_price / entry_price - 1.0) * 100.0

        trades.append(
            {
                "symbol": symbol,
                "signal_time": fmt_ts(ts),
                "entry_time": fmt_ts(entry_time),
                "exit_time": fmt_ts(exit_time),
                "setup": setup,
                "direction": "LONG" if is_long else "SHORT",
                "exit_reason": exit_reason,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return_pct": round(ret_pct, 2),
            }
        )

    return pd.DataFrame(trades)


# ======================
#  UI Header
# ======================

st.title("📉 Trend Squeeze Screener & Backtester")

n = now_ist()
mode_str = "🟢 LIVE MARKET" if market_open_now() else "🔵 OFF-MARKET"
st.caption(f"{mode_str} • Last refresh: {n.strftime('%d %b %Y • %H:%M:%S')} IST")

with st.sidebar:
    st.subheader("Settings")
    show_debug = st.checkbox("Show debug messages", value=False)

    st.markdown("---")
    st.subheader("Live coherence")
    catchup_candles = st.slider(
        "Catch-up window (candles)",
        min_value=8,
        max_value=64,
        value=32,
        step=4,
        help="Live will scan the last N 15m candles and log any setups it finds (deduped).",
    )
    retention_hours = st.slider(
        "Keep signals for (hours)",
        min_value=6,
        max_value=48,
        value=24,
        step=6,
        help="Display window for signals from Google Sheet.",
    )

st.markdown(
    "This app is now **coherent**: Live detects setups on candle timestamps (same as backtest) and stores them for viewing."
)

# ======================
# Zerodha auth
# ======================

api_key, api_secret, access_token = load_credentials_from_gsheet("ZerodhaTokenStore")
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# ======================
# Universe
# ======================

fallback_nifty50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE",
    "BAJAJFINSV","BHARTIARTL","BPCL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY",
    "EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK",
    "INFY","ITC","JSWSTEEL","KOTAKBANK","LT","LTIM","M&M","MARUTI","NESTLEIND","NTPC","ONGC",
    "POWERGRID","RELIANCE","SBIN","SBILIFE","SUNPHARMA","TATACONSUM","TMPV","TATASTEEL","TCS",
    "TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HEROMOTOCO","SHREECEM"
]

symbols_live = fetch_nifty50_symbols()
stock_list = symbols_live if symbols_live and len(symbols_live) >= 45 else fallback_nifty50
st.caption(f"Universe: NIFTY50 ({len(stock_list)} symbols).")

# ======================
# Shared squeeze thresholds
# ======================

c1, c2 = st.columns(2)
with c1:
    bbw_abs_threshold = st.slider(
        "BBW absolute threshold (max BBW)",
        0.01, 0.20, 0.05, step=0.005,
        help="Lower = tighter squeeze."
    )
with c2:
    bbw_pct_threshold = st.slider(
        "BBW percentile threshold",
        0.10, 0.80, 0.35, step=0.05,
        help="BBW must be in the bottom X% of last 20 bars."
    )

# ======================
# Tokens
# ======================

instrument_token_map = build_instrument_token_map(kite, stock_list)

# ======================
# Tabs
# ======================

live_tab, backtest_tab = st.tabs(["📺 Live Screener", "📜 Backtest"])

# ======================
# LIVE TAB (catch-up scan + dedup + sheet health)
# ======================

with live_tab:
    st.subheader("📺 Live Screener (Catch-up + 24h memory)")

    ws, ws_err = get_signals_worksheet()
    health_col1, health_col2 = st.columns([1, 4])

    with health_col1:
        if ws_err:
            st.error("Sheets ❌")
        else:
            st.success("Sheets ✅")

    with health_col2:
        if ws_err:
            st.caption(ws_err)
        else:
            st.caption(f"Using Google Sheet: **{SIGNAL_SHEET_NAME}** (sheet1)")

    # If sheet is unavailable, we still show current-detection; but coherence requires sheet.
    existing_keys = set()
    if ws and not ws_err:
        existing_keys = fetch_existing_keys_recent(ws, days_back=3)

    # --- Scan & build signal rows (from candle timestamps) ---
    rows_to_append = []
    signals_found = 0

    logged_at = fmt_dt(now_ist())

    for symbol in stock_list:
        token = instrument_token_map.get(symbol)
        if not token:
            if show_debug:
                st.warning(f"{symbol}: No instrument token found — skipping.")
            continue

        try:
            df = get_ohlc_15min(kite, token, show_debug=False)
            if df is None or df.shape[0] < 60:
                continue

            df_prepped = prepare_trend_squeeze(
                df,
                bbw_abs_threshold=bbw_abs_threshold,
                bbw_pct_threshold=bbw_pct_threshold,
            )

            # Scan last N candles to avoid missing setups when app wasn't open
            recent = df_prepped.tail(int(catchup_candles)).copy()
            recent = recent[recent["setup"] != ""]
            if recent.empty:
                continue

            for candle_ts, r in recent.iterrows():
                # Candle timestamp is the truth
                signal_time = ts_to_sheet_str(candle_ts)

                setup = r["setup"]
                trend = r.get("trend", "")
                ltp = float(r.get("close", np.nan))
                bbw = float(r.get("bbw", np.nan))
                bbw_rank = r.get("bbw_pct_rank", np.nan)
                rsi = float(r.get("rsi", np.nan))
                adx = float(r.get("adx", np.nan))

                # Continuation
                cont_bias = "LONG" if setup == "Bullish Squeeze" else "SHORT"
                cont_key = f"{symbol}|Continuation|{signal_time}|{setup}|{cont_bias}"
                if cont_key not in existing_keys:
                    rows_to_append.append([
                        cont_key, signal_time, logged_at, symbol, "Continuation", setup, cont_bias,
                        ltp, bbw, bbw_rank, rsi, adx, trend
                    ])
                    existing_keys.add(cont_key)
                    signals_found += 1

                # Reversal
                rev_bias = "LONG" if setup == "Bearish Squeeze" else "SHORT"
                rev_key = f"{symbol}|Reversal|{signal_time}|{setup}|{rev_bias}"
                if rev_key not in existing_keys:
                    rows_to_append.append([
                        rev_key, signal_time, logged_at, symbol, "Reversal", setup, rev_bias,
                        ltp, bbw, bbw_rank, rsi, adx, trend
                    ])
                    existing_keys.add(rev_key)
                    signals_found += 1

        except Exception as e:
            if show_debug:
                st.warning(f"{symbol}: error -> {e}")
            continue

    # Append signals (if possible)
    appended = 0
    append_err = None
    if ws and not ws_err:
        appended, append_err = append_signals(ws, rows_to_append, show_debug=show_debug)

    if ws_err:
        st.warning(
            "Google Sheet not available, so Live will only show what it detects right now. "
            "Coherence requires sheet access."
        )
    else:
        if append_err:
            st.warning(f"Could not write signals to sheet: {append_err}")
        else:
            st.caption(f"Logged **{appended}** new signals this refresh (deduped).")

    # Load and show last 24h window (the user's monitoring truth)
    if ws and not ws_err:
        df_recent = load_recent_signals(ws, hours=int(retention_hours))
    else:
        df_recent = pd.DataFrame()

    st.markdown(f"### 🧠 Signals in the last {int(retention_hours)} hours (deduped by Symbol+Mode)")
    if df_recent is None or df_recent.empty:
        st.info("No signals in the selected window.")
    else:
        # Split into continuation & reversal for clarity
        df_cont = df_recent[df_recent["Mode"] == "Continuation"].copy()
        df_rev = df_recent[df_recent["Mode"] == "Reversal"].copy()

        st.markdown("#### 🔵 Continuation (with trend)")
        if df_cont.empty:
            st.info("No continuation signals in the selected window.")
        else:
            st.dataframe(df_cont, use_container_width=True)

        st.markdown("#### 🟠 Reversal (against trend)")
        if df_rev.empty:
            st.info("No reversal signals in the selected window.")
        else:
            st.dataframe(df_rev, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Why this fixes your mismatch: Live now scans the last N candles every refresh and logs signals by candle timestamp. "
        "Backtest and Live are now referring to the same events."
    )

# ======================
# BACKTEST TAB
# ======================

with backtest_tab:
    st.subheader("📜 Trend Squeeze Backtest (15-minute)")

    if not instrument_token_map:
        st.error("No instrument tokens resolved. Cannot run backtest.")
    else:
        bt_symbol = st.selectbox("Select symbol to backtest", sorted(stock_list))

        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            lookback_days = st.slider(
                "Lookback period (days)",
                min_value=10, max_value=120, value=60, step=5,
                help="Calendar days of 15-minute history."
            )
        with col_bt2:
            hold_bars = st.slider(
                "Max holding period (bars)",
                min_value=2, max_value=24, value=8, step=1,
                help="15-minute bars to hold after entry."
            )

        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st_period = st.slider("Supertrend period", 7, 20, 10, 1)
        with col_st2:
            st_mult = st.slider("Supertrend multiplier", 1.5, 4.0, 3.0, 0.5)

        col_atr1, col_atr2 = st.columns(2)
        with col_atr1:
            atr_period = st.slider("ATR period", 7, 30, 14, 1)
        with col_atr2:
            atr_mult = st.slider("ATR multiple for hybrid stop", 0.5, 4.0, 2.0, 0.25)

        col_rsi1, col_rsi2 = st.columns(2)
        with col_rsi1:
            rsi_long_target = st.slider("RSI target for LONG", 55, 80, 70, 1)
        with col_rsi2:
            rsi_short_target = st.slider("RSI target for SHORT", 20, 45, 30, 1)

        trade_mode_label = st.radio(
            "How to trade the squeeze?",
            options=["Continuation (with trend)", "Reversal (against trend)"],
            index=0,
        )
        trade_mode = "Continuation" if "Continuation" in trade_mode_label else "Reversal"

        if st.button("Run Backtest"):
            token = instrument_token_map.get(bt_symbol)
            if not token:
                st.error(f"No instrument token found for {bt_symbol}.")
            else:
                st.info(f"Backtesting {bt_symbol} ({trade_mode}) with last {lookback_days} days 15m data...")

                df_bt = get_ohlc_15min(
                    kite,
                    token,
                    lookback_days=lookback_days,
                    min_candles=60,
                    show_debug=False,
                )

                if df_bt is None or df_bt.shape[0] < 60:
                    st.error("Not enough data to run backtest.")
                else:
                    trades = backtest_trend_squeeze(
                        df_bt,
                        bt_symbol,
                        hold_bars=hold_bars,
                        bbw_abs_threshold=bbw_abs_threshold,
                        bbw_pct_threshold=bbw_pct_threshold,
                        st_period=st_period,
                        st_mult=st_mult,
                        rsi_long_target=rsi_long_target,
                        rsi_short_target=rsi_short_target,
                        trade_mode=trade_mode,
                        atr_period=atr_period,
                        atr_mult=atr_mult,
                    )

                    if trades.empty:
                        st.warning("No signals found in the selected period with current thresholds.")
                    else:
                        total = len(trades)
                        wins = (trades["return_pct"] > 0).sum()
                        win_rate = (wins / total) * 100.0
                        avg_ret = trades["return_pct"].mean()
                        best = trades["return_pct"].max()
                        worst = trades["return_pct"].min()

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total trades", total)
                        m2.metric("Win rate (%)", f"{win_rate:.1f}")
                        m3.metric("Avg return / trade (%)", f"{avg_ret:.2f}")
                        m4.metric("Best / Worst (%)", f"{best:.2f} / {worst:.2f}")

                        st.markdown("#### Trade List")
                        st.dataframe(trades, use_container_width=True)
