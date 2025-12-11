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
from gspread.exceptions import SpreadsheetNotFound
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

# 🔁 Auto-refresh every 5 minutes (for Live tab)
st_autorefresh(interval=300000, key="auto_refresh")

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def market_open_now() -> bool:
    now = datetime.now(IST)
    return is_trading_day(now.date()) and MARKET_OPEN <= now.time() <= MARKET_CLOSE


def fetch_nifty50_symbols() -> list[str] | None:
    """
    Fetch the latest NIFTY 50 constituents from NSE's official CSV.

    Returns a list of SYMBOL strings (e.g. ['RELIANCE', 'HDFCBANK', ...])
    or None on failure.
    """
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
    except Exception as e:
        st.warning(f"Failed to download NIFTY 50 CSV from NSE: {e}")
        return None

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        st.warning(f"Failed to parse NIFTY 50 CSV: {e}")
        return None

    if "Symbol" not in df.columns:
        st.warning("NSE NIFTY 50 CSV does not contain 'Symbol' column. Format may have changed.")
        return None

    symbols = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )

    # Sort for stability
    symbols = sorted(symbols)

    if len(symbols) != 50:
        st.warning(
            f"Expected 50 NIFTY symbols, got {len(symbols)}. "
            "Proceeding, but please verify against NSE manually."
        )

    # 🔁 Post-demerger fix:
    # If NSE still shows TATAMOTORS but you trade TMPV, map it.
    if "TATAMOTORS" in symbols and "TMPV" not in symbols:
        symbols = ["TMPV" if s == "TATAMOTORS" else s for s in symbols]

    return symbols


def format_ts(ts: pd.Timestamp) -> str:
    """Format timestamps nicely, without timezone noise."""
    if isinstance(ts, pd.Timestamp):
        try:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Kolkata")
        except Exception:
            pass
        return ts.strftime("%d-%b-%Y %H:%M")
    return str(ts)


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute Supertrend (classic ATR-based) and return df with:
        - 'supertrend': line value
        - 'st_dir': +1 for bullish, -1 for bearish
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    hl2 = (high + low) / 2.0

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

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
            else:  # previous was final_lb
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
    """Ensure df has an 'rsi' column. If not, compute a basic RSI."""
    if "rsi" in df.columns:
        return df

    close = df["close"]
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df["rsi"] = rsi
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Average True Range (ATR) and add it as 'atr' column.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    df["atr"] = atr
    return df


# =========================
#   GOOGLE SHEETS HELPERS
# =========================

SIGNAL_SHEET_NAME = "TrendSqueezeSignals"
SIGNAL_COLUMNS = [
    "timestamp",     # ISO string
    "symbol",
    "mode",          # Continuation / Reversal
    "bias",          # LONG / SHORT
    "ltp",
    "bbw",
    "bbw_pct_rank",
    "rsi",
    "adx",
    "trend",
    "setup",
]


def get_signals_worksheet():
    """Get or create the Google Sheet used to store live signals."""
    try:
        service_account_info = st.secrets["gcp_service_account"]
    except Exception:
        # If secrets not present, just skip persistence gracefully
        return None

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    try:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, scope)
        client = gspread.authorize(creds)
    except Exception:
        return None

    try:
        sh = client.open(SIGNAL_SHEET_NAME)
    except SpreadsheetNotFound:
        # Create sheet in service account's Drive
        try:
            sh = client.create(SIGNAL_SHEET_NAME)
        except Exception:
            return None
    except Exception:
        return None

    ws = sh.sheet1

    # Ensure header row exists
    try:
        header = ws.row_values(1)
        if not header:
            ws.append_row(SIGNAL_COLUMNS)
    except Exception:
        # If anything weird, just try to reset header
        try:
            ws.clear()
            ws.append_row(SIGNAL_COLUMNS)
        except Exception:
            return None

    return ws


def persist_and_get_recent_signals(
    df_cont: pd.DataFrame,
    df_rev: pd.DataFrame,
    now_ist: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Persist current signals to Google Sheet and return two DataFrames:
      - recent continuation signals (last 24h, dedup by symbol+mode)
      - recent reversal signals (last 24h, dedup by symbol+mode)
    If sheet is not available, just return inputs (current scan only).
    """
    ws = get_signals_worksheet()
    if ws is None:
        # Persistence not available => behave like old version
        return df_cont, df_rev

    # Build new rows from current scan
    new_rows = []

    ts_str = now_ist.strftime("%Y-%m-%d %H:%M:%S")

    if not df_cont.empty:
        for _, r in df_cont.iterrows():
            new_rows.append(
                [
                    ts_str,
                    r["Symbol"],
                    "Continuation",
                    r["Bias"],
                    r["LTP"],
                    r["BBW"],
                    r["BBW %Rank (20)"],
                    r["RSI"],
                    r["ADX"],
                    r["Trend"],
                    r["Setup"],
                ]
            )

    if not df_rev.empty:
        for _, r in df_rev.iterrows():
            new_rows.append(
                [
                    ts_str,
                    r["Symbol"],
                    "Reversal",
                    r["Bias"],
                    r["LTP"],
                    r["BBW"],
                    r["BBW %Rank (20)"],
                    r["RSI"],
                    r["ADX"],
                    r["Trend"],
                    r["Setup"],
                ]
            )

    # Fetch existing rows
    try:
        records = ws.get_all_records()
    except Exception:
        return df_cont, df_rev

    if records:
        df_existing = pd.DataFrame(records)
    else:
        df_existing = pd.DataFrame(columns=SIGNAL_COLUMNS)

    # Append new rows to sheet (persistent store)
    if new_rows:
        try:
            ws.append_rows(new_rows)
        except Exception:
            # Even if append fails, we can still use existing for display
            pass

    # Build full DF for last 24h view (existing + newly created)
    df_new = pd.DataFrame(new_rows, columns=SIGNAL_COLUMNS) if new_rows else pd.DataFrame(columns=SIGNAL_COLUMNS)

    if df_existing.empty and df_new.empty:
        return df_cont, df_rev

    df_all = pd.concat([df_existing, df_new], ignore_index=True)

    # Parse timestamp
    try:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    except Exception:
        # If parsing fails, bail out to current-only
        return df_cont, df_rev

    # Filter last 24 hours (in IST, but timestamps stored without tz)
    cutoff = now_ist.replace(tzinfo=None) - timedelta(hours=24)
    df_recent = df_all[df_all["timestamp"] >= cutoff].copy()

    if df_recent.empty:
        return df_cont, df_rev

    # Build continuation and reversal views
    df_cont_recent = df_recent[df_recent["mode"] == "Continuation"].copy()
    df_rev_recent = df_recent[df_recent["mode"] == "Reversal"].copy()

    # Deduplicate by symbol+mode, keep latest timestamp
    if not df_cont_recent.empty:
        df_cont_recent = df_cont_recent.sort_values("timestamp", ascending=False)
        df_cont_recent = df_cont_recent.drop_duplicates(subset=["symbol", "mode"], keep="first")
        df_cont_recent["Timestamp"] = df_cont_recent["timestamp"].dt.strftime("%d-%b-%Y %H:%M")
        df_cont_recent.rename(
            columns={
                "symbol": "Symbol",
                "bias": "Bias",
                "ltp": "LTP",
                "bbw": "BBW",
                "bbw_pct_rank": "BBW %Rank (20)",
                "rsi": "RSI",
                "adx": "ADX",
                "trend": "Trend",
                "setup": "Setup",
            },
            inplace=True,
        )
        df_cont_display = df_cont_recent[
            [
                "Timestamp",
                "Symbol",
                "Bias",
                "LTP",
                "BBW",
                "BBW %Rank (20)",
                "RSI",
                "ADX",
                "Trend",
                "Setup",
            ]
        ]
    else:
        df_cont_display = pd.DataFrame(
            columns=["Timestamp", "Symbol", "Bias", "LTP", "BBW", "BBW %Rank (20)", "RSI", "ADX", "Trend", "Setup"]
        )

    if not df_rev_recent.empty:
        df_rev_recent = df_rev_recent.sort_values("timestamp", ascending=False)
        df_rev_recent = df_rev_recent.drop_duplicates(subset=["symbol", "mode"], keep="first")
        df_rev_recent["Timestamp"] = df_rev_recent["timestamp"].dt.strftime("%d-%b-%Y %H:%M")
        df_rev_recent.rename(
            columns={
                "symbol": "Symbol",
                "bias": "Bias",
                "ltp": "LTP",
                "bbw": "BBW",
                "bbw_pct_rank": "BBW %Rank (20)",
                "rsi": "RSI",
                "adx": "ADX",
                "trend": "Trend",
                "setup": "Setup",
            },
            inplace=True,
        )
        df_rev_display = df_rev_recent[
            [
                "Timestamp",
                "Symbol",
                "Bias",
                "LTP",
                "BBW",
                "BBW %Rank (20)",
                "RSI",
                "ADX",
                "Trend",
                "Setup",
            ]
        ]
    else:
        df_rev_display = pd.DataFrame(
            columns=["Timestamp", "Symbol", "Bias", "LTP", "BBW", "BBW %Rank (20)", "RSI", "ADX", "Trend", "Setup"]
        )

    return df_cont_display, df_rev_display


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
    Backtest Trend + Squeeze with:
      - Hybrid trailing stop based on Supertrend + ATR
      - RSI-based profit target
      - Time-based exit fallback

    trade_mode:
      - "Continuation": Bullish Squeeze = LONG, Bearish Squeeze = SHORT
      - "Reversal":     Bullish Squeeze = SHORT, Bearish Squeeze = LONG

    Entry: next bar’s open after signal.
    Exit: first of
      - TARGET_RSI
      - TRAIL_STOP_HYBRID (Supertrend + ATR trailing stop)
      - TIME_EXIT
    """

    # 1) Compute squeeze/trend logic
    df_prepped = prepare_trend_squeeze(
        df,
        bbw_abs_threshold=bbw_abs_threshold,
        bbw_pct_threshold=bbw_pct_threshold,
    )

    # 2) Ensure RSI
    df_prepped = ensure_rsi(df_prepped)

    # 3) Supertrend
    df_prepped = compute_supertrend(df_prepped, period=st_period, multiplier=st_mult)

    # 4) ATR
    df_prepped = compute_atr(df_prepped, period=atr_period)

    signals = df_prepped[df_prepped["setup"] != ""].copy()
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "signal_time",
                "entry_time",
                "exit_time",
                "setup",
                "direction",
                "exit_reason",
                "entry_price",
                "exit_price",
                "return_pct",
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
            continue  # can't enter on last bar

        max_exit_idx = min(entry_idx + hold_bars - 1, len(indexed_df) - 1)
        setup = signals.loc[ts, "setup"]

        # Direction depends on trade_mode
        if trade_mode == "Reversal":
            # Fade the signal: Bearish → LONG, Bullish → SHORT
            is_long = (setup == "Bearish Squeeze")
        else:
            # Continuation: follow the label
            is_long = (setup == "Bullish Squeeze")

        direction = 1 if is_long else -1

        entry_time = indexed_df.index[entry_idx]
        entry_price = indexed_df["open"].iloc[entry_idx]

        # Initial ATR & stop
        entry_atr = indexed_df["atr"].iloc[entry_idx]
        if np.isnan(entry_atr) or entry_atr <= 0:
            # skip trades with undefined ATR
            continue

        if is_long:
            trail_stop = entry_price - atr_mult * entry_atr
        else:
            trail_stop = entry_price + atr_mult * entry_atr

        exit_idx = max_exit_idx
        exit_reason = "TIME_EXIT"

        # Walk bar by bar
        for j in range(entry_idx, max_exit_idx + 1):
            row = indexed_df.iloc[j]
            c = row["close"]
            rsi = row["rsi"]
            st_line = row["supertrend"]
            atr_val = row["atr"]

            # --- Update hybrid trailing stop ---
            if not np.isnan(st_line) and not np.isnan(atr_val) and atr_val > 0:
                if is_long:
                    # Tightest stop: min(supertrend, price - ATR*mult)
                    candidate_st = st_line
                    candidate_atr = c - atr_mult * atr_val
                    proposed_stop = min(candidate_st, candidate_atr)
                    # Trail only upwards
                    trail_stop = max(trail_stop, proposed_stop)
                else:
                    # SHORT: tightest stop above price
                    candidate_st = st_line
                    candidate_atr = c + atr_mult * atr_val
                    proposed_stop = max(candidate_st, candidate_atr)
                    # Trail only downwards
                    trail_stop = min(trail_stop, proposed_stop)

            # --- Exit logic priority ---

            if is_long:
                # 1) RSI target
                if rsi >= rsi_long_target:
                    exit_idx = j
                    exit_reason = "TARGET_RSI"
                    break
                # 2) Hybrid trail stop
                if c <= trail_stop:
                    exit_idx = j
                    exit_reason = "TRAIL_STOP_HYBRID"
                    break
            else:
                # SHORT
                if rsi <= rsi_short_target:
                    exit_idx = j
                    exit_reason = "TARGET_RSI"
                    break
                if c >= trail_stop:
                    exit_idx = j
                    exit_reason = "TRAIL_STOP_HYBRID"
                    break

        exit_time = indexed_df.index[exit_idx]
        exit_price = indexed_df["close"].iloc[exit_idx]

        ret_pct = direction * (exit_price / entry_price - 1.0) * 100.0

        trades.append(
            {
                "symbol": symbol,
                "signal_time": format_ts(ts),
                "entry_time": format_ts(entry_time),
                "exit_time": format_ts(exit_time),
                "setup": setup,
                "direction": "LONG" if is_long else "SHORT",
                "exit_reason": exit_reason,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return_pct": round(ret_pct, 2),
            }
        )

    return pd.DataFrame(trades)


# =========================
#   PAGE HEADER
# =========================
st.title("📉 Trend Squeeze Screener & Backtester")
st.info("⏳ Live tab auto-refreshes every 5 minutes. Backtest tab uses historical data.")

now_ist = datetime.now(IST)
mode_str = (
    "🟢 LIVE MARKET"
    if market_open_now()
    else "🔵 EOD / OFF-MARKET (using last trading session data)"
)
st.caption(f"{mode_str} • Last refresh: {now_ist.strftime('%d %b %Y • %H:%M:%S')} IST")

# -----------------------------
# ⚙️ Sidebar controls
# -----------------------------
with st.sidebar:
    st.subheader("Settings")
    show_debug = st.checkbox("Show debug messages", value=False)

# -----------------------------
# 🔐 Zerodha credentials
# -----------------------------
api_key, api_secret, access_token = load_credentials_from_gsheet("ZerodhaTokenStore")
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# -----------------------------
# 📈 NIFTY 50 stock universe
# -----------------------------

# Fallback list in case NSE CSV fetch fails
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

# Try to fetch live NIFTY50 list from NSE
live_nifty50 = fetch_nifty50_symbols()

INVALID_SYMBOLS = {"DUMMYHDLVR", "DUMMY1", "DUMMY2", "DUMMY"}

if live_nifty50:
    clean_symbols = [
        s
        for s in live_nifty50
        if s.isalpha() and "DUMMY" not in s.upper() and s not in INVALID_SYMBOLS
    ]

    if len(clean_symbols) < len(live_nifty50) and show_debug:
        st.warning(
            "Some NSE symbols were invalid or temporary placeholders and were removed."
        )

    stock_list = clean_symbols
    st.caption(f"Universe: Latest NIFTY 50 from NSE (cleaned, {len(stock_list)} symbols).")
else:
    stock_list = fallback_nifty50
    st.caption("Universe: Fallback hard-coded NIFTY 50 list (NSE CSV unavailable).")

# -----------------------------
# 🎯 Squeeze configuration (shared by Live + Backtest)
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    bbw_abs_threshold = st.slider(
        "BBW absolute threshold (max BBW)",
        0.01,
        0.20,
        0.05,
        step=0.005,
        help="Maximum allowed BBW. Lower = tighter squeeze.",
    )
with col2:
    bbw_pct_threshold = st.slider(
        "BBW percentile threshold",
        0.10,
        0.80,
        0.35,
        step=0.05,
        help="BBW must be in the bottom X% of last 20 bars.",
    )

st.caption(
    "Squeeze = Bollinger inside Keltner AND BBW below both the absolute "
    "threshold and the rolling percentile threshold."
)

# -----------------------------
# 🧩 Instrument tokens (shared)
# -----------------------------
instrument_token_map = build_instrument_token_map(kite, stock_list)

# =========================
#   TABS: LIVE / BACKTEST
# =========================
live_tab, backtest_tab = st.tabs(["📺 Live Screener", "📜 Backtest"])


# =========================
#   LIVE SCREENER TAB
# =========================
with live_tab:
    continuation_rows: list[dict] = []
    reversal_rows: list[dict] = []

    if show_debug:
        st.info("📊 Fetching and analyzing NIFTY 50 stocks (live). Please wait...")
    else:
        st.info("📊 Scanning NIFTY 50 (live)... showing continuation and reversal candidates (last 24h).")

    for symbol in stock_list:
        token = instrument_token_map.get(symbol)
        if not token:
            if show_debug:
                st.warning(f"{symbol}: No instrument token found — skipping.")
            continue

        try:
            df = get_ohlc_15min(kite, token, show_debug=show_debug)

            if df is None or df.shape[0] < 60:
                if show_debug:
                    count = 0 if df is None else df.shape[0]
                    st.warning(f"{symbol}: Only {count} candles available — skipping.")
                continue

            df_prepped = prepare_trend_squeeze(
                df,
                bbw_abs_threshold=bbw_abs_threshold,
                bbw_pct_threshold=bbw_pct_threshold,
            )

            if df_prepped.isnull().values.any():
                if show_debug:
                    st.warning(f"{symbol}: Indicators contain NaNs — skipping.")
                continue

            latest = df_prepped.iloc[-1]

            setup = latest["setup"]
            if not setup:
                continue

            # Common fields
            base_row = {
                "Symbol": symbol,
                "LTP": round(latest["close"], 2),
                "BBW": round(latest["bbw"], 4),
                "BBW %Rank (20)": round(latest["bbw_pct_rank"], 2)
                if pd.notna(latest["bbw_pct_rank"])
                else None,
                "RSI": round(latest["rsi"], 1),
                "ADX": round(latest["adx"], 1),
                "Trend": latest["trend"],
                "Setup": setup,
            }

            # Continuation interpretation (with trend)
            cont_direction = "LONG" if setup == "Bullish Squeeze" else "SHORT"
            cont_row = base_row.copy()
            cont_row["Bias"] = cont_direction
            continuation_rows.append(cont_row)

            # Reversal interpretation (against trend)
            rev_direction = "LONG" if setup == "Bearish Squeeze" else "SHORT"
            rev_row = base_row.copy()
            rev_row["Bias"] = rev_direction
            reversal_rows.append(rev_row)

        except Exception as e:
            if show_debug:
                st.warning(f"{symbol}: Failed to fetch or compute — {str(e)}")
            continue

    # Current scan DataFrames (in case sheet is not available)
    df_cont_current = pd.DataFrame(continuation_rows) if continuation_rows else pd.DataFrame(
        columns=["Symbol", "LTP", "BBW", "BBW %Rank (20)", "RSI", "ADX", "Trend", "Setup", "Bias"]
    )
    df_rev_current = pd.DataFrame(reversal_rows) if reversal_rows else pd.DataFrame(
        columns=["Symbol", "LTP", "BBW", "BBW %Rank (20)", "RSI", "ADX", "Trend", "Setup", "Bias"]
    )

    # 🔁 Persist + show last 24h view
    df_cont_display, df_rev_display = persist_and_get_recent_signals(
        df_cont_current,
        df_rev_current,
        now_ist,
    )

    # Show continuation table
    st.markdown("### 🔵 Continuation Setups (with trend) – last 24 hours")
    if not df_cont_display.empty:
        st.success(f"{len(df_cont_display)} continuation candidates in the last 24 hours.")
        st.dataframe(df_cont_display, use_container_width=True)
    else:
        st.info("No continuation setups in the last 24 hours.")

    # Show reversal table
    st.markdown("### 🟠 Reversal Setups (against trend) – last 24 hours")
    if not df_rev_display.empty:
        st.success(f"{len(df_rev_display)} reversal candidates in the last 24 hours.")
        st.dataframe(df_rev_display, use_container_width=True)
    else:
        st.info("No reversal setups in the last 24 hours.")


# =========================
#   BACKTEST TAB
# =========================
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
                min_value=10,
                max_value=120,
                value=60,
                step=5,
                help="Number of calendar days of 15-minute history to use.",
            )
        with col_bt2:
            hold_bars = st.slider(
                "Max holding period (bars)",
                min_value=2,
                max_value=24,
                value=8,
                step=1,
                help="Maximum number of 15-minute bars to hold after entry.",
            )

        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st_period = st.slider(
                "Supertrend period",
                min_value=7,
                max_value=20,
                value=10,
                step=1,
            )
        with col_st2:
            st_mult = st.slider(
                "Supertrend multiplier",
                min_value=1.5,
                max_value=4.0,
                value=3.0,
                step=0.5,
            )

        col_atr1, col_atr2 = st.columns(2)
        with col_atr1:
            atr_period = st.slider(
                "ATR period",
                min_value=7,
                max_value=30,
                value=14,
                step=1,
            )
        with col_atr2:
            atr_mult = st.slider(
                "ATR multiple for hybrid stop",
                min_value=0.5,
                max_value=4.0,
                value=2.0,
                step=0.25,
            )

        col_rsi1, col_rsi2 = st.columns(2)
        with col_rsi1:
            rsi_long_target = st.slider(
                "RSI target for LONG",
                min_value=55,
                max_value=80,
                value=70,
                step=1,
            )
        with col_rsi2:
            rsi_short_target = st.slider(
                "RSI target for SHORT",
                min_value=20,
                max_value=45,
                value=30,
                step=1,
            )

        # 🔁 Trade mode – follow trend vs fade trend
        trade_mode_label = st.radio(
            "How to trade the squeeze?",
            options=[
                "Continuation (with trend)",
                "Reversal (against trend)",
            ],
            index=0,
            help="Reversal may work better for some stocks; test both.",
        )
        trade_mode = "Continuation" if "Continuation" in trade_mode_label else "Reversal"

        run_bt = st.button("Run Backtest")

        if run_bt:
            token = instrument_token_map.get(bt_symbol)
            if not token:
                st.error(f"No instrument token found for {bt_symbol}.")
            else:
                st.info(
                    f"Running {trade_mode.lower()} backtest for {bt_symbol} with "
                    f"{lookback_days} days of 15m data..."
                )

                df_bt = get_ohlc_15min(
                    kite,
                    token,
                    lookback_days=lookback_days,
                    min_candles=60,
                    show_debug=show_debug,
                )

                if df_bt is None or df_bt.shape[0] < 60:
                    st.error(
                        f"Not enough data to backtest {bt_symbol}. "
                        f"Got {0 if df_bt is None else df_bt.shape[0]} candles."
                    )
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
                        st.warning(
                            "No Trend Squeeze signals found in the selected period "
                            "for this symbol with current thresholds."
                        )
                    else:
                        total_trades = len(trades)
                        wins = (trades["return_pct"] > 0).sum()
                        losses = (trades["return_pct"] <= 0).sum()
                        win_rate = (wins / total_trades) * 100.0
                        avg_ret = trades["return_pct"].mean()
                        best = trades["return_pct"].max()
                        worst = trades["return_pct"].min()

                        st.success(
                            f"Generated {total_trades} trades for {bt_symbol} "
                            f"over last {lookback_days} days in {trade_mode} mode."
                        )

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total trades", total_trades)
                        m2.metric("Win rate (%)", f"{win_rate:.1f}")
                        m3.metric("Avg return / trade (%)", f"{avg_ret:.2f}")
                        m4.metric("Best / Worst (%)", f"{best:.2f} / {worst:.2f}")

                        st.markdown("#### Trade List")
                        st.dataframe(trades, use_container_width=True)
