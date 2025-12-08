import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

from utils.strategy import prepare_trend_squeeze


# --- CONFIGURATION ---

# Map symbols to CSV paths (15m candles, at least 6–12 months of data recommended)
# You can change this to your own folder structure.
SYMBOL_FILES: Dict[str, str] = {
    # Example:
    # "HDFCBANK": "data/HDFCBANK_15min.csv",
    # "RELIANCE": "data/RELIANCE_15min.csv",
}

HOLD_BARS = 8  # how many bars to hold after entry
MIN_BARS = 100  # minimum bars to run indicators

BBW_ABS_THRESHOLD = 0.05
BBW_PCT_THRESHOLD = 0.35
ADX_THRESHOLD = 20.0
RSI_BULL = 55.0
RSI_BEAR = 45.0
ROLLING_WINDOW = 20


def load_ohlcv(csv_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV.
    Expected columns: datetime (or date), open, high, low, close, volume.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(path)

    # Try to infer datetime column
    datetime_col = None
    for candidate in ["datetime", "date", "timestamp"]:
        if candidate in df.columns:
            datetime_col = candidate
            break

    if datetime_col is None:
        raise ValueError(
            f"No datetime column found in {csv_path}. "
            "Expected one of: datetime, date, timestamp."
        )

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)
    df.set_index(datetime_col, inplace=True)

    # Keep only required columns
    df = df[["open", "high", "low", "close", "volume"]]

    if df.shape[0] < MIN_BARS:
        raise ValueError(f"Not enough bars in {csv_path}: {df.shape[0]}")

    return df


def generate_signals(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Run trend-squeeze preparation and return rows where a setup is present.
    """
    df_prepped = prepare_trend_squeeze(
        df,
        bbw_abs_threshold=BBW_ABS_THRESHOLD,
        bbw_pct_threshold=BBW_PCT_THRESHOLD,
        adx_threshold=ADX_THRESHOLD,
        rsi_bull=RSI_BULL,
        rsi_bear=RSI_BEAR,
        rolling_window=ROLLING_WINDOW,
    )

    # Drop rows where indicators are incomplete
    df_prepped = df_prepped.dropna(subset=["setup"])

    # Mark signal rows
    signal_rows = df_prepped[df_prepped["setup"] != ""].copy()
    signal_rows["symbol"] = symbol
    return signal_rows


def backtest_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Basic time-based backtest:
    - Entry on next bar open after signal
    - Exit after HOLD_BARS bars (or last bar)
    - Long for Bullish Squeeze, short for Bearish Squeeze
    """
    signals = generate_signals(df, symbol)

    trades: List[dict] = []
    indexed_df = df.copy()

    for ts in signals.index:
        # Bar index of the signal
        try:
            idx = indexed_df.index.get_loc(ts)
        except KeyError:
            continue

        entry_idx = idx + 1
        if entry_idx >= len(indexed_df):
            continue  # can't enter after last bar

        exit_idx = min(entry_idx + HOLD_BARS - 1, len(indexed_df) - 1)

        entry_time = indexed_df.index[entry_idx]
        exit_time = indexed_df.index[exit_idx]

        entry_price = indexed_df["open"].iloc[entry_idx]
        exit_price = indexed_df["close"].iloc[exit_idx]

        setup = signals.loc[ts, "setup"]
        direction = 1 if setup == "Bullish Squeeze" else -1

        ret_pct = direction * (exit_price / entry_price - 1.0) * 100.0

        trades.append(
            {
                "symbol": symbol,
                "signal_time": ts,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "setup": setup,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": "LONG" if direction == 1 else "SHORT",
                "return_pct": ret_pct,
            }
        )

    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("No trades generated.")
        return

    total_trades = len(trades)
    wins = (trades["return_pct"] > 0).sum()
    losses = (trades["return_pct"] <= 0).sum()
    win_rate = wins / total_trades * 100.0

    avg_ret = trades["return_pct"].mean()
    median_ret = trades["return_pct"].median()
    best = trades["return_pct"].max()
    worst = trades["return_pct"].min()

    print("========== Trend Squeeze Backtest Summary ==========")
    print(f"Total trades: {total_trades}")
    print(f"Wins: {wins} | Losses: {losses} | Win rate: {win_rate:.1f}%")
    print(f"Avg return/trade: {avg_ret:.2f}%")
    print(f"Median return/trade: {median_ret:.2f}%")
    print(f"Best trade: {best:.2f}%")
    print(f"Worst trade: {worst:.2f}%")
    print("\nBy symbol:")
    by_symbol = (
        trades.groupby("symbol")["return_pct"]
        .agg(["count", "mean", "max", "min"])
        .sort_values("mean", ascending=False)
    )
    print(by_symbol)


def main():
    all_trades = []

    if not SYMBOL_FILES:
        print(
            "SYMBOL_FILES is empty. Please configure symbol -> CSV path mapping "
            "at the top of backtest_trend_squeeze.py."
        )
        return

    for symbol, path in SYMBOL_FILES.items():
        print(f"\n=== Processing {symbol} from {path} ===")
        try:
            df = load_ohlcv(path)
        except Exception as e:
            print(f"Failed to load {symbol}: {e}")
            continue

        trades = backtest_symbol(df, symbol)
        print(f"Generated {len(trades)} trades for {symbol}")
        all_trades.append(trades)

    if not all_trades:
        print("No trades generated for any symbol.")
        return

    trades_all = pd.concat(all_trades, ignore_index=True)
    summarize_trades(trades_all)


if __name__ == "__main__":
    main()
