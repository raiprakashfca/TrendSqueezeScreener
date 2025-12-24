import pandas as pd
import numpy as np
import ta


def prepare_trend_squeeze(
    df: pd.DataFrame,
    bbw_abs_threshold: float = 0.05,
    bbw_pct_threshold: float = 0.35,
    adx_threshold: float = 20.0,
    rsi_bull: float = 55.0,
    rsi_bear: float = 45.0,
    rolling_window: int = 20,
    # --- Breakout confirmation layer ---
    breakout_lookback: int = 20,
    require_bbw_expansion: bool = True,
    require_volume_spike: bool = False,
    volume_spike_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Adds indicators and classifies Trend + Squeeze, then confirms breakout.

    Outputs:
      - setup_forming: squeeze+trend present (setup forming)
      - setup: breakout-confirmed ("Bullish Breakout" / "Bearish Breakout")
      - stage: "FORMING" or "BREAKOUT"
      - range_high/range_low: consolidation box (rolling)
      - bbw_expanding, vol_sma20, vol_spike: confirmation helpers
    """

    df = df.copy()
    df = df.sort_index()

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    # --- EMAs ---
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # --- RSI / ADX ---
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()

    # --- Bollinger + BBW ---
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_hband"] = bb.bollinger_hband()
    df["bb_lband"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()
    df["bbw"] = (df["bb_hband"] - df["bb_lband"]) / df["bb_mavg"].replace(0, np.nan)

    # --- Keltner (squeeze) ---
    kc = ta.volatility.KeltnerChannel(high=df["high"], low=df["low"], close=df["close"], window=20, original_version=False)
    df["kc_hband"] = kc.keltner_channel_hband()
    df["kc_lband"] = kc.keltner_channel_lband()

    # --- BBW percentile rank ---
    df["bbw_pct_rank"] = df["bbw"].rolling(rolling_window).rank(pct=True)

    # --- Squeeze on ---
    df["squeeze_on"] = (
        (df["bbw"] < bbw_abs_threshold)
        & (df["bbw_pct_rank"] <= bbw_pct_threshold)
        & (df["bb_hband"] < df["kc_hband"])
        & (df["bb_lband"] > df["kc_lband"])
    )

    # --- Trend filters ---
    df["bull_trend"] = (
        (df["close"] > df["ema50"])
        & (df["ema50"] > df["ema200"])
        & (df["rsi"] >= rsi_bull)
        & (df["adx"] >= adx_threshold)
    )
    df["bear_trend"] = (
        (df["close"] < df["ema50"])
        & (df["ema50"] < df["ema200"])
        & (df["rsi"] <= rsi_bear)
        & (df["adx"] >= adx_threshold)
    )

    df["trend"] = np.where(df["bull_trend"], "Uptrend", np.where(df["bear_trend"], "Downtrend", ""))

    # --- Setup forming ---
    df["setup_forming"] = np.where(
        df["squeeze_on"] & df["bull_trend"],
        "Bullish Squeeze",
        np.where(df["squeeze_on"] & df["bear_trend"], "Bearish Squeeze", ""),
    )

    # --- Breakout confirmation ---
    lb = int(max(5, breakout_lookback))
    df["range_high"] = df["high"].rolling(lb).max().shift(1)
    df["range_low"] = df["low"].rolling(lb).min().shift(1)

    df["breakout_up"] = df["close"] > df["range_high"]
    df["breakout_down"] = df["close"] < df["range_low"]

    df["bbw_expanding"] = df["bbw"] > df["bbw"].shift(1)

    if "volume" in df.columns:
        df["vol_sma20"] = df["volume"].rolling(20).mean()
        df["vol_spike"] = df["volume"] > (volume_spike_mult * df["vol_sma20"])
    else:
        df["vol_sma20"] = np.nan
        df["vol_spike"] = False

    bbw_ok = df["bbw_expanding"] if require_bbw_expansion else True
    vol_ok = df["vol_spike"] if require_volume_spike else True

    df["bull_breakout"] = (df["setup_forming"] == "Bullish Squeeze") & df["breakout_up"] & bbw_ok & vol_ok
    df["bear_breakout"] = (df["setup_forming"] == "Bearish Squeeze") & df["breakout_down"] & bbw_ok & vol_ok

    df["setup"] = np.where(df["bull_breakout"], "Bullish Breakout", np.where(df["bear_breakout"], "Bearish Breakout", ""))
    df["stage"] = np.where(df["setup"] != "", "BREAKOUT", np.where(df["setup_forming"] != "", "FORMING", ""))

    return df
