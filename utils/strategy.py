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
) -> pd.DataFrame:
    """
    Add indicators and classify Trend + Squeeze in a single pass.

    True-ish "TTM-style" squeeze:
    - Bollinger Bands inside Keltner Channel
    - BBW in bottom X% of last N bars
    - Optional absolute BBW threshold
    - Trend filter using EMA50/EMA200 + RSI + ADX

    Returns df with extra columns:
    - ema50, ema200, rsi, adx
    - bb_mavg, bb_hband, bb_lband, bbw
    - kc_hband, kc_lband
    - bbw_pct_rank, squeeze_on
    - bull_trend, bear_trend
    - trend ("Uptrend"/"Downtrend"/"")
    - setup ("Bullish Squeeze"/"Bearish Squeeze"/"")
    """
    df = df.copy()

    # --- Core indicators ---
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_mavg"] = bb.bollinger_mavg()
    df["bb_hband"] = bb.bollinger_hband()
    df["bb_lband"] = bb.bollinger_lband()
    df["bbw"] = (df["bb_hband"] - df["bb_lband"]) / df["bb_mavg"]

    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=20,
        original_version=False,
    )
    df["kc_hband"] = kc.keltner_channel_hband()
    df["kc_lband"] = kc.keltner_channel_lband()

    # --- BBW percentile (rolling) ---
    df["bbw_pct_rank"] = (
        df["bbw"].rolling(rolling_window).rank(pct=True)
    )

    # --- True-ish squeeze condition ---
    # 1. BB inside Keltner
    # 2. BBW in bottom bbw_pct_threshold of last N bars
    # 3. Optional absolute BBW filter
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

    # --- Label trend / setup ---
    df["trend"] = np.where(
        df["bull_trend"],
        "Uptrend",
        np.where(df["bear_trend"], "Downtrend", ""),
    )

    df["setup"] = np.where(
        df["squeeze_on"] & df["bull_trend"],
        "Bullish Squeeze",
        np.where(
            df["squeeze_on"] & df["bear_trend"],
            "Bearish Squeeze",
            "",
        ),
    )

    return df
