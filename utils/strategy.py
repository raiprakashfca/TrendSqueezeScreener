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
    # --- NEW: Box breakout engine (range break without requiring TTM squeeze) ---
    engine: str = "hybrid",  # "squeeze" | "box" | "hybrid"
    box_width_pct_max: float = 0.012,  # 1.2% range width over lookback => consolidation
    require_di_confirmation: bool = True,  # uses +DI/-DI to avoid late/weak breaks
    adx_use_as_filter: bool = False,  # if True, requires ADX>=adx_threshold even for box mode (more conservative)
    # --- NEW: Anti-chase guardrails ---
    rsi_floor_short: float = 30.0,   # block shorts when RSI too low (avoid "sell the bottom")
    rsi_ceiling_long: float = 70.0,  # block longs when RSI too high (avoid "buy the top")
) -> pd.DataFrame:
    """
    Trend Squeeze + Breakout detection with an optional "Box Breakout" engine.

    Why Box engine?
      Visual consolidation is NOT always "BB inside KC" squeeze.
      Box breakout catches those earlier (like SBIN 24-Dec 13:15).

    Outputs (columns added):
      - ema50/ema200, rsi, adx, plus_di, minus_di, bbw, bbw_pct_rank
      - squeeze_on
      - range_high/range_low, box_width_pct, box_consolidating
      - breakout_up/breakout_down, bbw_expanding, vol_spike
      - setup_forming: "Bullish Squeeze"/"Bearish Squeeze"/"Bullish Box"/"Bearish Box"
      - setup: "Bullish Breakout"/"Bearish Breakout"/"Bullish Box Breakout"/"Bearish Box Breakout"
      - stage: "FORMING" or "BREAKOUT"
    """
    df = df.copy().sort_index()

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    engine = (engine or "hybrid").strip().lower()
    if engine not in {"squeeze", "box", "hybrid"}:
        engine = "hybrid"

    # --- EMAs ---
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # --- RSI / ADX / DI ---
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()

    # --- Bollinger + BBW ---
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_hband"] = bb.bollinger_hband()
    df["bb_lband"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()
    df["bbw"] = (df["bb_hband"] - df["bb_lband"]) / df["bb_mavg"].replace(0, np.nan)

    # --- Keltner (squeeze) ---
    kc = ta.volatility.KeltnerChannel(
        high=df["high"], low=df["low"], close=df["close"],
        window=20, original_version=False
    )
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

    # --- Trend filters (mainly for squeeze engine) ---
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

    # --- Setup forming (watchlist) ---
    df["setup_forming"] = ""

    if engine in {"squeeze", "hybrid"}:
        df.loc[df["squeeze_on"] & df["bull_trend"], "setup_forming"] = "Bullish Squeeze"
        df.loc[df["squeeze_on"] & df["bear_trend"], "setup_forming"] = "Bearish Squeeze"

    # --- Breakout box levels (used by BOTH engines) ---
    lb = int(max(10, breakout_lookback))
    df["range_high"] = df["high"].rolling(lb).max().shift(1)
    df["range_low"] = df["low"].rolling(lb).min().shift(1)

    df["box_width_pct"] = (df["range_high"] - df["range_low"]) / df["close"].replace(0, np.nan)
    df["box_consolidating"] = df["box_width_pct"] <= float(box_width_pct_max)

    df["breakout_up"] = df["close"] > df["range_high"]
    df["breakout_down"] = df["close"] < df["range_low"]

    # --- Confirmation helpers ---
    df["bbw_expanding"] = df["bbw"] > df["bbw"].shift(1)

    if "volume" in df.columns:
        df["vol_sma20"] = df["volume"].rolling(20).mean()
        df["vol_spike"] = df["volume"] > (float(volume_spike_mult) * df["vol_sma20"])
    else:
        df["vol_sma20"] = np.nan
        df["vol_spike"] = False

    bbw_ok = df["bbw_expanding"] if require_bbw_expansion else True
    vol_ok = df["vol_spike"] if require_volume_spike else True

    # --- Box engine forming state ---
    if engine in {"box", "hybrid"}:
        # Bullish box: consolidation while holding above EMA50; Bearish box: below EMA50
        df.loc[df["box_consolidating"] & (df["close"] >= df["ema50"]), "setup_forming"] = df["setup_forming"].where(
            df["setup_forming"] != "", "Bullish Box"
        )
        df.loc[df["box_consolidating"] & (df["close"] < df["ema50"]), "setup_forming"] = df["setup_forming"].where(
            df["setup_forming"] != "", "Bearish Box"
        )

    # --- Breakout rules ---
    df["setup"] = ""

    # Squeeze breakout
    if engine in {"squeeze", "hybrid"}:
        df.loc[
            (df["setup_forming"] == "Bullish Squeeze") & df["breakout_up"] & bbw_ok & vol_ok & (df["rsi"] <= rsi_ceiling_long),
            "setup"
        ] = "Bullish Breakout"

        df.loc[
            (df["setup_forming"] == "Bearish Squeeze") & df["breakout_down"] & bbw_ok & vol_ok & (df["rsi"] >= rsi_floor_short),
            "setup"
        ] = "Bearish Breakout"

    # Box breakout (range break)
    if engine in {"box", "hybrid"}:
        di_bull_ok = (df["plus_di"] >= df["minus_di"]) if require_di_confirmation else True
        di_bear_ok = (df["minus_di"] >= df["plus_di"]) if require_di_confirmation else True
        adx_ok = (df["adx"] >= adx_threshold) if adx_use_as_filter else True

        df.loc[
            df["box_consolidating"] & df["breakout_up"] & bbw_ok & vol_ok & di_bull_ok & adx_ok & (df["rsi"] <= rsi_ceiling_long),
            "setup"
        ] = df["setup"].where(df["setup"] != "", "Bullish Box Breakout")

        df.loc[
            df["box_consolidating"] & df["breakout_down"] & bbw_ok & vol_ok & di_bear_ok & adx_ok & (df["rsi"] >= rsi_floor_short),
            "setup"
        ] = df["setup"].where(df["setup"] != "", "Bearish Box Breakout")

    df["stage"] = np.where(df["setup"] != "", "BREAKOUT", np.where(df["setup_forming"] != "", "FORMING", ""))
    return df
