"""
Risk Calculator Module
Calculates position sizing, stop-loss levels, and risk metrics for trading signals.
"""

import pandas as pd
import numpy as np


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) - measures stock volatility.
    
    Simple explanation: ATR tells you how much a stock typically moves per candle.
    High ATR = volatile stock (moves a lot), Low ATR = stable stock (moves less).
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Number of candles to average (default 14 is standard)
    
    Returns:
        ATR value as float
    """
    if df is None or len(df) < period + 1:
        return np.nan
    
    # True Range is the largest of:
    # 1. Current high - current low
    # 2. Current high - previous close
    # 3. Previous close - current low
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else np.nan


def calculate_stop_loss(
    entry_price: float,
    bias: str,
    atr_value: float,
    atr_multiplier: float = 2.0,
    recent_swing_low: float = None,
    recent_swing_high: float = None,
    use_swing_points: bool = True
) -> dict:
    """
    Calculate stop-loss level for a trade.
    
    Two methods:
    1. ATR-based: Stop is X times ATR away from entry (dynamic, adapts to volatility)
    2. Swing-based: Stop below recent low for LONG, above recent high for SHORT (chart-based)
    
    Args:
        entry_price: Price where you enter the trade (LTP from signal)
        bias: "LONG" or "SHORT"
        atr_value: Average True Range value
        atr_multiplier: How many ATRs away to place stop (2.0 = 2 times ATR)
        recent_swing_low: Lowest price in recent candles (for LONG trades)
        recent_swing_high: Highest price in recent candles (for SHORT trades)
        use_swing_points: If True, uses swing points; if False, uses ATR only
    
    Returns:
        Dictionary with stop_loss price and method used
    """
    if pd.isna(entry_price) or entry_price <= 0:
        return {"stop_loss": None, "method": "invalid_entry_price"}
    
    if pd.isna(atr_value) or atr_value <= 0:
        return {"stop_loss": None, "method": "invalid_atr"}
    
    bias = bias.upper()
    
    # ATR-based stop loss
    atr_stop = entry_price - (atr_multiplier * atr_value) if bias == "LONG" else entry_price + (atr_multiplier * atr_value)
    
    # If swing points provided and enabled, use the tighter (more conservative) stop
    if use_swing_points:
        if bias == "LONG" and recent_swing_low is not None and not pd.isna(recent_swing_low):
            # For LONG: stop below recent swing low
            swing_stop = recent_swing_low * 0.995  # 0.5% buffer below swing low
            # Use tighter stop (closer to entry = less risk)
            stop_loss = max(swing_stop, atr_stop)
            method = "swing_based" if stop_loss == swing_stop else "atr_based"
        elif bias == "SHORT" and recent_swing_high is not None and not pd.isna(recent_swing_high):
            # For SHORT: stop above recent swing high
            swing_stop = recent_swing_high * 1.005  # 0.5% buffer above swing high
            # Use tighter stop (closer to entry = less risk)
            stop_loss = min(swing_stop, atr_stop)
            method = "swing_based" if stop_loss == swing_stop else "atr_based"
        else:
            stop_loss = atr_stop
            method = "atr_based"
    else:
        stop_loss = atr_stop
        method = "atr_based"
    
    return {
        "stop_loss": round(stop_loss, 2),
        "method": method,
        "atr_stop": round(atr_stop, 2)
    }


def calculate_position_size(
    account_capital: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss: float,
    bias: str
) -> dict:
    """
    Calculate how many shares to buy/sell based on your risk tolerance.
    
    The Kelly Criterion approach: Never risk more than X% of your capital on one trade.
    
    Example: 
    - Account: ₹10,00,000
    - Risk per trade: 1% = ₹10,000
    - Entry: ₹100, Stop: ₹95
    - Risk per share: ₹5
    - Position size: ₹10,000 / ₹5 = 2000 shares = ₹2,00,000 position
    
    Args:
        account_capital: Total trading capital (e.g., ₹10,00,000)
        risk_per_trade_pct: Risk percentage (0.5 to 2.0 recommended)
        entry_price: Entry price per share
        stop_loss: Stop-loss price per share
        bias: "LONG" or "SHORT"
    
    Returns:
        Dictionary with shares to trade, position value, rupee risk, and risk percentage
    """
    if pd.isna(stop_loss) or stop_loss is None:
        return {
            "shares": 0,
            "position_value": 0,
            "rupee_risk": 0,
            "risk_pct": 0,
            "error": "Invalid stop-loss"
        }
    
    if account_capital <= 0 or entry_price <= 0:
        return {
            "shares": 0,
            "position_value": 0,
            "rupee_risk": 0,
            "risk_pct": 0,
            "error": "Invalid capital or entry price"
        }
    
    # Maximum rupees you're willing to lose on this trade
    rupee_risk = account_capital * (risk_per_trade_pct / 100.0)
    
    # Risk per share = difference between entry and stop
    bias = bias.upper()
    if bias == "LONG":
        risk_per_share = entry_price - stop_loss
    else:  # SHORT
        risk_per_share = stop_loss - entry_price
    
    if risk_per_share <= 0:
        return {
            "shares": 0,
            "position_value": 0,
            "rupee_risk": 0,
            "risk_pct": 0,
            "error": "Stop-loss not protecting position (check bias direction)"
        }
    
    # Number of shares = Total risk / Risk per share
    shares = int(rupee_risk / risk_per_share)
    
    # Position value = shares × entry price
    position_value = shares * entry_price
    
    # Actual risk percentage based on position
    actual_risk_pct = (shares * risk_per_share) / account_capital * 100
    
    return {
        "shares": shares,
        "position_value": round(position_value, 2),
        "rupee_risk": round(rupee_risk, 2),
        "risk_pct": round(actual_risk_pct, 2),
        "risk_per_share": round(risk_per_share, 2),
        "error": None
    }


def calculate_targets(
    entry_price: float,
    stop_loss: float,
    bias: str,
    risk_reward_ratios: list = [1.5, 2.0, 3.0]
) -> dict:
    """
    Calculate target prices based on risk-reward ratio.
    
    Risk-Reward Ratio: If you risk ₹5 per share, how much do you aim to make?
    - 1:1 = ₹5 profit for ₹5 risk (break-even approach)
    - 2:1 = ₹10 profit for ₹5 risk (recommended minimum)
    - 3:1 = ₹15 profit for ₹5 risk (ideal)
    
    Args:
        entry_price: Entry price
        stop_loss: Stop-loss price
        bias: "LONG" or "SHORT"
        risk_reward_ratios: List of R:R ratios to calculate (e.g., [1.5, 2, 3])
    
    Returns:
        Dictionary with target prices for each ratio
    """
    if pd.isna(stop_loss) or stop_loss is None or entry_price <= 0:
        return {}
    
    bias = bias.upper()
    risk_per_share = abs(entry_price - stop_loss)
    
    targets = {}
    for rr in risk_reward_ratios:
        if bias == "LONG":
            target = entry_price + (risk_per_share * rr)
        else:  # SHORT
            target = entry_price - (risk_per_share * rr)
        
        targets[f"target_{rr}R"] = round(target, 2)
    
    return targets


def add_risk_metrics_to_signal(
    signal_row: dict,
    df_ohlc: pd.DataFrame,
    account_capital: float,
    risk_per_trade_pct: float = 1.0,
    atr_multiplier: float = 2.0,
    lookback_swing: int = 20
) -> dict:
    """
    Complete risk calculation for a trading signal.
    
    This is the MAIN function that brings everything together.
    Takes a signal and adds all risk management fields.
    
    Args:
        signal_row: Dictionary with signal data (symbol, bias, ltp, etc.)
        df_ohlc: OHLC DataFrame for the stock (used to calculate ATR and swings)
        account_capital: Your total trading capital
        risk_per_trade_pct: Risk percentage per trade (0.5-2% recommended)
        atr_multiplier: ATR multiplier for stop-loss (2.0 is standard)
        lookback_swing: How many candles back to find swing highs/lows
    
    Returns:
        Enhanced signal dictionary with all risk fields added
    """
    enhanced = signal_row.copy()
    
    entry_price = signal_row.get("ltp", None)
    bias = signal_row.get("bias", "").upper()
    
    if df_ohlc is None or df_ohlc.empty or entry_price is None or pd.isna(entry_price):
        enhanced.update({
            "atr": None,
            "stop_loss": None,
            "stop_method": "no_data",
            "shares": 0,
            "position_value": 0,
            "rupee_risk": 0,
            "risk_pct": 0,
            "target_1.5R": None,
            "target_2R": None,
            "target_3R": None,
        })
        return enhanced
    
    # 1. Calculate ATR
    atr = calculate_atr(df_ohlc, period=14)
    
    # 2. Find recent swing points
    recent_swing_low = df_ohlc['low'].tail(lookback_swing).min() if bias == "LONG" else None
    recent_swing_high = df_ohlc['high'].tail(lookback_swing).max() if bias == "SHORT" else None
    
    # 3. Calculate stop-loss
    stop_result = calculate_stop_loss(
        entry_price=entry_price,
        bias=bias,
        atr_value=atr,
        atr_multiplier=atr_multiplier,
        recent_swing_low=recent_swing_low,
        recent_swing_high=recent_swing_high,
        use_swing_points=True
    )
    
    # 4. Calculate position size
    pos_result = calculate_position_size(
        account_capital=account_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        entry_price=entry_price,
        stop_loss=stop_result.get("stop_loss"),
        bias=bias
    )
    
    # 5. Calculate targets
    targets = calculate_targets(
        entry_price=entry_price,
        stop_loss=stop_result.get("stop_loss"),
        bias=bias,
        risk_reward_ratios=[1.5, 2.0, 3.0]
    )
    
    # 6. Add all calculated fields to signal
    enhanced.update({
        "atr": round(atr, 2) if not pd.isna(atr) else None,
        "stop_loss": stop_result.get("stop_loss"),
        "stop_method": stop_result.get("method"),
        "shares": pos_result.get("shares", 0),
        "position_value": pos_result.get("position_value", 0),
        "rupee_risk": pos_result.get("rupee_risk", 0),
        "risk_pct": pos_result.get("risk_pct", 0),
        "risk_per_share": pos_result.get("risk_per_share", 0),
        **targets  # Adds target_1.5R, target_2R, target_3R
    })
    
    return enhanced
