"""
COMPATIBILITY SHIM

Your project previously imported FYERS helpers from:
    utils.zerodha_utils

You renamed/moved them to:
    utils.fyers_utils

This file re-exports everything so existing imports keep working.
"""

from __future__ import annotations

# Re-export the real implementation
from utils.fyers_utils import *  # noqa: F401,F403

# Backward-compatible function aliases (older naming without underscores)
try:
    initfyerssession = init_fyers_session  # type: ignore  # noqa: F405
    getohlc15min = get_ohlc_15min  # type: ignore  # noqa: F405
    getohlcdaily = get_ohlc_daily  # type: ignore  # noqa: F405
    istradingday = is_trading_day  # type: ignore  # noqa: F405
    getlasttradingday = get_last_trading_day  # type: ignore  # noqa: F405
    normalizeohlcindextoist = normalize_ohlc_index_to_ist  # type: ignore  # noqa: F405
except Exception:
    # If fyers_utils isn't ready yet, this will be fixed after you paste fyers_utils.py next.
    pass
