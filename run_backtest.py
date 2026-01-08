"""
Standalone Backtest Runner for Trend Squeeze Strategy

Usage (from repo root):
    python3 run_backtest.py

This uses:
- backtest_engine.BacktestEngine
- FYERS via utils.zerodha_utils.init_fyers_session (called inside BacktestEngine)
and assumes your .streamlit/secrets.toml holds fyers_app_id and fyers_access_token.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backtest_engine import BacktestEngine

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Test mode: "single" (recommended) or "multi"
TEST_MODE = "single"

# Single stock symbol for testing
TEST_SYMBOL = "RELIANCE"

# Optional multi-symbol universe (unused if TEST_MODE == "single")
NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
    "ICICIBANK", "KOTAKBANK", "SBIN", "BHARTIARTL", "ITC",
    "AXISBANK", "LT", "ASIANPAINT", "MARUTI", "TITAN",
    "SUNPHARMA", "ULTRACEMCO", "BAJFINANCE", "WIPRO", "NESTLEIND",
    "HCLTECH", "TECHM", "ONGC", "NTPC", "POWERGRID",
    "TATAMOTORS", "TATASTEEL", "M&M", "ADANIPORTS", "COALINDIA",
    "BAJAJFINSV", "DIVISLAB", "GRASIM", "JSWSTEEL", "DRREDDY",
    "INDUSINDBK", "CIPLA", "EICHERMOT", "APOLLOHOSP", "BRITANNIA",
    "TATACONSUM", "HINDALCO", "SBILIFE", "BPCL", "SHREECEM",
    "UPL", "HEROMOTOCO", "ADANIENT", "HDFCLIFE",
]

# Backtest period (default: last 6 months)
TODAY = datetime.now().date()
START_DATE = datetime(TODAY.year, TODAY.month, TODAY.day) - timedelta(days=180)
END_DATE = datetime(TODAY.year, TODAY.month, TODAY.day) - timedelta(days=1)

# Account settings
INITIAL_CAPITAL = 1_000_000.0  # 10 lakhs
RISK_PER_TRADE = 1.0           # 1% per trade
MAX_POSITIONS = 5

# Timeframe
TIMEFRAME = "15M"  # "15M" or "Daily"

# Strategy parameters (same semantics as your app)
STRATEGY_PARAMS = {
    "bbw_abs_threshold": 0.05,
    "bbw_pct_threshold": 0.35,
    "adx_threshold": 20.0,
    "rsi_bull": 55.0,
    "rsi_bear": 45.0,
    "rolling_window": 20,
    "breakout_lookback": 20,
    "require_bbw_expansion": True,
    "require_volume_spike": False,
    "volume_spike_mult": 1.5,
    "engine": "hybrid",          # "hybrid", "box", "squeeze"
    "box_width_pct_max": 0.012,  # 1.2% box width
    "require_di_confirmation": True,
    "rsi_floor_short": 30.0,
    "rsi_ceiling_long": 70.0,
}

# -----------------------------------------------------------------------------
# PRINT HELPERS
# -----------------------------------------------------------------------------
def print_config() -> None:
    print("\n" + "=" * 70)
    print(" " * 22 + "BACKTEST CONFIGURATION")
    print("=" * 70 + "\n")

    if TEST_MODE == "single":
        print("Mode:              🎯 SINGLE STOCK MODE (Fast)")
        print("Symbols:           1 stock(s)")
        print(f"                   {TEST_SYMBOL}")
    else:
        print("Mode:              🌐 MULTI-STOCK MODE")
        print(f"Symbols:           {len(NIFTY_50_SYMBOLS)} stock(s)")
        print("                   " + ", ".join(NIFTY_50_SYMBOLS[:5]) + "...")

    print(f"Timeframe:         {TIMEFRAME}")
    print(f"Period:            {START_DATE.date()} to {END_DATE.date()}")
    print(f"Initial Capital:   ₹{INITIAL_CAPITAL:,.0f}")
    print(f"Risk Per Trade:    {RISK_PER_TRADE:.1f}%")
    print(f"Max Positions:     {MAX_POSITIONS}")

    print("\n📈 STRATEGY PARAMETERS:")
    print(f"   BBW abs:              {STRATEGY_PARAMS['bbw_abs_threshold']}")
    print(f"   BBW pct:              {STRATEGY_PARAMS['bbw_pct_threshold']}")
    print(f"   ADX min:              {STRATEGY_PARAMS['adx_threshold']}")
    print(f"   RSI bull/bear:        {STRATEGY_PARAMS['rsi_bull']}/{STRATEGY_PARAMS['rsi_bear']}")
    print(f"   Engine:               {STRATEGY_PARAMS['engine']}")
    print(f"   BBW expansion:        {STRATEGY_PARAMS['require_bbw_expansion']}")
    print(f"   DI confirmation:      {STRATEGY_PARAMS['require_di_confirmation']}")

    print("\n" + "=" * 70)
    if TEST_MODE == "single":
        print("\n⏱️  Estimated time: ~30–60 seconds")
    else:
        est_min = int(len(NIFTY_50_SYMBOLS) * 1.5)
        print(f"\n⏱️  Estimated time: ~{est_min} minutes")

    print("💡 Tip: Start with single‑stock mode to validate quickly.\n")
    print("🚀 STARTING BACKTEST...\n")


def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 70)
    print(" " * 25 + "BACKTEST RESULTS")
    print("=" * 70 + "\n")

    print("💰 CAPITAL")
    print(f"   Initial Capital:        ₹{metrics['initial_capital']:>12,.0f}")
    print(f"   Final Capital:          ₹{metrics['final_capital']:>12,.0f}")
    print(f"   Total P&L:              ₹{metrics['total_pnl']:>12,.0f}")
    print(f"   Total Return:           {metrics['total_return_pct']:>12.2f}%\n")

    print("📊 TRADE STATISTICS")
    print(f"   Total Trades:           {metrics['total_trades']:>12}")
    print(f"   Winning Trades:         {metrics['winning_trades']:>12}")
    print(f"   Losing Trades:          {metrics['losing_trades']:>12}")
    print(f"   Win Rate:               {metrics['win_rate_pct']:>12.2f}%\n")

    print("💵 PROFIT/LOSS ANALYSIS")
    print(f"   Avg Win:                ₹{metrics['avg_win']:>12,.0f}")
    print(f"   Avg Loss:               ₹{metrics['avg_loss']:>12,.0f}")
    print(f"   Avg Win %:              {metrics['avg_win_pct']:>12.2f}%")
    print(f"   Avg Loss %:             {metrics['avg_loss_pct']:>12.2f}%")
    print(f"   Profit Factor:          {metrics['profit_factor']:>12.2f}\n")

    print("⚠️  RISK METRICS")
    print(f"   Max Drawdown:           {metrics['max_drawdown_pct']:>12.2f}%")
    print(f"   Sharpe Ratio:           {metrics['sharpe_ratio']:>12.2f}")
    print(f"   Avg Holding (days):     {metrics['avg_holding_days']:>12.1f}\n")

    if "exit_reasons" in metrics:
        print("🚪 EXIT BREAKDOWN")
        for reason, count in metrics["exit_reasons"].items():
            print(f"   {reason:20s} {count:>12}")
        print()

    if "quality_breakdown" in metrics and metrics["quality_breakdown"]:
        print("⭐ QUALITY BREAKDOWN")
        for q, stats in metrics["quality_breakdown"].items():
            print(f"   Grade {q}: {int(stats['count']):>3} trades | Avg P&L: ₹{stats['mean']:>10,.0f}")
        print()

    print("=" * 70)


def save_results(trades_df: pd.DataFrame, metrics: dict) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "single" if TEST_MODE == "single" else "multi"

    trades_file = f"backtest_trades_{suffix}_{ts}.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"\n💾 Trades saved to: {trades_file}")

    metrics_file = f"backtest_metrics_{suffix}_{ts}.json"
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            metrics_serializable[k] = float(v)
        else:
            metrics_serializable[k] = v
    with open(metrics_file, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f"💾 Metrics saved to: {metrics_file}")
    print("\n✅ BACKTEST COMPLETE!\n")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    print_config()

    symbols = [TEST_SYMBOL] if TEST_MODE == "single" else NIFTY_50_SYMBOLS

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        max_positions=MAX_POSITIONS,
    )

    # Credentials are pulled internally via init_fyers_session through your utils;
    # if you ever want to pass them explicitly, add fyers_app_id/access_token here.
    trades_df = engine.run_backtest(
        symbols=symbols,
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=TIMEFRAME,
        strategy_params=STRATEGY_PARAMS,
    )

    if trades_df.empty:
        print("\n" + "=" * 70)
        print("❌ NO TRADES GENERATED")
        print("   Possible reasons:")
        print("   1. Strategy parameters too strict (no signals)")
        print("   2. Date range too short or data unavailable")
        print("   3. FYERS intraday history / rate-limit issues")
        print("\n💡 Try:")
        print("   - Relaxing BBW / ADX thresholds")
        print("   - Testing another stock (e.g., INFY, HDFCBANK)")
        print("   - Checking FYERS token and data limits")
        print("=" * 70 + "\n")
        return

    metrics = engine.calculate_metrics()
    print_metrics(metrics)
    save_results(trades_df, metrics)

    print("📌 Next:")
    print("   • Inspect trades CSV for patterns")
    print("   • Focus on Sharpe > 1.0 and drawdown < 25% for robust strategies\n")


if __name__ == "__main__":
    main()
