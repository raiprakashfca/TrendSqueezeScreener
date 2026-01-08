"""
Run Backtest Script - Optimized for Single Stock Testing
Execute this to test your strategy on historical data
"""

import pandas as pd
from datetime import datetime, timedelta
from backtest_engine import BacktestEngine
import json
import sys

# =========================
# CONFIGURATION
# =========================

# Backtest period
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)

# Capital and risk settings
INITIAL_CAPITAL = 1000000.0  # 10 lakhs
RISK_PER_TRADE = 1.0  # 1% risk per trade
MAX_POSITIONS = 5  # Maximum concurrent trades

# =========================
# STOCK SELECTION MODES
# =========================

# MODE 1: Single stock (fastest - for quick testing)
SINGLE_STOCK_MODE = True  # Set to True to test just one stock
TEST_SYMBOL = "RELIANCE"  # Change this to test different stocks

# MODE 2: Small batch (moderate speed - test a few stocks)
BATCH_MODE = False  # Set to True to test a small batch
BATCH_SYMBOLS = [
    "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "BAJFINANCE", "KOTAKBANK", "LT"
]

# MODE 3: Full universe (slowest - comprehensive test)
FULL_MODE = False  # Set to True to test all NIFTY50 stocks
FULL_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BHARTIARTL", "BPCL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC",
    "JSWSTEEL", "KOTAKBANK", "LT", "LTIM", "M&M",
    "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "RELIANCE", "SBIN", "SBILIFE", "SUNPHARMA", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "ULTRACEMCO", "UPL", "WIPRO", "HEROMOTOCO", "SHREECEM"
]

# Timeframe selection
TIMEFRAME = "15M"  # "15M" or "Daily"

# Strategy parameters (adjust these to match your live settings)
STRATEGY_PARAMS = {
    'bbw_abs_threshold': 0.05,
    'bbw_pct_threshold': 0.35,
    'adx_threshold': 20.0,
    'rsi_bull': 55.0,
    'rsi_bear': 45.0,
    'rolling_window': 20,
    'breakout_lookback': 20,
    'require_bbw_expansion': True,
    'require_volume_spike': False,
    'volume_spike_mult': 1.5,
    'engine': 'hybrid',  # or 'box' or 'squeeze'
    'box_width_pct_max': 0.012,
    'require_di_confirmation': True,
    'rsi_floor_short': 30.0,
    'rsi_ceiling_long': 70.0,
}


def select_symbols():
    """
    Select which symbols to backtest based on mode.
    """
    if SINGLE_STOCK_MODE:
        return [TEST_SYMBOL]
    elif BATCH_MODE:
        return BATCH_SYMBOLS
    elif FULL_MODE:
        return FULL_SYMBOLS
    else:
        # Default to single stock if no mode selected
        print("⚠️  No mode selected, defaulting to SINGLE_STOCK_MODE")
        return [TEST_SYMBOL]


def print_metrics(metrics: dict):
    """Print metrics in formatted table"""
    print(f"\n{'='*70}")
    print(f"{'BACKTEST RESULTS':^70}")
    print(f"{'='*70}\n")
    
    print(f"💰 CAPITAL")
    print(f"   Initial Capital:        ₹{metrics['initial_capital']:>15,.0f}")
    print(f"   Final Capital:          ₹{metrics['final_capital']:>15,.0f}")
    print(f"   Total P&L:              ₹{metrics['total_pnl']:>15,.0f}")
    print(f"   Total Return:           {metrics['total_return_pct']:>15.2f}%\n")
    
    print(f"📊 TRADE STATISTICS")
    print(f"   Total Trades:           {metrics['total_trades']:>18,}")
    print(f"   Winning Trades:         {metrics['winning_trades']:>18,}")
    print(f"   Losing Trades:          {metrics['losing_trades']:>18,}")
    print(f"   Win Rate:               {metrics['win_rate_pct']:>15.2f}%\n")
    
    print(f"💵 PROFIT/LOSS ANALYSIS")
    print(f"   Avg Win:                ₹{metrics['avg_win']:>15,.0f}")
    print(f"   Avg Loss:               ₹{metrics['avg_loss']:>15,.0f}")
    print(f"   Avg Win %:              {metrics['avg_win_pct']:>15.2f}%")
    print(f"   Avg Loss %:             {metrics['avg_loss_pct']:>15.2f}%")
    print(f"   Profit Factor:          {metrics['profit_factor']:>18.2f}\n")
    
    print(f"⚠️  RISK METRICS")
    print(f"   Max Drawdown:           {metrics['max_drawdown_pct']:>15.2f}%")
    print(f"   Sharpe Ratio:           {metrics['sharpe_ratio']:>18.2f}")
    print(f"   Avg Holding (days):     {metrics['avg_holding_days']:>18.1f}\n")
    
    print(f"🎯 EXIT REASONS")
    for reason, count in metrics['exit_reasons'].items():
        print(f"   {reason:<20}    {count:>5,} trades")
    
    if metrics.get('quality_breakdown'):
        print(f"\n⭐ SIGNAL QUALITY PERFORMANCE")
        for quality, stats in metrics['quality_breakdown'].items():
            print(f"   Quality {quality}:")
            print(f"      Trades: {int(stats['count']):>4,}  |  Total P&L: ₹{stats['sum']:>10,.0f}  |  Avg: ₹{stats['mean']:>8,.0f}")
    
    print(f"\n{'='*70}")
    
    # Performance assessment
    print(f"\n🔍 ASSESSMENT:")
    if metrics['total_return_pct'] > 15 and metrics['sharpe_ratio'] > 1.5 and metrics['max_drawdown_pct'] > -25:
        print(f"   ✅ STRONG STRATEGY - Good returns with manageable risk")
    elif metrics['total_return_pct'] > 8 and metrics['sharpe_ratio'] > 1.0:
        print(f"   ⚠️  MODERATE STRATEGY - Acceptable but needs improvement")
    else:
        print(f"   ❌ WEAK STRATEGY - Requires significant optimization or redesign")
    
    if metrics['total_trades'] < 10:
        print(f"   ⚠️  WARNING: Very few trades ({metrics['total_trades']}). Results may not be statistically significant.")
    
    print(f"\n")


def print_config(symbols):
    """Print backtest configuration"""
    print(f"\n{'='*70}")
    print(f"{'BACKTEST CONFIGURATION':^70}")
    print(f"{'='*70}\n")
    
    # Mode
    if SINGLE_STOCK_MODE:
        mode = "🎯 SINGLE STOCK MODE (Fast)"
    elif BATCH_MODE:
        mode = "📦 BATCH MODE (Moderate)"
    elif FULL_MODE:
        mode = "🌐 FULL UNIVERSE MODE (Slow)"
    else:
        mode = "❓ Unknown"
    
    print(f"Mode:              {mode}")
    print(f"Symbols:           {len(symbols)} stock(s)")
    if len(symbols) <= 10:
        print(f"                   {', '.join(symbols)}")
    print(f"Timeframe:         {TIMEFRAME}")
    print(f"Period:            {START_DATE.date()} to {END_DATE.date()}")
    print(f"Initial Capital:   ₹{INITIAL_CAPITAL:,.0f}")
    print(f"Risk Per Trade:    {RISK_PER_TRADE}%")
    print(f"Max Positions:     {MAX_POSITIONS}")
    
    print(f"\n📈 STRATEGY PARAMETERS:")
    print(f"   BBW Threshold:        {STRATEGY_PARAMS['bbw_abs_threshold']}")
    print(f"   ADX Threshold:        {STRATEGY_PARAMS['adx_threshold']}")
    print(f"   RSI Bull/Bear:        {STRATEGY_PARAMS['rsi_bull']}/{STRATEGY_PARAMS['rsi_bear']}")
    print(f"   Engine:               {STRATEGY_PARAMS['engine']}")
    print(f"   BBW Expansion:        {STRATEGY_PARAMS['require_bbw_expansion']}")
    print(f"   DI Confirmation:      {STRATEGY_PARAMS['require_di_confirmation']}")
    
    print(f"\n{'='*70}\n")
    
    # Estimated time
    if len(symbols) == 1:
        est_time = "~30-60 seconds"
    elif len(symbols) <= 10:
        est_time = "~5-10 minutes"
    else:
        est_time = "~15-30 minutes"
    
    print(f"⏱️  Estimated time: {est_time}")
    print(f"💡 Tip: Start with single stock mode to validate quickly!\n")


def main():
    """Run the backtest"""
    
    # Select symbols based on mode
    symbols = select_symbols()
    
    # Print configuration
    print_config(symbols)
    
    # Confirm before running full mode
    if FULL_MODE:
        confirm = input("⚠️  Full mode selected. This will take 15-30 minutes. Continue? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("❌ Backtest cancelled.")
            return
    
    print(f"🚀 STARTING BACKTEST...\n")
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        brokerage_per_trade=20.0,
        slippage_pct=0.05,
        max_positions=MAX_POSITIONS,
    )
    
    # Run backtest
    trades_df = engine.run_backtest(
        symbols=symbols,
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=TIMEFRAME,
        strategy_params=STRATEGY_PARAMS
    )
    
    # Check if we got any trades
    if trades_df.empty:
        print(f"\n❌ NO TRADES GENERATED")
        print(f"   Possible reasons:")
        print(f"   1. Strategy parameters too strict (no signals)")
        print(f"   2. Date range too short")
        print(f"   3. Data fetch issues")
        print(f"\n💡 Try:")
        print(f"   - Relaxing bbw_abs_threshold or adx_threshold")
        print(f"   - Testing a different stock (e.g., RELIANCE, INFY)")
        print(f"   - Checking if FYERS token is valid")
        return
    
    # Calculate metrics
    metrics = engine.calculate_metrics()
    
    # Print results
    print_metrics(metrics)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "single" if SINGLE_STOCK_MODE else ("batch" if BATCH_MODE else "full")
    
    # Save trades CSV
    trades_file = f"backtest_trades_{mode_suffix}_{timestamp}.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"💾 Trades saved to: {trades_file}")
    
    # Save metrics JSON
    metrics_file = f"backtest_metrics_{mode_suffix}_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"💾 Metrics saved to: {metrics_file}")
    
    # Print sample trades
    print(f"\n📋 SAMPLE TRADES (First 5):")
    print(trades_df.head(5).to_string(index=False))
    
    print(f"\n✅ BACKTEST COMPLETE!\n")
    
    # Next steps suggestion
    if SINGLE_STOCK_MODE:
        print(f"💡 NEXT STEPS:")
        print(f"   1. Review the results above")
        print(f"   2. If good, test on BATCH_MODE (10 stocks)")
        print(f"   3. If batch results are consistent, run FULL_MODE")
        print(f"\n   To switch modes: Edit run_backtest.py")
        print(f"   - Set SINGLE_STOCK_MODE = False")
        print(f"   - Set BATCH_MODE = True")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Backtest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ BACKTEST FAILED")
        print(f"Error: {str(e)}")
        print(f"\nFull error details:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
