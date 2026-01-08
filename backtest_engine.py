"""
Backtest Engine for Trend Squeeze Strategy
Tests strategy performance on historical data with realistic costs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from utils.zerodha_utils import get_ohlc_15min, get_ohlc_daily, normalize_ohlc_index_to_ist
from utils.strategy import prepare_trend_squeeze
from utils.risk_calculator import add_risk_metrics_to_signal


class BacktestEngine:
    """
    Backtests the Trend Squeeze strategy with realistic trading costs.
    
    Transaction costs included:
    - Brokerage: ₹20 per trade (flat fee realistic for discount brokers)
    - STT (Securities Transaction Tax): 0.025% on sell side only
    - Stamp Duty: 0.003% on buy side
    - GST: 18% on brokerage
    - Slippage: 0.05% per trade (realistic for liquid stocks)
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        risk_per_trade: float = 1.0,
        brokerage_per_trade: float = 20.0,
        slippage_pct: float = 0.05,
        max_positions: int = 5,
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital in rupees (default 10 lakhs)
            risk_per_trade: Risk percentage per trade (default 1%)
            brokerage_per_trade: Flat brokerage per trade (default ₹20)
            slippage_pct: Slippage as percentage (default 0.05%)
            max_positions: Maximum concurrent positions (default 5)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.brokerage_per_trade = brokerage_per_trade
        self.slippage_pct = slippage_pct / 100.0  # Convert to decimal
        self.max_positions = max_positions
        
        self.trades = []  # List of all completed trades
        self.open_positions = {}  # Currently open positions
        self.equity_curve = []  # Daily account value
        self.daily_returns = []  # Daily return percentages
        
    def calculate_transaction_costs(self, price: float, quantity: int, side: str) -> float:
        """
        Calculate total transaction costs for a trade.
        
        Args:
            price: Trade price
            quantity: Number of shares
            side: 'BUY' or 'SELL'
        
        Returns:
            Total transaction cost in rupees
        """
        trade_value = price * quantity
        
        # Brokerage (flat ₹20 per trade)
        brokerage = self.brokerage_per_trade
        
        # GST on brokerage (18%)
        gst = brokerage * 0.18
        
        # STT (0.025% on sell side only)
        stt = trade_value * 0.00025 if side == 'SELL' else 0
        
        # Stamp duty (0.003% on buy side, 0.015% on sell side)
        stamp_duty = trade_value * 0.00003 if side == 'BUY' else trade_value * 0.00015
        
        # Slippage (applies both ways)
        slippage = trade_value * self.slippage_pct
        
        total_cost = brokerage + gst + stt + stamp_duty + slippage
        return round(total_cost, 2)
    
    def open_position(
        self,
        symbol: str,
        entry_date: datetime,
        entry_price: float,
        stop_loss: float,
        target: float,
        shares: int,
        bias: str,
        setup: str,
        timeframe: str,
        signal_quality: str
    ):
        """
        Open a new position.
        
        Returns True if position opened successfully, False if rejected.
        """
        # Check if max positions reached
        if len(self.open_positions) >= self.max_positions:
            return False
        
        # Check if symbol already has open position
        if symbol in self.open_positions:
            return False
        
        # Calculate position value
        position_value = entry_price * shares
        
        # Apply slippage to entry (slippage works against you)
        if bias == "LONG":
            actual_entry = entry_price * (1 + self.slippage_pct)
        else:  # SHORT
            actual_entry = entry_price * (1 - self.slippage_pct)
        
        # Calculate entry costs
        entry_costs = self.calculate_transaction_costs(actual_entry, shares, 'BUY')
        
        # Check if enough capital
        required_capital = position_value + entry_costs
        if required_capital > self.current_capital:
            return False
        
        # Deduct capital
        self.current_capital -= required_capital
        
        # Store position
        self.open_positions[symbol] = {
            'entry_date': entry_date,
            'entry_price': actual_entry,
            'stop_loss': stop_loss,
            'target': target,
            'shares': shares,
            'bias': bias,
            'setup': setup,
            'timeframe': timeframe,
            'quality': signal_quality,
            'entry_costs': entry_costs,
            'position_value': position_value,
        }
        
        return True
    
    def check_exit(
        self,
        symbol: str,
        current_date: datetime,
        high: float,
        low: float,
        close: float
    ) -> Tuple[bool, str, float]:
        """
        Check if position should be exited (stop or target hit).
        
        Returns:
            (should_exit, exit_reason, exit_price)
        """
        if symbol not in self.open_positions:
            return False, None, None
        
        pos = self.open_positions[symbol]
        bias = pos['bias']
        
        if bias == "LONG":
            # Check stop-loss hit
            if low <= pos['stop_loss']:
                return True, 'STOP', pos['stop_loss']
            # Check target hit
            if high >= pos['target']:
                return True, 'TARGET', pos['target']
        else:  # SHORT
            # Check stop-loss hit
            if high >= pos['stop_loss']:
                return True, 'STOP', pos['stop_loss']
            # Check target hit
            if low <= pos['target']:
                return True, 'TARGET', pos['target']
        
        return False, None, None
    
    def close_position(
        self,
        symbol: str,
        exit_date: datetime,
        exit_price: float,
        exit_reason: str
    ):
        """
        Close an open position and record the trade.
        """
        if symbol not in self.open_positions:
            return
        
        pos = self.open_positions[symbol]
        bias = pos['bias']
        shares = pos['shares']
        
        # Apply slippage to exit
        if bias == "LONG":
            actual_exit = exit_price * (1 - self.slippage_pct)
        else:  # SHORT
            actual_exit = exit_price * (1 + self.slippage_pct)
        
        # Calculate exit costs
        exit_costs = self.calculate_transaction_costs(actual_exit, shares, 'SELL')
        
        # Calculate P&L
        if bias == "LONG":
            gross_pnl = (actual_exit - pos['entry_price']) * shares
        else:  # SHORT
            gross_pnl = (pos['entry_price'] - actual_exit) * shares
        
        total_costs = pos['entry_costs'] + exit_costs
        net_pnl = gross_pnl - total_costs
        
        # Return capital + profit/loss
        self.current_capital += pos['position_value'] + net_pnl
        
        # Calculate returns
        return_pct = (net_pnl / pos['position_value']) * 100
        
        # Calculate holding period
        holding_days = (exit_date - pos['entry_date']).days
        
        # Record trade
        trade = {
            'symbol': symbol,
            'timeframe': pos['timeframe'],
            'setup': pos['setup'],
            'quality': pos['quality'],
            'bias': bias,
            'entry_date': pos['entry_date'],
            'exit_date': exit_date,
            'holding_days': holding_days,
            'entry_price': pos['entry_price'],
            'exit_price': actual_exit,
            'stop_loss': pos['stop_loss'],
            'target': pos['target'],
            'shares': shares,
            'position_value': pos['position_value'],
            'gross_pnl': round(gross_pnl, 2),
            'total_costs': round(total_costs, 2),
            'net_pnl': round(net_pnl, 2),
            'return_pct': round(return_pct, 2),
            'exit_reason': exit_reason,
            'capital_after': round(self.current_capital, 2),
        }
        
        self.trades.append(trade)
        
        # Remove from open positions
        del self.open_positions[symbol]
    
    def run_backtest(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "15M",
        strategy_params: Dict = None
    ) -> pd.DataFrame:
        """
        Run backtest on list of symbols.
        
        Args:
            symbols: List of stock symbols to test
            start_date: Backtest start date
            end_date: Backtest end date
            timeframe: "15M" or "Daily"
            strategy_params: Dictionary of strategy parameters
        
        Returns:
            DataFrame of all trades
        """
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        print(f"Timeframe: {timeframe} | Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"{'='*60}\n")
        
        if strategy_params is None:
            strategy_params = {}
        
        # Track daily equity for equity curve
        equity_dates = []
        equity_values = []
        
        # Process each symbol
        for idx, symbol in enumerate(symbols, 1):
            print(f"[{idx}/{len(symbols)}] Processing {symbol}...", end=" ")
            
            try:
                # Get historical data
                if timeframe == "15M":
                    df = get_ohlc_15min(symbol, days_back=365)
                else:
                    df = get_ohlc_daily(symbol, lookback_days=730)
                
                df = normalize_ohlc_index_to_ist(df)
                
                if df is None or len(df) < 200:
                    print("❌ Insufficient data")
                    continue
                
                # Filter date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                if df.empty:
                    print("❌ No data in range")
                    continue
                
                # Apply strategy
                df_signals = prepare_trend_squeeze(df, **strategy_params)
                
                # Find signal rows
                df_signals = df_signals[df_signals['setup'] != ""].copy()
                
                if df_signals.empty:
                    print("⚪ No signals")
                    continue
                
                signals_count = 0
                
                # Process each signal
                for sig_date, sig_row in df_signals.iterrows():
                    setup = sig_row['setup']
                    bias = "LONG" if str(setup).startswith("Bullish") else "SHORT"
                    ltp = float(sig_row['close'])
                    
                    # Calculate risk metrics for this signal
                    signal_dict = {
                        'ltp': ltp,
                        'bias': bias,
                    }
                    
                    enhanced = add_risk_metrics_to_signal(
                        signal_row=signal_dict,
                        df_ohlc=df.loc[:sig_date],
                        account_capital=self.current_capital,
                        risk_per_trade_pct=self.risk_per_trade,
                        atr_multiplier=2.0,
                        lookback_swing=20
                    )
                    
                    stop_loss = enhanced.get('stop_loss')
                    shares = enhanced.get('shares', 0)
                    target = enhanced.get('target_2R')  # Use 2R target
                    
                    if stop_loss is None or shares == 0 or target is None:
                        continue
                    
                    # Try to open position
                    opened = self.open_position(
                        symbol=symbol,
                        entry_date=sig_date,
                        entry_price=ltp,
                        stop_loss=stop_loss,
                        target=target,
                        shares=shares,
                        bias=bias,
                        setup=setup,
                        timeframe=timeframe,
                        signal_quality=sig_row.get('quality_score', 'C')
                    )
                    
                    if opened:
                        signals_count += 1
                    
                    # Check exits for all open positions on subsequent bars
                    future_df = df.loc[sig_date:]
                    for future_date, future_row in future_df.iterrows():
                        if future_date <= sig_date:
                            continue
                        
                        should_exit, exit_reason, exit_price = self.check_exit(
                            symbol=symbol,
                            current_date=future_date,
                            high=future_row['high'],
                            low=future_row['low'],
                            close=future_row['close']
                        )
                        
                        if should_exit:
                            self.close_position(
                                symbol=symbol,
                                exit_date=future_date,
                                exit_price=exit_price,
                                exit_reason=exit_reason
                            )
                            break
                
                print(f"✅ {signals_count} signals")
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                continue
        
        # Close any remaining open positions at end date
        print(f"\nClosing {len(self.open_positions)} remaining positions...")
        for symbol in list(self.open_positions.keys()):
            pos = self.open_positions[symbol]
            # Use last known price (approximation)
            self.close_position(
                symbol=symbol,
                exit_date=end_date,
                exit_price=pos['entry_price'],  # Exit at entry (neutral)
                exit_reason='END_OF_BACKTEST'
            )
        
        return pd.DataFrame(self.trades)
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with all performance metrics
        """
        if not self.trades:
            return {'error': 'No trades to analyze'}
        
        df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['net_pnl'] > 0])
        losing_trades = len(df[df['net_pnl'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = df['net_pnl'].sum()
        avg_win = df[df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0
        
        # Risk-reward
        avg_win_pct = df[df['net_pnl'] > 0]['return_pct'].mean() if winning_trades > 0 else 0
        avg_loss_pct = df[df['net_pnl'] < 0]['return_pct'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Drawdown calculation
        df = df.sort_values('exit_date')
        df['cumulative_pnl'] = df['net_pnl'].cumsum()
        df['equity'] = self.initial_capital + df['cumulative_pnl']
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak'] * 100
        max_drawdown = df['drawdown'].min()
        
        # Returns
        final_capital = self.current_capital
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Sharpe ratio (simplified - using trade returns)
        if len(df) > 1:
            returns = df['return_pct'].values
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        else:
            sharpe = 0
        
        # Holding period
        avg_holding_days = df['holding_days'].mean()
        
        # Exit reasons breakdown
        exit_reasons = df['exit_reason'].value_counts().to_dict()
        
        # Quality breakdown
        quality_breakdown = df.groupby('quality')['net_pnl'].agg(['count', 'sum', 'mean']).to_dict('index')
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': round(final_capital, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return, 2),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_win_pct': round(avg_win_pct, 2),
            'avg_loss_pct': round(avg_loss_pct, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            'avg_holding_days': round(avg_holding_days, 1),
            'exit_reasons': exit_reasons,
            'quality_breakdown': quality_breakdown,
        }
        
        return metrics
