"""
Backtest Engine for Trend Squeeze Strategy
Tests strategy performance on historical data with realistic costs.

This module is designed to be reusable:
- from run_backtest.py (local CLI)
- from Streamlit app (single-symbol on-demand backtest)

It depends only on your existing utility functions and strategy:
- utils.zerodha_utils: get_ohlc_15min, get_ohlc_daily, normalize_ohlc_index_to_ist, init_fyers_session
- utils.strategy: prepare_trend_squeeze
- utils.risk_calculator: add_risk_metrics_to_signal
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

from utils.zerodha_utils import (
    get_ohlc_15min,
    get_ohlc_daily,
    normalize_ohlc_index_to_ist,
    init_fyers_session,
)
from utils.strategy import prepare_trend_squeeze
from utils.risk_calculator import add_risk_metrics_to_signal


class BacktestEngine:
    """
    Backtests the Trend Squeeze strategy with realistic trading costs.

    Transaction costs included:
    - Brokerage: ₹20 per trade (flat fee, realistic for discount brokers)
    - STT: 0.025% on sell side only
    - Stamp Duty: 0.003% on buy, 0.015% on sell (approximation)
    - GST: 18% on brokerage
    - Slippage: configurable %, applied against the trader
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        risk_per_trade: float = 1.0,
        brokerage_per_trade: float = 20.0,
        slippage_pct: float = 0.05,
        max_positions: int = 5,
    ) -> None:
        """
        Args:
            initial_capital: Starting capital in rupees (default 10 lakhs)
            risk_per_trade: Risk percentage per trade (default 1%)
            brokerage_per_trade: Flat brokerage per trade (default ₹20)
            slippage_pct: Slippage as percentage (default 0.05%)
            max_positions: Maximum concurrent positions
        """
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.risk_per_trade = float(risk_per_trade)
        self.brokerage_per_trade = float(brokerage_per_trade)
        self.slippage_pct = float(slippage_pct) / 100.0  # convert to decimal
        self.max_positions = int(max_positions)

        self.trades: List[Dict] = []
        self.open_positions: Dict[str, Dict] = {}

    # -------------------------------------------------------------------------
    # COSTS
    # -------------------------------------------------------------------------
    def calculate_transaction_costs(self, price: float, quantity: int, side: str) -> float:
        """
        Calculate total transaction costs for a trade leg.

        side: "BUY" or "SELL"
        """
        price = float(price)
        quantity = int(quantity)
        trade_value = price * quantity

        # Brokerage
        brokerage = self.brokerage_per_trade

        # GST on brokerage
        gst = brokerage * 0.18

        # STT (sell side only)
        stt = trade_value * 0.00025 if side.upper() == "SELL" else 0.0

        # Stamp duty
        if side.upper() == "BUY":
            stamp_duty = trade_value * 0.00003
        else:
            stamp_duty = trade_value * 0.00015

        # Slippage
        slippage = trade_value * self.slippage_pct

        total = brokerage + gst + stt + stamp_duty + slippage
        return round(total, 2)

    # -------------------------------------------------------------------------
    # POSITION MANAGEMENT
    # -------------------------------------------------------------------------
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
        quality: str,
    ) -> bool:
        """
        Open a new position, if capital & slot are available.
        """
        if len(self.open_positions) >= self.max_positions:
            return False

        if symbol in self.open_positions:
            return False

        bias = bias.upper()
        shares = int(shares)
        entry_price = float(entry_price)

        position_value = entry_price * shares

        # Slippage against the trader on entry
        if bias == "LONG":
            actual_entry = entry_price * (1.0 + self.slippage_pct)
        else:  # SHORT
            actual_entry = entry_price * (1.0 - self.slippage_pct)

        entry_costs = self.calculate_transaction_costs(actual_entry, shares, "BUY")

        required_capital = position_value + entry_costs
        if required_capital > self.current_capital:
            return False

        self.current_capital -= required_capital

        self.open_positions[symbol] = {
            "entry_date": entry_date,
            "entry_price": actual_entry,
            "stop_loss": float(stop_loss),
            "target": float(target),
            "shares": shares,
            "bias": bias,
            "setup": setup,
            "timeframe": timeframe,
            "quality": quality,
            "entry_costs": entry_costs,
            "position_value": position_value,
        }
        return True

    def check_exit(
        self,
        symbol: str,
        current_date: datetime,
        high: float,
        low: float,
        close: float,
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check if a position should be exited based on OHLC bar.
        """
        if symbol not in self.open_positions:
            return False, None, None

        pos = self.open_positions[symbol]
        bias = pos["bias"]
        high = float(high)
        low = float(low)
        close = float(close)

        if bias == "LONG":
            if low <= pos["stop_loss"]:
                return True, "STOP", pos["stop_loss"]
            if high >= pos["target"]:
                return True, "TARGET", pos["target"]
        else:  # SHORT
            if high >= pos["stop_loss"]:
                return True, "STOP", pos["stop_loss"]
            if low <= pos["target"]:
                return True, "TARGET", pos["target"]

        return False, None, None

    def close_position(
        self,
        symbol: str,
        exit_date: datetime,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """
        Close an open position and record the trade.
        """
        if symbol not in self.open_positions:
            return

        pos = self.open_positions[symbol]
        bias = pos["bias"]
        shares = pos["shares"]

        # Slippage on exit, again against the trader
        if bias == "LONG":
            actual_exit = float(exit_price) * (1.0 - self.slippage_pct)
        else:
            actual_exit = float(exit_price) * (1.0 + self.slippage_pct)

        exit_costs = self.calculate_transaction_costs(actual_exit, shares, "SELL")

        if bias == "LONG":
            gross_pnl = (actual_exit - pos["entry_price"]) * shares
        else:
            gross_pnl = (pos["entry_price"] - actual_exit) * shares

        total_costs = pos["entry_costs"] + exit_costs
        net_pnl = gross_pnl - total_costs

        self.current_capital += pos["position_value"] + net_pnl

        return_pct = net_pnl / pos["position_value"] * 100.0
        holding_days = (exit_date - pos["entry_date"]).days

        trade = {
            "symbol": symbol,
            "timeframe": pos["timeframe"],
            "setup": pos["setup"],
            "quality": pos["quality"],
            "bias": bias,
            "entry_date": pos["entry_date"],
            "exit_date": exit_date,
            "holding_days": holding_days,
            "entry_price": pos["entry_price"],
            "exit_price": actual_exit,
            "stop_loss": pos["stop_loss"],
            "target": pos["target"],
            "shares": shares,
            "position_value": pos["position_value"],
            "gross_pnl": round(gross_pnl, 2),
            "total_costs": round(total_costs, 2),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(return_pct, 2),
            "exit_reason": exit_reason,
            "capital_after": round(self.current_capital, 2),
        }

        self.trades.append(trade)
        del self.open_positions[symbol]

    # -------------------------------------------------------------------------
    # MAIN BACKTEST
    # -------------------------------------------------------------------------
    def _fetch_price_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLC data for a symbol between start_date and end_date.
        Uses your existing utility functions, then normalizes to IST.
        """
        try:
            if timeframe.upper() == "15M":
                # Delegate "how many days" to the caller (run_backtest/app)
                df = get_ohlc_15min(symbol, days_back=365)
            else:
                df = get_ohlc_daily(symbol, lookback_days=730)

            df = normalize_ohlc_index_to_ist(df)

            if df is None or df.empty:
                return None

            df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
            if df.empty:
                return None

            return df
        except Exception:
            return None

    def run_backtest(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "15M",
        strategy_params: Optional[Dict] = None,
        fyers_app_id: Optional[str] = None,
        fyers_access_token: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run backtest across symbols.

        - If fyers_app_id & fyers_access_token are provided, initializes FYERS once.
        - Otherwise assumes data utilities are already wired to a working session.

        Returns:
            DataFrame of all executed trades.
        """
        print(f"\n{'=' * 70}")
        print(f"BACKTESTING: {len(symbols)} symbol(s) from {start_date.date()} to {end_date.date()}")
        print(f"Timeframe: {timeframe} | Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"{'=' * 70}\n")

        # Initialize FYERS session once (if credentials passed)
        if fyers_app_id and fyers_access_token:
            try:
                init_fyers_session(fyers_app_id, fyers_access_token)
                print("✅ FYERS session initialized\n")
            except Exception as e:
                print(f"⚠️ FYERS init error: {str(e)[:80]}\n")

        if strategy_params is None:
            strategy_params = {}

        for idx, symbol in enumerate(symbols, 1):
            print(f"[{idx}/{len(symbols)}] Processing {symbol}...", end=" ")

            try:
                df = self._fetch_price_data(symbol, start_date, end_date, timeframe)
                if df is None or len(df) < 50:
                    print("❌ Insufficient data")
                    continue

                # Strategy
                df_sig = prepare_trend_squeeze(df, **strategy_params)
                df_sig = df_sig[df_sig["setup"] != ""].copy()

                if df_sig.empty:
                    print("⚪ No signals")
                    continue

                signals_used = 0

                for ts, row in df_sig.iterrows():
                    setup = str(row["setup"])
                    bias = "LONG" if setup.startswith("Bullish") else "SHORT"
                    ltp = float(row["close"])

                    # Risk sizing
                    signal_dict = {"ltp": ltp, "bias": bias, "symbol": symbol}
                    enhanced = add_risk_metrics_to_signal(
                        signal_row=signal_dict,
                        df_ohlc=df.loc[:ts],
                        account_capital=self.current_capital,
                        risk_per_trade_pct=self.risk_per_trade,
                        atr_multiplier=2.0,
                        lookback_swing=20,
                    )

                    stop_loss = enhanced.get("stop_loss")
                    shares = int(enhanced.get("shares", 0))
                    target = enhanced.get("target_2R")

                    if not stop_loss or shares <= 0 or not target:
                        continue

                    opened = self.open_position(
                        symbol=symbol,
                        entry_date=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        entry_price=ltp,
                        stop_loss=float(stop_loss),
                        target=float(target),
                        shares=shares,
                        bias=bias,
                        setup=setup,
                        timeframe=timeframe,
                        quality=row.get("quality_score", "C"),
                    )

                    if not opened:
                        continue

                    signals_used += 1

                    # Iterate forward bars to find exit
                    future_df = df.loc[ts:]
                    for fut_ts, fut_row in future_df.iterrows():
                        if fut_ts <= ts:
                            continue

                        should_exit, reason, exit_price = self.check_exit(
                            symbol=symbol,
                            current_date=fut_ts.to_pydatetime() if hasattr(fut_ts, "to_pydatetime") else fut_ts,
                            high=fut_row["high"],
                            low=fut_row["low"],
                            close=fut_row["close"],
                        )
                        if should_exit:
                            self.close_position(
                                symbol=symbol,
                                exit_date=fut_ts.to_pydatetime() if hasattr(fut_ts, "to_pydatetime") else fut_ts,
                                exit_price=float(exit_price),
                                exit_reason=reason,
                            )
                            break

                print(f"✅ {signals_used} signals")

            except Exception as e:
                print(f"❌ Error: {str(e)[:80]}")
                continue

        # Force close remaining positions at end_date at entry price (neutral)
        if self.open_positions:
            print(f"\nClosing {len(self.open_positions)} remaining positions at end-of-backtest...")
            for symbol in list(self.open_positions.keys()):
                pos = self.open_positions[symbol]
                self.close_position(
                    symbol=symbol,
                    exit_date=end_date,
                    exit_price=pos["entry_price"],
                    exit_reason="END_OF_BACKTEST",
                )

        return pd.DataFrame(self.trades)

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------
    def calculate_metrics(self) -> Dict:
        """
        Calculate performance statistics on self.trades.
        """
        if not self.trades:
            return {"error": "No trades to analyze"}

        df = pd.DataFrame(self.trades).sort_values("exit_date")

        total_trades = len(df)
        winning_trades = (df["net_pnl"] > 0).sum()
        losing_trades = (df["net_pnl"] < 0).sum()
        win_rate = (winning_trades / total_trades * 100.0) if total_trades else 0.0

        total_pnl = float(df["net_pnl"].sum())
        avg_win = float(df[df["net_pnl"] > 0]["net_pnl"].mean()) if winning_trades else 0.0
        avg_loss = float(df[df["net_pnl"] < 0]["net_pnl"].mean()) if losing_trades else 0.0

        avg_win_pct = float(df[df["net_pnl"] > 0]["return_pct"].mean()) if winning_trades else 0.0
        avg_loss_pct = float(df[df["net_pnl"] < 0]["return_pct"].mean()) if losing_trades else 0.0

        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        df["cumulative_pnl"] = df["net_pnl"].cumsum()
        df["equity"] = self.initial_capital + df["cumulative_pnl"]
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"] * 100.0
        max_drawdown = float(df["drawdown"].min())

        final_capital = float(self.current_capital)
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100.0

        if df["return_pct"].std() > 0:
            sharpe = (df["return_pct"].mean() / df["return_pct"].std()) * np.sqrt(252.0)
        else:
            sharpe = 0.0

        avg_holding_days = float(df["holding_days"].mean())

        exit_reasons = df["exit_reason"].value_counts().to_dict()
        quality_breakdown = df.groupby("quality")["net_pnl"].agg(["count", "sum", "mean"]).to_dict("index")

        return {
            "initial_capital": float(self.initial_capital),
            "final_capital": round(final_capital, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "total_trades": int(total_trades),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "win_rate_pct": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_win_pct": round(avg_win_pct, 2),
            "avg_loss_pct": round(avg_loss_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "avg_holding_days": round(avg_holding_days, 1),
            "exit_reasons": exit_reasons,
            "quality_breakdown": quality_breakdown,
        }
