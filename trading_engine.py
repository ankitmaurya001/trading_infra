#!/usr/bin/env python3
"""
Trading Engine - Handles trade execution, position management, and performance tracking
Separates trading logic from the live trading simulator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import os
import json
import logging

from strategies import BaseStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEngine:
    """
    Trading engine that handles:
    - Trade execution and management
    - Position tracking
    - Performance calculation
    - Logging and reporting
    """
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trade_history = []
        self.active_trades = []
        self.session_id = None
        self.symbol = None
        
        # Logging setup
        self.trade_log_file = None
        self.decision_log_file = None
        self.data_log_file = None
        
        # Performance tracking
        self.total_pnl = 0
        self.unrealized_pnl = 0
        
    def setup_logging(self, session_id: str, symbol: str):
        """
        Setup logging files for the trading session.
        
        Args:
            session_id: Unique session identifier
            symbol: Trading symbol
        """
        self.session_id = session_id
        self.symbol = symbol
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Setup log files
        self.trade_log_file = f"logs/live_trades_{session_id}.csv"
        self.decision_log_file = f"logs/live_decisions_{session_id}.csv"
        self.data_log_file = f"logs/live_data_{session_id}.csv"
        
        # Initialize trade log
        trade_log_df = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'strategy', 'action', 'price', 'quantity', 
            'balance', 'pnl', 'trade_id', 'status'
        ])
        trade_log_df.to_csv(self.trade_log_file, index=False)
        
        # Initialize decision log
        decision_log_df = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'strategy', 'signal', 'signal_name', 'current_price',
            'current_balance', 'active_trades_count', 'position_type', 'take_profit', 'stop_loss', 'trade_status'
        ])
        decision_log_df.to_csv(self.decision_log_file, index=False)
        
        print(f"📁 Logging setup complete:")
        print(f"   Trade Log: {self.trade_log_file}")
        print(f"   Decision Log: {self.decision_log_file}")
        print(f"   Data Log: {self.data_log_file}")
    
    def execute_trade(self, strategy: BaseStrategy, action: str, price: float, timestamp: datetime) -> Dict:
        """
        Execute a new trade.
        
        Args:
            strategy: Strategy instance that generated the signal
            action: Trade action ('BUY' or 'SELL')
            price: Execution price
            timestamp: Trade timestamp
            
        Returns:
            Dictionary containing trade details
        """
        # Calculate position size (use full available balance for maximum compounding)
        position_size = self.current_balance
        quantity = position_size / price
        
        # Create trade record
        trade = {
            'id': len(self.trade_history) + 1,
            'strategy': strategy.name,
            'action': action,
            'entry_price': price,
            'entry_time': timestamp,
            'quantity': quantity,
            'status': 'open',
            'pnl': 0
        }
        
        # Update balance (all balance is now in the position)
        self.current_balance = 0
        
        # Add to active trades
        self.active_trades.append(trade)
        self.trade_history.append(trade)
        
        # Log trade
        self._log_trade(trade, price, timestamp)
        
        # Print trade execution log
        print(f"🔄 [{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - {action} {quantity:.4f} {self.symbol} @ ${price:.2f}")
        print(f"💰 Position Size: ${position_size:.2f} (Full Balance), New Balance: ${self.current_balance:.2f}")
        
        return trade
    
    def close_trades(self, strategy: BaseStrategy, position_type: str, price: float, timestamp: datetime) -> List[Dict]:
        """
        Close existing trades.
        
        Args:
            strategy: Strategy instance
            position_type: Type of position to close ('LONG' or 'SHORT')
            price: Current price
            timestamp: Close timestamp
            
        Returns:
            List of closed trades
        """
        closed_trades = []
        
        for trade in self.active_trades[:]:  # Copy list to avoid modification during iteration
            if trade['strategy'] == strategy.name and trade['status'] == 'open':
                if (position_type == 'LONG' and trade['action'] == 'BUY') or \
                   (position_type == 'SHORT' and trade['action'] == 'SELL'):
                    
                    # Calculate PnL
                    if trade['action'] == 'BUY':
                        pnl = (price - trade['entry_price']) / trade['entry_price']
                    else:
                        pnl = (trade['entry_price'] - price) / trade['entry_price']
                    
                    # Update trade
                    trade['exit_price'] = price
                    trade['exit_time'] = timestamp
                    trade['pnl'] = pnl
                    
                    # Get detailed status from strategy's trade history
                    strategy_trades = strategy.get_trade_history()
                    if not strategy_trades.empty:
                        # Find the most recent closed trade from the strategy
                        recent_trades = strategy_trades.tail(1)
                        if not recent_trades.empty:
                            detailed_status = recent_trades['Status'].iloc[0]
                            trade['status'] = detailed_status
                        else:
                            trade['status'] = 'closed'
                    else:
                        trade['status'] = 'closed'
                    
                    # Update balance - add back original position value plus profit/loss
                    original_position_value = trade['quantity'] * trade['entry_price']
                    
                    if trade['action'] == 'BUY':
                        # For LONG positions: profit = current_value - original_value
                        current_position_value = trade['quantity'] * price
                        profit_loss = current_position_value - original_position_value
                    else:
                        # For SHORT positions: profit = original_value - current_value
                        # We sold at entry_price, buy back at price
                        current_position_value = trade['quantity'] * price
                        profit_loss = original_position_value - current_position_value
                    
                    # Add back the original position value plus any profit/loss
                    self.current_balance += original_position_value + profit_loss
                    
                    # Remove from active trades
                    self.active_trades.remove(trade)
                    closed_trades.append(trade)
                    
                    # Log trade closure
                    self._log_trade(trade, price, timestamp, is_exit=True)
                    
                    # Print trade closure log
                    print(f"✅ [{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - CLOSED {trade['action']} position")
                    print(f"📈 PnL: {pnl:.2%} (${profit_loss:.2f})")
                    print(f"💰 Original Position: ${original_position_value:.2f}, Profit/Loss: ${profit_loss:.2f}")
                    print(f"💰 New Balance: ${self.current_balance:.2f}")
        
        return closed_trades
    
    def process_strategy_signals(self, strategy: BaseStrategy, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Process strategy signals and execute trades.
        
        Args:
            strategy: Strategy instance
            data: Market data
            current_time: Current timestamp
            
        Returns:
            Dictionary containing signal processing results
        """
        # Generate signals
        signals_data = strategy.generate_signals(data)
        
        if signals_data is None or signals_data.empty:
            return {'signal': None, 'action': None, 'trades_executed': []}
            
        # Get the latest signal from the 'Signal' column
        latest_signal = int(signals_data['Signal'].iloc[-1])
        current_price = float(data['Close'].iloc[-1])
        
        # Log signal generation
        signal_name = {
            0: "HOLD",
            1: "LONG_ENTRY", 
            -1: "SHORT_ENTRY",
            2: "LONG_EXIT",
            -2: "SHORT_EXIT"
        }.get(latest_signal, f"UNKNOWN({latest_signal})")
        
        print(f"📊 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - Signal: {signal_name} @ ${current_price:.2f}")
        
        # Log decision to CSV
        self._log_decision(strategy, latest_signal, signal_name, current_price, current_time, signals_data)
        
        trades_executed = []
        action_taken = None
        
        # Check for entry signals (only if no active trades exist and sufficient balance)
        if latest_signal == 1 and self.current_balance > 0 and len(self.active_trades) == 0:  # Long entry
            print(f"🟢 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - LONG ENTRY signal detected")
            trade = self.execute_trade(strategy, 'BUY', current_price, current_time)
            trades_executed.append(trade)
            action_taken = 'LONG_ENTRY'
        elif latest_signal == -1 and self.current_balance > 0 and len(self.active_trades) == 0:  # Short entry
            print(f"🔴 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - SHORT ENTRY signal detected")
            trade = self.execute_trade(strategy, 'SELL', current_price, current_time)
            trades_executed.append(trade)
            action_taken = 'SHORT_ENTRY'
        elif (latest_signal == 1 or latest_signal == -1) and len(self.active_trades) > 0:
            # Signal ignored because active trade exists
            signal_type = "LONG ENTRY" if latest_signal == 1 else "SHORT ENTRY"
            print(f"⚠️  [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - {signal_type} signal IGNORED (active trade exists)")
        elif (latest_signal == 1 or latest_signal == -1) and self.current_balance <= 0:
            # Signal ignored because insufficient balance
            signal_type = "LONG ENTRY" if latest_signal == 1 else "SHORT ENTRY"
            print(f"⚠️  [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - {signal_type} signal IGNORED (insufficient balance: ${self.current_balance:.2f})")
        
        # Check for exit signals
        if latest_signal == 2:  # Long exit
            print(f"🟢 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - LONG EXIT signal detected")
            closed_trades = self.close_trades(strategy, 'LONG', current_price, current_time)
            trades_executed.extend(closed_trades)
            action_taken = 'LONG_EXIT'
        elif latest_signal == -2:  # Short exit
            print(f"🔴 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - SHORT EXIT signal detected")
            closed_trades = self.close_trades(strategy, 'SHORT', current_price, current_time)
            trades_executed.extend(closed_trades)
            action_taken = 'SHORT_EXIT'
        
        return {
            'signal': latest_signal,
            'signal_name': signal_name,
            'action': action_taken,
            'trades_executed': trades_executed,
            'current_price': current_price
        }
    
    def get_current_status(self, current_data: pd.DataFrame = None) -> Dict:
        """
        Get current trading status.
        
        Args:
            current_data: Current market data for unrealized PnL calculation
            
        Returns:
            Dictionary containing current trading status
        """
        # Calculate unrealized PnL from active trades
        unrealized_pnl = 0
        active_trade_info = None
        
        for trade in self.active_trades:
            if trade['status'] == 'open':
                # Calculate current value of the position
                current_price = current_data['Close'].iloc[-1] if current_data is not None and not current_data.empty else trade['entry_price']
                current_value = trade['quantity'] * current_price
                original_value = trade['quantity'] * trade['entry_price']
                
                if trade['action'] == 'BUY':
                    unrealized_pnl += current_value - original_value
                else:  # SELL (short)
                    unrealized_pnl += original_value - current_value
                
                # Store active trade info for display
                active_trade_info = {
                    'strategy': trade['strategy'],
                    'action': trade['action'],
                    'entry_price': trade['entry_price'],
                    'quantity': trade['quantity'],
                    'entry_time': trade['entry_time']
                }
        
        total_pnl = self.current_balance - self.initial_balance
        total_value = self.current_balance + unrealized_pnl
        
        return {
            'current_balance': self.current_balance,
            'total_pnl': total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_value': total_value,
            'active_trades': len(self.active_trades),
            'total_trades': len(self.trade_history),
            'active_trade_info': active_trade_info,
            'can_trade': len(self.active_trades) == 0 and self.current_balance > 0
        }
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.
        
        Returns:
            DataFrame containing trade history
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics from trade history.
        
        Returns:
            Dictionary containing performance metrics
        """
        trade_history_df = self.get_trade_history_df()
        
        if trade_history_df.empty:
            return self._get_empty_metrics()
        
        # Filter closed trades
        closed_trades = trade_history_df[trade_history_df['status'] == 'closed'].copy()
        
        if closed_trades.empty:
            return self._get_empty_metrics()
        
        # Sort by exit time for proper chronological order
        closed_trades = closed_trades.sort_values('exit_time')
        
        # Separate winning and losing trades
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        losing_trades = closed_trades[closed_trades['pnl'] < 0]
        
        # Basic metrics
        total_trades = len(closed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_return = closed_trades['pnl'].mean() if total_trades > 0 else 0
        
        # Calculate total PnL using compounding
        total_multiplier = 1.0
        for _, trade in closed_trades.iterrows():
            total_multiplier *= (1 + trade['pnl'])
        total_pnl = total_multiplier - 1.0
        
        # Geometric mean return
        geometric_mean_return = (total_multiplier ** (1.0 / total_trades)) - 1.0 if total_trades > 0 else 0
        
        # Win/Loss metrics
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        largest_win = closed_trades['pnl'].max() if total_trades > 0 else 0
        largest_loss = closed_trades['pnl'].min() if total_trades > 0 else 0
        
        # Risk-reward ratio (win/loss ratio)
        risk_reward_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
        
        # Profit factor
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf') if total_profit > 0 else 0
        
        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_consecutive_wins = 0
        current_consecutive_losses = 0
        
        for _, trade in closed_trades.iterrows():
            if trade['pnl'] > 0:
                current_consecutive_wins += 1
                current_consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
            else:
                current_consecutive_losses += 1
                current_consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
        
        # Sharpe ratio
        returns = closed_trades['pnl'].tolist()
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            sharpe_ratio = mean_return / max(std_return, 1e-8)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        if len(returns) > 1:
            mean_return = np.mean(returns)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_deviation = np.std(downside_returns, ddof=1)
                sortino_ratio = mean_return / max(downside_deviation, 1e-8)
            else:
                sortino_ratio = float('inf') if mean_return > 0 else 0
        else:
            sortino_ratio = 0
        
        # Max drawdown
        cumulative_returns = []
        cumulative_return = 1.0
        for _, trade in closed_trades.iterrows():
            cumulative_return *= (1 + trade['pnl'])
            cumulative_returns.append(cumulative_return)
        
        max_drawdown = 0
        peak = 1.0
        for cr in cumulative_returns:
            if cr > peak:
                peak = cr
            drawdown = (peak - cr) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calmar ratio
        if len(closed_trades) > 1:
            first_trade_date = closed_trades['exit_time'].min()
            last_trade_date = closed_trades['exit_time'].max()
            trading_days = (last_trade_date - first_trade_date).days
            if trading_days > 0:
                annualized_return = (total_multiplier ** (365 / trading_days)) - 1
            else:
                annualized_return = (1 + geometric_mean_return) ** 252 - 1
        else:
            annualized_return = (1 + geometric_mean_return) ** 252 - 1
        
        calmar_ratio = annualized_return / max(max_drawdown, 1e-8)
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "total_pnl": total_pnl,
            "geometric_mean_return": geometric_mean_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "consecutive_wins": max_consecutive_wins,
            "consecutive_losses": max_consecutive_losses,
            "risk_reward_ratio": risk_reward_ratio,
            "calmar_ratio": calmar_ratio,
            "sortino_ratio": sortino_ratio
        }
    
    def _get_empty_metrics(self) -> Dict:
        """
        Get empty metrics structure.
        
        Returns:
            Dictionary with zero values for all metrics
        """
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "total_pnl": 0,
            "geometric_mean_return": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "profit_factor": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "risk_reward_ratio": 0,
            "calmar_ratio": 0,
            "sortino_ratio": 0
        }
    
    def _log_trade(self, trade: Dict, price: float, timestamp: datetime, is_exit: bool = False):
        """
        Log trade to CSV file.
        
        Args:
            trade: Trade dictionary
            price: Trade price
            timestamp: Trade timestamp
            is_exit: Whether this is an exit trade
        """
        if self.trade_log_file:
            log_entry = {
                'timestamp': timestamp,
                'symbol': self.symbol,
                'strategy': trade['strategy'],
                'action': 'EXIT' if is_exit else trade['action'],
                'price': price,
                'quantity': trade['quantity'],
                'balance': self.current_balance,
                'pnl': trade.get('pnl', 0),
                'trade_id': trade['id'],
                'status': trade['status']
            }
            
            # Append to CSV
            log_df = pd.DataFrame([log_entry])
            log_df.to_csv(self.trade_log_file, mode='a', header=False, index=False)
    
    def _log_decision(self, strategy: BaseStrategy, signal: int, signal_name: str, 
                     current_price: float, current_time: datetime, signals_data: pd.DataFrame):
        """
        Log algorithm decision to CSV file.
        
        Args:
            strategy: Strategy instance
            signal: Signal value
            signal_name: Human-readable signal name
            current_price: Current price
            current_time: Current timestamp
            signals_data: Signals data from strategy
        """
        if self.decision_log_file:
            # Get current position info
            active_trades_count = len(self.active_trades)
            position_type = "NONE"
            take_profit = None
            stop_loss = None
            
            # Get take profit and stop loss from signals data if available
            if 'Take_Profit' in signals_data.columns and 'Stop_Loss' in signals_data.columns:
                take_profit = signals_data['Take_Profit'].iloc[-1]
                stop_loss = signals_data['Stop_Loss'].iloc[-1]
                
                # Handle NaN values
                if pd.isna(take_profit):
                    take_profit = None
                if pd.isna(stop_loss):
                    stop_loss = None
            
            # Find active trade for this strategy
            trade_status = "NONE"
            for trade in self.active_trades:
                if trade['strategy'] == strategy.name:
                    position_type = trade['action']
                    trade_status = trade.get('status', 'open')
                    break
            
            log_entry = {
                'timestamp': current_time,
                'symbol': self.symbol,
                'strategy': strategy.name,
                'signal': signal,
                'signal_name': signal_name,
                'current_price': current_price,
                'current_balance': self.current_balance,
                'active_trades_count': active_trades_count,
                'position_type': position_type,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'trade_status': trade_status
            }
            
            # Append to CSV
            log_df = pd.DataFrame([log_entry])
            log_df.to_csv(self.decision_log_file, mode='a', header=False, index=False)
    
    def save_data_to_csv(self, data: pd.DataFrame, fetch_time: datetime):
        """
        Save fetched data to CSV file.
        
        Args:
            data: Market data
            fetch_time: Data fetch timestamp
        """
        if self.data_log_file and not data.empty:
            # Add fetch timestamp to data
            data_copy = data.copy()
            data_copy['fetch_timestamp'] = fetch_time
            
            # Save to CSV (append mode)
            data_copy.to_csv(self.data_log_file, mode='a', header=not os.path.exists(self.data_log_file), index=True)
            
            print(f"💾 [{fetch_time.strftime('%Y-%m-%d %H:%M:%S')}] Data saved to {self.data_log_file}")
    
    def get_log_files_info(self) -> Dict:
        """
        Get information about log files.
        
        Returns:
            Dictionary containing log file paths
        """
        return {
            'trade_log': self.trade_log_file,
            'data_log': self.data_log_file,
            'decision_log': self.decision_log_file,
            'session_id': self.session_id
        }
    
    def get_decision_log_df(self) -> pd.DataFrame:
        """
        Get decision log as DataFrame.
        
        Returns:
            DataFrame containing decision log
        """
        if not self.decision_log_file or not os.path.exists(self.decision_log_file):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.decision_log_file)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error reading decision log: {e}")
            return pd.DataFrame()
    
    def reset(self):
        """
        Reset the trading engine to initial state.
        """
        self.current_balance = self.initial_balance
        self.trade_history = []
        self.active_trades = []
        self.total_pnl = 0
        self.unrealized_pnl = 0
        print("🔄 Trading engine reset to initial state.")
