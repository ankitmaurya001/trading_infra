#!/usr/bin/env python3
"""
Live Trading Simulator with Real-time Data Polling and Trade Execution
Combines strategy optimization with live trading simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

from data_fetcher import DataFetcher
from strategies import MovingAverageCrossover, RSIStrategy, DonchianChannelBreakout
from strategy_optimizer import (
    optimize_moving_average_crossover,
    optimize_rsi_strategy,
    optimize_donchian_channel
)
from visualizer import Visualizer

class LiveTradingSimulator:
    def __init__(self):
        self.is_running = False
        self.current_data = pd.DataFrame()
        self.trade_history = []
        self.initial_balance = 10000  # $10,000 starting balance
        self.current_balance = self.initial_balance
        self.active_trades = []
        self.strategies = []
        self.optimized_params = {}
        self.data_fetcher = DataFetcher()
        self.last_update = None
        self.trade_log_file = None
        self.data_log_file = None
        self.decision_log_file = None
        self.session_id = None
        
        # Mock live trading settings
        self.mock_mode = False
        self.mock_data = pd.DataFrame()
        self.mock_current_index = 0
        self.mock_start_date = None
        self.mock_end_date = None
        
    def set_manual_parameters(self, ma_params=None, rsi_params=None, donchian_params=None):
        """Set strategy parameters manually (from optimization results)"""
        self.optimized_params = {}
        
        if ma_params:
            self.optimized_params['ma'] = ma_params
        if rsi_params:
            self.optimized_params['rsi'] = rsi_params
        if donchian_params:
            self.optimized_params['donchian'] = donchian_params
            
        st.success("Strategy parameters set successfully!")
        return True
    
    def initialize_strategies(self, enabled_strategies):
        """Initialize strategies with optimized parameters"""
        self.strategies = []
        
        if 'ma' in enabled_strategies and 'ma' in self.optimized_params:
            params = self.optimized_params['ma']
            strategy = MovingAverageCrossover(
                short_window=params['short_window'],
                long_window=params['long_window'],
                risk_reward_ratio=params['risk_reward_ratio'],
                trading_fee=params['trading_fee']
            )
            self.strategies.append(strategy)
            
        if 'rsi' in enabled_strategies and 'rsi' in self.optimized_params:
            params = self.optimized_params['rsi']
            strategy = RSIStrategy(
                period=params['period'],
                overbought=params['overbought'],
                oversold=params['oversold'],
                risk_reward_ratio=params['risk_reward_ratio'],
                trading_fee=params['trading_fee']
            )
            self.strategies.append(strategy)
            
        if 'donchian' in enabled_strategies and 'donchian' in self.optimized_params:
            params = self.optimized_params['donchian']
            strategy = DonchianChannelBreakout(
                channel_period=params['channel_period'],
                risk_reward_ratio=params['risk_reward_ratio'],
                trading_fee=params['trading_fee']
            )
            self.strategies.append(strategy)
    
    def start_live_trading(self, symbol, interval, polling_frequency, enabled_strategies, mock_mode=False, mock_days_back=10, mock_delay=0.01):
        """Start the live trading simulation"""
        if not self.strategies:
            st.error("No strategies initialized. Please set parameters first.")
            return
            
        self.is_running = True
        self.symbol = symbol
        self.interval = interval
        self.polling_frequency = polling_frequency
        self.mock_mode = mock_mode
        
        # Create session ID and log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_mock" if mock_mode else "_live"
        self.session_id = f"{symbol}_{timestamp}{mode_suffix}"
        
        # Trade log file
        self.trade_log_file = f"logs/live_trades_{self.session_id}.csv"
        
        # Data log file
        self.data_log_file = f"logs/live_data_{self.session_id}.csv"
        
        # Decision log file
        self.decision_log_file = f"logs/live_decisions_{self.session_id}.csv"
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Initialize trade log
        trade_log_df = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'strategy', 'action', 'price', 'quantity', 
            'balance', 'pnl', 'trade_id', 'status'
        ])
        trade_log_df.to_csv(self.trade_log_file, index=False)
        
        # Initialize decision log
        decision_log_df = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'strategy', 'signal', 'signal_name', 'current_price',
            'current_balance', 'active_trades_count', 'position_type', 'take_profit', 'stop_loss'
        ])
        decision_log_df.to_csv(self.decision_log_file, index=False)
        
        # Setup mock data if in mock mode
        if mock_mode:
            self._setup_mock_data(symbol, interval, mock_days_back)
            self.mock_delay = mock_delay
        
        print(f"🚀 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Live trading started!")
        print(f"📊 Symbol: {symbol}")
        print(f"⏱️  Interval: {interval}")
        print(f"📡 Polling Frequency: {polling_frequency} seconds")
        print(f"🎭 Mode: {'MOCK' if mock_mode else 'LIVE'}")
        if mock_mode:
            print(f"📅 Mock Data: {mock_days_back} days back from today")
        print(f"📋 Trade Log: {self.trade_log_file}")
        print(f"📊 Data Log: {self.data_log_file}")
        print(f"🎯 Decision Log: {self.decision_log_file}")
        print(f"🎯 Active Strategies: {[s.name for s in self.strategies]}")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print("=" * 60)
        
        st.success(f"Live trading started! Trades will be logged to {self.trade_log_file}")
        
        # Start the trading loop in a separate thread
        trading_thread = threading.Thread(
            target=self._trading_loop,
            args=(symbol, interval, polling_frequency)
        )
        trading_thread.daemon = True
        trading_thread.start()
    
    def _setup_mock_data(self, symbol, interval, mock_days_back):
        """Setup mock data for testing"""
        print(f"🎭 Setting up mock data for {symbol}...")
        
        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        if interval in ["5m", "15m", "30m", "1h"]:
            # For intraday data, fetch more historical data
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        else:
            # For daily data, fetch more historical data
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        print(f"📥 Fetching historical data from {start_date} to {end_date}")
        
        # Fetch all historical data
        data = self.data_fetcher.fetch_data(symbol, start_date, end_date, interval=interval)
        
        if data.empty:
            print(f"❌ No data fetched for {symbol}")
            return
        
        print(f"✅ Fetched {len(data)} data points")
        
        # Calculate the mock start point (mock_days_back from the end)
        if interval in ["5m", "15m", "30m", "1h"]:
            # For intraday, calculate approximate data points
            if interval == "5m":
                points_per_day = 288  # 24 * 12
            elif interval == "15m":
                points_per_day = 96   # 24 * 4
            elif interval == "30m":
                points_per_day = 48   # 24 * 2
            elif interval == "1h":
                points_per_day = 24   # 24 * 1
            
            mock_points_back = mock_days_back * points_per_day
            self.mock_current_index = max(0, len(data) - mock_points_back)
        else:
            # For daily data
            self.mock_current_index = max(0, len(data) - mock_days_back)
        
        # Store the mock data
        self.mock_data = data
        self.mock_start_date = data.index[self.mock_current_index]
        self.mock_end_date = data.index[-1]
        
        print(f"🎭 Mock simulation will start from: {self.mock_start_date}")
        print(f"🎭 Mock simulation will end at: {self.mock_end_date}")
        print(f"🎭 Total mock data points: {len(data) - self.mock_current_index}")
        print(f"🎭 Historical data points: {self.mock_current_index}")
    
    def stop_live_trading(self):
        """Stop the live trading simulation"""
        self.is_running = False
        print(f"🛑 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Live trading stopped.")
        print(f"📊 Final Balance: ${self.current_balance:,.2f}")
        print(f"📈 Total PnL: ${self.current_balance - self.initial_balance:,.2f}")
        print(f"📋 Total Trades: {len(self.trade_history)}")
        print("=" * 60)
        st.info("Live trading stopped.")
    
    def _trading_loop(self, symbol, interval, polling_frequency):
        """Main trading loop that runs in a separate thread"""
        print(f"🚀 Starting {'mock' if self.mock_mode else 'live'} trading loop for {symbol} with {interval} interval")
        print(f"📡 Polling frequency: {polling_frequency} seconds")
        
        if self.mock_mode:
            self._mock_trading_loop(symbol, interval, polling_frequency)
        else:
            self._live_trading_loop(symbol, interval, polling_frequency)
    
    def _live_trading_loop(self, symbol, interval, polling_frequency):
        """Live trading loop that fetches real-time data"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Calculate date range for data fetching
                # For live trading, we need enough historical data for indicators
                if interval in ["5m", "15m", "30m", "1h"]:
                    # For intraday data, fetch last 7 days
                    start_date = (current_time - timedelta(days=7)).strftime("%Y-%m-%d")
                else:
                    # For daily data, fetch last 60 days
                    start_date = (current_time - timedelta(days=60)).strftime("%Y-%m-%d")
                
                end_date = current_time.strftime("%Y-%m-%d")
                
                print(f"📥 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Fetching {interval} data for {symbol} from {start_date} to {end_date}")
                
                data = self.data_fetcher.fetch_data(symbol, start_date, end_date, interval=interval)
                
                if not data.empty:
                    print(f"✅ [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Successfully fetched {len(data)} data points")
                    print(f"📊 Latest data: {data.index[-1]} - Close: ${data['Close'].iloc[-1]:.2f}")
                    
                    # Save data to CSV
                    self._save_data_to_csv(data, current_time)
                    
                    self.current_data = data
                    self.last_update = current_time
                    
                    # Process each strategy
                    for strategy in self.strategies:
                        self._process_strategy(strategy, data)
                else:
                    print(f"⚠️  [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] No data received for {symbol}")
                
                # Wait for next polling cycle
                print(f"⏳ [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting {polling_frequency} seconds until next update...")
                time.sleep(polling_frequency)
                
            except Exception as e:
                error_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"❌ [{error_time}] Error in trading loop: {str(e)}")
                time.sleep(polling_frequency)
    
    def _mock_trading_loop(self, symbol, interval, polling_frequency):
        """Mock trading loop that processes historical data sequentially"""
        print(f"🎭 Starting mock trading simulation...")
        print(f"⚡ Mock mode: Running as fast as possible (ignoring polling frequency)")
        
        start_time = time.time()
        
        while self.is_running and self.mock_current_index < len(self.mock_data):
            try:
                # Get current data point
                current_data_point = self.mock_data.iloc[self.mock_current_index]
                current_time = current_data_point.name  # This is the timestamp
                
                # Create data up to current point for strategy processing
                data = self.mock_data.iloc[:self.mock_current_index + 1]
                
                # Calculate progress
                progress = (self.mock_current_index / len(self.mock_data)) * 100
                print(f"🎭 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Processing mock data point {self.mock_current_index + 1}/{len(self.mock_data)} ({progress:.1f}%)")
                print(f"📊 Mock data: {current_time} - Close: ${current_data_point['Close']:.2f}")
                
                # Save data to CSV
                self._save_data_to_csv(data, current_time)
                
                self.current_data = data
                self.last_update = current_time
                
                # Process each strategy
                for strategy in self.strategies:
                    self._process_strategy(strategy, data)
                
                # Move to next data point
                self.mock_current_index += 1
                
                # Use configurable delay in mock mode
                time.sleep(self.mock_delay)
                
            except Exception as e:
                error_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"❌ [{error_time}] Error in mock trading loop: {str(e)}")
                time.sleep(0.1)  # Small delay on error
        
        end_time = time.time()
        simulation_duration = end_time - start_time
        
        if self.mock_current_index >= len(self.mock_data):
            print(f"🎭 Mock trading simulation completed!")
            print(f"📊 Processed all {len(self.mock_data)} data points")
            print(f"⏱️  Simulation duration: {simulation_duration:.2f} seconds")
            print(f"🚀 Average speed: {len(self.mock_data)/simulation_duration:.1f} data points/second")
            self.is_running = False
    
    def _process_strategy(self, strategy, data):
        """Process a single strategy and execute trades"""
        # Generate signals
        signals_data = strategy.generate_signals(data)
        
        if signals_data is None or signals_data.empty:
            return
            
        # Get the latest signal from the 'Signal' column
        latest_signal = int(signals_data['Signal'].iloc[-1])  # Convert to integer
        current_price = float(data['Close'].iloc[-1])  # Convert to float
        current_time = data.index[-1]
        
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
        
        # Check for entry signals (only if no active trades exist and sufficient balance)
        if latest_signal == 1 and self.current_balance > 0 and len(self.active_trades) == 0:  # Long entry
            print(f"🟢 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - LONG ENTRY signal detected")
            self._execute_trade(strategy, 'BUY', current_price, current_time)
        elif latest_signal == -1 and self.current_balance > 0 and len(self.active_trades) == 0:  # Short entry
            print(f"🔴 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - SHORT ENTRY signal detected")
            self._execute_trade(strategy, 'SELL', current_price, current_time)
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
            self._close_trades(strategy, 'LONG', current_price, current_time)
        elif latest_signal == -2:  # Short exit
            print(f"🔴 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - SHORT EXIT signal detected")
            self._close_trades(strategy, 'SHORT', current_price, current_time)
    
    def _execute_trade(self, strategy, action, price, timestamp):
        """Execute a new trade"""
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
    
    def _close_trades(self, strategy, position_type, price, timestamp):
        """Close existing trades"""
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
                    trade['status'] = 'closed'
                    trade['pnl'] = pnl
                    
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
                    
                    # Log trade closure
                    self._log_trade(trade, price, timestamp, is_exit=True)
                    
                    # Print trade closure log
                    print(f"✅ [{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {strategy.name} - CLOSED {trade['action']} position")
                    print(f"📈 PnL: {pnl:.2%} (${profit_loss:.2f})")
                    print(f"💰 Original Position: ${original_position_value:.2f}, Profit/Loss: ${profit_loss:.2f}")
                    print(f"💰 New Balance: ${self.current_balance:.2f}")
    
    def _log_trade(self, trade, price, timestamp, is_exit=False):
        """Log trade to CSV file"""
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
    
    def _save_data_to_csv(self, data, fetch_time):
        """Save fetched data to CSV file"""
        if self.data_log_file and not data.empty:
            # Add fetch timestamp to data
            data_copy = data.copy()
            data_copy['fetch_timestamp'] = fetch_time
            
            # Save to CSV (append mode)
            data_copy.to_csv(self.data_log_file, mode='a', header=not os.path.exists(self.data_log_file), index=True)
            
            print(f"💾 [{fetch_time.strftime('%Y-%m-%d %H:%M:%S')}] Data saved to {self.data_log_file}")
    
    def _log_decision(self, strategy, signal, signal_name, current_price, current_time, signals_data):
        """Log algorithm decision to CSV file"""
        if self.decision_log_file:
            # Get current position info
            active_trades_count = len(self.active_trades)
            position_type = "NONE"
            take_profit = None
            stop_loss = None
            
            # Find active trade for this strategy
            for trade in self.active_trades:
                if trade['strategy'] == strategy.name:
                    position_type = trade['action']
                    # Get take profit and stop loss from signals data if available
                    if 'Take_Profit' in signals_data.columns and 'Stop_Loss' in signals_data.columns:
                        take_profit = signals_data['Take_Profit'].iloc[-1]
                        stop_loss = signals_data['Stop_Loss'].iloc[-1]
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
                'stop_loss': stop_loss
            }
            
            # Append to CSV
            log_df = pd.DataFrame([log_entry])
            log_df.to_csv(self.decision_log_file, mode='a', header=False, index=False)
    
    def get_current_status(self):
        """Get current trading status"""
        # Calculate unrealized PnL from active trades
        unrealized_pnl = 0
        active_trade_info = None
        
        for trade in self.active_trades:
            if trade['status'] == 'open':
                # Calculate current value of the position
                current_price = self.current_data['Close'].iloc[-1] if not self.current_data.empty else trade['entry_price']
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
        
        return {
            'is_running': self.is_running,
            'current_balance': self.current_balance,
            'total_pnl': self.current_balance - self.initial_balance,
            'unrealized_pnl': unrealized_pnl,
            'total_value': self.current_balance + unrealized_pnl,
            'active_trades': len(self.active_trades),
            'total_trades': len(self.trade_history),
            'last_update': self.last_update,
            'active_trade_info': active_trade_info,
            'can_trade': len(self.active_trades) == 0 and self.current_balance > 0
        }
    
    def get_trade_history_df(self):
        """Get trade history as DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def get_log_files_info(self):
        """Get information about log files"""
        return {
            'trade_log': self.trade_log_file,
            'data_log': self.data_log_file,
            'decision_log': self.decision_log_file,
            'session_id': self.session_id
        }
    
    def get_decision_log_df(self):
        """Get decision log as DataFrame"""
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
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics from trade history"""
        trade_history_df = self.get_trade_history_df()
        
        if trade_history_df.empty:
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
        
        # Filter closed trades
        closed_trades = trade_history_df[trade_history_df['status'] == 'closed'].copy()
        
        if closed_trades.empty:
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

def main():
    st.set_page_config(page_title="Live Trading Simulator", layout="wide")
    st.title("🚀 Live Trading Simulator")
    st.markdown("Real-time trading simulation with optimized strategies")
    
    # Initialize simulator
    if 'simulator' not in st.session_state:
        st.session_state.simulator = LiveTradingSimulator()
    
    simulator = st.session_state.simulator
    
    # Sidebar configuration
    st.sidebar.header("📊 Configuration")
    
    # Symbol and data settings
    symbol = st.sidebar.text_input("Symbol", value="BTC-USD")
    
    interval = st.sidebar.selectbox(
        "Data Interval",
        options=["1d", "5m", "15m", "30m", "1h"],
        index=2,
        help="Select the timeframe for live data polling"
    )
    
    # Strategy selection
    st.sidebar.subheader("🎯 Strategies")
    enabled_strategies = st.sidebar.multiselect(
        "Select strategies to use",
        options=['ma', 'rsi', 'donchian'],
        default=['ma', 'rsi'],
        help="Select which strategies to use for trading"
    )
    
    # Manual Parameter Input
    st.sidebar.subheader("⚙️ Strategy Parameters")
    
    # Moving Average Parameters
    if 'ma' in enabled_strategies:
        st.sidebar.markdown("**Moving Average Crossover**")
        ma_short = st.sidebar.number_input("Short MA Period", min_value=5, max_value=50, value=20, key="ma_short")
        ma_long = st.sidebar.number_input("Long MA Period", min_value=10, max_value=200, value=50, key="ma_long")
        ma_risk_reward = st.sidebar.slider("Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="ma_rr")
        ma_trading_fee = st.sidebar.slider("Trading Fee (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="ma_fee")
    
    # RSI Parameters
    if 'rsi' in enabled_strategies:
        st.sidebar.markdown("**RSI Strategy**")
        rsi_period = st.sidebar.number_input("RSI Period", min_value=5, max_value=50, value=14, key="rsi_period")
        rsi_overbought = st.sidebar.slider("Overbought Level", min_value=50, max_value=90, value=70, key="rsi_over")
        rsi_oversold = st.sidebar.slider("Oversold Level", min_value=10, max_value=50, value=30, key="rsi_under")
        rsi_risk_reward = st.sidebar.slider("Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="rsi_rr")
        rsi_trading_fee = st.sidebar.slider("Trading Fee (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="rsi_fee")
    
    # Donchian Channel Parameters
    if 'donchian' in enabled_strategies:
        st.sidebar.markdown("**Donchian Channel**")
        donchian_period = st.sidebar.number_input("Channel Period", min_value=5, max_value=50, value=20, key="donchian_period")
        donchian_risk_reward = st.sidebar.slider("Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="donchian_rr")
        donchian_trading_fee = st.sidebar.slider("Trading Fee (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="donchian_fee")
    
    # Trading settings
    st.sidebar.subheader("💰 Trading Settings")
    initial_balance = st.sidebar.number_input(
        "Initial Balance ($)",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000
    )
    
    polling_frequency = st.sidebar.slider(
        "Polling Frequency (seconds)",
        min_value=1,
        max_value=300,
        value=60,
        step=5
    )
    
    # Mock mode settings
    st.sidebar.subheader("🎭 Testing Mode")
    mock_mode = st.sidebar.checkbox(
        "Enable Mock Mode",
        value=False,
        help="Use historical data to simulate live trading for testing"
    )
    
    if mock_mode:
        mock_days_back = st.sidebar.slider(
            "Mock Days Back",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            help="How many days back to start the mock simulation"
        )
        
        mock_speed = st.sidebar.selectbox(
            "Mock Simulation Speed",
            options=["Ultra Fast (10ms)", "Fast (100ms)", "Normal (1s)", "Slow (5s)"],
            index=0,
            help="Speed of mock simulation (ignores polling frequency)"
        )
        
        # Convert speed selection to delay
        speed_delays = {
            "Ultra Fast (10ms)": 0.01,
            "Fast (100ms)": 0.1,
            "Normal (1s)": 1.0,
            "Slow (5s)": 5.0
        }
        mock_delay = speed_delays[mock_speed]
        
        st.sidebar.info(f"🎭 Mock mode will simulate trading from {mock_days_back} days ago")
        st.sidebar.info(f"⚡ Speed: {mock_speed}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Live Trading Dashboard")
        
        # Current time display
        current_time = datetime.now()
        st.caption(f"🕐 Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Status display
        status = simulator.get_current_status()
        
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        
        with status_col1:
            status_text = "🟢 Running" if status['is_running'] else "🔴 Stopped"
            if status['is_running'] and simulator.mock_mode:
                status_text += " (Mock)"
            st.metric(
                "Status",
                status_text
            )
        
        with status_col2:
            st.metric(
                "Balance",
                f"${status['current_balance']:,.2f}",
                f"{status['total_pnl']:+.2f}"
            )
            st.caption(f"Total Value: ${status['total_value']:,.2f}")
            if status['unrealized_pnl'] != 0:
                st.caption(f"Unrealized: {status['unrealized_pnl']:+.2f}")
            st.caption("💡 Full balance used per trade for maximum compounding")
        
        with status_col3:
            st.metric(
                "Active Trades",
                status['active_trades']
            )
            # Show active trade details if any
            if status['active_trade_info']:
                trade_info = status['active_trade_info']
                st.caption(f"📈 {trade_info['strategy']} - {trade_info['action']}")
                st.caption(f"Entry: ${trade_info['entry_price']:.2f}")
                st.caption(f"Qty: {trade_info['quantity']:.4f}")
                st.caption(f"Time: {trade_info['entry_time'].strftime('%H:%M:%S')}")
            elif status['active_trades'] == 0 and status['can_trade']:
                st.caption("✅ Ready to trade")
            elif status['active_trades'] == 0 and not status['can_trade']:
                st.caption("⚠️ No balance available")
        
        with status_col4:
            st.metric(
                "Total Trades",
                status['total_trades']
            )
        
        # Last update and mock progress
        if status['last_update']:
            update_text = f"Last update: {status['last_update'].strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Add mock progress if in mock mode
            if simulator.mock_mode and simulator.mock_data is not None and len(simulator.mock_data) > 0:
                progress = (simulator.mock_current_index / len(simulator.mock_data)) * 100
                update_text += f" | Mock Progress: {progress:.1f}% ({simulator.mock_current_index}/{len(simulator.mock_data)})"
            
            st.caption(update_text)
        
        # Display current parameters
        if simulator.optimized_params:
            st.subheader("📋 Current Strategy Parameters")
            param_col1, param_col2, param_col3 = st.columns(3)
            
            with param_col1:
                if 'ma' in simulator.optimized_params:
                    st.markdown("**Moving Average Crossover**")
                    params = simulator.optimized_params['ma']
                    st.write(f"Short MA: {params['short_window']}")
                    st.write(f"Long MA: {params['long_window']}")
                    st.write(f"Risk/Reward: {params['risk_reward_ratio']}")
                    st.write(f"Trading Fee: {params['trading_fee']:.3f}")
            
            with param_col2:
                if 'rsi' in simulator.optimized_params:
                    st.markdown("**RSI Strategy**")
                    params = simulator.optimized_params['rsi']
                    st.write(f"Period: {params['period']}")
                    st.write(f"Overbought: {params['overbought']}")
                    st.write(f"Oversold: {params['oversold']}")
                    st.write(f"Risk/Reward: {params['risk_reward_ratio']}")
                    st.write(f"Trading Fee: {params['trading_fee']:.3f}")
            
            with param_col3:
                if 'donchian' in simulator.optimized_params:
                    st.markdown("**Donchian Channel**")
                    params = simulator.optimized_params['donchian']
                    st.write(f"Channel Period: {params['channel_period']}")
                    st.write(f"Risk/Reward: {params['risk_reward_ratio']}")
                    st.write(f"Trading Fee: {params['trading_fee']:.3f}")
        else:
            st.info("No strategy parameters set. Please configure parameters and click 'Set Parameters'.")
    
    with col2:
        st.subheader("🎮 Controls")
        
        # Set Parameters button
        if st.button("⚙️ Set Parameters", type="primary"):
            # Collect parameters from sidebar
            ma_params = None
            rsi_params = None
            donchian_params = None
            
            if 'ma' in enabled_strategies:
                ma_params = {
                    'short_window': ma_short,
                    'long_window': ma_long,
                    'risk_reward_ratio': ma_risk_reward,
                    'trading_fee': ma_trading_fee / 100  # Convert percentage to decimal
                }
            
            if 'rsi' in enabled_strategies:
                rsi_params = {
                    'period': rsi_period,
                    'overbought': rsi_overbought,
                    'oversold': rsi_oversold,
                    'risk_reward_ratio': rsi_risk_reward,
                    'trading_fee': rsi_trading_fee / 100  # Convert percentage to decimal
                }
            
            if 'donchian' in enabled_strategies:
                donchian_params = {
                    'channel_period': donchian_period,
                    'risk_reward_ratio': donchian_risk_reward,
                    'trading_fee': donchian_trading_fee / 100  # Convert percentage to decimal
                }
            
            # Set parameters and initialize strategies
            success = simulator.set_manual_parameters(ma_params, rsi_params, donchian_params)
            if success:
                simulator.initialize_strategies(enabled_strategies)
                print(f"⚙️ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Strategy parameters set successfully!")
                print(f"🎯 Active strategies: {enabled_strategies}")
        
        # Start/Stop trading
        if not status['is_running']:
            if st.button("▶️ Start Trading", type="primary"):
                mock_delay_param = mock_delay if mock_mode else 0.01
                simulator.start_live_trading(symbol, interval, polling_frequency, enabled_strategies, mock_mode, mock_days_back if mock_mode else 10, mock_delay_param)
                st.rerun()
        else:
            if st.button("⏹️ Stop Trading", type="secondary"):
                simulator.stop_live_trading()
                st.rerun()
    
    # Log files information
    if status['is_running']:
        st.subheader("📁 Log Files")
        log_info = simulator.get_log_files_info()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption("Trade Log")
            st.code(log_info['trade_log'], language=None)
            
        with col2:
            st.caption("Data Log")
            st.code(log_info['data_log'], language=None)
            
        with col3:
            st.caption("Decision Log")
            st.code(log_info['decision_log'], language=None)
    
    # Decision Log Viewer
    if status['is_running']:
        st.subheader("🎯 Decision Log Viewer")
        
        # Auto-refresh decision log
        if st.button("🔄 Refresh Decision Log", type="secondary"):
            st.rerun()
        
        decision_log_df = simulator.get_decision_log_df()
        
        if not decision_log_df.empty:
            # Decision log controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Strategy filter
                strategies = decision_log_df['strategy'].unique()
                selected_strategy = st.selectbox(
                    "Filter by Strategy",
                    options=["All"] + list(strategies),
                    index=0
                )
            
            with col2:
                # Signal filter
                signals = decision_log_df['signal_name'].unique()
                selected_signal = st.selectbox(
                    "Filter by Signal",
                    options=["All"] + list(signals),
                    index=0
                )
            
            with col3:
                # Date range filter
                if len(decision_log_df) > 1:
                    min_date = decision_log_df['timestamp'].min()
                    max_date = decision_log_df['timestamp'].max()
                    date_range = st.date_input(
                        "Date Range",
                        value=(min_date.date(), max_date.date()),
                        min_value=min_date.date(),
                        max_value=max_date.date()
                    )
                else:
                    date_range = None
            
            # Apply filters
            filtered_df = decision_log_df.copy()
            
            if selected_strategy != "All":
                filtered_df = filtered_df[filtered_df['strategy'] == selected_strategy]
            
            if selected_signal != "All":
                filtered_df = filtered_df[filtered_df['signal_name'] == selected_signal]
            
            if date_range and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df['timestamp'].dt.date >= start_date) &
                    (filtered_df['timestamp'].dt.date <= end_date)
                ]
            
            # Display decision log
            if not filtered_df.empty:
                # Format the display
                display_df = filtered_df.copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                display_df['current_price'] = display_df['current_price'].round(2)
                display_df['current_balance'] = display_df['current_balance'].round(2)
                
                # Color code signals
                def color_signal(val):
                    if val == 'LONG_ENTRY':
                        return 'background-color: lightgreen'
                    elif val == 'SHORT_ENTRY':
                        return 'background-color: lightcoral'
                    elif val == 'LONG_EXIT':
                        return 'background-color: lightblue'
                    elif val == 'SHORT_EXIT':
                        return 'background-color: lightyellow'
                    elif val == 'HOLD':
                        return 'background-color: lightgray'
                    else:
                        return ''
                
                styled_df = display_df.style.applymap(color_signal, subset=['signal_name'])
                
                # Show decision log
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Decision statistics
                st.subheader("📊 Decision Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_decisions = len(filtered_df)
                    st.metric("Total Decisions", total_decisions)
                
                with col2:
                    if not filtered_df.empty:
                        signal_counts = filtered_df['signal_name'].value_counts()
                        most_common = signal_counts.index[0] if len(signal_counts) > 0 else "None"
                        st.metric("Most Common Signal", most_common)
                
                with col3:
                    if not filtered_df.empty:
                        avg_price = filtered_df['current_price'].mean()
                        st.metric("Avg Price", f"${avg_price:.2f}")
                
                with col4:
                    if not filtered_df.empty:
                        price_change = filtered_df['current_price'].iloc[-1] - filtered_df['current_price'].iloc[0]
                        st.metric("Price Change", f"${price_change:.2f}")
                
                # Signal distribution chart
                if not filtered_df.empty:
                    st.subheader("📈 Signal Distribution")
                    
                    signal_counts = filtered_df['signal_name'].value_counts()
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=signal_counts.index,
                            y=signal_counts.values,
                            marker_color=['lightgreen', 'lightcoral', 'lightblue', 'lightyellow', 'lightgray'][:len(signal_counts)]
                        )
                    ])
                    
                    fig.update_layout(
                        title="Signal Distribution",
                        xaxis_title="Signal Type",
                        yaxis_title="Count",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="signal_distribution_chart")
                
                # Strategy performance comparison
                if len(strategies) > 1:
                    st.subheader("🎯 Strategy Comparison")
                    
                    strategy_stats = []
                    for strategy in strategies:
                        strategy_df = filtered_df[filtered_df['strategy'] == strategy]
                        if not strategy_df.empty:
                            stats = {
                                'Strategy': strategy,
                                'Total Decisions': len(strategy_df),
                                'Avg Price': strategy_df['current_price'].mean(),
                                'Price Change': strategy_df['current_price'].iloc[-1] - strategy_df['current_price'].iloc[0],
                                'Most Common Signal': strategy_df['signal_name'].mode().iloc[0] if len(strategy_df['signal_name'].mode()) > 0 else "None"
                            }
                            strategy_stats.append(stats)
                    
                    if strategy_stats:
                        strategy_df = pd.DataFrame(strategy_stats)
                        st.dataframe(strategy_df, use_container_width=True, hide_index=True)
            else:
                st.info("No decisions match the selected filters.")
        else:
            st.info("No decision log data available yet. Decisions will appear here as they are made.")
        
        # Decision log summary
        if not decision_log_df.empty:
            st.subheader("📋 Decision Log Summary")
            
            # Show recent decisions
            recent_decisions = decision_log_df.tail(10)
            if not recent_decisions.empty:
                st.caption("Recent Decisions:")
                for _, row in recent_decisions.iterrows():
                    signal_color = {
                        'LONG_ENTRY': '🟢',
                        'SHORT_ENTRY': '🔴', 
                        'LONG_EXIT': '🔵',
                        'SHORT_EXIT': '🟡',
                        'HOLD': '⚪'
                    }.get(row['signal_name'], '❓')
                    
                    st.write(f"{signal_color} **{row['strategy']}** - {row['signal_name']} @ ${row['current_price']:.2f} ({row['timestamp'].strftime('%H:%M:%S')})")
    
    # Trading history
    st.subheader("📋 Trade History")
    
    trade_history_df = simulator.get_trade_history_df()
    
    if not trade_history_df.empty:
        # Format the display
        display_df = trade_history_df.copy()
        display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['pnl_pct'] = (display_df['pnl'] * 100).round(2)
        
        # Color code PnL
        def color_pnl(val):
            if val > 0:
                return 'color: green; font-weight: bold'
            elif val < 0:
                return 'color: red; font-weight: bold'
            return 'color: black'
        
        styled_df = display_df.style.map(color_pnl, subset=['pnl_pct'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download trade history
        csv = trade_history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History",
            data=csv,
            file_name=f"trade_history_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Download log files if trading is active
        if status['is_running']:
            log_info = simulator.get_log_files_info()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if os.path.exists(log_info['trade_log']):
                    with open(log_info['trade_log'], 'r') as f:
                        trade_csv = f.read()
                    st.download_button(
                        label="📥 Download Trade Log",
                        data=trade_csv,
                        file_name=f"trade_log_{log_info['session_id']}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if os.path.exists(log_info['data_log']):
                    with open(log_info['data_log'], 'r') as f:
                        data_csv = f.read()
                    st.download_button(
                        label="📥 Download Data Log",
                        data=data_csv,
                        file_name=f"data_log_{log_info['session_id']}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if os.path.exists(log_info['decision_log']):
                    with open(log_info['decision_log'], 'r') as f:
                        decision_csv = f.read()
                    st.download_button(
                        label="📥 Download Decision Log",
                        data=decision_csv,
                        file_name=f"decision_log_{log_info['session_id']}.csv",
                        mime="text/csv"
                    )
    else:
        st.info("No trades executed yet. Start live trading to see trade history.")
    
    # Performance Metrics
    if not trade_history_df.empty:
        st.subheader("📊 Performance Metrics")
        
        # Calculate metrics
        metrics = simulator.calculate_performance_metrics()
        
        # Main performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", metrics["total_trades"])
            st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
        with col2:
            st.metric("Total PnL", f"{metrics['total_pnl']:.2%}")
            st.metric("Avg Return per Trade", f"{metrics['avg_return']:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
            st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.3f}")
        with col4:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        
        # Risk-adjusted metrics
        st.subheader("Risk-Adjusted Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
            st.metric("Geometric Mean Return", f"{metrics['geometric_mean_return']:.2%}")
        with col2:
            st.metric("Avg Win", f"{metrics['avg_win']:.2%}")
            st.metric("Avg Loss", f"{metrics['avg_loss']:.2%}")
        with col3:
            st.metric("Win/Loss Ratio", f"{metrics['risk_reward_ratio']:.2f}")
            st.metric("Largest Win", f"{metrics['largest_win']:.2%}")
        with col4:
            st.metric("Largest Loss", f"{metrics['largest_loss']:.2%}")
            st.metric("Consecutive Wins", metrics['consecutive_wins'])
        
        # Performance summary with color coding
        st.subheader("Performance Summary")
        
        # Color-code performance based on metrics
        sharpe_color = "green" if metrics['sharpe_ratio'] > 1.0 else "orange" if metrics['sharpe_ratio'] > 0.5 else "red"
        calmar_color = "green" if metrics['calmar_ratio'] > 1.0 else "orange" if metrics['calmar_ratio'] > 0.5 else "red"
        profit_factor_color = "green" if metrics['profit_factor'] > 1.5 else "orange" if metrics['profit_factor'] > 1.0 else "red"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; border-radius: 5px; background-color: {sharpe_color}20;'>
                <h4>Risk-Adjusted Returns</h4>
                <p style='font-size: 24px; color: {sharpe_color};'>{metrics['sharpe_ratio']:.2f}</p>
                <p>Sharpe Ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; border-radius: 5px; background-color: {calmar_color}20;'>
                <h4>Return vs Drawdown</h4>
                <p style='font-size: 24px; color: {calmar_color};'>{metrics['calmar_ratio']:.2f}</p>
                <p>Calmar Ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; border-radius: 5px; background-color: {profit_factor_color}20;'>
                <h4>Profit Efficiency</h4>
                <p style='font-size: 24px; color: {profit_factor_color};'>{metrics['profit_factor']:.2f}</p>
                <p>Profit Factor</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance charts
    if not trade_history_df.empty:
        st.subheader("📊 Performance Analysis")
        
        # Create performance chart
        fig = go.Figure()
        
        # Add cumulative PnL line
        cumulative_pnl = []
        balance_history = []
        dates = []
        
        # Sort trades by exit time to ensure chronological order
        sorted_trades = trade_history_df[trade_history_df['status'] == 'closed'].sort_values('exit_time')
        
        # Calculate cumulative PnL with proper compounding
        cumulative_pnl = []
        balance_history = []
        dates = []
        
        current_balance = initial_balance
        for _, trade in sorted_trades.iterrows():
            # Calculate the actual dollar profit/loss for this trade
            position_value = trade['quantity'] * trade['entry_price']
            trade_profit_loss = position_value * trade['pnl']  # Convert percentage to dollars
            
            # Update balance with this trade's profit/loss
            current_balance += trade_profit_loss
            
            # Calculate cumulative return from initial balance
            cumulative_return = (current_balance - initial_balance) / initial_balance
            
            cumulative_pnl.append(cumulative_return)
            balance_history.append(current_balance)
            dates.append(trade['exit_time'])
        
        if dates:
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_pnl,
                mode='lines+markers',
                name='Cumulative PnL',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>PnL:</b> %{y:.2%}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Cumulative Performance",
                xaxis_title="Date",
                yaxis_title="Cumulative PnL",
                yaxis_tickformat='.1%',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key="cumulative_performance_chart")
    
    # Market data display
    if not simulator.current_data.empty:
        st.subheader("📈 Current Market Data")
        
        # Show latest data
        latest_data = simulator.current_data.tail(10)
        st.dataframe(latest_data[['Open', 'High', 'Low', 'Close', 'Volume']], use_container_width=True)

def load_optimization_results():
    """
    Helper function to load optimization results from simple_optimization_example.py
    Run this function to get the optimized parameters and copy them to the simulator
    """
    print("🔧 Loading optimization results...")
    print("=" * 60)
    
    # Example of how to run optimization and get results
    from data_fetcher import DataFetcher
    from strategy_optimizer import (
        optimize_moving_average_crossover,
        optimize_rsi_strategy,
        optimize_donchian_channel
    )
    
    # Configuration
    symbol = "BTC-USD"
    start_date = "2025-07-30"
    end_date = "2025-08-24"
    interval = "15m"
    trading_fee = 0.001
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    
    # Fetch data
    fetcher = DataFetcher()
    data = fetcher.fetch_data(symbol, start_date, end_date, interval=interval)
    
    if data.empty:
        print(f"No data fetched for {symbol}")
        return None
    
    print(f"Successfully fetched {len(data)} data points")
    
    # Run optimizations
    print("\nOptimizing Moving Average Crossover...")
    best_params_ma, best_metrics_ma = optimize_moving_average_crossover(
        data=data,
        short_window_range=[5, 10, 15, 20, 25, 30],
        long_window_range=[20, 30, 40, 50, 60, 70],
        risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        trading_fee=trading_fee,
        sharpe_threshold=0.1
    )
    
    print("\nOptimizing RSI Strategy...")
    best_params_rsi, best_metrics_rsi = optimize_rsi_strategy(
        data=data,
        period_range=[10, 14, 20, 30],
        overbought_range=[65, 70, 75, 80],
        oversold_range=[20, 25, 30, 35],
        risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        trading_fee=trading_fee,
        sharpe_threshold=0.1
    )
    
    print("\nOptimizing Donchian Channel...")
    best_params_donchian, best_metrics_donchian = optimize_donchian_channel(
        data=data,
        channel_period_range=[10, 15, 20, 25, 30],
        risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        trading_fee=trading_fee,
        sharpe_threshold=0.1
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS - COPY THESE TO THE SIMULATOR")
    print("=" * 60)
    
    print(f"\n📊 Moving Average Crossover:")
    print(f"   Short MA: {best_params_ma['short_window']}")
    print(f"   Long MA: {best_params_ma['long_window']}")
    print(f"   Risk/Reward: {best_params_ma['risk_reward_ratio']}")
    print(f"   Trading Fee: {best_params_ma['trading_fee']}")
    print(f"   Sharpe Ratio: {best_metrics_ma['sharpe_ratio']:.3f}")
    
    print(f"\n📊 RSI Strategy:")
    print(f"   Period: {best_params_rsi['period']}")
    print(f"   Overbought: {best_params_rsi['overbought']}")
    print(f"   Oversold: {best_params_rsi['oversold']}")
    print(f"   Risk/Reward: {best_params_rsi['risk_reward_ratio']}")
    print(f"   Trading Fee: {best_params_rsi['trading_fee']}")
    print(f"   Sharpe Ratio: {best_metrics_rsi['sharpe_ratio']:.3f}")
    
    print(f"\n📊 Donchian Channel:")
    print(f"   Channel Period: {best_params_donchian['channel_period']}")
    print(f"   Risk/Reward: {best_params_donchian['risk_reward_ratio']}")
    print(f"   Trading Fee: {best_params_donchian['trading_fee']}")
    print(f"   Sharpe Ratio: {best_metrics_donchian['sharpe_ratio']:.3f}")
    
    print("\n" + "=" * 60)
    print("Copy these values to the Live Trading Simulator sidebar!")
    print("=" * 60)
    
    return {
        'ma': best_params_ma,
        'rsi': best_params_rsi,
        'donchian': best_params_donchian
    }

if __name__ == "__main__":
    main()
