#!/usr/bin/env python3
"""
Trading Dashboard - Streamlit UI
Reads trading data from log folder and displays comprehensive trading information
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional

from streamlit_components import (
    display_trading_status,
    display_trade_history,
    display_performance_metrics,
    create_performance_chart,
    display_decision_log_viewer,
    display_decision_log_summary,
    display_log_files_info,
    display_market_data,
    create_live_trading_chart
)


class TradingDashboard:
    """
    Trading dashboard that reads data from log files and displays trading information.
    """
    
    def __init__(self, log_folder: str = "logs"):
        self.log_folder = log_folder
        self.current_session_id = None
        self.status_data = {}
        self.trade_history_df = pd.DataFrame()
        self.decision_log_df = pd.DataFrame()
        self.market_data_df = pd.DataFrame()
        
    def get_available_sessions(self) -> List[Dict]:
        """Get list of available trading sessions from log folder."""
        if not os.path.exists(self.log_folder):
            return []
        
        sessions = []
        
        # Look for session folders (each folder is a session)
        session_folders = [d for d in os.listdir(self.log_folder) 
                          if os.path.isdir(os.path.join(self.log_folder, d))]
        
        for session_folder in session_folders:
            try:
                # Look for status.json in the session folder
                status_file = os.path.join(self.log_folder, session_folder, "status.json")
                
                if os.path.exists(status_file):
                    with open(status_file, 'r') as f:
                        status_data = json.load(f)
                    
                    session_id = status_data.get('session_id', session_folder)
                    symbol = status_data.get('symbol', 'unknown')
                    start_time = status_data.get('last_update', 'unknown')
                    is_running = status_data.get('is_running', False)
                    mock_mode = status_data.get('mock_mode', False)
                    execution_mode = status_data.get('execution_mode', 'unknown')
                    
                    # Get folder modification time as fallback
                    folder_mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(self.log_folder, session_folder)))
                    
                    sessions.append({
                        'session_id': session_id,
                        'symbol': symbol,
                        'start_time': start_time,
                        'file_mtime': folder_mtime,
                        'is_running': is_running,
                        'mock_mode': mock_mode,
                        'execution_mode': execution_mode,
                        'status_file': status_file,
                        'session_folder': os.path.join(self.log_folder, session_folder)
                    })
                
            except Exception as e:
                st.warning(f"Error reading session folder {session_folder}: {e}")
                continue
        
        # Sort by folder modification time (newest first)
        sessions.sort(key=lambda x: x['file_mtime'], reverse=True)
        
        return sessions
    
    def load_session_data(self, session_id: str) -> bool:
        """Load all data for a specific trading session."""
        print(f"🔄 Loading session data for: {session_id}")
        print(f"📁 Log folder: {self.log_folder}")
        
        try:
            self.current_session_id = session_id
            
            # Find the session folder
            print(f"🔍 Looking for session folder...")
            session_folder = None
            available_folders = os.listdir(self.log_folder) if os.path.exists(self.log_folder) else []
            print(f"📂 Available folders: {available_folders}")
            
            for folder in available_folders:
                if folder == session_id or folder.startswith(session_id):
                    session_folder = os.path.join(self.log_folder, folder)
                    print(f"✅ Found session folder: {session_folder}")
                    break
            
            if not session_folder or not os.path.exists(session_folder):
                print(f"❌ Session folder not found for: {session_id}")
                st.error(f"Session folder not found for: {session_id}")
                return False
            
            # Load status data
            print(f"📊 Loading status data...")
            status_file = os.path.join(session_folder, "status.json")
            print(f"📄 Status file path: {status_file}")
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    self.status_data = json.load(f)
                print(f"✅ Status data loaded: {len(self.status_data)} fields")
                print(f"   Status keys: {list(self.status_data.keys())}")
            else:
                print(f"❌ Status file not found: {status_file}")
                st.error(f"Status file not found: {status_file}")
                return False
            
            # Load trade history
            print(f"📈 Loading trade history...")
            trade_file = os.path.join(session_folder, "trades.csv")
            print(f"📄 Trade file path: {trade_file}")
            if os.path.exists(trade_file):
                self.trade_history_df = pd.read_csv(trade_file)
                print(f"✅ Trade history loaded: {len(self.trade_history_df)} trades")
                if not self.trade_history_df.empty:
                    print(f"   Trade columns: {list(self.trade_history_df.columns)}")
                    print(f"   Trade statuses: {self.trade_history_df['status'].value_counts().to_dict()}")
                    
                    # Convert datetime columns
                    datetime_cols = ['entry_time', 'exit_time', 'timestamp']
                    for col in datetime_cols:
                        if col in self.trade_history_df.columns:
                            self.trade_history_df[col] = pd.to_datetime(self.trade_history_df[col])
                            print(f"   Converted datetime column: {col}")
                    
                    # Handle different trade data formats
                    # If we have the new format (timestamp, price, etc.), map to expected columns
                    if 'timestamp' in self.trade_history_df.columns and 'entry_time' not in self.trade_history_df.columns:
                        print(f"🔄 Mapping new format to expected format...")
                        # Map new format to expected format
                        self.trade_history_df['entry_time'] = self.trade_history_df['timestamp']
                        if 'price' in self.trade_history_df.columns:
                            self.trade_history_df['entry_price'] = self.trade_history_df['price']
                            self.trade_history_df['exit_price'] = self.trade_history_df['price']  # For now, use same price
                            print(f"   Mapped price → entry_price, exit_price")
                        if 'exit_time' not in self.trade_history_df.columns:
                            self.trade_history_df['exit_time'] = self.trade_history_df['timestamp']  # For now, use same time
                            print(f"   Mapped timestamp → exit_time")
                        print(f"   Final trade columns: {list(self.trade_history_df.columns)}")
            else:
                print(f"❌ Trade file not found: {trade_file}")
                self.trade_history_df = pd.DataFrame()
            
            # Load decision log
            print(f"🎯 Loading decision log...")
            decision_file = os.path.join(session_folder, "decisions.csv")
            print(f"📄 Decision file path: {decision_file}")
            if os.path.exists(decision_file):
                self.decision_log_df = pd.read_csv(decision_file)
                print(f"✅ Decision log loaded: {len(self.decision_log_df)} decisions")
                if not self.decision_log_df.empty:
                    print(f"   Decision columns: {list(self.decision_log_df.columns)}")
                    # Convert datetime columns
                    if 'timestamp' in self.decision_log_df.columns:
                        self.decision_log_df['timestamp'] = pd.to_datetime(self.decision_log_df['timestamp'])
                        print(f"   Converted timestamp column")
            else:
                print(f"❌ Decision file not found: {decision_file}")
                self.decision_log_df = pd.DataFrame()
            
            # Load market data
            print(f"📊 Loading market data...")
            data_file = os.path.join(session_folder, "market_data.csv")
            print(f"📄 Market data file path: {data_file}")
            if os.path.exists(data_file):
                self.market_data_df = pd.read_csv(data_file, index_col=0)
                print(f"✅ Market data loaded: {len(self.market_data_df)} data points")
                if not self.market_data_df.empty:
                    print(f"   Market data columns: {list(self.market_data_df.columns)}")
                    self.market_data_df.index = pd.to_datetime(self.market_data_df.index)
                    print(f"   Converted index to datetime")
            else:
                print(f"❌ Market data file not found: {data_file}")
                self.market_data_df = pd.DataFrame()
            
            print(f"🎉 Session data loading completed successfully!")
            return True
            
        except Exception as e:
            print(f"💥 Error loading session data: {e}")
            import traceback
            traceback.print_exc()
            st.error(f"Error loading session data: {e}")
            return False
    
    def get_current_status(self) -> Dict:
        """Get current trading status."""
        return self.status_data
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        return self.trade_history_df
    
    def get_decision_log_df(self) -> pd.DataFrame:
        """Get decision log as DataFrame."""
        return self.decision_log_df
    
    def get_market_data_df(self) -> pd.DataFrame:
        """Get market data as DataFrame."""
        return self.market_data_df
    
    def process_trade_data(self) -> pd.DataFrame:
        """
        Process trade data to combine entry and exit records into single trade records.
        Each trade currently has two rows: one for entry (BUY/SELL) and one for exit (EXIT).
        This function combines them into one row per trade.
        """
        print(f"🔄 Processing trade data to combine entry/exit records...")
        
        if self.trade_history_df.empty:
            return pd.DataFrame()
        
        # Group by trade_id to combine entry and exit records
        processed_trades = []
        
        for trade_id, group in self.trade_history_df.groupby('trade_id'):
            if len(group) < 2:
                print(f"   ⚠️ Trade {trade_id} has only {len(group)} record(s), skipping")
                continue
            
            # Get entry and exit records with proper empty checks
            entry_records = group[group['action'].isin(['BUY', 'SELL'])]
            exit_records = group[group['action'] == 'EXIT']
            
            if entry_records.empty:
                print(f"   ⚠️ Trade {trade_id} has no entry record (BUY/SELL), skipping")
                continue
            
            if exit_records.empty:
                print(f"   ⚠️ Trade {trade_id} has no exit record (EXIT), skipping")
                continue
            
            entry_record = entry_records.iloc[0]
            exit_record = exit_records.iloc[0]
            
            # Create combined trade record
            combined_trade = {
                'trade_id': trade_id,
                'symbol': entry_record['symbol'],
                'strategy': entry_record['strategy'],
                'direction': 'LONG' if entry_record['action'] == 'BUY' else 'SHORT',
                'entry_action': entry_record['action'],
                'entry_time': entry_record['timestamp'],
                'entry_price': entry_record['price'],
                'quantity': entry_record['quantity'],
                'leverage': entry_record.get('leverage', 1.0),
                'position_size': entry_record.get('position_size', entry_record['quantity'] * entry_record['price']),
                'atr': entry_record.get('atr', 0.0),
                'exit_time': exit_record['timestamp'],
                'exit_price': exit_record['price'],
                'exit_status': exit_record['status'],
                'pnl': exit_record['pnl'],
                'balance': exit_record['balance']
            }
            
            processed_trades.append(combined_trade)
        
        processed_df = pd.DataFrame(processed_trades)
        print(f"   ✅ Processed {len(processed_df)} complete trades from {len(self.trade_history_df)} records")
        
        return processed_df

    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics from trade history."""
        print(f"📊 Calculating performance metrics...")
        print(f"   Trade history shape: {self.trade_history_df.shape}")
        
        if self.trade_history_df.empty:
            print(f"   ❌ No trade history data available")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_return': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'calmar_ratio': 0,
                'sortino_ratio': 0,
                'geometric_mean_return': 0,
                'risk_reward_ratio': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'consecutive_wins': 0
            }
        
        # Process trade data to combine entry/exit records
        processed_trades = self.process_trade_data()
        
        if processed_trades.empty:
            print(f"   ❌ No complete trades found after processing")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_return': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'calmar_ratio': 0,
                'sortino_ratio': 0,
                'geometric_mean_return': 0,
                'risk_reward_ratio': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'consecutive_wins': 0
            }
        
        print(f"   Processed trades: {len(processed_trades)}")
        print(f"   Processed trade columns: {list(processed_trades.columns)}")
        
        # Use processed trades for calculations
        closed_trades = processed_trades  # All processed trades are closed trades
        
        if closed_trades.empty:
            print(f"   ❌ No closed trades found")
            return {
                'total_trades': len(self.trade_history_df),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_return': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'calmar_ratio': 0,
                'sortino_ratio': 0,
                'geometric_mean_return': 0,
                'risk_reward_ratio': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'consecutive_wins': 0
            }
        
        # Calculate metrics
        print(f"   Calculating metrics from {len(closed_trades)} closed trades...")
        
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
        losing_trades = len(closed_trades[closed_trades['pnl'] < 0])
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
        
        print(f"   Total trades: {total_trades}")
        print(f"   Winning trades: {winning_trades}")
        print(f"   Losing trades: {losing_trades}")
        print(f"   Win rate: {win_rate:.2%}")
        
        # Calculate total PnL based on actual balance change from status data
        # This matches the actual balance progression in the system
        current_balance = self.status_data.get('current_balance', 10000)
        initial_balance = self.status_data.get('initial_balance', 10000)
        total_pnl = (current_balance - initial_balance) / initial_balance
        avg_win = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = closed_trades[closed_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        print(f"   Total PnL: {total_pnl:.4f}")
        print(f"   Average win: {avg_win:.4f}")
        print(f"   Average loss: {avg_loss:.4f}")
        
        # Profit factor
        gross_profit = closed_trades[closed_trades['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(closed_trades[closed_trades['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        print(f"   Gross profit: {gross_profit:.4f}")
        print(f"   Gross loss: {gross_loss:.4f}")
        print(f"   Profit factor: {profit_factor:.4f}")
        
        # Max drawdown (simplified)
        cumulative_pnl = closed_trades['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        print(f"   Max drawdown: {max_drawdown:.4f}")
        
        # Sharpe ratio (simplified)
        if len(closed_trades) > 1:
            returns = closed_trades['pnl'].pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        print(f"   Sharpe ratio: {sharpe_ratio:.4f}")
        
        # Calculate additional metrics
        avg_return = total_pnl / total_trades if total_trades > 0 else 0
        calmar_ratio = total_pnl / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Calculate Sortino ratio (similar to Sharpe but only considers downside deviation)
        if len(closed_trades) > 1:
            returns = closed_trades['pnl'].pct_change().dropna()
            downside_returns = returns[returns < 0]
            sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        else:
            sortino_ratio = 0
        
        # Calculate geometric mean return
        if len(closed_trades) > 0:
            geometric_mean_return = (1 + total_pnl) ** (1 / total_trades) - 1
        else:
            geometric_mean_return = 0
        
        # Calculate risk-reward ratio (win/loss ratio)
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Calculate largest win and loss
        largest_win = closed_trades[closed_trades['pnl'] > 0]['pnl'].max() if winning_trades > 0 else 0
        largest_loss = closed_trades[closed_trades['pnl'] < 0]['pnl'].min() if losing_trades > 0 else 0
        
        # Calculate consecutive wins (simplified)
        consecutive_wins = 0
        if len(closed_trades) > 0:
            current_streak = 0
            for pnl in closed_trades['pnl']:
                if pnl > 0:
                    current_streak += 1
                    consecutive_wins = max(consecutive_wins, current_streak)
                else:
                    current_streak = 0
        
        print(f"   Average return per trade: {avg_return:.4f}")
        print(f"   Calmar ratio: {calmar_ratio:.4f}")
        print(f"   Sortino ratio: {sortino_ratio:.4f}")
        print(f"   Geometric mean return: {geometric_mean_return:.4f}")
        print(f"   Risk-reward ratio: {risk_reward_ratio:.4f}")
        print(f"   Largest win: {largest_win:.4f}")
        print(f"   Largest loss: {largest_loss:.4f}")
        print(f"   Consecutive wins: {consecutive_wins}")
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_return': avg_return,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'geometric_mean_return': geometric_mean_return,
            'risk_reward_ratio': risk_reward_ratio,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'consecutive_wins': consecutive_wins
        }
        
        print(f"✅ Performance metrics calculated successfully!")
        return metrics
    
    def get_log_files_info(self) -> Dict:
        """Get information about log files for current session."""
        if not self.current_session_id:
            return {}
        
        # Find the session folder
        session_folder = None
        for folder in os.listdir(self.log_folder):
            if folder == self.current_session_id or folder.startswith(self.current_session_id):
                session_folder = os.path.join(self.log_folder, folder)
                break
        
        if not session_folder:
            return {}
        
        log_files = {}
        
        # Check for different log file types in session folder
        file_types = {
            'status': 'status.json',
            'trades': 'trades.csv',
            'decisions': 'decisions.csv',
            'data': 'market_data.csv',
            'engine_log': 'trading_engine.log'
        }
        
        for file_type, filename in file_types.items():
            filepath = os.path.join(session_folder, filename)
            if os.path.exists(filepath):
                stat = os.stat(filepath)
                log_files[file_type] = {
                    'filename': filename,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'path': filepath
                }
        
        # Add the log content keys that display_log_files_info expects
        # Read the actual log content for display
        try:
            # Trade log content
            trade_file = os.path.join(session_folder, "trades.csv")
            if os.path.exists(trade_file):
                with open(trade_file, 'r') as f:
                    log_files['trade_log'] = f.read()
            else:
                log_files['trade_log'] = "No trade log available"
            
            # Data log content (market data)
            data_file = os.path.join(session_folder, "market_data.csv")
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    # Only show first 1000 characters to avoid overwhelming the UI
                    content = f.read()
                    log_files['data_log'] = content[:1000] + "..." if len(content) > 1000 else content
            else:
                log_files['data_log'] = "No data log available"
            
            # Decision log content
            decision_file = os.path.join(session_folder, "decisions.csv")
            if os.path.exists(decision_file):
                with open(decision_file, 'r') as f:
                    # Only show first 1000 characters to avoid overwhelming the UI
                    content = f.read()
                    log_files['decision_log'] = content[:1000] + "..." if len(content) > 1000 else content
            else:
                log_files['decision_log'] = "No decision log available"
                
        except Exception as e:
            print(f"⚠️ Error reading log files: {e}")
            log_files['trade_log'] = f"Error reading trade log: {e}"
            log_files['data_log'] = f"Error reading data log: {e}"
            log_files['decision_log'] = f"Error reading decision log: {e}"
        
        return log_files


    def load_session_strategy_parameters(self) -> Dict:
        """Load strategy parameters from the session-specific strategy_parameters.json file."""
        if not self.current_session_id:
            print(f"   ❌ No current session ID")
            return {}
        
        # Find the session folder
        session_folder = None
        for folder in os.listdir(self.log_folder):
            if folder == self.current_session_id or folder.startswith(self.current_session_id):
                session_folder = os.path.join(self.log_folder, folder)
                break
        
        if not session_folder:
            print(f"   ❌ Session folder not found for: {self.current_session_id}")
            return {}
        
        # Look for strategy_parameters.json file
        params_file = os.path.join(session_folder, "strategy_parameters.json")
        
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    strategy_params = json.load(f)
                print(f"   ✅ Loaded strategy parameters from: {params_file}")
                print(f"   📊 Parameters: {strategy_params.get('strategy_parameters', {})}")
                return strategy_params.get('strategy_parameters', {})
            except Exception as e:
                print(f"   ❌ Error loading strategy parameters: {e}")
                return {}
        else:
            print(f"   ⚠️ Strategy parameters file not found: {params_file}")
            print(f"   📊 Falling back to default parameters")
            return {}

    def display_current_trade_history(self, trade_history_df: pd.DataFrame, symbol: str):
        """
        Display trade history with combined entry/exit records.
        This processes the raw trade data to show one row per complete trade.
        """
        print(f"📋 Displaying current trade history for {symbol}...")
        print(f"📋 Trade history shape: {trade_history_df.shape}")
        st.subheader("📋 Trade History")
        
        if not trade_history_df.empty:
            print(f"📋 Processing trade history data...")
            
            # Process trade data to combine entry/exit records
            processed_trades = self.process_trade_data()
            
            if processed_trades.empty:
                print(f"📋 No complete trades found after processing")
                st.info("No complete trades found. Trades need both entry and exit records.")
                return
            
            # Create a display-friendly version
            display_df = processed_trades.copy()
            
            # Format time columns
            display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
            display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Add PnL percentage column
            display_df['pnl_pct'] = (display_df['pnl'] * 100).round(2)
            
            # Add price change column
            display_df['price_change'] = display_df['exit_price'] - display_df['entry_price']
            display_df['price_change_pct'] = ((display_df['exit_price'] - display_df['entry_price']) / display_df['entry_price'] * 100).round(2)
            
            # Select and reorder columns for display
            display_columns = [
                'trade_id', 'direction', 'strategy', 'entry_time', 'exit_time',
                'entry_price', 'exit_price', 'price_change', 'price_change_pct',
                'quantity', 'leverage', 'position_size', 'atr', 'pnl', 'pnl_pct', 'exit_status'
            ]
            
            display_df = display_df[display_columns]
            
            # Rename columns for better display
            display_df.columns = [
                'Trade ID', 'Direction', 'Strategy', 'Entry Time', 'Exit Time',
                'Entry Price', 'Exit Price', 'Price Change', 'Price Change %',
                'Quantity', 'Leverage', 'Position Size', 'ATR', 'PnL', 'PnL %', 'Exit Status'
            ]
            
            # Color code PnL
            def color_pnl(val):
                if val > 0:
                    return 'color: green; font-weight: bold'
                elif val < 0:
                    return 'color: red; font-weight: bold'
                return 'color: black'
            
            # Apply styling to PnL columns
            styled_df = display_df.style.map(color_pnl, subset=['PnL %'])
            
            print(f"📋 Rendering processed trade history dataframe...")
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download trade history
            print(f"📋 Creating download button...")
            csv = processed_trades.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade History",
                data=csv,
                file_name=f"trade_history_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            print(f"✅ Trade history displayed successfully!")
        else:
            print(f"📋 No trade history data available")
            st.info("No trades executed yet. Start live trading to see trade history.")


def main():
    """Main Streamlit application."""
    print(f"🚀 Starting Trading Dashboard...")
    st.set_page_config(
        page_title="Trading Dashboard", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📊 Trading Dashboard")
    st.markdown("Monitor live and historical trading sessions")
    print(f"✅ Dashboard header rendered")
    
    # Add custom CSS for better chart display
    st.markdown("""
     <style>
     .main .block-container {
         padding-top: 1rem;
         padding-bottom: 1rem;
         padding-left: 1rem;
         padding-right: 1rem;
         max-width: 100%;
     }
     .stPlotlyChart {
         width: 100% !important;
         max-width: 100% !important;
         min-width: 100% !important;
         margin: 0 auto !important;
     }
     .stPlotlyChart iframe {
         width: 100% !important;
         max-width: 100% !important;
         min-width: 100% !important;
     }
     </style>
     """, unsafe_allow_html=True)
    
    # Initialize dashboard
    print(f"🔧 Initializing dashboard...")
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = TradingDashboard()
        print(f"✅ Dashboard created")
    else:
        print(f"✅ Dashboard already exists")
    
    dashboard = st.session_state.dashboard
    print(f"✅ Dashboard initialized")
    
    # Sidebar configuration
    st.sidebar.header("📊 Session Selection")
    
    # Log folder selection
    log_folder = st.sidebar.text_input(
        "Log Folder Path", 
        value="logs",
        help="Path to the folder containing trading logs"
    )
    
    # Update dashboard log folder if changed
    if dashboard.log_folder != log_folder:
        dashboard.log_folder = log_folder
        st.rerun()
    
    # Get available sessions
    print(f"🔍 Getting available sessions...")
    sessions = dashboard.get_available_sessions()
    print(f"🔍 Found {len(sessions)} sessions")
    
    if not sessions:
        print(f"❌ No sessions found")
        st.warning(f"No trading sessions found in folder: {log_folder}")
        st.info("Make sure the trading engine is running and generating logs.")
        return
    
    # Session selection with manual input option
    st.sidebar.subheader("🎯 Session Selection")
    
    # Option to manually enter session ID
    manual_session_id = st.sidebar.text_input(
        "Or Enter Session ID Manually",
        value="",
        help="Enter a specific session ID to view (e.g., BTCUSDT_20250105_143022_live)",
        placeholder="BTCUSDT_20250105_143022_live"
    )
    
    # Auto-refresh settings
    st.sidebar.subheader("🔄 Auto-Refresh Settings")
    auto_refresh = st.sidebar.checkbox(
        "Enable Auto-Refresh",
        value=False,  # Disabled by default to prevent infinite loops
        help="Automatically refresh data for live sessions"
    )
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=5,
            max_value=300,
            value=30,
            step=5,
            help="How often to refresh data for live sessions"
        )
    
    # Chart scaling settings
    st.sidebar.subheader("📊 Chart Scaling")
    
    # Chart height control
    chart_height = st.sidebar.slider(
        "Chart Height (pixels)",
        min_value=400,
        max_value=1500,
        value=800,
        step=50,
        help="Adjust the overall height of the trading chart"
    )
    
    # Main chart vs RSI ratio
    main_chart_ratio = st.sidebar.slider(
        "Main Chart Ratio",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        step=0.05,
        help="Ratio of main chart vs RSI subplot (higher = more space for price chart)"
    )
    
    # Vertical spacing between subplots
    vertical_spacing = st.sidebar.slider(
        "Subplot Spacing",
        min_value=0.02,
        max_value=0.2,
        value=0.1,
        step=0.01,
        help="Vertical spacing between main chart and RSI subplot"
    )
    
    # Price axis scaling
    price_axis_padding = st.sidebar.slider(
        "Price Axis Padding (%)",
        min_value=0.0,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Extra padding above and below price range (as percentage of range)"
    )
    
    # RSI axis range control
    rsi_auto_range = st.sidebar.checkbox(
        "Auto RSI Range",
        value=True,
        help="Automatically set RSI range to 0-100, or use custom range"
    )
    
    if not rsi_auto_range:
        rsi_min = st.sidebar.slider(
            "RSI Min",
            min_value=0,
            max_value=50,
            value=0,
            step=5
        )
        rsi_max = st.sidebar.slider(
            "RSI Max",
            min_value=50,
            max_value=100,
            value=100,
            step=5
        )
    else:
        rsi_min, rsi_max = 0, 100
    
    # Chart reset button
    if st.sidebar.button("🔄 Reset Chart Settings", type="secondary"):
        # Reset to default values by rerunning
        st.rerun()
    
    # Session selection
    print(f"🎯 Processing session selection...")
    if manual_session_id.strip():
        print(f"🎯 Using manual session ID: {manual_session_id}")
        # Use manually entered session ID
        selected_session = None
        manual_id = manual_session_id.strip()
        
        # Try exact match first
        for session in sessions:
            if session['session_id'] == manual_id:
                selected_session = session
                break
        
        # If no exact match, try partial match
        if not selected_session:
            matching_sessions = []
            for session in sessions:
                if manual_id.lower() in session['session_id'].lower():
                    matching_sessions.append(session)
            
            if len(matching_sessions) == 1:
                selected_session = matching_sessions[0]
                st.sidebar.success(f"Found matching session: {selected_session['session_id']}")
            elif len(matching_sessions) > 1:
                st.sidebar.warning(f"Multiple sessions match '{manual_id}':")
                for session in matching_sessions[:3]:  # Show first 3 matches
                    st.sidebar.write(f"• {session['session_id']}")
                if len(matching_sessions) > 3:
                    st.sidebar.write(f"• ... and {len(matching_sessions) - 3} more")
                return
            else:
                st.sidebar.error(f"Session ID '{manual_id}' not found")
                st.sidebar.info("Available sessions:")
                for session in sessions[:5]:  # Show first 5 sessions
                    st.sidebar.write(f"• {session['session_id']}")
                return
    else:
        print(f"🎯 Using dropdown selection")
        # Use dropdown selection
        session_options = []
        for session in sessions:
            status_icon = "🟢" if session['is_running'] else "🔴"
            # Build a friendly mode label
            exec_mode = session.get('execution_mode', 'unknown')
            if exec_mode == 'mock':
                mode_icon = "🎭"
                mode_label = "Mock"
            elif exec_mode == 'paper':
                mode_icon = "🧪"
                mode_label = "Paper"
            elif exec_mode == 'binance_testnet':
                mode_icon = "🧪📡"
                mode_label = "Testnet"
            elif exec_mode == 'binance_live':
                mode_icon = "⚡📡"
                mode_label = "Live"
            else:
                mode_icon = "❓"
                mode_label = exec_mode

            display_name = f"{status_icon} {mode_icon} {session['symbol']} - {mode_label} - {session['session_id']}"
            session_options.append(display_name)
        
        selected_index = st.sidebar.selectbox(
            "Select Trading Session",
            options=range(len(sessions)),
            format_func=lambda x: session_options[x],
            help="Select a trading session to view"
        )
        
        selected_session = sessions[selected_index]
        print(f"🎯 Selected session: {selected_session['session_id']}")
    
    # Load session data
    if dashboard.current_session_id != selected_session['session_id']:
        print(f"🔄 Loading session data for: {selected_session['session_id']}")
        with st.spinner("Loading session data..."):
            success = dashboard.load_session_data(selected_session['session_id'])
            if not success:
                print(f"❌ Failed to load session data for: {selected_session['session_id']}")
                st.error("Failed to load session data")
                return
            else:
                print(f"✅ Successfully loaded session data for: {selected_session['session_id']}")
    
    # Auto-refresh for live sessions
    print(f"🔄 Checking auto-refresh settings...")
    print(f"   Session running: {selected_session['is_running']}")
    print(f"   Auto-refresh enabled: {auto_refresh}")
    
    if selected_session['is_running'] and auto_refresh:
        print(f"🔄 Auto-refresh enabled for live session")
        st.sidebar.info(f"🔄 Live session - auto-refreshing every {refresh_interval}s")
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now", type="primary"):
            print(f"🔄 Manual refresh button clicked")
            dashboard.load_session_data(selected_session['session_id'])
            st.rerun()
        
        # Note: Auto-refresh is disabled to prevent infinite loops
        # Users can manually refresh using the button above
    elif selected_session['is_running']:
        print(f"🔄 Live session but auto-refresh disabled")
        # Live session but auto-refresh disabled
        st.sidebar.warning("🔄 Live session detected - auto-refresh disabled")
        if st.sidebar.button("🔄 Refresh Now", type="primary"):
            print(f"🔄 Manual refresh button clicked")
            dashboard.load_session_data(selected_session['session_id'])
            st.rerun()
    else:
        print(f"📊 Static session - manual refresh only")
        # Static session - manual refresh only
        st.sidebar.info("📊 Static session - manual refresh only")
        if st.sidebar.button("🔄 Refresh Data", type="secondary"):
            print(f"🔄 Manual refresh button clicked")
            dashboard.load_session_data(selected_session['session_id'])
            st.rerun()
    
    # Main content area
    print(f"🎨 Rendering main content area...")
    st.subheader(f"📈 Trading Session: {selected_session['symbol']}")
    
    # Session info with expandable details
    print(f"📊 Rendering session info metrics...")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Symbol", selected_session['symbol'])
    with col2:
        st.metric("Status", "🟢 Live" if selected_session['is_running'] else "🔴 Stopped")
    with col3:
        exec_mode = selected_session.get('execution_mode', 'unknown')
        if exec_mode == 'mock':
            st.metric("Mode", "🎭 Mock")
        elif exec_mode == 'paper':
            st.metric("Mode", "🧪 Paper")
        elif exec_mode == 'binance_testnet':
            st.metric("Mode", "🧪📡 Testnet")
        elif exec_mode == 'binance_live':
            st.metric("Mode", "⚡📡 Live")
        else:
            st.metric("Mode", f"❓ {exec_mode}")
    with col4:
        st.metric("Session ID", selected_session['session_id'][:8] + "...")
    
    # Expandable session details
    with st.expander("📋 Session Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Full Session ID:** {selected_session['session_id']}")
            st.write(f"**Session Folder:** {selected_session.get('session_folder', 'N/A')}")
            st.write(f"**Last Update:** {selected_session.get('start_time', 'N/A')}")
        with col2:
            st.write(f"**Folder Modified:** {selected_session['file_mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Auto-Refresh:** {'✅ Enabled' if auto_refresh and selected_session['is_running'] else '❌ Disabled'}")
            if auto_refresh and selected_session['is_running']:
                st.write(f"**Refresh Interval:** {refresh_interval} seconds")
    
    # Current time and refresh status
    current_time = datetime.now()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"🕐 Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        if selected_session['is_running'] and auto_refresh:
            st.caption(f"🔄 Auto-refresh: {refresh_interval}s")
        elif selected_session['is_running']:
            st.caption("🔄 Manual refresh")
        else:
            st.caption("📊 Static session")
    
    # Status display
    print(f"📊 Getting current status...")
    status = dashboard.get_current_status()
    print(f"📊 Status data: {status}")
    if status:
        print(f"📊 Displaying trading status...")
        # Pass None for simulator since we don't have a simulator object
        display_trading_status(status, None)
        
        # Last update with refresh indicator
        if status.get('last_update'):
            update_text = f"Last update: {status['last_update']}"
            if status.get('mock_mode') and status.get('mock_progress') is not None:
                update_text += f" | Mock Progress: {status['mock_progress']:.1f}%"
            
            # Add refresh indicator
            if selected_session['is_running']:
                update_text += " | 🔄 Live"
            
            st.caption(update_text)
    
    # Trading Chart
    print(f"📊 Getting market data...")
    market_data = dashboard.get_market_data_df()
    print(f"📊 Market data shape: {market_data.shape}")
    if not market_data.empty:
        print(f"📊 Rendering trading chart...")
        st.subheader("📊 Trading Chart")
        
        # Enhanced CSS for chart display
        st.markdown("""
         <style>
         .stPlotlyChart {
             width: 100% !important;
             max-width: 100% !important;
             min-width: 100% !important;
             height: auto !important;
         }
         .stPlotlyChart iframe {
             width: 100% !important;
             max-width: 100% !important;
             min-width: 100% !important;
             height: auto !important;
         }
         </style>
         """, unsafe_allow_html=True)
        
        # Create and display the enhanced trading chart
        trade_history = dashboard.get_trade_history_df()
        
        # Process trade data to get complete trades
        processed_trades = dashboard.process_trade_data()
        
        # Load strategy parameters from session-specific file
        print(f"📊 Loading strategy parameters from session...")
        strategy_params = dashboard.load_session_strategy_parameters()
        
        # Calculate technical indicators using actual strategy parameters
        print(f"📊 Calculating technical indicators with actual parameters...")
        
        # Get MA parameters
        ma_params = strategy_params.get('ma', {})
        short_window = ma_params.get('short_window', 50)  # fallback to 50 if not found
        long_window = ma_params.get('long_window', 200)   # fallback to 200 if not found
        
        print(f"   MA Parameters: Short={short_window}, Long={long_window}")
        
        # Calculate Moving Averages with actual parameters
        market_data[f'MA_{short_window}'] = market_data['Close'].rolling(window=short_window).mean()
        market_data[f'MA_{long_window}'] = market_data['Close'].rolling(window=long_window).mean()
        
        # Get RSI parameters
        rsi_params = strategy_params.get('rsi', {})
        rsi_period = rsi_params.get('period', 14)  # fallback to 14 if not found
        rsi_overbought = rsi_params.get('overbought', 70)  # fallback to 70 if not found
        rsi_oversold = rsi_params.get('oversold', 30)      # fallback to 30 if not found
        
        print(f"   RSI Parameters: Period={rsi_period}, Overbought={rsi_overbought}, Oversold={rsi_oversold}")
        
        # Calculate RSI with actual parameters
        def calculate_rsi(prices, period=rsi_period):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            # Prevent division by zero: use small epsilon when loss is 0
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        market_data['RSI'] = calculate_rsi(market_data['Close'])

        # Display the strategy parameters in use
        with st.expander("⚙️ Strategy Parameters In Use", expanded=False):
            st.caption("Effective parameters applied to indicators and trade markers")
            ma_col1, ma_col2 = st.columns(2)
            with ma_col1:
                st.metric("Short MA Window", int(short_window))
            with ma_col2:
                st.metric("Long MA Window", int(long_window))

            rsi_c1, rsi_c2, rsi_c3 = st.columns(3)
            with rsi_c1:
                st.metric("RSI Period", int(rsi_period))
            with rsi_c2:
                st.metric("RSI Overbought", int(rsi_overbought))
            with rsi_c3:
                st.metric("RSI Oversold", int(rsi_oversold))

            if strategy_params:
                st.caption("Raw session parameters (from strategy_parameters.json)")
                st.json(strategy_params)
            else:
                st.info("No session-specific parameters found; using dashboard defaults.")
        
        # Create enhanced candlestick chart with detailed trade markers and indicators
        # Calculate RSI subplot ratio based on main chart ratio
        rsi_ratio = 1.0 - main_chart_ratio
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=vertical_spacing,
            subplot_titles=(f"{selected_session['symbol']} Price Chart with Indicators", "RSI"),
            row_heights=[main_chart_ratio, rsi_ratio]
        )
        
        # Add candlestick chart to main subplot
        fig.add_trace(go.Candlestick(
            x=market_data.index,
            open=market_data['Open'],
            high=market_data['High'],
            low=market_data['Low'],
            close=market_data['Close'],
            name="Price",
            showlegend=True
        ), row=1, col=1)
        
        # Add Moving Averages with dynamic column names
        fig.add_trace(go.Scatter(
            x=market_data.index,
            y=market_data[f'MA_{short_window}'],
            mode='lines',
            name=f'MA {short_window}',
            line=dict(color='orange', width=2),
            opacity=0.8
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=market_data.index,
            y=market_data[f'MA_{long_window}'],
            mode='lines',
            name=f'MA {long_window}',
            line=dict(color='purple', width=2),
            opacity=0.8
        ), row=1, col=1)
        
        # Add RSI to second subplot
        fig.add_trace(go.Scatter(
            x=market_data.index,
            y=market_data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='blue', width=2)
        ), row=2, col=1)
        
        # Add RSI overbought/oversold levels with actual parameters
        fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
        fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
        
        # Add trade markers if we have processed trades
        if not processed_trades.empty:
            print(f"📊 Adding {len(processed_trades)} trade markers to chart...")
            
            # Separate LONG and SHORT trades for different styling
            long_trades = processed_trades[processed_trades['direction'] == 'LONG']
            short_trades = processed_trades[processed_trades['direction'] == 'SHORT']
            
            # Add LONG trade entry points
            if not long_trades.empty:
                fig.add_trace(go.Scatter(
                    x=long_trades['entry_time'],
                    y=long_trades['entry_price'],
                    mode='markers+text',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    text=[f"L{i}" for i in long_trades['trade_id']],
                    textposition='top center',
                    textfont=dict(size=10, color='darkgreen'),
                    name='LONG Entry',
                    hovertemplate='<b>LONG Entry</b><br>' +
                                'Trade ID: %{customdata[0]}<br>' +
                                'Time: %{x}<br>' +
                                'Price: $%{y:.2f}<br>' +
                                'Strategy: %{customdata[1]}<extra></extra>',
                    customdata=list(zip(long_trades['trade_id'], long_trades['strategy']))
                ), row=1, col=1)
            
            # Add SHORT trade entry points
            if not short_trades.empty:
                fig.add_trace(go.Scatter(
                    x=short_trades['entry_time'],
                    y=short_trades['entry_price'],
                    mode='markers+text',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    text=[f"S{i}" for i in short_trades['trade_id']],
                    textposition='bottom center',
                    textfont=dict(size=10, color='darkred'),
                    name='SHORT Entry',
                    hovertemplate='<b>SHORT Entry</b><br>' +
                                'Trade ID: %{customdata[0]}<br>' +
                                'Time: %{x}<br>' +
                                'Price: $%{y:.2f}<br>' +
                                'Strategy: %{customdata[1]}<extra></extra>',
                    customdata=list(zip(short_trades['trade_id'], short_trades['strategy']))
                ), row=1, col=1)
            
            # Add exit points for all trades
            fig.add_trace(go.Scatter(
                x=processed_trades['exit_time'],
                y=processed_trades['exit_price'],
                mode='markers+text',
                marker=dict(
                    symbol='circle',
                    size=10,
                    color=processed_trades['pnl'].apply(lambda x: 'lightgreen' if x > 0 else 'lightcoral'),
                    line=dict(
                        width=2,
                        color=processed_trades['pnl'].apply(lambda x: 'green' if x > 0 else 'red')
                    )
                ),
                text=[f"E{i}" for i in processed_trades['trade_id']],
                textposition='middle center',
                textfont=dict(size=9, color='black'),
                name='Exit',
                hovertemplate='<b>Exit</b><br>' +
                            'Trade ID: %{customdata[0]}<br>' +
                            'Time: %{x}<br>' +
                            'Price: $%{y:.2f}<br>' +
                            'PnL: %{customdata[1]:.2%}<br>' +
                            'Status: %{customdata[2]}<extra></extra>',
                customdata=list(zip(
                    processed_trades['trade_id'],
                    processed_trades['pnl'],
                    processed_trades['exit_status']
                ))
            ), row=1, col=1)
            
            # Add trade lines connecting entry and exit points
            for _, trade in processed_trades.iterrows():
                # Determine line color based on PnL
                line_color = 'green' if trade['pnl'] > 0 else 'red'
                line_width = 2 if trade['pnl'] > 0 else 1
                
                fig.add_trace(go.Scatter(
                    x=[trade['entry_time'], trade['exit_time']],
                    y=[trade['entry_price'], trade['exit_price']],
                    mode='lines',
                    line=dict(color=line_color, width=line_width, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip',
                    opacity=0.6
                ), row=1, col=1)
        
        # Update layout with enhanced styling for subplots
        fig.update_layout(
            title=f"{selected_session['symbol']} Trading Chart with Technical Indicators",
            height=chart_height,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='closest',
            template='plotly_white'
        )
        
        # Update x-axis labels with better formatting
        fig.update_xaxes(
            title_text="Time", 
            row=2, col=1,
            tickformat="%m/%d %H:%M",
            tickangle=45
        )
        
        # Update y-axes with better formatting for zoomed views
        fig.update_yaxes(
            title_text="Price ($)", 
            row=1, col=1,
            tickformat=".2f",
            tickfont=dict(size=10),
            title_font=dict(size=12)
        )
        
        fig.update_yaxes(
            title_text="RSI", 
            row=2, col=1,
            tickformat=".0f",
            tickfont=dict(size=10),
            title_font=dict(size=12)
        )
        
        # Set RSI y-axis range based on user settings
        fig.update_yaxes(range=[rsi_min, rsi_max], row=2, col=1)
        
        # Add price axis padding for better visibility when zoomed
        if not market_data.empty and price_axis_padding > 0:
            price_min = market_data['Low'].min()
            price_max = market_data['High'].max()
            price_range = price_max - price_min
            padding = price_range * price_axis_padding
            
            fig.update_yaxes(
                range=[price_min - padding, price_max + padding], 
                row=1, col=1
            )
        
        # Add annotations for trade count and indicator info
        if not processed_trades.empty:
            winning_trades = len(processed_trades[processed_trades['pnl'] > 0])
            total_trades = len(processed_trades)
            win_rate = (winning_trades / total_trades) * 100
            
            fig.add_annotation(
                text=f"Trades: {total_trades} | Win Rate: {win_rate:.1f}%",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        
        # Add RSI level annotations with actual parameters
        fig.add_annotation(
            text=f"RSI Levels: {rsi_overbought} (Overbought) | {rsi_oversold} (Oversold) | Period: {rsi_period}",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        # Enhanced chart configuration for better zooming and scaling
        chart_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'responsive': True,
            'autosize': True,
            'fillFrame': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{selected_session["symbol"]}_trading_chart',
                'height': chart_height,
                'width': None,
                'scale': 2  # Higher resolution for better quality
            },
            'scrollZoom': True,  # Enable scroll zoom
            'doubleClick': 'reset+autosize',  # Double-click to reset zoom
            'showTips': True
        }
        
        st.plotly_chart(
            fig, 
            use_container_width=True, 
            key="enhanced_trading_chart",
            config=chart_config
        )
        
        # Display active trade information
        if status.get('active_trade_info'):
            st.subheader("🎯 Active Trade")
            trade_info = status['active_trade_info']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Strategy", trade_info['strategy'])
            with col2:
                st.metric("Action", trade_info['action'])
            with col3:
                st.metric("Entry Price", f"${trade_info['entry_price']:.2f}")
            with col4:
                st.metric("Quantity", f"{trade_info['quantity']:.4f}")
            
            # Calculate current PnL if we have current data
            if not market_data.empty:
                current_price = market_data['Close'].iloc[-1]
                if trade_info['action'] == 'BUY':
                    current_pnl = (current_price - trade_info['entry_price']) / trade_info['entry_price']
                else:  # SELL (short)
                    current_pnl = (trade_info['entry_price'] - current_price) / trade_info['entry_price']
                
                st.metric(
                    "Current PnL",
                    f"{current_pnl:.2%}",
                    f"${current_price:.2f}",
                    delta_color="normal"
                )
    
    # Decision Log Viewer
    print(f"🎯 Getting decision log...")
    decision_log = dashboard.get_decision_log_df()
    print(f"🎯 Decision log shape: {decision_log.shape}")
    if not decision_log.empty:
        print(f"🎯 Rendering decision log...")
        st.subheader("🎯 Decision Log")
        
        # Decision log controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.caption(f"📊 {len(decision_log)} decisions logged")
        with col2:
            if st.button("🔄 Refresh Log", type="secondary", key="refresh_decision_log"):
                dashboard.load_session_data(selected_session['session_id'])
                st.rerun()
        with col3:
            # Show last decision time
            if not decision_log.empty and 'timestamp' in decision_log.columns:
                last_decision = decision_log['timestamp'].max()
                st.caption(f"Last: {last_decision.strftime('%H:%M:%S')}")
        
        display_decision_log_viewer(decision_log)
        display_decision_log_summary(decision_log)
    
    # Trading history
    print(f"📈 Getting trade history...")
    trade_history = dashboard.get_trade_history_df()
    print(f"📈 Trade history shape: {trade_history.shape}")
    
    # Debug information (can be removed later)
    print(f"🔍 Rendering debug information...")
    with st.expander("🔍 Debug Information", expanded=False):
        st.write("**Trade History DataFrame Info:**")
        if not trade_history.empty:
            st.write(f"Shape: {trade_history.shape}")
            st.write(f"Columns: {list(trade_history.columns)}")
            st.write("**First few rows:**")
            st.dataframe(trade_history.head())
            st.write("**Status value counts:**")
            st.write(trade_history['status'].value_counts())
        else:
            st.write("No trade history data available")
    
    print(f"📈 Displaying trade history...")
    dashboard.display_current_trade_history(trade_history, selected_session['symbol'])
    
    # Performance Metrics
    print(f"📊 Rendering performance metrics section...")
    st.subheader("📊 Performance Metrics")
    print(f"🔄 Calculating performance metrics...")
    metrics = dashboard.calculate_performance_metrics()
    print(f"📊 Displaying performance metrics: {metrics}")
    print(f"📊 Calling display_performance_metrics...")
    display_performance_metrics(metrics)
    print(f"✅ Performance metrics displayed successfully!")
    
    # Performance charts and recent trades
    if not trade_history.empty:
        # Performance charts
        st.subheader("📊 Performance Analysis")
        initial_balance = status.get('initial_balance', 10000)
        performance_fig = create_performance_chart(trade_history, initial_balance)
        if performance_fig:
            st.plotly_chart(performance_fig, use_container_width=True, key="cumulative_performance_chart")
        
        # Recent Trades Summary
        st.subheader("📋 Recent Trades")
        
        # Use processed trade data for recent trades
        processed_trades = dashboard.process_trade_data()
        recent_trades = processed_trades.tail(10) if not processed_trades.empty else pd.DataFrame()
        
        if not recent_trades.empty:
            # Display recent trades in a more visual way
            for _, trade in recent_trades.iterrows():
                trade_color = "🟢" if trade['pnl'] > 0 else "🔴"
                trade_direction = "📈" if trade['direction'] == 'LONG' else "📉"
                
                with st.container():
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    with col1:
                        st.write(f"{trade_direction} {trade['strategy']}")
                    with col2:
                        st.write(f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f}")
                    with col3:
                        st.write(f"{trade['pnl']:.2%}")
                    with col4:
                        # Show leverage and position size
                        leverage = trade.get('leverage', 1.0)
                        position_size = trade.get('position_size', trade['quantity'] * trade['entry_price'])
                        st.write(f"{leverage:.1f}x (${position_size:.0f})")
                    with col5:
                        # Format times
                        entry_time = pd.to_datetime(trade['entry_time'])
                        exit_time = pd.to_datetime(trade['exit_time'])
                        entry_str = entry_time.strftime('%m/%d %H:%M')
                        exit_str = exit_time.strftime('%m/%d %H:%M')
                        st.write(f"{entry_str} → {exit_str}")
                    with col6:
                        # Calculate dollar PnL
                        # PnL percentage is already calculated based on margin used
                        # So dollar PnL should be: pnl_percentage × margin_used
                        leverage = trade.get('leverage', 1.0)
                        position_size = trade.get('position_size', trade['quantity'] * trade['entry_price'])
                        margin_used = position_size / leverage
                        dollar_pnl = trade['pnl'] * margin_used
                        st.write(f"{trade_color} ${dollar_pnl:.2f}")
        else:
            st.info("No closed trades yet. Performance metrics will appear here once trades are executed.")
    else:
        st.info("No trades executed yet. Performance metrics will appear here once trading begins.")
    
    # Market data display
    display_market_data(market_data)
    
    # Session management section
    st.sidebar.subheader("📁 Session Management")
    
    # Show current session info
    st.sidebar.info(f"**Current Session:**\n{selected_session['session_id']}")
    
    # Session actions
    if st.sidebar.button("🔄 Reload All Sessions", type="secondary"):
        # Force reload of session list
        st.rerun()
    
    # Log files information
    log_info = dashboard.get_log_files_info()
    if log_info:
        with st.sidebar.expander("📄 Log Files Info", expanded=False):
            display_log_files_info(log_info)


if __name__ == "__main__":
    main()
