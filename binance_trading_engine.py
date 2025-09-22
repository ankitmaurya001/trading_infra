#!/usr/bin/env python3
"""
Pure Python Trading Engine for Binance
No UI dependencies - runs as a standalone script with config file support
"""

import pandas as pd
import numpy as np
import time
import threading
import json
import os
import signal
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

from strategy_manager import StrategyManager
from trading_engine import TradingEngine
from data_fetcher import BinanceDataFetcher
import config as cfg


class BinanceTradingEngine:
    """
    Pure Python trading engine with no UI dependencies.
    Reads configuration from file and saves all status to logs.
    """
    
    def __init__(self, config_file: str = "trading_config.json"):
        """
        Initialize the trading engine.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Initialize components
        self.strategy_manager = StrategyManager()
        self.trading_engine = TradingEngine(
            initial_balance=self.config.get('initial_balance', 10000),
            max_leverage=self.config.get('max_leverage', 10.0),
            max_loss_percent=self.config.get('max_loss_percent', 2.0)
        )
        self.data_fetcher = BinanceDataFetcher(
            api_key=cfg.BINANCE_API_KEY, 
            api_secret=cfg.BINANCE_SECRET_KEY
        )
        
        # Trading state
        self.is_running = False
        self.current_data = pd.DataFrame()
        self.last_update = None
        
        # Mock trading settings
        self.mock_mode = self.config.get('mock_mode', False)
        self.mock_data = pd.DataFrame()
        self.mock_current_index = 0
        self.mock_start_date = None
        self.mock_end_date = None
        self.mock_delay = self.config.get('mock_delay', 0.01)
        
        # Trading configuration
        self.symbol = self.config.get('symbol', 'BTCUSDT')
        self.interval = self.config.get('interval', '15m')
        self.polling_frequency = self.config.get('polling_frequency', 60)
        self.session_id = None
        
        # Data tracking for efficient processing
        self.last_processed_timestamp = None
        self.last_processed_index = -1
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_file):
            logger.info(f"Creating default config file: {self.config_file}")
            # Create default config if it doesn't exist
            default_config = {
                "symbol": "BTCUSDT",
                "interval": "15m",
                "polling_frequency": 60,
                "initial_balance": 10000,
                "max_leverage": 10,
                "max_loss_percent": 2.0,
                "mock_mode": False,
                "mock_days_back": 10,
                "mock_delay": 0.01,
                "enabled_strategies": ["ma", "rsi"],
                "ma_params": {
                    "short_window": 10,
                    "long_window": 20,
                    "risk_reward_ratio": 2.0,
                    "trading_fee": 0.0
                },
                "rsi_params": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30,
                    "risk_reward_ratio": 2.0,
                    "trading_fee": 0.0
                },
                "donchian_params": {
                    "channel_period": 20,
                    "risk_reward_ratio": 2.0,
                    "trading_fee": 0.0
                },
                "log_level": "INFO",
                "log_folder": "logs"
            }
            self._save_config(default_config)
            print(f"Created default config file: {self.config_file}")
            return default_config
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from: {self.config_file}")
            return config
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    
    def _save_config(self, config: Dict):
        """Save configuration to JSON file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        base_log_folder = self.config.get('log_folder', 'logs')
        
        # Create session ID for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_mock" if self.mock_mode else "_live"
        self.session_id = f"{self.symbol}_{timestamp}{mode_suffix}"
        
        # Create session-specific folder
        self.session_folder = os.path.join(base_log_folder, self.session_id)
        os.makedirs(self.session_folder, exist_ok=True)
        
        # Setup file logging
        log_file = os.path.join(self.session_folder, "trading_engine.log")
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Trading engine initialized. Session ID: {self.session_id}")
        self.logger.info(f"Session folder: {self.session_folder}")
        
        # Setup trading engine logging with session folder
        self.trading_engine.setup_logging(self.session_id, self.symbol, self.session_folder)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.stop_trading()
        sys.exit(0)
    
    def setup_strategies(self) -> bool:
        """Setup strategies based on configuration."""
        enabled_strategies = self.config.get('enabled_strategies', ['ma', 'rsi'])
        ma_params = self.config.get('ma_params')
        rsi_params = self.config.get('rsi_params')
        donchian_params = self.config.get('donchian_params')
        
        self.logger.info(f"Setting up strategies: {enabled_strategies}")
        self.logger.info(f"MA params: {ma_params}")
        self.logger.info(f"RSI params: {rsi_params}")
        self.logger.info(f"Donchian params: {donchian_params}")
        
        # Set manual parameters
        success = self.strategy_manager.set_manual_parameters(
            ma_params=ma_params,
            rsi_params=rsi_params,
            donchian_params=donchian_params
        )
        
        if not success:
            self.logger.error("Failed to set strategy parameters.")
            return False
        
        self.logger.info(f"Strategy manager optimized_params: {self.strategy_manager.optimized_params}")
        
        # Initialize strategies
        strategies = self.strategy_manager.initialize_strategies(enabled_strategies)
        
        if not strategies:
            self.logger.error("No strategies initialized. Please check parameters.")
            return False
        
        self.logger.info(f"✅ {len(strategies)} strategies initialized successfully!")
        
        # Save strategy parameters to session folder for dashboard reference
        self.save_strategy_parameters()
        
        return True
    
    def save_strategy_parameters(self):
        """Save strategy parameters used in this session to a JSON file."""
        try:
            # Get the actual parameters used by the strategy manager
            strategy_params = {
                'session_id': self.session_id,
                'symbol': self.symbol,
                'interval': self.config.get('interval', '15m'),
                'enabled_strategies': self.config.get('enabled_strategies', []),
                'strategy_parameters': self.strategy_manager.optimized_params,
                'config_timestamp': datetime.now().isoformat(),
                'description': 'Strategy parameters used during this trading session'
            }
            
            # Save to session folder
            params_file = os.path.join(self.session_folder, "strategy_parameters.json")
            with open(params_file, 'w') as f:
                json.dump(strategy_params, f, indent=2, default=str)
            
            self.logger.info(f"Strategy parameters saved to: {params_file}")
            self.logger.info(f"Parameters: {strategy_params}")
            
        except Exception as e:
            self.logger.error(f"Failed to save strategy parameters: {e}")
    
    def start_trading(self) -> bool:
        """Start the trading engine."""
        if not self.setup_strategies():
            return False
        if not self.strategy_manager.get_strategies():
            self.logger.error("No strategies initialized. Please setup strategies first.")
            return False
        
        # if not self.setup_strategies():
        #     return False
        
        self.is_running = True
        mock_days_back = self.config.get('mock_days_back', 10)
        
        # Reset data tracking for new session
        self.last_processed_timestamp = None
        self.last_processed_index = -1
        
        self.logger.info("🚀 Live trading started!")
        self.logger.info(f"📊 Symbol: {self.symbol}")
        self.logger.info(f"⏱️  Interval: {self.interval}")
        self.logger.info(f"📡 Polling Frequency: {self.polling_frequency} seconds")
        self.logger.info(f"🎭 Mode: {'MOCK' if self.mock_mode else 'LIVE'}")
        self.logger.info(f"🎯 Active Strategies: {[s.name for s in self.strategy_manager.get_strategies()]}")
        self.logger.info(f"💰 Initial Balance: ${self.trading_engine.initial_balance:,.2f}")
        self.logger.info("=" * 60)
        
        # Setup mock data if in mock mode
        if self.mock_mode:
            self._setup_mock_data(mock_days_back)
        
        # Start the trading loop in a separate thread
        trading_thread = threading.Thread(
            target=self._trading_loop,
            daemon=True
        )
        trading_thread.start()
        
        return True
    
    def stop_trading(self):
        """Stop the trading engine."""
        self.is_running = False
        self.logger.info("🛑 Live trading stopped.")
        
        status = self.trading_engine.get_current_status()
        self.logger.info(f"📊 Final Balance: ${status['current_balance']:,.2f}")
        self.logger.info(f"📈 Total PnL: ${status['total_pnl']:,.2f}")
        self.logger.info(f"📋 Total Trades: {status['total_trades']}")
        self.logger.info("=" * 60)
    
    def _setup_mock_data(self, mock_days_back: int):
        """Setup mock data for testing."""
        self.logger.info(f"🎭 Setting up mock data for {self.symbol}...")
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if self.interval in ["5m", "15m", "30m", "1h"]:
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d %H:%M:%S')
        else:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d %H:%M:%S')
        
        self.logger.info(f"📥 Fetching historical data from {start_date} to {end_date}")
        
        # Fetch all historical data
        data = self.data_fetcher.fetch_historical_data(
            self.symbol, start_date, end_date, interval=self.interval
        )
        
        if data.empty:
            self.logger.error(f"❌ No data fetched for {self.symbol}")
            return
        
        self.logger.info(f"✅ Fetched {len(data)} data points")
        
        # Calculate the mock start point
        if self.interval in ["5m", "15m", "30m", "1h"]:
            points_per_day = {
                "5m": 288,   # 24 * 12
                "15m": 96,   # 24 * 4
                "30m": 48,   # 24 * 2
                "1h": 24     # 24 * 1
            }
            mock_points_back = mock_days_back * points_per_day.get(self.interval, 96)
            self.mock_current_index = max(0, len(data) - mock_points_back)
        else:
            self.mock_current_index = max(0, len(data) - mock_days_back)
        
        # Store the mock data
        self.mock_data = data
        self.mock_start_date = data.index[self.mock_current_index]
        self.mock_end_date = data.index[-1]
        
        self.logger.info(f"🎭 Mock simulation will start from: {self.mock_start_date}")
        self.logger.info(f"🎭 Mock simulation will end at: {self.mock_end_date}")
        self.logger.info(f"🎭 Total mock data points: {len(data) - self.mock_current_index}")
    
    def _trading_loop(self):
        """Main trading loop that runs in a separate thread."""
        self.logger.info(f"🚀 Starting {'mock' if self.mock_mode else 'live'} trading loop for {self.symbol}")
        
        if self.mock_mode:
            self._mock_trading_loop()
        else:
            self._live_trading_loop()
    
    def _live_trading_loop(self):
        """Live trading loop that fetches real-time data."""
        while self.is_running:
            try:
                current_time = datetime.now()
                self.logger.debug(f"🕐 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}]")
                
                # Calculate date range for data fetching
                if self.interval in ["5m", "15m", "30m", "1h"]:
                    start_date = (current_time - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    start_date = (current_time - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
                
                end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                self.logger.debug(f"📥 Fetching {self.interval} data for {self.symbol}")
                
                data = self.data_fetcher.fetch_historical_data(
                    self.symbol, start_date, end_date, interval=self.interval
                )
                # Ignore last tick of data, as tick not yet closed
                data = data.iloc[:-1]
                
                if not data.empty:
                    self.logger.debug(f"✅ Successfully fetched {len(data)} data points")
                    self.logger.debug(f"📊 Latest data: {data.index[-1]} - Close: ${data['Close'].iloc[-1]:.2f}")
                    
                    # Check if we have new data
                    latest_data_timestamp = data.index[-1]
                    has_new_data = (self.last_processed_timestamp is None or 
                                  latest_data_timestamp > self.last_processed_timestamp)
                    
                    if has_new_data:
                        self.logger.info(f"🆕 New data detected! Processing strategies...")
                        
                        # Save only new data points to CSV
                        if self.last_processed_timestamp is not None:
                            new_data = data[data.index > self.last_processed_timestamp]
                        else:
                            new_data = data
                        
                        if not new_data.empty:
                            self.trading_engine.save_data_to_csv(new_data, current_time)
                        
                        self.current_data = data
                        self.last_update = current_time
                        
                        # Process each strategy only when new data is available
                        for strategy in self.strategy_manager.get_strategies():
                            self.trading_engine.process_strategy_signals(strategy, data, current_time)
                        
                        # Update the last processed timestamp
                        self.last_processed_timestamp = latest_data_timestamp
                        self.logger.info(f"✅ Strategies processed for new data at {latest_data_timestamp}")
                        
                        # Save current status to file
                        self._save_status_to_file()
                    else:
                        self.logger.debug(f"⏸️  No new data available. Skipping strategy processing.")
                        self.current_data = data
                        self.last_update = current_time
                else:
                    self.logger.warning(f"⚠️  No data received for {self.symbol}")
                
                # Calculate dynamic polling frequency
                if not has_new_data:
                    dynamic_polling_frequency = 5
                else:
                    dynamic_polling_frequency = self._calculate_dynamic_polling_frequency()
                
                # Wait for next polling cycle
                self.logger.debug(f"⏳ Waiting {dynamic_polling_frequency} seconds until next update...")
                time.sleep(dynamic_polling_frequency)
                
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {str(e)}")
                time.sleep(self.polling_frequency)
    
    def _mock_trading_loop(self):
        """Mock trading loop that processes historical data sequentially."""
        self.logger.info(f"🎭 Starting mock trading simulation...")
        
        start_time = time.time()
        
        while self.is_running and self.mock_current_index < len(self.mock_data):
            try:
                # Check if we have a new data point to process
                if self.mock_current_index > self.last_processed_index:
                    # Get current data point
                    current_data_point = self.mock_data.iloc[self.mock_current_index]
                    current_time = current_data_point.name
                    
                    # Create data up to current point for strategy processing
                    data = self.mock_data.iloc[:self.mock_current_index + 1]
                    
                    # Calculate progress
                    progress = (self.mock_current_index / len(self.mock_data)) * 100
                    self.logger.info(f"🎭 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Processing mock data point {self.mock_current_index + 1}/{len(self.mock_data)} ({progress:.1f}%)")
                    
                    # Save only the new data point to CSV
                    new_data_point = pd.DataFrame([current_data_point])
                    self.trading_engine.save_data_to_csv(new_data_point, current_time)
                    
                    self.current_data = data
                    self.last_update = current_time
                    
                    # Process each strategy only for new data
                    for strategy in self.strategy_manager.get_strategies():
                        self.trading_engine.process_strategy_signals(strategy, data, current_time)
                    
                    # Update the last processed index
                    self.last_processed_index = self.mock_current_index
                    
                    # Save current status to file
                    self._save_status_to_file()
                
                # Move to next data point
                self.mock_current_index += 1
                
                # Use configurable delay in mock mode
                # time.sleep(self.mock_delay)
                
            except Exception as e:
                self.logger.error(f"❌ Error in mock trading loop: {str(e)}")
                time.sleep(0.1)
        
        end_time = time.time()
        simulation_duration = end_time - start_time
        
        if self.mock_current_index >= len(self.mock_data):
            self.logger.info(f"🎭 Mock trading simulation completed!")
            self.logger.info(f"📊 Processed all {len(self.mock_data)} data points")
            self.logger.info(f"⏱️  Simulation duration: {simulation_duration:.2f} seconds")
            self.is_running = False
    
    def _calculate_dynamic_polling_frequency(self) -> int:
        """Calculate dynamic polling frequency based on time until next tick."""
        try:
            current_time = datetime.now()
            
            # Convert interval to minutes
            interval_minutes = {
                "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30,
                "60m": 60, "1h": 60, "1d": 1440
            }
            
            if self.interval not in interval_minutes:
                return self.polling_frequency
            
            minutes = interval_minutes[self.interval]
            
            # Calculate next tick time
            if self.interval in ["1m", "2m", "5m", "15m", "30m", "60m", "1h"]:
                if minutes == 60:  # 1h
                    next_tick = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    current_minute = current_time.minute
                    next_minute = ((current_minute // minutes) + 1) * minutes
                    if next_minute >= 60:
                        next_tick = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, minutes=next_minute % 60)
                    else:
                        next_tick = current_time.replace(minute=next_minute, second=0, microsecond=0)
            else:
                next_tick = current_time + timedelta(minutes=minutes)
            
            time_until_next_tick = (next_tick - current_time).total_seconds()
            polling_frequency = max(1, int(time_until_next_tick + 5))  # 5 second buffer
            
            return polling_frequency
            
        except Exception as e:
            self.logger.warning(f"Error calculating dynamic polling frequency: {e}")
            return self.polling_frequency
    
    def _save_status_to_file(self):
        """Save current trading status to JSON file for UI consumption."""
        try:
            status = self.trading_engine.get_current_status(self.current_data)
            status['is_running'] = self.is_running
            status['last_update'] = self.last_update.isoformat() if self.last_update else None
            status['session_id'] = self.session_id
            status['symbol'] = self.symbol
            status['interval'] = self.interval
            status['mock_mode'] = self.mock_mode
            
            # Add mock progress if in mock mode
            if self.mock_mode and not self.mock_data.empty:
                status['mock_progress'] = (self.mock_current_index / len(self.mock_data)) * 100
                status['mock_total_points'] = len(self.mock_data)
                status['mock_current_point'] = self.mock_current_index
            
            # Save to status file in session folder
            status_file = os.path.join(self.session_folder, "status.json")
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving status to file: {e}")
    
    def run(self):
        """Main run method - starts trading and keeps the process alive."""
        try:
            if self.start_trading():
                self.logger.info("Trading engine started successfully. Press Ctrl+C to stop.")
                
                # Keep the main thread alive
                while self.is_running:
                    time.sleep(1)
            else:
                self.logger.error("Failed to start trading engine.")
                sys.exit(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt. Shutting down...")
            self.stop_trading()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.stop_trading()
            sys.exit(1)


def main():
    """Main entry point for the trading engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Binance Trading Engine")
    parser.add_argument(
        "--config", 
        default="trading_config.json",
        help="Path to configuration file (default: trading_config.json)"
    )
    
    args = parser.parse_args()
    print(f"Using config file: {args.config}")
    
    # Create and run the trading engine
    engine = BinanceTradingEngine(config_file=args.config)
    engine.run()


if __name__ == "__main__":
    main()
