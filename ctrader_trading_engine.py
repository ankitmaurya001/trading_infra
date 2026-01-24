#!/usr/bin/env python3
"""
Pure Python Trading Engine for cTrader (Forex)
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
from typing import List, Dict, Optional, Tuple
import logging
import pytz

from strategy_manager import StrategyManager
from trading_engine import TradingEngine
from data_fetcher import CTraderDataFetcher
from brokers import CTraderForexBroker
import config as cfg


class CTraderTradingEngine:
    """
    Pure Python trading engine for cTrader (Forex) with no UI dependencies.
    Reads configuration from file and saves all status to logs.
    """
    
    def __init__(self, config_file: str = "trading_config_ctrader.json"):
        """
        Initialize the trading engine.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Trading configuration
        self.symbol = self.config.get('symbol', 'EURUSD')
        self.interval = self.config.get('interval', '15m')
        self.polling_frequency = self.config.get('polling_frequency', 60)
        self.session_id = None
        
        # Forex markets use UTC timezone
        self.utc_timezone = pytz.UTC
        
        # Check if live trading is enabled (default: False for safety)
        self.live_trading = self.config.get('live_trading', False)
        
        # Get cTrader credentials from config
        try:
            self.client_id = cfg.CTRADER_CLIENT_ID
            self.client_secret = cfg.CTRADER_CLIENT_SECRET
            self.access_token = cfg.CTRADER_ACCESS_TOKEN
            self.account_id = cfg.CTRADER_ACCOUNT_ID
            self.demo = cfg.CTRADER_DEMO
        except AttributeError as e:
            print(f"❌ Missing cTrader credentials in config.py: {e}")
            sys.exit(1)
        
        # Initialize data fetcher
        print("\n🔐 Initializing cTrader data fetcher...")
        self.data_fetcher = CTraderDataFetcher(
            client_id=self.client_id,
            client_secret=self.client_secret,
            access_token=self.access_token,
            account_id=self.account_id,
            demo=self.demo
        )
        print(f"✅ Data fetcher initialized! Mode: {'Demo' if self.demo else 'Live'}")
        
        # Initialize cTrader Forex Broker
        self.broker = CTraderForexBroker(
            data_fetcher=self.data_fetcher,
            demo=self.demo
        )
        
        # Fetch actual available balance (stub for now - forex balance tracking is internal)
        config_balance = self.config.get('initial_balance', None)
        actual_balance = self._fetch_actual_balance()
        
        # Determine initial balance:
        # - If config has a value, use minimum of config and actual (for risk limiting)
        # - If config is None or 0, use actual balance or default
        if config_balance and config_balance > 0:
            initial_balance = min(config_balance, actual_balance) if actual_balance > 0 else config_balance
            if actual_balance > 0 and config_balance < actual_balance:
                print(f"💰 Using configured balance limit: ${initial_balance:,.2f} (actual available: ${actual_balance:,.2f})")
            elif actual_balance > 0:
                print(f"💰 Using actual available balance: ${initial_balance:,.2f} (config limit: ${config_balance:,.2f})")
        else:
            initial_balance = actual_balance if actual_balance > 0 else 10000
            print(f"💰 Using initial balance: ${initial_balance:,.2f}")
        
        # Initialize components with actual balance
        self.strategy_manager = StrategyManager()
        self.trading_engine = TradingEngine(
            initial_balance=initial_balance,
            max_leverage=self.config.get('max_leverage', 10.0),
            max_loss_percent=self.config.get('max_loss_percent', 2.0),
            atr_buffer_percent=self.config.get('atr_buffer_percent', 0.0)
        )
        
        # Enable broker in trading engine only if live trading is enabled
        self.trading_engine.broker = self.broker
        self.trading_engine.use_broker = self.live_trading
        self.trading_engine.symbol = self.symbol
        
        # Forex trading configuration (simpler than commodity - no margin monitoring needed)
        forex_config = self.config.get('forex_trading', {})
        self.enable_balance_monitoring = forex_config.get('enable_balance_monitoring', True)
        self.balance_check_interval = forex_config.get('balance_check_interval_seconds', 300)
        self.balance_alert_threshold = forex_config.get('balance_alert_threshold_percent', 50)  # Alert if balance drops 50%
        
        # Balance monitoring state
        self.balance_monitor_thread = None
        self.balance_monitor_running = False
        
        # Trading state
        self.is_running = False
        self.current_data = pd.DataFrame()
        self.last_update = None
        
        # Data tracking for efficient processing
        self.last_processed_timestamp = None
        self.last_processed_index = -1
        
        # Token refresh tracking
        self.last_auth_error_time = None
        self.auth_error_count = 0
        
        # Setup logging - MUST be done before using self.logger
        self._setup_logging()
        
        # Log trading mode (now self.logger is available)
        if self.live_trading:
            self.logger.warning("⚠️  LIVE TRADING ENABLED - Real orders will be placed!")
            self.logger.warning("⚠️  NOTE: cTrader broker currently uses stub implementation")
            self.logger.warning("⚠️  Implement REST API integration for actual order placement")
        else:
            self.logger.info("📊 Virtual trading mode - No real orders will be placed")
            self.logger.info("   - Connects to cTrader for data")
            self.logger.info("   - Validates balance requirements")
            self.logger.info("   - Tracks PnL internally")
            self.logger.info("   - No orders placed on exchange")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _fetch_actual_balance(self) -> float:
        """
        Fetch actual available balance from cTrader.
        
        Note: cTrader Open API doesn't provide balance directly.
        This is a stub that returns config value or default.
        
        Returns:
            float: Available balance in USD, or 0 if unable to fetch
        """
        try:
            # Try to get balance from broker (stub implementation)
            balances = self.broker.get_balances()
            available = balances.get('available', 0.0)
            
            if available > 0:
                return available
            
            # Fallback to config or default
            config_balance = self.config.get('initial_balance', None)
            if config_balance and config_balance > 0:
                return config_balance
            
            return 10000.0  # Default balance
                
        except Exception as e:
            print(f"⚠️ Could not fetch balance from cTrader: {e}")
            return 0.0
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_file):
            print(f"Creating default config file: {self.config_file}")
            # Create default config if it doesn't exist
            default_config = {
                "symbol": "EURUSD",
                "interval": "15m",
                "polling_frequency": 60,
                "initial_balance": 10000,
                "max_leverage": 10,
                "max_loss_percent": 2.0,
                "live_trading": False,
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
                "forex_trading": {
                    "enable_balance_monitoring": True,
                    "balance_check_interval_seconds": 300,
                    "balance_alert_threshold_percent": 50
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
        
        # Create session ID for this run using UTC timezone
        timestamp = datetime.now(self.utc_timezone).strftime("%Y%m%d_%H%M%S")
        trading_suffix = "_live" if self.live_trading else "_virtual"
        self.session_id = f"{self.symbol}_{timestamp}{trading_suffix}"
        
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
    
    def _is_market_open(self) -> bool:
        """
        Check if the forex market is currently open.
        
        Forex markets are open 24 hours a day, 5 days a week (Monday-Friday).
        Closed on weekends (Saturday and Sunday).
        
        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            # Get current time in UTC (forex markets use UTC)
            utc_now = datetime.now(self.utc_timezone)
            current_weekday = utc_now.weekday()  # 0=Monday, 6=Sunday
            
            # Forex markets are closed on weekends
            if current_weekday >= 5:  # Saturday or Sunday
                return False
            
            # Forex markets are open 24/5 (Monday-Friday)
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            # On error, assume market is closed to be safe
            return False
    
    def _get_time_until_market_open(self) -> Optional[timedelta]:
        """
        Calculate time until forex market opens next.
        
        Returns:
            timedelta: Time until market opens, or None if market is already open
        """
        try:
            if self._is_market_open():
                return None
            
            utc_now = datetime.now(self.utc_timezone)
            current_weekday = utc_now.weekday()
            
            # Calculate next Monday (forex opens Monday 00:00 UTC)
            if current_weekday == 5:  # Saturday
                days_until_monday = 2
            elif current_weekday == 6:  # Sunday
                days_until_monday = 1
            else:
                # Shouldn't happen if _is_market_open() is correct
                return None
            
            next_monday = utc_now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_until_monday)
            time_until_open = next_monday - utc_now
            
            return time_until_open
            
        except Exception as e:
            self.logger.error(f"Error calculating time until market open: {e}")
            return None
    
    def _refresh_token_if_needed(self, error_message: str = None):
        """
        Refresh authentication token if we're getting authentication errors.
        
        Note: cTrader uses OAuth tokens that may need refreshing.
        This is a stub implementation.
        
        Args:
            error_message: Error message from the API call
        """
        try:
            # Check if error is related to authentication
            auth_keywords = ['api_key', 'access_token', 'authentication', 'unauthorized', 'token', 'expired']
            is_auth_error = error_message and any(keyword in str(error_message).lower() for keyword in auth_keywords)
            
            if is_auth_error:
                current_time = time.time()
                
                # Only refresh if we haven't refreshed recently (avoid infinite loops)
                if (self.last_auth_error_time is None or 
                    current_time - self.last_auth_error_time > 300):  # 5 minutes cooldown
                    
                    self.logger.warning("🔄 Authentication error detected. Token refresh may be needed...")
                    self.auth_error_count += 1
                    self.last_auth_error_time = current_time
                    
                    # TODO: Implement token refresh for cTrader OAuth
                    self.logger.warning("⚠️  Token refresh not yet implemented for cTrader")
                    self.logger.warning("⚠️  Please manually refresh your access token")
                    self.auth_error_count = 0  # Reset on acknowledgment
                else:
                    self.logger.warning(f"⚠️  Authentication error but too soon to refresh again. Waiting...")
            else:
                # Reset error count if it's not an auth error
                self.auth_error_count = 0
                
        except Exception as e:
            self.logger.error(f"❌ Error refreshing token: {e}")
            # If refresh fails multiple times, wait longer
            if self.auth_error_count >= 3:
                self.logger.error("❌ Multiple authentication failures. Waiting 30 minutes before retry...")
                time.sleep(1800)  # Wait 30 minutes
                self.auth_error_count = 0
    
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
                'config_timestamp': datetime.now(self.utc_timezone).isoformat(),
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
        
        self.is_running = True
        
        # Reset data tracking for new session
        self.last_processed_timestamp = None
        
        self.logger.info("🚀 Trading engine started!")
        self.logger.info(f"📊 Symbol: {self.symbol}")
        self.logger.info(f"⏱️  Interval: {self.interval}")
        self.logger.info(f"📡 Polling Frequency: {self.polling_frequency} seconds")
        if self.live_trading:
            self.logger.info(f"💰 Trading: LIVE (Real Orders)")
            self.logger.warning("⚠️  NOTE: Broker uses stub implementation - orders not actually placed")
        else:
            self.logger.info(f"💰 Trading: VIRTUAL (No Real Orders)")
            self.logger.info(f"   - Connects to cTrader for data")
            self.logger.info(f"   - Validates balance requirements")
            self.logger.info(f"   - Tracks PnL internally")
            self.logger.info(f"   - No orders placed on exchange")
        self.logger.info(f"🎯 Active Strategies: {[s.name for s in self.strategy_manager.get_strategies()]}")
        self.logger.info(f"💰 Initial Balance: ${self.trading_engine.initial_balance:,.2f}")
        
        # Start balance monitoring thread (if enabled)
        if self.enable_balance_monitoring:
            self._start_balance_monitor()
            self.logger.info("💰 Balance monitoring: ENABLED")
        else:
            self.logger.warning("⚠️  Balance monitoring: DISABLED (for testing only)")
        
        self.logger.info("=" * 60)
        
        # Save initial status with is_running = True
        self.last_update = datetime.now(self.utc_timezone)
        self._save_status_to_file()
        
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
        
        # Stop balance monitoring
        self._stop_balance_monitor()
        
        self.logger.info("🛑 Trading stopped.")
        
        status = self.trading_engine.get_current_status()
        self.logger.info(f"📊 Final Balance: ${status['current_balance']:,.2f}")
        self.logger.info(f"📈 Total PnL: ${status['total_pnl']:,.2f}")
        self.logger.info(f"📋 Total Trades: {status['total_trades']}")
        self.logger.info("=" * 60)
        
        # Save final status with is_running = False
        self._save_status_to_file()
    
    def _trading_loop(self):
        """Main trading loop that runs in a separate thread."""
        self.logger.info(f"🚀 Starting trading loop for {self.symbol}")
        self._live_trading_loop()
    
    def _live_trading_loop(self):
        """Live trading loop that fetches real-time data."""
        while self.is_running:
            try:
                # Use UTC timezone consistently for all operations
                utc_now = datetime.now(self.utc_timezone)
                current_time = utc_now  # Use UTC time for consistency
                
                # Check if market is open before polling
                if not self._is_market_open():
                    time_until_open = self._get_time_until_market_open()
                    if time_until_open:
                        hours_until = time_until_open.total_seconds() / 3600
                        if hours_until > 1:
                            self.logger.info(f"🌙 Market is closed (weekend). Next open in {hours_until:.1f} hours. Waiting...")
                            # Sleep for 1 hour or until market opens, whichever is shorter
                            sleep_time = min(3600, time_until_open.total_seconds())
                            time.sleep(sleep_time)
                        else:
                            # Less than 1 hour, sleep for 5 minutes and check again
                            self.logger.info(f"🌙 Market is closed. Next open in {time_until_open}. Waiting 5 minutes...")
                            time.sleep(300)
                    else:
                        # Shouldn't happen, but sleep anyway
                        self.logger.warning("⚠️  Market status check failed. Waiting 5 minutes...")
                        time.sleep(300)
                    continue
                
                self.logger.debug(f"🕐 [{utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}] Market is open ✓")
                
                # Calculate date range for data fetching using UTC timezone consistently
                # Fetch enough data for strategy calculations
                if self.interval in ["1m", "5m", "15m", "30m", "1h"]:
                    start_date = (utc_now - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    start_date = (utc_now - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
                
                # For end_date, add 1 day to ensure we get the latest tick data
                end_date = (utc_now + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
                
                self.logger.debug(f"📥 Fetching {self.interval} data for {self.symbol}")
                
                try:
                    data = self.data_fetcher.fetch_historical_data(
                        self.symbol, start_date, end_date, interval=self.interval
                    )
                except Exception as fetch_error:
                    error_msg = str(fetch_error)
                    self.logger.error(f"❌ Error fetching data: {error_msg}")
                    # Check if it's an authentication error and refresh token
                    self._refresh_token_if_needed(error_msg)
                    data = pd.DataFrame()  # Set empty DataFrame to continue loop
                
                # Initialize has_new_data to False
                has_new_data = False
                
                # Ignore last tick of data, as tick not yet closed
                if not data.empty:
                    data = data.iloc[:-1]
                
                if not data.empty:
                    self.logger.debug(f"✅ Successfully fetched {len(data)} data points")
                    self.logger.debug(f"📊 Latest data: {data.index[-1]} - Close: ${data['Close'].iloc[-1]:.5f}")
                    
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
                        
                        # Sync positions with broker to detect if stop-loss orders were triggered
                        if self.live_trading:
                            self._sync_positions_with_broker(current_time)
                        
                        # Process each strategy only when new data is available
                        for strategy in self.strategy_manager.get_strategies():
                            self.trading_engine.process_strategy_signals(strategy, data, current_time)
                        
                        # Check take-profit for open positions (after strategy processing)
                        self._check_take_profit(data, current_time)
                        
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
                    # Refresh token if we're getting empty data (might be auth issue)
                    self._refresh_token_if_needed("Empty data returned")
                
                # Calculate dynamic polling frequency
                if not data.empty and has_new_data:
                    dynamic_polling_frequency = self._calculate_dynamic_polling_frequency()
                else:
                    # If no new data or empty data, poll less frequently
                    dynamic_polling_frequency = min(60, self.polling_frequency)
                
                # Wait for next polling cycle
                self.logger.debug(f"⏳ Waiting {dynamic_polling_frequency} seconds until next update...")
                time.sleep(dynamic_polling_frequency)
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"❌ Error in trading loop: {error_msg}")
                
                # Check if it's an authentication error
                self._refresh_token_if_needed(error_msg)
                
                # Wait before retrying
                time.sleep(min(60, self.polling_frequency))
    
    def _calculate_dynamic_polling_frequency(self) -> int:
        """Calculate dynamic polling frequency based on time until next tick."""
        try:
            # Use UTC timezone for all time calculations
            current_time = datetime.now(self.utc_timezone)
            
            # Convert interval to minutes
            interval_minutes = {
                "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                "1h": 60, "4h": 240, "1d": 1440
            }
            
            if self.interval not in interval_minutes:
                return self.polling_frequency
            
            minutes = interval_minutes[self.interval]
            
            # Calculate next tick time
            if self.interval in ["1m", "5m", "15m", "30m"]:
                current_minute = current_time.minute
                next_minute = ((current_minute // minutes) + 1) * minutes
                if next_minute >= 60:
                    next_tick = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, minutes=next_minute % 60)
                else:
                    next_tick = current_time.replace(minute=next_minute, second=0, microsecond=0)
            elif self.interval == "1h":
                next_tick = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            elif self.interval == "4h":
                current_hour = current_time.hour
                next_hour = ((current_hour // 4) + 1) * 4
                if next_hour >= 24:
                    next_tick = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    next_tick = current_time.replace(hour=next_hour, minute=0, second=0, microsecond=0)
            elif self.interval == "1d":
                next_tick = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_tick = current_time + timedelta(minutes=minutes)
            
            time_until_next_tick = (next_tick - current_time).total_seconds()
            polling_frequency = max(1, int(time_until_next_tick + 5))  # 5 second buffer
            
            return polling_frequency
            
        except Exception as e:
            self.logger.warning(f"Error calculating dynamic polling frequency: {e}")
            return self.polling_frequency
    
    def _check_take_profit(self, data: pd.DataFrame, current_time: datetime):
        """
        Check take-profit levels for open positions when new candle closes.
        
        Args:
            data: Latest market data
            current_time: Current timestamp
        """
        if data.empty:
            return
        
        # Get current price from latest candle close
        current_price = float(data['Close'].iloc[-1])
        
        # Check all open positions
        for trade in self.trading_engine.active_trades[:]:  # Copy list to avoid modification during iteration
            if trade.get('status') != 'open':
                continue
            
            take_profit = trade.get('take_profit')
            if take_profit is None:
                continue
            
            # Determine position type
            is_long = trade.get('action') == 'BUY'
            
            # Check if take-profit is hit
            take_profit_hit = False
            if is_long:
                # LONG position: take-profit hit if current_price >= take_profit
                take_profit_hit = current_price >= take_profit
            else:
                # SHORT position: take-profit hit if current_price <= take_profit
                take_profit_hit = current_price <= take_profit
            
            if take_profit_hit:
                self.logger.info(f"🎯 Take-profit hit for trade {trade['id']}: {trade['action']} @ {current_price:.5f} (target: {take_profit:.5f})")
                
                # Close the position
                position_type = 'LONG' if is_long else 'SHORT'
                closed_trades = self.trading_engine.close_trades(
                    strategy=self.strategy_manager.get_strategy_by_name(trade['strategy']),
                    position_type=position_type,
                    price=current_price,
                    timestamp=current_time,
                    exit_type='tp_hit'  # Explicitly pass exit type
                )
                
                if closed_trades:
                    self.logger.info(f"✅ Position closed (TP hit): {closed_trades[0].get('status', 'closed')}")
    
    def _sync_positions_with_broker(self, current_time: datetime):
        """
        Sync positions with broker to detect if any stop-loss orders were triggered.
        
        Note: This is a stub implementation since cTrader broker doesn't
        actually track positions. Positions are tracked internally.
        
        Args:
            current_time: Current timestamp
        """
        if not self.live_trading or not self.broker:
            return
        
        # For cTrader stub broker, positions are tracked internally
        # This method is kept for interface compatibility
        pass
    
    def _balance_monitor_loop(self):
        """
        Background thread that continuously monitors account balance.
        Alerts if balance falls below threshold.
        """
        self.logger.info("💰 Balance monitoring thread started")
        
        initial_balance = self.trading_engine.initial_balance
        
        while self.balance_monitor_running and self.is_running:
            try:
                # Get current balance from trading engine
                status = self.trading_engine.get_current_status()
                current_balance = status.get('current_balance', initial_balance)
                
                # Calculate balance drop percentage
                balance_drop_pct = ((initial_balance - current_balance) / initial_balance) * 100 if initial_balance > 0 else 0
                
                # Check threshold
                if balance_drop_pct >= self.balance_alert_threshold:
                    self.logger.warning(
                        f"⚠️  Balance alert: Balance dropped {balance_drop_pct:.1f}% "
                        f"(Current: ${current_balance:,.2f}, Initial: ${initial_balance:,.2f})"
                    )
                else:
                    self.logger.debug(
                        f"💰 Balance OK: ${current_balance:,.2f} "
                        f"({balance_drop_pct:.1f}% drop from initial)"
                    )
                
                # Sleep until next check
                time.sleep(self.balance_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in balance monitor loop: {e}")
                time.sleep(self.balance_check_interval)
        
        self.logger.info("💰 Balance monitoring thread stopped")
    
    def _start_balance_monitor(self):
        """Start the balance monitoring thread."""
        if not self.balance_monitor_running:
            self.balance_monitor_running = True
            self.balance_monitor_thread = threading.Thread(
                target=self._balance_monitor_loop,
                daemon=True
            )
            self.balance_monitor_thread.start()
            self.logger.info("💰 Balance monitoring thread started")
    
    def _stop_balance_monitor(self):
        """Stop the balance monitoring thread."""
        if self.balance_monitor_running:
            self.balance_monitor_running = False
            self.logger.info("💰 Balance monitoring thread stopped")
    
    def _save_status_to_file(self):
        """Save current trading status to JSON file for UI consumption."""
        try:
            status = self.trading_engine.get_current_status(self.current_data)
            status['is_running'] = self.is_running
            status['last_update'] = self.last_update.isoformat() if self.last_update else None
            status['session_id'] = self.session_id
            status['symbol'] = self.symbol
            status['interval'] = self.interval
            status['live_trading'] = self.live_trading
            
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
    
    parser = argparse.ArgumentParser(description="cTrader Trading Engine")
    parser.add_argument(
        "--config", 
        default="trading_config_ctrader.json",
        help="Path to configuration file (default: trading_config_ctrader.json)"
    )
    
    args = parser.parse_args()
    print(f"Using config file: {args.config}")
    
    # Create and run the trading engine
    engine = CTraderTradingEngine(config_file=args.config)
    engine.run()


if __name__ == "__main__":
    main()

