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
from typing import List, Dict, Optional, Tuple
import logging
import pytz

from strategy_manager import StrategyManager
from trading_engine import TradingEngine
from data_fetcher import KiteDataFetcher
from brokers import KiteCommodityBroker
import config as cfg


class KiteTradingEngine:
    """
    Pure Python trading engine with no UI dependencies.
    Reads configuration from file and saves all status to logs.
    """
    
    def __init__(self, config_file: str = "trading_config_kite.json"):
        """
        Initialize the trading engine.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Trading configuration - MUST be set before broker initialization
        self.symbol = self.config.get('symbol', 'TATAMOTORS')
        self.interval = self.config.get('interval', '15minute')
        self.polling_frequency = self.config.get('polling_frequency', 60)
        self.session_id = None
        
        # Exchange and market hours configuration - MUST be set before broker initialization
        self.exchange = cfg.KITE_EXCHANGE
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        
        # Check if live trading is enabled (default: False for safety)
        self.live_trading = self.config.get('live_trading', False)
        
        # Initialize data fetcher and authenticate first (needed for margin check)
        self.data_fetcher = KiteDataFetcher(
            cfg.KITE_CREDENTIALS, 
            cfg.KITE_EXCHANGE
        )

        # Authenticate with Kite
        print("\n🔐 Authenticating with Kite Connect...")
        self.data_fetcher.authenticate()
        print("✅ Authentication successful!")
        
        # Initialize Kite Commodity Broker (self.exchange is now defined)
        self.broker = KiteCommodityBroker(
            kite=self.data_fetcher.kite,
            exchange=self.exchange
        )
        
        # Fetch actual available margin from Kite
        config_balance = self.config.get('initial_balance', None)
        actual_balance = self._fetch_actual_balance()
        
        # Determine initial balance:
        # - If config has a value, use minimum of config and actual (for risk limiting)
        # - If config is None or 0, use actual balance
        if config_balance and config_balance > 0:
            initial_balance = min(config_balance, actual_balance) if actual_balance > 0 else config_balance
            if actual_balance > 0 and config_balance < actual_balance:
                print(f"💰 Using configured balance limit: ₹{initial_balance:,.2f} (actual available: ₹{actual_balance:,.2f})")
            elif actual_balance > 0:
                print(f"💰 Using actual available balance: ₹{initial_balance:,.2f} (config limit: ₹{config_balance:,.2f})")
        else:
            initial_balance = actual_balance if actual_balance > 0 else 10000
            print(f"💰 Using actual available balance: ₹{initial_balance:,.2f}")
        
        # Get margin required for 1 lot (this is the actual capital at risk per trade)
        self.lot_margin = self._fetch_lot_margin()
        if self.lot_margin > 0:
            print(f"📊 Margin per lot ({self.symbol}): ₹{self.lot_margin:,.2f}")
        
        # Initialize components with actual balance
        self.strategy_manager = StrategyManager()
        self.trading_engine = TradingEngine(
            initial_balance=initial_balance,
            max_leverage=self.config.get('max_leverage', 10.0),
            max_loss_percent=self.config.get('max_loss_percent', 2.0),
            atr_buffer_percent=self.config.get('atr_buffer_percent', 0.0)
        )
        
        # Store lot margin in trading engine for proper position sizing
        self.trading_engine.lot_margin = self.lot_margin if self.lot_margin > 0 else None
        
        # Enable broker in trading engine only if live trading is enabled
        self.trading_engine.broker = self.broker
        self.trading_engine.use_broker = self.live_trading
        self.trading_engine.symbol = self.symbol  # self.symbol is now defined
        
        # Commodity trading configuration
        commodity_config = self.config.get('commodity_trading', {})
        self.margin_buffer_percent = commodity_config.get('margin_buffer_percent', 20)
        self.margin_check_interval = commodity_config.get('margin_check_interval_seconds', 300)
        self.margin_alert_threshold = commodity_config.get('margin_alert_threshold_percent', 150)
        self.margin_critical_threshold = commodity_config.get('margin_critical_threshold_percent', 110)
        self.use_gtt_for_stop_loss = commodity_config.get('use_gtt_for_stop_loss', True)
        
        # Margin monitoring state
        self.margin_monitor_thread = None
        self.margin_monitor_running = False
        
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
        else:
            self.logger.info("📊 Virtual trading mode - No real orders will be placed")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _fetch_actual_balance(self) -> float:
        """
        Fetch actual available balance from Kite.
        Uses single ledger facility (equity funds available for commodity trading).
        
        Returns:
            float: Available margin in INR, or 0 if unable to fetch
        """
        try:
            margins = self.broker.check_margins()
            available = margins.get('available', 0.0)
            
            if available > 0:
                return available
            
            # If commodity margin is 0, check equity (single ledger)
            try:
                all_margins = self.broker.kite.margins()
                equity_margin = all_margins.get('equity', {})
                
                # Try different fields for available balance
                equity_available = equity_margin.get('available', {})
                if isinstance(equity_available, dict):
                    available = equity_available.get('live_balance', 0) or equity_available.get('cash', 0) or equity_available.get('opening_balance', 0)
                else:
                    available = equity_available
                
                if available == 0:
                    available = equity_margin.get('net', 0)
                
                return float(available)
            except Exception as e:
                print(f"⚠️ Could not fetch equity margin: {e}")
                return 0.0
                
        except Exception as e:
            print(f"⚠️ Could not fetch balance from Kite: {e}")
            return 0.0
    
    def _fetch_lot_margin(self) -> float:
        """
        Fetch margin required for 1 lot of the current symbol.
        This represents the actual capital at risk per trade.
        
        Returns:
            float: Margin required for 1 lot in INR, or 0 if unable to fetch
        """
        try:
            # Get current price for margin calculation
            current_price = self.broker.get_price(self.symbol)
            
            if current_price <= 0:
                print(f"⚠️ Could not get price for {self.symbol}")
                return 0.0
            
            # Get margin for both BUY and SELL to check if they differ
            # For commodity futures, they're typically the same
            buy_margins = self.broker.get_order_margins(
                symbol=self.symbol,
                transaction_type='BUY',
                quantity=1,
                price=current_price,
                order_type='MARKET'
            )
            buy_margin = buy_margins.get('total', 0.0)
            
            sell_margins = self.broker.get_order_margins(
                symbol=self.symbol,
                transaction_type='SELL',
                quantity=1,
                price=current_price,
                order_type='MARKET'
            )
            sell_margin = sell_margins.get('total', 0.0)
            
            # Use the maximum of both (to be safe)
            if buy_margin != sell_margin and buy_margin > 0 and sell_margin > 0:
                print(f"   📊 BUY margin: ₹{buy_margin:,.2f}, SELL margin: ₹{sell_margin:,.2f}")
            
            return max(buy_margin, sell_margin)
            
        except Exception as e:
            print(f"⚠️ Could not fetch lot margin: {e}")
            return 0.0
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_file):
            print(f"Creating default config file: {self.config_file}")
            # Create default config if it doesn't exist
            default_config = {
                "symbol": "TATAMOTORS",
                "interval": "15minute",
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
        
        # Create session ID for this run using IST timezone
        timestamp = datetime.now(self.ist_timezone).strftime("%Y%m%d_%H%M%S")
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
        Check if the market is currently open based on exchange type.
        
        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            # Get current time in IST
            ist_now = datetime.now(self.ist_timezone)
            current_time = ist_now.time()
            current_weekday = ist_now.weekday()  # 0=Monday, 6=Sunday
            
            # Check if it's a weekday (Monday=0 to Friday=4)
            if current_weekday >= 5:  # Saturday or Sunday
                return False
            
            # Define market hours based on exchange
            if self.exchange == "MCX":
                # MCX: 9:00 AM to 11:55 PM IST (weekdays)
                market_open = datetime.strptime("09:00", "%H:%M").time()
                market_close = datetime.strptime("23:55", "%H:%M").time()
            elif self.exchange in ["NSE", "BSE"]:
                # NSE/BSE: 9:15 AM to 3:30 PM IST (weekdays)
                market_open = datetime.strptime("09:15", "%H:%M").time()
                market_close = datetime.strptime("15:30", "%H:%M").time()
            else:
                # Default to NSE hours for unknown exchanges
                market_open = datetime.strptime("09:15", "%H:%M").time()
                market_close = datetime.strptime("15:30", "%H:%M").time()
            
            # Check if current time is within market hours
            is_open = market_open <= current_time <= market_close
            
            return is_open
            
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            # On error, assume market is closed to be safe
            return False
    
    def _get_time_until_market_open(self) -> Optional[timedelta]:
        """
        Calculate time until market opens next.
        
        Returns:
            timedelta: Time until market opens, or None if market is already open
        """
        try:
            if self._is_market_open():
                return None
            
            ist_now = datetime.now(self.ist_timezone)
            current_time = ist_now.time()
            current_weekday = ist_now.weekday()
            
            # Define market hours based on exchange
            if self.exchange == "MCX":
                market_open_time = datetime.strptime("09:00", "%H:%M").time()
            elif self.exchange in ["NSE", "BSE"]:
                market_open_time = datetime.strptime("09:15", "%H:%M").time()
            else:
                market_open_time = datetime.strptime("09:15", "%H:%M").time()
            
            # Calculate next market open time
            if current_weekday >= 5:  # Weekend
                # Next Monday
                days_until_monday = (7 - current_weekday) % 7
                if days_until_monday == 0:
                    days_until_monday = 7
                next_open = ist_now.replace(hour=market_open_time.hour, 
                                          minute=market_open_time.minute, 
                                          second=0, 
                                          microsecond=0) + timedelta(days=days_until_monday)
            elif current_time > market_open_time:
                # Market closed for today, next open is tomorrow
                next_open = ist_now.replace(hour=market_open_time.hour, 
                                          minute=market_open_time.minute, 
                                          second=0, 
                                          microsecond=0) + timedelta(days=1)
            else:
                # Market opens today
                next_open = ist_now.replace(hour=market_open_time.hour, 
                                          minute=market_open_time.minute, 
                                          second=0, 
                                          microsecond=0)
            
            time_until_open = next_open - ist_now
            return time_until_open
            
        except Exception as e:
            self.logger.error(f"Error calculating time until market open: {e}")
            return None
    
    def _refresh_token_if_needed(self, error_message: str = None):
        """
        Refresh authentication token if we're getting authentication errors.
        
        Args:
            error_message: Error message from the API call
        """
        try:
            # Check if error is related to authentication
            auth_keywords = ['api_key', 'access_token', 'authentication', 'unauthorized', 'token']
            is_auth_error = error_message and any(keyword in str(error_message).lower() for keyword in auth_keywords)
            
            if is_auth_error:
                current_time = time.time()
                
                # Only refresh if we haven't refreshed recently (avoid infinite loops)
                if (self.last_auth_error_time is None or 
                    current_time - self.last_auth_error_time > 300):  # 5 minutes cooldown
                    
                    self.logger.warning("🔄 Authentication error detected. Attempting to refresh token...")
                    self.auth_error_count += 1
                    self.last_auth_error_time = current_time
                    
                    # Re-authenticate
                    self.data_fetcher.authenticate()
                    self.logger.info("✅ Token refreshed successfully!")
                    self.auth_error_count = 0  # Reset on success
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
                'interval': self.config.get('interval', '15minute'),
                'enabled_strategies': self.config.get('enabled_strategies', []),
                'strategy_parameters': self.strategy_manager.optimized_params,
                'config_timestamp': datetime.now(self.ist_timezone).isoformat(),
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
        
        # Reset data tracking for new session
        self.last_processed_timestamp = None
        
        self.logger.info("🚀 Trading engine started!")
        self.logger.info(f"📊 Symbol: {self.symbol}")
        self.logger.info(f"⏱️  Interval: {self.interval}")
        self.logger.info(f"📡 Polling Frequency: {self.polling_frequency} seconds")
        if self.live_trading:
            self.logger.info(f"💰 Trading: LIVE (Real Orders)")
        else:
            self.logger.info(f"💰 Trading: VIRTUAL (No Real Orders)")
            self.logger.info(f"   - Still connects to Kite for data and margin checks")
            self.logger.info(f"   - Validates margin requirements")
            self.logger.info(f"   - Tracks PnL internally")
            self.logger.info(f"   - No orders placed on exchange")
        self.logger.info(f"🎯 Active Strategies: {[s.name for s in self.strategy_manager.get_strategies()]}")
        self.logger.info(f"💰 Initial Balance: ${self.trading_engine.initial_balance:,.2f}")
        
        # Start margin monitoring thread (always for live data)
        self._start_margin_monitor()
        
        self.logger.info("=" * 60)
        
        # Save initial status with is_running = True
        self.last_update = datetime.now(self.ist_timezone)
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
        
        # Stop margin monitoring
        self._stop_margin_monitor()
        
        self.logger.info("🛑 Live trading stopped.")
        
        status = self.trading_engine.get_current_status()
        self.logger.info(f"📊 Final Balance: ${status['current_balance']:,.2f}")
        self.logger.info(f"📈 Total PnL: ${status['total_pnl']:,.2f}")
        self.logger.info(f"📋 Total Trades: {status['total_trades']}")
        self.logger.info("=" * 60)
        
        # Save final status with is_running = False
        self._save_status_to_file()
    
    def _trading_loop(self):
        """Main trading loop that runs in a separate thread."""
        self.logger.info(f"🚀 Starting live trading loop for {self.symbol}")
        self._live_trading_loop()
    
    def _live_trading_loop(self):
        """Live trading loop that fetches real-time data."""
        while self.is_running:
            try:
                # Use IST timezone consistently for all operations
                ist_now = datetime.now(self.ist_timezone)
                current_time = ist_now  # Use IST time for consistency
                
                # Check if market is open before polling
                if not self._is_market_open():
                    time_until_open = self._get_time_until_market_open()
                    if time_until_open:
                        hours_until = time_until_open.total_seconds() / 3600
                        if hours_until > 1:
                            self.logger.info(f"🌙 Market is closed. Next open in {hours_until:.1f} hours. Waiting...")
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
                
                self.logger.debug(f"🕐 [{ist_now.strftime('%Y-%m-%d %H:%M:%S IST')}] Market is open ✓")
                
                # Calculate date range for data fetching using IST timezone consistently
                # Use IST for all date calculations to avoid timezone issues across different machines
                if self.interval in ["5minute", "15minute", "30minute", "1hour"]:
                    start_date = (ist_now - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    start_date = (ist_now - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
                
                # For end_date, add 1 day to ensure we get the latest tick data
                # This handles timezone edge cases where the API might need tomorrow's date
                # to return today's last tick, especially near day boundaries
                end_date = (ist_now + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
                
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
                        
                        # Sync positions with broker to detect if GTT stop-loss orders were triggered
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
            # Use IST timezone for all time calculations
            current_time = datetime.now(self.ist_timezone)
            
            # Convert interval to minutes
            interval_minutes = {
                "1minute": 1, "2minute": 2, "5minute": 5, "15minute": 15, "30minute": 30,
                "60m": 60, "1h": 60, "1d": 1440
            }
            
            if self.interval not in interval_minutes:
                return self.polling_frequency
            
            minutes = interval_minutes[self.interval]
            
            # Calculate next tick time
            if self.interval in ["1minute", "2minute", "5minute", "15minute", "30minute", "60minute", "1hour"]:
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
                self.logger.info(f"🎯 Take-profit hit for trade {trade['id']}: {trade['action']} @ {current_price:.2f} (target: {take_profit:.2f})")
                
                # Close the position - pass exit_type='tp_hit' to ensure GTT is cancelled
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
        Sync positions with broker to detect if any GTT stop-loss orders were triggered.
        This handles the case where the exchange closes a position via GTT while the engine wasn't monitoring.
        
        Args:
            current_time: Current timestamp
        """
        if not self.live_trading or not self.broker:
            return
        
        try:
            # Get current positions from broker
            broker_positions = self.broker.get_positions()
            
            # Create a map of broker positions by symbol
            broker_position_map = {}
            for pos in broker_positions:
                symbol = pos.get('tradingsymbol')
                qty = pos.get('quantity', 0)
                if symbol:
                    broker_position_map[symbol] = qty
            
            # Check if any active trades no longer have a corresponding broker position
            for trade in self.trading_engine.active_trades[:]:
                if trade.get('status') != 'open':
                    continue
                
                # Check if we have a broker order for this trade
                if not trade.get('broker_order_id'):
                    continue
                
                # Get broker position for this symbol
                broker_qty = broker_position_map.get(self.symbol, 0)
                
                # Determine if position should exist
                expected_qty = 1 if trade.get('action') == 'BUY' else -1  # 1 lot for LONG, -1 for SHORT
                
                # Check if GTT might have triggered (position closed but we think it's open)
                if broker_qty == 0:
                    # Position was closed on broker side (likely GTT triggered)
                    self.logger.warning(f"🔔 Detected position closed by broker (likely GTT triggered): Trade {trade['id']}")
                    
                    # Get current price to calculate PnL
                    try:
                        current_price = self.broker.get_price(self.symbol)
                    except:
                        current_price = trade.get('stop_loss', trade.get('entry_price'))
                    
                    # Close the trade internally with sl_hit status
                    strategy = self.strategy_manager.get_strategy_by_name(trade['strategy'])
                    if strategy:
                        position_type = 'LONG' if trade.get('action') == 'BUY' else 'SHORT'
                        closed_trades = self.trading_engine.close_trades(
                            strategy=strategy,
                            position_type=position_type,
                            price=current_price,
                            timestamp=current_time,
                            exit_type='sl_hit'  # Assume GTT stop-loss triggered
                        )
                        if closed_trades:
                            self.logger.info(f"✅ Trade {trade['id']} marked as closed (GTT stop-loss triggered)")
                elif (trade.get('action') == 'BUY' and broker_qty < 0) or \
                     (trade.get('action') == 'SELL' and broker_qty > 0):
                    # Position direction mismatch - something's wrong
                    self.logger.error(f"⚠️  Position direction mismatch for trade {trade['id']}! "
                                     f"Expected: {expected_qty}, Broker: {broker_qty}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to sync positions with broker: {e}")
    
    def _check_margin_before_order(self, required_margin: float) -> Tuple[bool, str]:
        """
        Check if sufficient margin is available before placing an order.
        
        Args:
            required_margin: Required margin amount
            
        Returns:
            Tuple of (is_sufficient, message)
        """
        try:
            margins = self.broker.check_margins()
            available_margin = margins.get('available', 0.0)
            
            # Add buffer
            required_with_buffer = required_margin * (1 + self.margin_buffer_percent / 100)
            
            if available_margin >= required_with_buffer:
                return True, f"Sufficient margin: {available_margin:.2f} >= {required_with_buffer:.2f}"
            else:
                return False, f"Insufficient margin: {available_margin:.2f} < {required_with_buffer:.2f} (required: {required_margin:.2f})"
                
        except Exception as e:
            self.logger.error(f"Error checking margin: {e}")
            return False, f"Error checking margin: {e}"
    
    def _margin_monitor_loop(self):
        """
        Background thread that continuously monitors margin requirements.
        Alerts if margin falls below threshold and exits if critically low.
        
        IMPORTANT: For commodity trading, margin is BLOCKED by the broker when position is opened.
        The 'available' margin is what remains AFTER blocking. We need to check:
        1. Total margin (available + utilised) vs required - is position properly funded?
        2. Buffer margin (available) - do we have enough excess for M2M fluctuations?
        """
        self.logger.info("💰 Margin monitoring thread started")
        
        while self.margin_monitor_running and self.is_running:
            try:
                # Check if we have open positions
                if not self.trading_engine.active_trades:
                    time.sleep(self.margin_check_interval)
                    continue
                
                # Get current margin status
                margins = self.broker.check_margins()
                available_margin = margins.get('available', 0.0)
                utilised_margin = margins.get('utilised', 0.0)
                
                # Get raw margin data for detailed analysis
                raw_margins = margins.get('raw', {})
                equity_raw = None
                
                # Try to get equity margin details (for single ledger)
                # This is important because utilised_margin from check_margins() might be stale
                try:
                    all_margins = self.broker.kite.margins()
                    equity_raw = all_margins.get('equity', {})
                    utilised_details = equity_raw.get('utilised', {})
                    
                    # Get M2M (unrealized P&L) - this is the key metric for margin calls
                    m2m_unrealised = utilised_details.get('m2m_unrealised', 0)
                    span_margin = utilised_details.get('span', 0)
                    exposure_margin = utilised_details.get('exposure', 0)
                    total_blocked = utilised_details.get('debits', 0)
                    
                    # IMPORTANT: Use equity utilised if it's higher (single ledger case)
                    # This ensures we capture the blocked margin correctly
                    if total_blocked > utilised_margin:
                        utilised_margin = total_blocked
                        self.logger.debug(f"Using equity utilised margin: {utilised_margin:.2f}")
                    
                except Exception as e:
                    self.logger.debug(f"Could not get detailed margin info: {e}")
                    m2m_unrealised = 0
                    span_margin = 0
                    exposure_margin = 0
                    total_blocked = utilised_margin
                
                # Calculate total margin AFTER potential update from equity utilised
                total_margin = available_margin + utilised_margin
                
                # Calculate required margin for current positions at current price
                total_required_margin = 0.0
                for trade in self.trading_engine.active_trades:
                    if trade.get('status') == 'open':
                        try:
                            current_price = self.broker.get_price(self.symbol)
                            if current_price > 0:
                                # Get actual margin requirement using order_margins API
                                transaction_type = 'BUY' if trade.get('action') == 'BUY' else 'SELL'
                                order_margins = self.broker.get_order_margins(
                                    symbol=self.symbol,
                                    transaction_type=transaction_type,
                                    quantity=1,  # 1 lot
                                    price=current_price,
                                    order_type='MARKET'
                                )
                                actual_margin = order_margins.get('total', 0.0)
                                if actual_margin > 0:
                                    total_required_margin += actual_margin
                                else:
                                    # Fallback to estimated calculation if API fails
                                    position_size = trade.get('position_size', 0)
                                    leverage = trade.get('leverage', 1.0)
                                    margin_required = position_size / leverage if leverage > 0 else position_size
                                    total_required_margin += margin_required
                                    self.logger.warning(f"Could not get actual margin for trade {trade.get('id')}, using estimate")
                        except Exception as e:
                            self.logger.warning(f"Error calculating margin for trade {trade.get('id')}: {e}")
                
                if total_required_margin > 0:
                    # CORRECT LOGIC: Check if TOTAL margin (blocked + available) covers requirement
                    # The broker has already blocked the required margin in 'utilised'
                    # We need to check:
                    # 1. Is blocked margin sufficient? (total_blocked >= total_required_margin)
                    # 2. Is there buffer for M2M? (available_margin > some threshold)
                    
                    # Margin coverage: How much of required margin is covered by total funds
                    margin_coverage = (total_margin / total_required_margin) * 100
                    
                    # Buffer ratio: Available margin as % of required (for M2M fluctuations)
                    buffer_ratio = (available_margin / total_required_margin) * 100
                    
                    # Log margin status
                    self.logger.debug(
                        f"💰 Margin Status: "
                        f"Total={total_margin:.2f}, Blocked={total_blocked:.2f}, "
                        f"Available={available_margin:.2f}, Required={total_required_margin:.2f}, "
                        f"Coverage={margin_coverage:.1f}%, Buffer={buffer_ratio:.1f}%, "
                        f"M2M={m2m_unrealised:.2f}"
                    )
                    
                    # Check thresholds using COVERAGE (not just available)
                    # Coverage should be >= 100% (total funds cover required margin)
                    # Buffer should be >= some threshold for M2M fluctuations
                    
                    # Critical: Total margin doesn't cover required (position under-funded)
                    if margin_coverage < self.margin_critical_threshold:
                        if self.live_trading:
                            self.logger.error(
                                f"🚨 CRITICAL: Margin coverage critically low! "
                                f"Coverage={margin_coverage:.1f}% (Total={total_margin:.2f}, Required={total_required_margin:.2f})"
                            )
                            self._exit_all_positions("margin_critical")
                        else:
                            self.logger.warning(
                                f"🚨 CRITICAL: Margin coverage would be critically low ({margin_coverage:.1f}%) - Would exit in live mode"
                            )
                    
                    # Alert: Low buffer for M2M fluctuations (less than 10% extra)
                    elif buffer_ratio < 10:
                        self.logger.warning(
                            f"⚠️  Low margin buffer: Only {buffer_ratio:.1f}% available for M2M fluctuations "
                            f"(Available={available_margin:.2f}, M2M={m2m_unrealised:.2f})"
                        )
                    
                    # OK: Sufficient margin
                    else:
                        self.logger.debug(
                            f"💰 Margin OK: Coverage={margin_coverage:.1f}%, Buffer={buffer_ratio:.1f}%"
                        )
                
                # Sleep until next check
                time.sleep(self.margin_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in margin monitor loop: {e}")
                time.sleep(self.margin_check_interval)
        
        self.logger.info("💰 Margin monitoring thread stopped")
    
    def _exit_all_positions(self, reason: str):
        """
        Exit all open positions.
        
        Args:
            reason: Reason for exit (e.g., 'margin_critical')
        """
        self.logger.warning(f"🛑 Exiting all positions. Reason: {reason}")
        
        try:
            # Get current price
            current_price = self.broker.get_price(self.symbol)
            if current_price <= 0:
                self.logger.error("Cannot get current price for exit")
                return
            
            current_time = datetime.now(self.ist_timezone)
            
            # Close all open trades
            for trade in self.trading_engine.active_trades[:]:
                if trade.get('status') == 'open':
                    position_type = 'LONG' if trade.get('action') == 'BUY' else 'SHORT'
                    strategy = self.strategy_manager.get_strategy_by_name(trade['strategy'])
                    if strategy:
                        self.trading_engine.close_trades(
                            strategy=strategy,
                            position_type=position_type,
                            price=current_price,
                            timestamp=current_time,
                            exit_type=reason  # Pass the exit reason (e.g., 'margin_critical')
                        )
        except Exception as e:
            self.logger.error(f"Error exiting positions: {e}")
    
    def _start_margin_monitor(self):
        """Start the margin monitoring thread."""
        if not self.margin_monitor_running:
            self.margin_monitor_running = True
            self.margin_monitor_thread = threading.Thread(
                target=self._margin_monitor_loop,
                daemon=True
            )
            self.margin_monitor_thread.start()
            self.logger.info("💰 Margin monitoring thread started")
    
    def _stop_margin_monitor(self):
        """Stop the margin monitoring thread."""
        if self.margin_monitor_running:
            self.margin_monitor_running = False
            self.logger.info("💰 Margin monitoring thread stopped")
    
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
    
    parser = argparse.ArgumentParser(description="Binance Trading Engine")
    parser.add_argument(
        "--config", 
        default="trading_config_kite.json",
        help="Path to configuration file (default: kite_trading_config.json)"
    )
    
    args = parser.parse_args()
    print(f"Using config file: {args.config}")
    
    # Create and run the trading engine
    engine = KiteTradingEngine(config_file=args.config)
    engine.run()


if __name__ == "__main__":
    main()
