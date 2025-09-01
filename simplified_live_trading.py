#!/usr/bin/env python3
"""
Simplified Live Trading Simulator
Uses StrategyManager and TradingEngine for modular, clean architecture
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
from typing import List, Dict

from strategy_manager import StrategyManager
from trading_engine import TradingEngine
from data_fetcher import DataFetcher
from streamlit_components import (
    display_trading_status,
    display_strategy_parameters,
    display_trade_history,
    display_performance_metrics,
    create_performance_chart,
    display_decision_log_viewer,
    display_decision_log_summary,
    display_log_files_info,
    display_market_data,
    create_strategy_parameter_inputs,
    create_live_trading_chart
)

def calculate_next_tick_time(interval: str, current_time: datetime = None) -> datetime:
    """
    Calculate the next tick time based on the interval.
    
    Args:
        interval: Data interval (e.g., "5m", "15m", "30m", "1h", "1d")
        current_time: Current time (defaults to now)
        
    Returns:
        Next tick time as datetime
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Convert interval to minutes
    interval_minutes = {
        "1m": 1,
        "2m": 2,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "60m": 60,
        "1h": 60,
        "1d": 1440,  # 24 * 60
        "1wk": 10080,  # 7 * 24 * 60
        "1mo": 43200,  # 30 * 24 * 60 (approximate)
    }
    
    if interval not in interval_minutes:
        raise ValueError(f"Unsupported interval: {interval}")
    
    minutes = interval_minutes[interval]
    
    # For intraday intervals, calculate next tick based on market hours
    if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "1h"]:
        # Market hours: 9:30 AM - 4:00 PM ET (simplified)
        # For now, we'll assume 24/7 trading for crypto
        # Calculate the next tick time
        current_minute = current_time.minute
        current_hour = current_time.hour
        
        # Find the next tick time
        if minutes == 60:  # 1h
            next_hour = current_hour + 1
            next_tick = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            # For other intervals, find the next multiple
            next_minute = ((current_minute // minutes) + 1) * minutes
            if next_minute >= 60:
                next_hour = current_hour + 1
                next_minute = next_minute % 60
                next_tick = current_time.replace(hour=next_hour, minute=next_minute, second=0, microsecond=0)
            else:
                next_tick = current_time.replace(minute=next_minute, second=0, microsecond=0)
    
    elif interval == "1d":
        # For daily data, next tick is tomorrow at market open (9:30 AM ET)
        next_tick = current_time.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)
    
    else:
        # For other intervals, use a simple approach
        next_tick = current_time + timedelta(minutes=minutes)
    
    return next_tick

def calculate_dynamic_polling_frequency(interval: str, current_time: datetime = None, buffer_seconds: int = 1) -> int:
    """
    Calculate dynamic polling frequency based on time until next tick.
    
    Args:
        interval: Data interval
        current_time: Current time (defaults to now)
        buffer_seconds: Buffer time to add after next tick (default: 1 second)
        
    Returns:
        Polling frequency in seconds
    """
    if current_time is None:
        current_time = datetime.now()
    
    next_tick_time = calculate_next_tick_time(interval, current_time)
    time_until_next_tick = (next_tick_time - current_time).total_seconds()
    
    # Add buffer time
    polling_frequency = max(1, int(time_until_next_tick + buffer_seconds))
    
    return polling_frequency

class SimplifiedLiveTradingSimulator:
    """
    Simplified live trading simulator using modular components.
    """
    
    def __init__(self, initial_balance: float = 10000):
        self.strategy_manager = StrategyManager()
        self.trading_engine = TradingEngine(initial_balance)
        self.data_fetcher = DataFetcher()
        
        # Trading state
        self.is_running = False
        self.current_data = pd.DataFrame()
        self.last_update = None
        
        # Mock trading settings
        self.mock_mode = False
        self.mock_data = pd.DataFrame()
        self.mock_current_index = 0
        self.mock_start_date = None
        self.mock_end_date = None
        self.mock_delay = 0.01
        
        # Trading configuration
        self.symbol = None
        self.interval = None
        self.polling_frequency = None
        self.session_id = None
        
        # Data tracking for efficient processing
        self.last_processed_timestamp = None
        self.last_processed_index = -1
    
    def setup_strategies(self, 
                        symbol: str,
                        start_date: str = None,
                        end_date: str = None,
                        interval: str = "15m",
                        enabled_strategies: List[str] = None,
                        ma_params: Dict = None,
                        rsi_params: Dict = None,
                        donchian_params: Dict = None) -> bool:
        """
        Setup strategies with manual parameters.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data fetching
            end_date: End date for data fetching
            interval: Data interval
            enabled_strategies: List of strategies to use
            ma_params: Moving Average parameters
            rsi_params: RSI parameters
            donchian_params: Donchian Channel parameters
            
        Returns:
            True if setup was successful
        """
        if enabled_strategies is None:
            enabled_strategies = ['ma', 'rsi']
        
        self.symbol = symbol
        self.interval = interval
        
        print(f"🚀 Setting up strategies for {symbol}...")
        
        # Set manual parameters
        success = self.strategy_manager.set_manual_parameters(
            ma_params=ma_params,
            rsi_params=rsi_params,
            donchian_params=donchian_params
        )
        
        if not success:
            st.error("Failed to set strategy parameters.")
            return False
        
        # Initialize strategies
        strategies = self.strategy_manager.initialize_strategies(enabled_strategies)
        
        if not strategies:
            st.error("No strategies initialized. Please check parameters.")
            return False
        
        print(f"✅ {len(strategies)} strategies initialized successfully!")
        return True
    
    def set_manual_parameters(self, 
                            ma_params: Dict = None,
                            rsi_params: Dict = None,
                            donchian_params: Dict = None,
                            enabled_strategies: List[str] = None) -> bool:
        """
        Set strategy parameters manually.
        
        Args:
            ma_params: Moving Average parameters
            rsi_params: RSI parameters
            donchian_params: Donchian Channel parameters
            enabled_strategies: List of strategies to enable
            
        Returns:
            True if parameters were set successfully
        """
        if enabled_strategies is None:
            enabled_strategies = []
            if ma_params:
                enabled_strategies.append('ma')
            if rsi_params:
                enabled_strategies.append('rsi')
            if donchian_params:
                enabled_strategies.append('donchian')
        
        success = self.strategy_manager.set_manual_parameters(
            ma_params=ma_params,
            rsi_params=rsi_params,
            donchian_params=donchian_params
        )
        
        if success:
            strategies = self.strategy_manager.initialize_strategies(enabled_strategies)
            if strategies:
                print(f"✅ {len(strategies)} strategies initialized with manual parameters!")
                return True
        
        return False
    
    def start_live_trading(self, 
                          symbol: str,
                          interval: str,
                          polling_frequency: int,
                          mock_mode: bool = False,
                          mock_days_back: int = 10,
                          mock_delay: float = 0.01) -> bool:
        """
        Start live trading simulation.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            polling_frequency: Polling frequency in seconds
            mock_mode: Whether to use mock mode
            mock_days_back: Days back for mock data
            mock_delay: Delay between mock data points
            
        Returns:
            True if trading started successfully
        """
        if not self.strategy_manager.get_strategies():
            st.error("No strategies initialized. Please setup strategies first.")
            return False
        
        self.is_running = True
        self.symbol = symbol
        self.interval = interval
        self.polling_frequency = polling_frequency
        self.mock_mode = mock_mode
        self.mock_delay = mock_delay
        
        # Reset data tracking for new session
        self.last_processed_timestamp = None
        self.last_processed_index = -1
        
        # Create session ID and setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_mock" if mock_mode else "_live"
        self.session_id = f"{symbol}_{timestamp}{mode_suffix}"
        
        self.trading_engine.setup_logging(self.session_id, symbol)
        
        # Setup mock data if in mock mode
        if mock_mode:
            self._setup_mock_data(symbol, interval, mock_days_back)
        
        print(f"🚀 Live trading started!")
        print(f"📊 Symbol: {symbol}")
        print(f"⏱️  Interval: {interval}")
        print(f"📡 Polling Frequency: {polling_frequency} seconds")
        print(f"🎭 Mode: {'MOCK' if mock_mode else 'LIVE'}")
        print(f"🎯 Active Strategies: {[s.name for s in self.strategy_manager.get_strategies()]}")
        print(f"💰 Initial Balance: ${self.trading_engine.initial_balance:,.2f}")
        print("=" * 60)
        
        st.success(f"Live trading started! Session ID: {self.session_id}")
        
        # Start the trading loop in a separate thread
        trading_thread = threading.Thread(
            target=self._trading_loop,
            args=(symbol, interval, polling_frequency)
        )
        trading_thread.daemon = True
        trading_thread.start()
        
        return True
    
    def stop_live_trading(self):
        """Stop the live trading simulation."""
        self.is_running = False
        print(f"🛑 Live trading stopped.")
        
        status = self.trading_engine.get_current_status()
        print(f"📊 Final Balance: ${status['current_balance']:,.2f}")
        print(f"📈 Total PnL: ${status['total_pnl']:,.2f}")
        print(f"📋 Total Trades: {status['total_trades']}")
        print("=" * 60)
        
        st.info("Live trading stopped.")
    
    def _setup_mock_data(self, symbol: str, interval: str, mock_days_back: int):
        """Setup mock data for testing."""
        print(f"🎭 Setting up mock data for {symbol}...")
        
        # Calculate date range
        # end_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
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
    
    def _trading_loop(self, symbol: str, interval: str, polling_frequency: int):
        """Main trading loop that runs in a separate thread."""
        print(f"🚀 Starting {'mock' if self.mock_mode else 'live'} trading loop for {symbol}")
        
        if self.mock_mode:
            self._mock_trading_loop(symbol, interval, polling_frequency)
        else:
            self._live_trading_loop(symbol, interval, polling_frequency)
    
    def _live_trading_loop(self, symbol: str, interval: str, polling_frequency: int):
        """
        Live trading loop that fetches real-time data with dynamic polling frequency.
        
        The polling frequency is calculated dynamically based on the time until the next tick,
        with the provided polling_frequency used as a fallback in case of errors.
        """
        while self.is_running:
            try:
                current_time = datetime.now()
                print(f"🕐 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}]")
                        
                # Calculate date range for data fetching
                if interval in ["5m", "15m", "30m", "1h"]:
                    start_date = (current_time - timedelta(days=7)).strftime("%Y-%m-%d")
                else:
                    start_date = (current_time - timedelta(days=60)).strftime("%Y-%m-%d")
                
                # end_date = current_time.strftime("%Y-%m-%d")
                end_date = (current_time + timedelta(days=1)).strftime("%Y-%m-%d")
                
                print(f"📥 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Fetching {interval} data for {symbol}")
                
                data = self.data_fetcher.fetch_data(symbol, start_date, end_date, interval=interval)
                # ignore last tick of data, as tick not yet closed
                data = data.iloc[:-1]
                
                if not data.empty:
                    print(f"✅ Successfully fetched {len(data)} data points")
                    print(f"📊 Latest data: {data.index[-1]} - Close: ${data['Close'].iloc[-1]:.2f}")
                    
                    # Check if we have new data
                    latest_data_timestamp = data.index[-1]
                    has_new_data = self.last_processed_timestamp is None or latest_data_timestamp > self.last_processed_timestamp
                    
                    if has_new_data:
                        print(f"🆕 New data detected! Processing strategies...")
                        
                        # Save data to CSV
                        self.trading_engine.save_data_to_csv(data, current_time)
                        
                        self.current_data = data
                        self.last_update = current_time
                        
                        # Process each strategy only when new data is available
                        for strategy in self.strategy_manager.get_strategies():
                            self.trading_engine.process_strategy_signals(strategy, data, current_time)
                        
                        # Update the last processed timestamp
                        self.last_processed_timestamp = latest_data_timestamp
                        print(f"✅ Strategies processed for new data at {latest_data_timestamp}")
                    else:
                        print(f"⏸️  No new data available. Skipping strategy processing.")
                        # Still update current data for display purposes
                        self.current_data = data
                        self.last_update = current_time
                else:
                    print(f"⚠️  No data received for {symbol}")

                if not has_new_data:
                    dynamic_polling_frequency = 5
                else:
                    # Calculate dynamic polling frequency based on next tick time
                    dynamic_polling_frequency = calculate_dynamic_polling_frequency(interval, datetime.now(), buffer_seconds=5)
                    
                # Wait for next polling cycle using dynamic frequency
                print(f"⏳ Waiting {dynamic_polling_frequency} seconds until next update...")
                time.sleep(dynamic_polling_frequency)
                
            except Exception as e:
                error_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"❌ Error in trading loop: {str(e)}")
                # Use dynamic polling frequency even in error case
                try:
                    dynamic_polling_frequency = calculate_dynamic_polling_frequency(interval, datetime.now(), buffer_seconds=5)
                    time.sleep(dynamic_polling_frequency)
                except:
                    # Fallback to original polling frequency if dynamic calculation fails
                    time.sleep(polling_frequency)
    
    def _mock_trading_loop(self, symbol: str, interval: str, polling_frequency: int):
        """Mock trading loop that processes historical data sequentially."""
        print(f"🎭 Starting mock trading simulation...")
        print(f"⚡ Mock mode: Processing new data points only")
        
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
                    # print(f"🎭 [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Processing NEW mock data point {self.mock_current_index + 1}/{len(self.mock_data)} ({progress:.1f}%)")
                    # print(f"📊 Mock data: {current_time} - Close: ${current_data_point['Close']:.2f}")
                    
                    # Save data to CSV
                    self.trading_engine.save_data_to_csv(data, current_time)
                    
                    self.current_data = data
                    self.last_update = current_time
                    
                    # Process each strategy only for new data
                    for strategy in self.strategy_manager.get_strategies():
                        self.trading_engine.process_strategy_signals(strategy, data, current_time)
                    
                    # Update the last processed index
                    self.last_processed_index = self.mock_current_index
                    # print(f"✅ Strategies processed for new data point {self.mock_current_index + 1}")
                else:
                    # No new data to process, just wait
                    print(f"⏸️  No new data point to process. Waiting...")
                
                # Move to next data point
                self.mock_current_index += 1
                
                # Use configurable delay in mock mode
                time.sleep(self.mock_delay)
                
            except Exception as e:
                error_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"❌ Error in mock trading loop: {str(e)}")
                time.sleep(0.1)
        
        end_time = time.time()
        simulation_duration = end_time - start_time
        
        if self.mock_current_index >= len(self.mock_data):
            print(f"🎭 Mock trading simulation completed!")
            print(f"📊 Processed all {len(self.mock_data)} data points")
            print(f"⏱️  Simulation duration: {simulation_duration:.2f} seconds")
            self.is_running = False
    
    def get_current_status(self) -> Dict:
        """Get current trading status."""
        status = self.trading_engine.get_current_status(self.current_data)
        status['is_running'] = self.is_running
        status['last_update'] = self.last_update
        return status
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        return self.trading_engine.get_trade_history_df()
    
    def get_decision_log_df(self) -> pd.DataFrame:
        """Get decision log as DataFrame."""
        return self.trading_engine.get_decision_log_df()
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        return self.trading_engine.calculate_performance_metrics()
    
    def get_log_files_info(self) -> Dict:
        """Get information about log files."""
        return self.trading_engine.get_log_files_info()
    
    def get_strategy_parameters(self) -> Dict:
        """Get current strategy parameters."""
        return self.strategy_manager.get_all_parameters()

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Simplified Live Trading Simulator", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🚀 Simplified Live Trading Simulator")
    st.markdown("Modular trading simulation with optimized strategies")
    
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
    
    # Initialize simulator
    if 'simulator' not in st.session_state:
        st.session_state.simulator = SimplifiedLiveTradingSimulator()
    
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
    
    # Strategy parameters using modular component
    params = create_strategy_parameter_inputs(enabled_strategies)
    ma_params = params['ma_params']
    rsi_params = params['rsi_params']
    donchian_params = params['donchian_params']
    
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
            step=1
        )
        
        mock_speed = st.sidebar.selectbox(
            "Mock Simulation Speed",
            options=["Ultra Fast (10ms)", "Fast (100ms)", "Normal (1s)", "Slow (5s)"],
            index=0
        )
        
        speed_delays = {
            "Ultra Fast (10ms)": 0.01,
            "Fast (100ms)": 0.1,
            "Normal (1s)": 1.0,
            "Slow (5s)": 5.0
        }
        mock_delay = speed_delays[mock_speed]
    
    # Main content area - Use full width for better chart display
    st.subheader("📈 Live Trading Dashboard")
    
    # Current time display
    current_time = datetime.now()
    st.caption(f"🕐 Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Status display using modular component
    status = simulator.get_current_status()
    display_trading_status(status, simulator)
    
    # Last update
    if status['last_update']:
        update_text = f"Last update: {status['last_update'].strftime('%Y-%m-%d %H:%M:%S')}"
        if simulator.mock_mode and simulator.mock_data is not None and len(simulator.mock_data) > 0:
            progress = (simulator.mock_current_index / len(simulator.mock_data)) * 100
            update_text += f" | Mock Progress: {progress:.1f}%"
        st.caption(update_text)
    
    # Display current parameters using modular component
    params = simulator.get_strategy_parameters()
    display_strategy_parameters(params)
        
    # Live Trading Chart
    if not simulator.current_data.empty:
        st.subheader("📊 Live Trading Chart")
        
                 # Enhanced CSS to ensure proper width without breaking layout
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
         .stPlotlyChart > div {
             width: 100% !important;
             max-width: 100% !important;
             min-width: 100% !important;
         }
         [data-testid="stPlotlyChart"] {
             width: 100% !important;
             max-width: 100% !important;
             min-width: 100% !important;
         }
         </style>
         """, unsafe_allow_html=True)
        
        # Get strategies for chart
        strategies = simulator.strategy_manager.get_strategies()
        
        # Create and display the live chart
        live_chart = create_live_trading_chart(
            current_data=simulator.current_data,
            trade_history_df=simulator.get_trade_history_df(),
            strategies=strategies,
            symbol=symbol,
            active_trade_info=status.get('active_trade_info'),
            session_id=simulator.session_id
        )
        
        if live_chart:
                         # Use a simple container for the chart
            with st.container():
                 # No additional CSS needed - let it use the global styles
                
                st.plotly_chart(
                    live_chart, 
                    use_container_width=True, 
                    key="live_trading_chart", 
                    config={
                        'displayModeBar': True, 
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                        'responsive': True,
                        'autosize': True,
                        'fillFrame': True,
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'{symbol}_live_chart',
                            'height': 1000,
                            'width': None,
                            'scale': 1
                        }
                    }
                )
            
            # Add auto-refresh for live chart
            if status['is_running']:
                st.caption("🔄 Chart auto-updates with new data")
            
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
                if not simulator.current_data.empty:
                    current_price = simulator.current_data['Close'].iloc[-1]
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

    # Controls section in sidebar for better organization
    st.sidebar.subheader("🎮 Trading Controls")
    
    # Setup strategies button
    if st.sidebar.button("⚙️ Setup Strategies", type="primary"):
        with st.spinner("Setting up strategies..."):
            success = simulator.setup_strategies(
                symbol=symbol,
                interval=interval,
                enabled_strategies=enabled_strategies,
                ma_params=ma_params,
                rsi_params=rsi_params,
                donchian_params=donchian_params
            )
            if success:
                st.success("Strategies setup completed!")
            else:
                st.error("Strategy setup failed!")
    
    # Start/Stop trading
    if not status['is_running']:
        if st.sidebar.button("▶️ Start Trading", type="primary"):
            mock_delay_param = mock_delay if mock_mode else 0.01
            success = simulator.start_live_trading(
                symbol=symbol,
                interval=interval,
                polling_frequency=polling_frequency,
                mock_mode=mock_mode,
                mock_days_back=mock_days_back if mock_mode else 10,
                mock_delay=mock_delay_param
            )
            if success:
                st.rerun()
            else:
                st.error("Failed to start trading!")
    else:
        if st.sidebar.button("⏹️ Stop Trading", type="secondary"):
            simulator.stop_live_trading()
            st.rerun()
    
    # Log files information
    if status['is_running']:
        log_info = simulator.get_log_files_info()
        display_log_files_info(log_info)
    
    # Decision Log Viewer
    if status['is_running']:
        st.subheader("🎯 Decision Log Viewer")
        
        # Auto-refresh decision log
        if st.button("🔄 Refresh Decision Log", type="secondary"):
            st.rerun()
        
        decision_log_df = simulator.get_decision_log_df()
        display_decision_log_viewer(decision_log_df)
        display_decision_log_summary(decision_log_df)
    
    # Trading history using modular component
    trade_history_df = simulator.get_trade_history_df()
    display_trade_history(trade_history_df, symbol)
    
    # Performance Metrics using modular component
    if not trade_history_df.empty:
        metrics = simulator.calculate_performance_metrics()
        display_performance_metrics(metrics)
        
        # Performance charts
        st.subheader("📊 Performance Analysis")
        performance_fig = create_performance_chart(trade_history_df, initial_balance)
        if performance_fig:
            st.plotly_chart(performance_fig, use_container_width=True, key="cumulative_performance_chart")
        
        # Recent Trades Summary
        st.subheader("📋 Recent Trades")
        closed_trades = trade_history_df[trade_history_df['status'] == 'closed'].tail(10)
        
        # Check if required columns exist
        required_columns = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'action', 'strategy']
        missing_columns = [col for col in required_columns if col not in closed_trades.columns]
        
        if missing_columns:
            st.warning(f"Trade history missing required columns: {missing_columns}. Cannot display recent trades.")
        elif not closed_trades.empty:
            # Display recent trades in a more visual way
            for _, trade in closed_trades.iterrows():
                trade_color = "🟢" if trade['pnl'] > 0 else "🔴"
                trade_direction = "📈" if trade['action'] == 'BUY' else "📉"
                
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.write(f"{trade_direction} {trade['strategy']}")
                    with col2:
                        st.write(f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f}")
                    with col3:
                        st.write(f"{trade['pnl']:.2%}")
                    with col4:
                        st.write(f"{trade['entry_time'].strftime('%m/%d %H:%M')} → {trade['exit_time'].strftime('%m/%d %H:%M')}")
                    with col5:
                        st.write(f"{trade_color} ${trade['pnl'] * trade['quantity'] * trade['entry_price']:.2f}")
    
    # Market data display
    display_market_data(simulator.current_data)

if __name__ == "__main__":
    main()
