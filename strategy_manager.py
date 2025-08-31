#!/usr/bin/env python3
"""
Strategy Manager - Centralized strategy management and optimization
Handles strategy optimization, parameter management, and strategy initialization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging

from data_fetcher import DataFetcher
from strategies import MovingAverageCrossover, RSIStrategy, DonchianChannelBreakout
from strategy_optimizer import (
    optimize_moving_average_crossover,
    optimize_rsi_strategy,
    optimize_donchian_channel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Centralized strategy management system that handles:
    - Strategy optimization
    - Parameter management
    - Strategy initialization
    - Performance tracking
    """
    
    def __init__(self):
        self.optimized_params = {}
        self.optimization_results = {}
        self.strategies = []
        self.data_fetcher = DataFetcher()
        
        # Default parameter ranges for optimization
        self.default_param_ranges = {
            'ma': {
                'short_window_range': [5, 10, 15, 20, 25, 30],
                'long_window_range': [20, 30, 40, 50, 60, 70],
                'risk_reward_range': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                'trading_fee': 0.001
            },
            'rsi': {
                'period_range': [10, 14, 20, 30],
                'overbought_range': [65, 70, 75, 80],
                'oversold_range': [20, 25, 30, 35],
                'risk_reward_range': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                'trading_fee': 0.001
            },
            'donchian': {
                'channel_period_range': [10, 15, 20, 25, 30],
                'risk_reward_range': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                'trading_fee': 0.001
            }
        }
    
    def optimize_strategies(self, 
                          symbol: str,
                          start_date: str,
                          end_date: str,
                          interval: str = "15m",
                          enabled_strategies: List[str] = None,
                          custom_param_ranges: Dict = None,
                          sharpe_threshold: float = 0.1) -> Dict:
        """
        Optimize all enabled strategies and return results.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            start_date: Start date for optimization data
            end_date: End date for optimization data
            interval: Data interval (e.g., "15m", "1h", "1d")
            enabled_strategies: List of strategies to optimize
            custom_param_ranges: Custom parameter ranges to override defaults
            sharpe_threshold: Minimum Sharpe ratio threshold
            
        Returns:
            Dictionary containing optimization results for each strategy
        """
        if enabled_strategies is None:
            enabled_strategies = ['ma', 'rsi', 'donchian']
        
        print(f"🚀 Starting strategy optimization for {symbol}")
        print(f"📅 Data period: {start_date} to {end_date}")
        print(f"⏱️  Interval: {interval}")
        print(f"🎯 Strategies: {enabled_strategies}")
        print("=" * 60)
        
        # Fetch data for optimization
        print(f"📥 Fetching optimization data...")
        data = self.data_fetcher.fetch_data(symbol, start_date, end_date, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data fetched for {symbol}")
        
        print(f"✅ Fetched {len(data)} data points")
        print(f"📊 Data range: {data.index[0]} to {data.index[-1]}")
        
        # Merge custom parameter ranges with defaults
        param_ranges = self._merge_param_ranges(custom_param_ranges)
        
        results = {}
        
        # Optimize each strategy
        for strategy_name in enabled_strategies:
            if strategy_name not in ['ma', 'rsi', 'donchian']:
                print(f"⚠️  Unknown strategy: {strategy_name}, skipping...")
                continue
            
            print(f"\n🎯 Optimizing {strategy_name.upper()} strategy...")
            
            try:
                if strategy_name == 'ma':
                    best_params, best_metrics = optimize_moving_average_crossover(
                        data=data,
                        short_window_range=param_ranges['ma']['short_window_range'],
                        long_window_range=param_ranges['ma']['long_window_range'],
                        risk_reward_range=param_ranges['ma']['risk_reward_range'],
                        trading_fee=param_ranges['ma']['trading_fee'],
                        sharpe_threshold=sharpe_threshold
                    )
                elif strategy_name == 'rsi':
                    best_params, best_metrics = optimize_rsi_strategy(
                        data=data,
                        period_range=param_ranges['rsi']['period_range'],
                        overbought_range=param_ranges['rsi']['overbought_range'],
                        oversold_range=param_ranges['rsi']['oversold_range'],
                        risk_reward_range=param_ranges['rsi']['risk_reward_range'],
                        trading_fee=param_ranges['rsi']['trading_fee'],
                        sharpe_threshold=sharpe_threshold
                    )
                elif strategy_name == 'donchian':
                    best_params, best_metrics = optimize_donchian_channel(
                        data=data,
                        channel_period_range=param_ranges['donchian']['channel_period_range'],
                        risk_reward_range=param_ranges['donchian']['risk_reward_range'],
                        trading_fee=param_ranges['donchian']['trading_fee'],
                        sharpe_threshold=sharpe_threshold
                    )
                
                results[strategy_name] = {
                    'parameters': best_params,
                    'metrics': best_metrics,
                    'optimization_date': datetime.now().isoformat(),
                    'data_period': f"{start_date} to {end_date}",
                    'symbol': symbol,
                    'interval': interval
                }
                
                # Store optimized parameters
                self.optimized_params[strategy_name] = best_params
                self.optimization_results[strategy_name] = results[strategy_name]
                
                print(f"✅ {strategy_name.upper()} optimization completed!")
                print(f"📊 Best Sharpe Ratio: {best_metrics['sharpe_ratio']:.3f}")
                print(f"📈 Total PnL: {best_metrics['total_pnl']:.2%}")
                print(f"🎯 Win Rate: {best_metrics['win_rate']:.2%}")
                
            except Exception as e:
                print(f"❌ Error optimizing {strategy_name}: {str(e)}")
                results[strategy_name] = {
                    'error': str(e),
                    'optimization_date': datetime.now().isoformat()
                }
        
        print("\n" + "=" * 60)
        print("🎯 OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        for strategy_name, result in results.items():
            if 'error' not in result:
                metrics = result['metrics']
                print(f"\n📊 {strategy_name.upper()}:")
                print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                print(f"   Total PnL: {metrics['total_pnl']:.2%}")
                print(f"   Win Rate: {metrics['win_rate']:.2%}")
                print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
                print(f"   Total Trades: {metrics['total_trades']}")
            else:
                print(f"\n❌ {strategy_name.upper()}: {result['error']}")
        
        return results
    
    def set_manual_parameters(self, 
                            ma_params: Dict = None,
                            rsi_params: Dict = None,
                            donchian_params: Dict = None) -> bool:
        """
        Set strategy parameters manually (from optimization results or user input).
        
        Args:
            ma_params: Moving Average parameters
            rsi_params: RSI parameters
            donchian_params: Donchian Channel parameters
            
        Returns:
            True if parameters were set successfully
        """
        self.optimized_params = {}
        
        if ma_params:
            self.optimized_params['ma'] = ma_params
        if rsi_params:
            self.optimized_params['rsi'] = rsi_params
        if donchian_params:
            self.optimized_params['donchian'] = donchian_params
        
        print(f"✅ Strategy parameters set successfully!")
        print(f"📋 Active strategies: {list(self.optimized_params.keys())}")
        
        return True
    
    def initialize_strategies(self, enabled_strategies: List[str]) -> List:
        """
        Initialize strategies with optimized parameters.
        
        Args:
            enabled_strategies: List of strategy names to initialize
            
        Returns:
            List of initialized strategy instances
        """
        self.strategies = []
        
        for strategy_name in enabled_strategies:
            if strategy_name not in self.optimized_params:
                print(f"⚠️  No parameters found for {strategy_name}, skipping...")
                continue
            
            params = self.optimized_params[strategy_name]
            
            try:
                if strategy_name == 'ma':
                    strategy = MovingAverageCrossover(
                        short_window=params['short_window'],
                        long_window=params['long_window'],
                        risk_reward_ratio=params['risk_reward_ratio'],
                        trading_fee=params['trading_fee']
                    )
                elif strategy_name == 'rsi':
                    strategy = RSIStrategy(
                        period=params['period'],
                        overbought=params['overbought'],
                        oversold=params['oversold'],
                        risk_reward_ratio=params['risk_reward_ratio'],
                        trading_fee=params['trading_fee']
                    )
                elif strategy_name == 'donchian':
                    strategy = DonchianChannelBreakout(
                        channel_period=params['channel_period'],
                        risk_reward_ratio=params['risk_reward_ratio'],
                        trading_fee=params['trading_fee']
                    )
                else:
                    print(f"⚠️  Unknown strategy: {strategy_name}, skipping...")
                    continue
                
                self.strategies.append(strategy)
                print(f"✅ Initialized {strategy_name}: {strategy.name}")
                
            except Exception as e:
                print(f"❌ Error initializing {strategy_name}: {str(e)}")
        
        print(f"🎯 Total strategies initialized: {len(self.strategies)}")
        return self.strategies
    
    def get_strategy_parameters(self, strategy_name: str) -> Dict:
        """
        Get parameters for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary of strategy parameters
        """
        return self.optimized_params.get(strategy_name, {})
    
    def get_all_parameters(self) -> Dict:
        """
        Get all strategy parameters.
        
        Returns:
            Dictionary of all strategy parameters
        """
        return self.optimized_params.copy()
    
    def get_optimization_results(self, strategy_name: str = None) -> Dict:
        """
        Get optimization results for strategies.
        
        Args:
            strategy_name: Specific strategy name (optional)
            
        Returns:
            Dictionary of optimization results
        """
        if strategy_name:
            return self.optimization_results.get(strategy_name, {})
        return self.optimization_results.copy()
    
    def print_optimization_summary(self):
        """
        Print a summary of all optimization results.
        """
        if not self.optimization_results:
            print("📋 No optimization results available.")
            return
        
        print("\n" + "=" * 80)
        print("🎯 OPTIMIZATION RESULTS SUMMARY")
        print("=" * 80)
        
        for strategy_name, result in self.optimization_results.items():
            if 'error' in result:
                print(f"\n❌ {strategy_name.upper()}: {result['error']}")
                continue
            
            metrics = result['metrics']
            params = result['parameters']
            
            print(f"\n📊 {strategy_name.upper()} Strategy:")
            print(f"   📅 Optimization Date: {result['optimization_date']}")
            print(f"   📈 Data Period: {result['data_period']}")
            print(f"   💰 Symbol: {result['symbol']}")
            print(f"   ⏱️  Interval: {result['interval']}")
            
            print(f"\n   ⚙️  Best Parameters:")
            for param, value in params.items():
                if param == 'trading_fee':
                    print(f"      {param}: {value:.4f} ({value*100:.2f}%)")
                else:
                    print(f"      {param}: {value}")
            
            print(f"\n   📊 Performance Metrics:")
            print(f"      Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"      Total PnL: {metrics['total_pnl']:.2%}")
            print(f"      Win Rate: {metrics['win_rate']:.2%}")
            print(f"      Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"      Total Trades: {metrics['total_trades']}")
            print(f"      Calmar Ratio: {metrics['calmar_ratio']:.3f}")
            print(f"      Profit Factor: {metrics['profit_factor']:.2f}")
    
    def export_optimization_results(self, filename: str = None) -> str:
        """
        Export optimization results to a JSON file.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to the exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        import json
        
        # Convert datetime objects to strings for JSON serialization
        export_data = {}
        for strategy_name, result in self.optimization_results.items():
            export_data[strategy_name] = result.copy()
            # Ensure all values are JSON serializable
            for key, value in export_data[strategy_name].items():
                if isinstance(value, (np.integer, np.floating)):
                    export_data[strategy_name][key] = float(value)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Optimization results exported to: {filename}")
        return filename
    
    def _merge_param_ranges(self, custom_ranges: Dict = None) -> Dict:
        """
        Merge custom parameter ranges with defaults.
        
        Args:
            custom_ranges: Custom parameter ranges to override defaults
            
        Returns:
            Merged parameter ranges
        """
        merged = self.default_param_ranges.copy()
        
        if custom_ranges:
            for strategy_name, ranges in custom_ranges.items():
                if strategy_name in merged:
                    merged[strategy_name].update(ranges)
        
        return merged
    
    def get_strategies(self) -> List:
        """
        Get the list of initialized strategies.
        
        Returns:
            List of strategy instances
        """
        return self.strategies.copy()
    
    def clear_strategies(self):
        """
        Clear all initialized strategies.
        """
        self.strategies = []
        print("🧹 All strategies cleared.")
    
    def validate_strategy_parameters(self, strategy_name: str, params: Dict) -> bool:
        """
        Validate strategy parameters.
        
        Args:
            strategy_name: Name of the strategy
            params: Parameters to validate
            
        Returns:
            True if parameters are valid
        """
        try:
            if strategy_name == 'ma':
                required_params = ['short_window', 'long_window', 'risk_reward_ratio', 'trading_fee']
                if not all(param in params for param in required_params):
                    return False
                if params['short_window'] >= params['long_window']:
                    return False
            elif strategy_name == 'rsi':
                required_params = ['period', 'overbought', 'oversold', 'risk_reward_ratio', 'trading_fee']
                if not all(param in params for param in required_params):
                    return False
                if params['oversold'] >= params['overbought']:
                    return False
            elif strategy_name == 'donchian':
                required_params = ['channel_period', 'risk_reward_ratio', 'trading_fee']
                if not all(param in params for param in required_params):
                    return False
            
            return True
        except Exception:
            return False
