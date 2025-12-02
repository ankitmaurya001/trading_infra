#!/usr/bin/env python3
"""
Comprehensive Strategy Validation Pipeline
Combines parameter optimization with train-test validation and mock trading testing.

This pipeline:
1. Fetches historical data and splits into train/test sets
2. Optimizes parameters on training data
3. Validates optimized parameters on test data using mock trading
4. Compares performance across strategies
5. Prepares best strategies for live trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_fetcher import BinanceDataFetcher
from strategy_optimizer import (
    optimize_moving_average_crossover,
    optimize_rsi_strategy,
    optimize_donchian_channel
)
from strategy_manager import StrategyManager
from trading_engine import TradingEngine
import config as cfg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveStrategyValidator:
    """
    Comprehensive strategy validation pipeline that combines optimization with train-test validation.
    """
    
    def __init__(self, 
                 initial_balance: float = 10000,
                 max_leverage: float = 10.0,
                 max_loss_percent: float = 2.0,
                 trading_fee: float = 0):
        """
        Initialize the validator.
        
        Args:
            initial_balance: Initial trading balance
            max_leverage: Maximum leverage allowed
            max_loss_percent: Maximum loss percentage per trade
            trading_fee: Trading fee as decimal (0.001 = 0.1%)
        """
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.max_loss_percent = max_loss_percent
        self.trading_fee = trading_fee
        
        # Initialize components
        self.data_fetcher = BinanceDataFetcher(
            api_key=cfg.BINANCE_API_KEY, 
            api_secret=cfg.BINANCE_SECRET_KEY
        )
        self.strategy_manager = StrategyManager()
        self.trading_engine = TradingEngine(
            initial_balance, 
            max_leverage, 
            max_loss_percent
        )
        
        # Results storage
        self.optimization_results = {}
        self.validation_results = {}
        self.final_recommendations = {}
        
        # Data storage
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.symbol = None
        self.interval = None
        
    def fetch_and_split_data(self, 
                           symbol: str,
                           start_date: str,
                           end_date: str,
                           interval: str = "15m",
                           train_ratio: float = 0.7,
                           random_shift_test_percentage: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch historical data and split into train/test sets.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval
            train_ratio: Ratio of data to use for training (0.7 = 70%)
            
        Returns:
            Tuple of (train_data, test_data)
        """
        print(f"📥 Fetching data for {symbol} from {start_date} to {end_date}...")
        print(f"📊 Interval: {interval}")
        print(f"🔀 Train/Test split: {train_ratio:.1%}/{1-train_ratio:.1%}")
        
        # Fetch historical data
        data = self.data_fetcher.fetch_historical_data(
            symbol, start_date, end_date, interval=interval
        )
        
        if data.empty:
            raise ValueError(f"No data fetched for {symbol}")
        
        print(f"✅ Successfully fetched {len(data)} data points")
        print(f"📅 Data range: {data.index[0]} to {data.index[-1]}")
        
        # Split data
        split_index = int(len(data) * train_ratio)
        self.train_data = data.iloc[:split_index].copy()
        self.test_data = data.iloc[int(random_shift_test_percentage*split_index):].copy()
        
        self.symbol = symbol
        self.interval = interval
        
        print(f"📚 Training data: {len(self.train_data)} points ({self.train_data.index[0]} to {self.train_data.index[-1]})")
        print(f"🧪 Test data: {len(self.test_data)} points ({self.test_data.index[0]} to {self.test_data.index[-1]})")
        
        return self.train_data, self.test_data
    
    def optimize_strategies_on_train_data(self, 
                                        strategies_to_optimize: List[str] = None,
                                        custom_param_ranges: Dict = None) -> Dict:
        """
        Optimize strategies on training data.
        
        Args:
            strategies_to_optimize: List of strategies to optimize ['ma', 'rsi', 'donchian']
            custom_param_ranges: Custom parameter ranges for optimization
            
        Returns:
            Dictionary with optimization results
        """
        if strategies_to_optimize is None:
            strategies_to_optimize = ['ma', 'rsi', 'donchian']
        
        if self.train_data.empty:
            raise ValueError("No training data available. Run fetch_and_split_data() first.")
        
        print("\n" + "="*80)
        print("🔍 PARAMETER OPTIMIZATION ON TRAINING DATA")
        print("="*80)
        
        optimization_results = {}
        
        # Default parameter ranges
        default_ranges = {
            'ma': {
                'short_window_range': [5, 10, 15, 20, 25, 30],
                'long_window_range': [20, 30, 40, 50, 60, 70],
                'risk_reward_range': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            },
            'rsi': {
                'period_range': [10, 14, 20, 30],
                'overbought_range': [65, 70, 75, 80],
                'oversold_range': [20, 25, 30, 35],
                'risk_reward_range': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            },
            'donchian': {
                'channel_period_range': [10, 15, 20, 25, 30],
                'risk_reward_range': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            }
        }
        
        # Use custom ranges if provided
        if custom_param_ranges:
            for strategy, ranges in custom_param_ranges.items():
                if strategy in default_ranges:
                    default_ranges[strategy].update(ranges)
        
        # Optimize Moving Average Crossover
        if 'ma' in strategies_to_optimize:
            print("\n📈 OPTIMIZING MOVING AVERAGE CROSSOVER")
            print("-" * 50)
            
            try:
                # Use the optimizer directly to get top 5 results
                from strategy_optimizer import StrategyOptimizer, MovingAverageCrossover
                
                param_ranges = {
                    'short_window': default_ranges['ma']['short_window_range'],
                    'long_window': default_ranges['ma']['long_window_range'],
                    'risk_reward_ratio': default_ranges['ma']['risk_reward_range'],
                    'trading_fee': [self.trading_fee]
                }
                
                optimizer = StrategyOptimizer(
                    data=self.train_data,
                    strategy_class=MovingAverageCrossover,
                    param_ranges=param_ranges,
                    optimization_metric="composite_score",
                    min_trades=10,
                    max_drawdown_threshold=0.4,
                    sharpe_threshold=0.1
                )
                
                # Run optimization
                best_params, best_metrics = optimizer.optimize()
                optimizer.print_optimization_summary()
                
                # Get top 5 results
                top_5_results = optimizer.get_top_results(5)
                
                optimization_results['ma'] = {
                    'best_params': best_params,
                    'best_metrics': best_metrics,
                    'top_5_results': top_5_results,
                    'strategy_name': 'Moving Average Crossover'
                }
                
                print(f"✅ MA Optimization completed")
                print(f"🏆 Best Parameters: {best_params}")
                print(f"📊 Best Sharpe Ratio: {best_metrics.get('sharpe_ratio', 'N/A'):.3f}")
                print(f"📈 Best Total PnL: {best_metrics.get('total_pnl', 'N/A'):.2%}")
                
            except Exception as e:
                print(f"❌ MA Optimization failed: {e}")
                optimization_results['ma'] = {'error': str(e)}
        
        # Optimize RSI Strategy
        if 'rsi' in strategies_to_optimize:
            print("\n📊 OPTIMIZING RSI STRATEGY")
            print("-" * 50)
            
            try:
                # Use the optimizer directly to get top 5 results
                from strategy_optimizer import StrategyOptimizer, RSIStrategy
                
                param_ranges = {
                    'period': default_ranges['rsi']['period_range'],
                    'overbought': default_ranges['rsi']['overbought_range'],
                    'oversold': default_ranges['rsi']['oversold_range'],
                    'risk_reward_ratio': default_ranges['rsi']['risk_reward_range'],
                    'trading_fee': [self.trading_fee]
                }
                
                optimizer = StrategyOptimizer(
                    data=self.train_data,
                    strategy_class=RSIStrategy,
                    param_ranges=param_ranges,
                    optimization_metric="composite_score",
                    min_trades=10,
                    max_drawdown_threshold=0.4,
                    sharpe_threshold=0.1
                )
                
                # Run optimization
                best_params, best_metrics = optimizer.optimize()
                optimizer.print_optimization_summary()
                
                # Get top 5 results
                top_5_results = optimizer.get_top_results(5)
                
                optimization_results['rsi'] = {
                    'best_params': best_params,
                    'best_metrics': best_metrics,
                    'top_5_results': top_5_results,
                    'strategy_name': 'RSI Strategy'
                }
                
                print(f"✅ RSI Optimization completed")
                print(f"🏆 Best Parameters: {best_params}")
                print(f"📊 Best Sharpe Ratio: {best_metrics.get('sharpe_ratio', 'N/A'):.3f}")
                print(f"📈 Best Total PnL: {best_metrics.get('total_pnl', 'N/A'):.2%}")
                
            except Exception as e:
                print(f"❌ RSI Optimization failed: {e}")
                optimization_results['rsi'] = {'error': str(e)}
        
        # Optimize Donchian Channel
        if 'donchian' in strategies_to_optimize:
            print("\n📉 OPTIMIZING DONCHIAN CHANNEL")
            print("-" * 50)
            
            try:
                # Use the optimizer directly to get top 5 results
                from strategy_optimizer import StrategyOptimizer, DonchianChannelBreakout
                
                param_ranges = {
                    'channel_period': default_ranges['donchian']['channel_period_range'],
                    'risk_reward_ratio': default_ranges['donchian']['risk_reward_range'],
                    'trading_fee': [self.trading_fee]
                }
                
                optimizer = StrategyOptimizer(
                    data=self.train_data,
                    strategy_class=DonchianChannelBreakout,
                    param_ranges=param_ranges,
                    optimization_metric="composite_score",
                    min_trades=10,
                    max_drawdown_threshold=0.4,
                    sharpe_threshold=0.1
                )
                
                # Run optimization
                best_params, best_metrics = optimizer.optimize()
                optimizer.print_optimization_summary()
                
                # Get top 5 results
                top_5_results = optimizer.get_top_results(5)
                
                optimization_results['donchian'] = {
                    'best_params': best_params,
                    'best_metrics': best_metrics,
                    'top_5_results': top_5_results,
                    'strategy_name': 'Donchian Channel'
                }
                
                print(f"✅ Donchian Optimization completed")
                print(f"🏆 Best Parameters: {best_params}")
                print(f"📊 Best Sharpe Ratio: {best_metrics.get('sharpe_ratio', 'N/A'):.3f}")
                print(f"📈 Best Total PnL: {best_metrics.get('total_pnl', 'N/A'):.2%}")
                
            except Exception as e:
                print(f"❌ Donchian Optimization failed: {e}")
                optimization_results['donchian'] = {'error': str(e)}
        
        self.optimization_results = optimization_results
        return optimization_results
    
    def validate_strategies_on_test_data(self, 
                                       strategies_to_validate: List[str] = None,
                                       mock_trading_delay: float = 0.01,
                                       top_n_params: int = 5) -> Dict:
        """
        Validate optimized strategies on test data using mock trading.
        
        Args:
            strategies_to_validate: List of strategies to validate
            mock_trading_delay: Delay between mock data points (seconds)
            top_n_params: Number of top parameter sets to validate per strategy
            
        Returns:
            Dictionary with validation results
        """
        if strategies_to_validate is None:
            strategies_to_validate = list(self.optimization_results.keys())
        
        if self.test_data.empty:
            raise ValueError("No test data available. Run fetch_and_split_data() first.")
        
        if not self.optimization_results:
            raise ValueError("No optimization results available. Run optimize_strategies_on_train_data() first.")
        
        print("\n" + "="*80)
        print("🧪 VALIDATING STRATEGIES ON TEST DATA")
        print("="*80)
        
        validation_results = {}
        
        for strategy_key in strategies_to_validate:
            if strategy_key not in self.optimization_results:
                print(f"⚠️  Skipping {strategy_key} - no optimization results available")
                continue
            
            if 'error' in self.optimization_results[strategy_key]:
                print(f"⚠️  Skipping {strategy_key} - optimization failed")
                continue
            
            print(f"\n🎯 VALIDATING {strategy_key.upper()} STRATEGY")
            print("-" * 50)
            
            try:
                # Get top 5 parameter sets
                top_5_results = self.optimization_results[strategy_key].get('top_5_results', [])
                strategy_name = self.optimization_results[strategy_key]['strategy_name']
                
                if not top_5_results:
                    print(f"⚠️  No top 5 results available for {strategy_key}")
                    continue
                
                print(f"📋 Testing top {min(len(top_5_results), top_n_params)} parameter sets...")
                
                # Validate each of the top parameter sets
                strategy_validation_results = []
                
                for i, result in enumerate(top_5_results[:top_n_params]):
                    params = result['parameters']
                    score = result['score']
                    
                    print(f"\n🔍 Testing parameter set {i+1}/{min(len(top_5_results), top_n_params)}")
                    print(f"📊 Score: {score:.4f}")
                    print(f"⚙️  Parameters: {params}")
                    
                    # Setup strategy with current parameters
                    if strategy_key == 'ma':
                        success = self.strategy_manager.set_manual_parameters(ma_params=params)
                    elif strategy_key == 'rsi':
                        success = self.strategy_manager.set_manual_parameters(rsi_params=params)
                    elif strategy_key == 'donchian':
                        success = self.strategy_manager.set_manual_parameters(donchian_params=params)
                    else:
                        print(f"❌ Unknown strategy key: {strategy_key}")
                        continue
                    
                    if not success:
                        print(f"❌ Failed to setup {strategy_key} strategy with parameters {i+1}")
                        continue
                    
                    # Initialize strategy
                    strategies = self.strategy_manager.initialize_strategies([strategy_key])
                    if not strategies:
                        print(f"❌ Failed to initialize {strategy_key} strategy with parameters {i+1}")
                        continue
                    
                    strategy = strategies[0]
                    
                    # Create new trading engine for validation
                    validation_engine = TradingEngine(
                        self.initial_balance, 
                        self.max_leverage, 
                        self.max_loss_percent
                    )
                    
                    # Setup logging for validation
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    session_id = f"{self.symbol}_{strategy_key}_param_{i+1}_validation_{timestamp}"
                    validation_engine.setup_logging(session_id, self.symbol)
                    
                    # Run mock trading simulation
                    print(f"🚀 Running mock trading simulation...")
                    
                    start_time = datetime.now()
                    
                    # Process each data point sequentially
                    for j in range(len(self.test_data)):
                        # Get data up to current point
                        current_data = self.test_data.iloc[:j+1]
                        current_time = self.test_data.index[j]
                        
                        # Process strategy signals
                        validation_engine.process_strategy_signals(strategy, current_data, current_time)
                        
                        # Small delay to simulate real-time processing
                        if mock_trading_delay > 0:
                            import time
                            time.sleep(mock_trading_delay)
                    
                    end_time = datetime.now()
                    simulation_duration = (end_time - start_time).total_seconds()
                    
                    # Get validation results
                    final_status = validation_engine.get_current_status()
                    trade_history = validation_engine.get_trade_history_df()
                    performance_metrics = validation_engine.calculate_performance_metrics()
                    
                    # Store results for this parameter set
                    param_result = {
                        'parameter_set': i + 1,
                        'parameters': params,
                        'optimization_score': score,
                        'final_status': final_status,
                        'trade_history': trade_history,
                        'performance_metrics': performance_metrics,
                        'simulation_duration': simulation_duration,
                        'session_id': session_id
                    }
                    
                    strategy_validation_results.append(param_result)
                    
                    print(f"✅ Parameter set {i+1} validation completed!")
                    print(f"📊 Final Balance: ${final_status['current_balance']:,.2f}")
                    print(f"📈 Total PnL: {performance_metrics.get('total_pnl', 0):.2%}")
                    print(f"📋 Total Trades: {final_status['total_trades']}")
                    
                    # Display key metrics
                    if 'sharpe_ratio' in performance_metrics:
                        print(f"📊 Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
                    if 'win_rate' in performance_metrics:
                        print(f"🎯 Win Rate: {performance_metrics['win_rate']:.2%}")
                    if 'max_drawdown' in performance_metrics:
                        print(f"📉 Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
                
                # Store all validation results for this strategy
                validation_results[strategy_key] = {
                    'strategy_name': strategy_name,
                    'parameter_sets_tested': len(strategy_validation_results),
                    'validation_results': strategy_validation_results
                }
                
                print(f"\n✅ {strategy_key.upper()} validation completed for {len(strategy_validation_results)} parameter sets!")
                
            except Exception as e:
                print(f"❌ {strategy_key.upper()} validation failed: {e}")
                validation_results[strategy_key] = {'error': str(e)}
        
        self.validation_results = validation_results
        return validation_results
    
    def compare_strategies(self) -> Dict:
        """
        Compare all validated strategies and provide recommendations.
        
        Returns:
            Dictionary with comparison results and recommendations
        """
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_strategies_on_test_data() first.")
        
        print("\n" + "="*80)
        print("📊 STRATEGY COMPARISON AND RECOMMENDATIONS")
        print("="*80)
        
        comparison_results = {}
        strategy_scores = []
        
        # Compare strategies - now with multiple parameter sets per strategy
        print(f"\n{'Strategy':<15} {'Param Set':<10} {'Sharpe':<8} {'Calmar':<8} {'Win Rate':<10} {'Max DD':<8} {'Total PnL':<10} {'Trades':<8}")
        print("-" * 100)
        
        for strategy_key, results in self.validation_results.items():
            if 'error' in results:
                print(f"{strategy_key:<15} {'ERROR':<10} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10} {'ERROR':<8} {'ERROR':<10} {'ERROR':<8}")
                continue
            
            strategy_name = results.get('strategy_name', strategy_key)
            validation_results_list = results.get('validation_results', [])
            
            if not validation_results_list:
                print(f"{strategy_key:<15} {'NO DATA':<10} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<8} {'N/A':<10} {'N/A':<8}")
                continue
            
            # Process each parameter set for this strategy
            for param_result in validation_results_list:
                metrics = param_result['performance_metrics']
                status = param_result['final_status']
                param_set = param_result['parameter_set']
                
                # Extract key metrics
                sharpe = metrics.get('sharpe_ratio', 0)
                calmar = metrics.get('calmar_ratio', 0)
                win_rate = metrics.get('win_rate', 0)
                max_dd = metrics.get('max_drawdown', 0)
                # Use percentage PnL from performance metrics, not dollar amount from status
                total_pnl = metrics.get('total_pnl', 0)
                trades = status.get('total_trades', 0)
                
                print(f"{strategy_key:<15} {f'Set {param_set}':<10} {sharpe:<8.3f} {calmar:<8.3f} {win_rate:<10.2%} {max_dd:<8.2%} {total_pnl:<10.2%} {trades:<8}")
                
                # Calculate composite score for ranking
                # Weighted combination of key metrics
                composite_score = (
                    0.3 * sharpe +           # Risk-adjusted returns
                    0.2 * calmar +             # Risk-adjusted returns (alternative)
                    0.2 * win_rate +           # Win rate
                    0.1 * (1 - max_dd) +       # Drawdown (inverted)
                    0.2 * total_pnl            # Total returns
                )
                
                strategy_scores.append({
                    'strategy_key': strategy_key,
                    'strategy_name': strategy_name,
                    'parameter_set': param_set,
                    'composite_score': composite_score,
                    'sharpe_ratio': sharpe,
                    'calmar_ratio': calmar,
                    'win_rate': win_rate,
                    'max_drawdown': max_dd,
                    'total_pnl': total_pnl,
                    'total_trades': trades,
                    'parameters': param_result.get('parameters', {}),
                    'final_status': status,
                    'optimization_score': param_result.get('optimization_score', 0)
                })
        
        # Sort by composite score
        strategy_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Generate recommendations
        recommendations = {
            'best_strategy': strategy_scores[0] if strategy_scores else None,
            'all_strategies_ranked': strategy_scores,
            'comparison_summary': {
                'total_strategies_tested': len(strategy_scores),
                'best_composite_score': strategy_scores[0]['composite_score'] if strategy_scores else 0,
                'best_sharpe_ratio': max([s['sharpe_ratio'] for s in strategy_scores]) if strategy_scores else 0,
                'best_win_rate': max([s['win_rate'] for s in strategy_scores]) if strategy_scores else 0,
                'lowest_drawdown': min([s['max_drawdown'] for s in strategy_scores]) if strategy_scores else 0
            }
        }
        
        # Display recommendations
        if strategy_scores:
            best = strategy_scores[0]
            print(f"\n🏆 BEST STRATEGY: {best['strategy_name']} (Parameter Set {best['parameter_set']})")
            print(f"📊 Composite Score: {best['composite_score']:.4f}")
            print(f"📈 Sharpe Ratio: {best['sharpe_ratio']:.3f}")
            print(f"🎯 Win Rate: {best['win_rate']:.2%}")
            print(f"📉 Max Drawdown: {best['max_drawdown']:.2%}")
            print(f"💰 Total PnL: {best['total_pnl']:.2%}")
            print(f"📋 Total Trades: {best['total_trades']}")
            print(f"⚙️  Parameters: {best['parameters']}")
            print(f"🔍 Optimization Score: {best['optimization_score']:.4f}")
            
            # Show top 3 overall
            print(f"\n🥇 TOP 3 OVERALL PERFORMERS:")
            for i, result in enumerate(strategy_scores[:3]):
                print(f"{i+1}. {result['strategy_name']} (Set {result['parameter_set']}) - Score: {result['composite_score']:.4f}")
            
            # Show best parameter set for each strategy
            print(f"\n🏆 BEST PARAMETER SET PER STRATEGY:")
            strategy_best = {}
            for result in strategy_scores:
                strategy_key = result['strategy_key']
                if strategy_key not in strategy_best or result['composite_score'] > strategy_best[strategy_key]['composite_score']:
                    strategy_best[strategy_key] = result
            
            for strategy_key, result in strategy_best.items():
                print(f"  {result['strategy_name']}: Set {result['parameter_set']} (Score: {result['composite_score']:.4f})")
        
        self.final_recommendations = recommendations
        return recommendations
    
    def save_results(self, output_dir: str = "validation_results"):
        """
        Save all results to files.
        
        Args:
            output_dir: Directory to save results
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save optimization results
        if self.optimization_results:
            opt_file = os.path.join(output_dir, f"optimization_results_{timestamp}.json")
            with open(opt_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json.dump(self.optimization_results, f, indent=2, default=str)
            print(f"💾 Optimization results saved to {opt_file}")
        
        # Save validation results
        if self.validation_results:
            val_file = os.path.join(output_dir, f"validation_results_{timestamp}.json")
            with open(val_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            print(f"💾 Validation results saved to {val_file}")
        
        # Save recommendations
        if self.final_recommendations:
            rec_file = os.path.join(output_dir, f"recommendations_{timestamp}.json")
            with open(rec_file, 'w') as f:
                json.dump(self.final_recommendations, f, indent=2, default=str)
            print(f"💾 Recommendations saved to {rec_file}")
        
        # Save data splits
        if not self.train_data.empty:
            train_file = os.path.join(output_dir, f"train_data_{timestamp}.csv")
            self.train_data.to_csv(train_file)
            print(f"💾 Training data saved to {train_file}")
        
        if not self.test_data.empty:
            test_file = os.path.join(output_dir, f"test_data_{timestamp}.csv")
            self.test_data.to_csv(test_file)
            print(f"💾 Test data saved to {test_file}")
    
    def get_live_trading_setup(self, strategy_key: str = None) -> Dict:
        """
        Get the best strategy setup for live trading.
        
        Args:
            strategy_key: Specific strategy to use (if None, uses best strategy)
            
        Returns:
            Dictionary with live trading setup
        """
        if not self.final_recommendations:
            raise ValueError("No recommendations available. Run compare_strategies() first.")
        
        if strategy_key is None:
            best_strategy = self.final_recommendations['best_strategy']
            if best_strategy is None:
                raise ValueError("No best strategy found.")
            strategy_key = best_strategy['strategy_key']
        
        if strategy_key not in self.validation_results:
            raise ValueError(f"Strategy {strategy_key} not found in validation results.")
        
        validation_result = self.validation_results[strategy_key]
        
        # If we have multiple parameter sets, find the best one
        if 'validation_results' in validation_result and validation_result['validation_results']:
            # Find the best parameter set for this strategy
            best_param_result = max(validation_result['validation_results'], 
                                 key=lambda x: x['performance_metrics'].get('sharpe_ratio', 0))
            
            live_setup = {
                'strategy_key': strategy_key,
                'strategy_name': validation_result['strategy_name'],
                'parameters': best_param_result['parameters'],
                'performance_metrics': best_param_result['performance_metrics'],
                'parameter_set': best_param_result['parameter_set'],
                'symbol': self.symbol,
                'interval': self.interval,
                'initial_balance': self.initial_balance,
                'max_leverage': self.max_leverage,
                'max_loss_percent': self.max_loss_percent,
                'trading_fee': self.trading_fee,
                'session_id': best_param_result['session_id']
            }
        else:
            # Fallback to old structure
            live_setup = {
                'strategy_key': strategy_key,
                'strategy_name': validation_result['strategy_name'],
                'parameters': validation_result.get('parameters', {}),
                'performance_metrics': validation_result.get('performance_metrics', {}),
                'symbol': self.symbol,
                'interval': self.interval,
                'initial_balance': self.initial_balance,
                'max_leverage': self.max_leverage,
                'max_loss_percent': self.max_loss_percent,
                'trading_fee': self.trading_fee,
                'session_id': validation_result.get('session_id', 'unknown')
            }
        
        print(f"\n🚀 LIVE TRADING SETUP FOR {strategy_key.upper()}")
        print("="*50)
        print(f"📊 Strategy: {live_setup['strategy_name']}")
        print(f"⚙️  Parameters: {live_setup['parameters']}")
        if 'parameter_set' in live_setup:
            print(f"🔢 Parameter Set: {live_setup['parameter_set']}")
        print(f"📈 Expected Performance:")
        print(f"  - Sharpe Ratio: {live_setup['performance_metrics'].get('sharpe_ratio', 'N/A'):.3f}")
        print(f"  - Win Rate: {live_setup['performance_metrics'].get('win_rate', 'N/A'):.2%}")
        print(f"  - Max Drawdown: {live_setup['performance_metrics'].get('max_drawdown', 'N/A'):.2%}")
        print(f"  - Total PnL: {live_setup['performance_metrics'].get('total_pnl', 'N/A'):.2%}")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"📊 Symbol: {self.symbol}")
        print(f"⏱️  Interval: {self.interval}")
        
        return live_setup


def main():
    """
    Main function demonstrating the comprehensive validation pipeline.
    """
    print("🚀 COMPREHENSIVE STRATEGY VALIDATION PIPELINE")
    print("="*80)
    
    # Initialize validator
    validator = ComprehensiveStrategyValidator(
        initial_balance=10000,
        max_leverage=10.0,
        max_loss_percent=2.0,
        trading_fee=0.0
    )
    
    # Configuration
    symbol = "ETHUSDT"
    end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    interval = "15m"
    train_ratio = 0.7
    
    try:
        # Step 1: Fetch and split data
        print("\n📥 STEP 1: FETCHING AND SPLITTING DATA")
        print("="*50)
        train_data, test_data = validator.fetch_and_split_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            train_ratio=train_ratio
        )
        
        # Step 2: Optimize strategies on training data
        print("\n🔍 STEP 2: OPTIMIZING STRATEGIES ON TRAINING DATA")
        print("="*50)
        # optimization_results = validator.optimize_strategies_on_train_data(
        #     strategies_to_optimize=['ma', 'rsi', 'donchian']
        # )
        optimization_results = validator.optimize_strategies_on_train_data(
            strategies_to_optimize=['ma']
        )
        
        # Step 3: Validate strategies on test data
        print("\n🧪 STEP 3: VALIDATING STRATEGIES ON TEST DATA")
        print("="*50)
        # validation_results = validator.validate_strategies_on_test_data(
        #     strategies_to_validate=['ma', 'rsi', 'donchian'],
        #     mock_trading_delay=0
        # )
        validation_results = validator.validate_strategies_on_test_data(
            strategies_to_validate=['ma'],
            mock_trading_delay=0
        )
        
        # Step 4: Compare strategies and get recommendations
        print("\n📊 STEP 4: COMPARING STRATEGIES")
        print("="*50)
        recommendations = validator.compare_strategies()
        
        # Step 5: Get live trading setup
        print("\n🚀 STEP 5: LIVE TRADING SETUP")
        print("="*50)
        live_setup = validator.get_live_trading_setup()
        
        # Step 6: Save results
        print("\n💾 STEP 6: SAVING RESULTS")
        print("="*50)
        validator.save_results()
        
        print("\n✅ COMPREHENSIVE VALIDATION COMPLETED!")
        print("="*80)
        print("🎯 Next steps:")
        print("1. Review the saved results in 'validation_results' directory")
        print("2. Use the live trading setup for actual trading")
        print("3. Monitor performance and adjust parameters as needed")
        
    except Exception as e:
        logger.error(f"Error in comprehensive validation: {e}")
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your Binance API credentials")
        print("2. Ensure you have sufficient data for the date range")
        print("3. Verify the symbol is correct and trading is active")


if __name__ == "__main__":
    main()
