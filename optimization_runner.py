#!/usr/bin/env python3
"""
Optimization Runner - Simplified interface for running strategy optimizations
Uses the same approach as simple_optimization_example.py
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import argparse
import json

from data_fetcher import DataFetcher
from strategy_optimizer import (
    optimize_moving_average_crossover,
    optimize_rsi_strategy,
    optimize_donchian_channel
)

def run_optimization(symbol: str = "BTC-USD",
                   start_date: str = None,
                   end_date: str = None,
                   interval: str = "15m",
                   enabled_strategies: List[str] = None,
                   trading_fee: float = 0.001,
                   sharpe_threshold: float = 0.1) -> Dict:
    """
    Run optimization for specified strategies using the same approach as simple_optimization_example.py
    
    Args:
        symbol: Trading symbol (e.g., "BTC-USD")
        start_date: Start date for optimization data (default: 30 days ago)
        end_date: End date for optimization data (default: today)
        interval: Data interval (e.g., "15m", "1h", "1d")
        enabled_strategies: List of strategies to optimize
        trading_fee: Trading fee as decimal (e.g., 0.001 for 0.1%)
        sharpe_threshold: Minimum Sharpe ratio threshold
        
    Returns:
        Dictionary containing optimization results for each strategy
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    if enabled_strategies is None:
        enabled_strategies = ['ma', 'rsi', 'donchian']
    
    print("ROBUST STRATEGY OPTIMIZATION WITH FETCH_DATA")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    print(f"Strategies: {enabled_strategies}")
    print(f"Trading Fee: {trading_fee}")
    print(f"Sharpe Threshold: {sharpe_threshold}")
    print("="*60)
    
    # Fetch data
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    fetcher = DataFetcher()
    data = fetcher.fetch_data(symbol, start_date, end_date, interval=interval)
    
    if data.empty:
        print(f"No data fetched for {symbol}")
        return {'error': f"No data fetched for {symbol}"}
    
    print(f"Successfully fetched {len(data)} data points")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Columns: {list(data.columns)}")
    
    # Save data to CSV (optional)
    filename = f"{symbol}_{start_date}_{end_date}_{interval}.csv"
    data.to_csv(filename)
    print(f"Data saved to {filename}")
    
    results = {}
    
    # Run optimizations for each enabled strategy
    if 'ma' in enabled_strategies:
        print("\n" + "="*60)
        print("MOVING AVERAGE CROSSOVER OPTIMIZATION")
        print("="*60)
        
        best_params_ma, best_metrics_ma = optimize_moving_average_crossover(
            data=data,
            short_window_range=[5, 10, 15, 20, 25, 30],
            long_window_range=[20, 30, 40, 50, 60, 70],
            risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            trading_fee=trading_fee,
            sharpe_threshold=sharpe_threshold
        )
        
        results['ma'] = {
            'parameters': best_params_ma,
            'metrics': best_metrics_ma,
            'optimization_date': datetime.now().isoformat(),
            'data_period': f"{start_date} to {end_date}",
            'symbol': symbol,
            'interval': interval
        }
        
        print(f"\nBEST MA PARAMETERS: {best_params_ma}")
        print(f"BEST MA METRICS:")
        print(f"  Total Trades: {best_metrics_ma['total_trades']}")
        print(f"  Win Rate: {best_metrics_ma['win_rate']:.2%}")
        print(f"  Sharpe Ratio: {best_metrics_ma['sharpe_ratio']:.3f}")
        print(f"  Calmar Ratio: {best_metrics_ma['calmar_ratio']:.3f}")
        print(f"  Max Drawdown: {best_metrics_ma['max_drawdown']:.2%}")
        print(f"  Total PnL: {best_metrics_ma['total_pnl']:.2%}")
    
    if 'rsi' in enabled_strategies:
        print("\n" + "="*60)
        print("RSI STRATEGY OPTIMIZATION")
        print("="*60)
        
        best_params_rsi, best_metrics_rsi = optimize_rsi_strategy(
            data=data,
            period_range=[10, 14, 20, 30],
            overbought_range=[65, 70, 75, 80],
            oversold_range=[20, 25, 30, 35],
            risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            trading_fee=trading_fee,
            sharpe_threshold=sharpe_threshold
        )
        
        results['rsi'] = {
            'parameters': best_params_rsi,
            'metrics': best_metrics_rsi,
            'optimization_date': datetime.now().isoformat(),
            'data_period': f"{start_date} to {end_date}",
            'symbol': symbol,
            'interval': interval
        }
        
        print(f"\nBEST RSI PARAMETERS: {best_params_rsi}")
        print(f"BEST RSI METRICS:")
        print(f"  Total Trades: {best_metrics_rsi['total_trades']}")
        print(f"  Win Rate: {best_metrics_rsi['win_rate']:.2%}")
        print(f"  Sharpe Ratio: {best_metrics_rsi['sharpe_ratio']:.3f}")
        print(f"  Calmar Ratio: {best_metrics_rsi['calmar_ratio']:.3f}")
        print(f"  Max Drawdown: {best_metrics_rsi['max_drawdown']:.2%}")
        print(f"  Total PnL: {best_metrics_rsi['total_pnl']:.2%}")
    
    if 'donchian' in enabled_strategies:
        print("\n" + "="*60)
        print("DONCHIAN CHANNEL OPTIMIZATION")
        print("="*60)
        
        best_params_donchian, best_metrics_donchian = optimize_donchian_channel(
            data=data,
            channel_period_range=[10, 15, 20, 25, 30],
            risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            trading_fee=trading_fee,
            sharpe_threshold=sharpe_threshold
        )
        
        results['donchian'] = {
            'parameters': best_params_donchian,
            'metrics': best_metrics_donchian,
            'optimization_date': datetime.now().isoformat(),
            'data_period': f"{start_date} to {end_date}",
            'symbol': symbol,
            'interval': interval
        }
        
        print(f"\nBEST DONCHIAN PARAMETERS: {best_params_donchian}")
        print(f"BEST DONCHIAN METRICS:")
        print(f"  Total Trades: {best_metrics_donchian['total_trades']}")
        print(f"  Win Rate: {best_metrics_donchian['win_rate']:.2%}")
        print(f"  Sharpe Ratio: {best_metrics_donchian['sharpe_ratio']:.3f}")
        print(f"  Calmar Ratio: {best_metrics_donchian['calmar_ratio']:.3f}")
        print(f"  Max Drawdown: {best_metrics_donchian['max_drawdown']:.2%}")
        print(f"  Total PnL: {best_metrics_donchian['total_pnl']:.2%}")
    
    # Compare strategies
    if len(results) > 1:
        print("\n" + "="*60)
        print("STRATEGY COMPARISON")
        print("="*60)
        
        strategies = []
        if 'ma' in results:
            strategies.append(("Moving Average", results['ma']['metrics']))
        if 'rsi' in results:
            strategies.append(("RSI", results['rsi']['metrics']))
        if 'donchian' in results:
            strategies.append(("Donchian Channel", results['donchian']['metrics']))
        
        print(f"{'Strategy':<20} {'Sharpe':<8} {'Calmar':<8} {'Win Rate':<10} {'Max DD':<8} {'Total PnL':<10}")
        print("-" * 70)
        
        for name, metrics in strategies:
            print(f"{name:<20} {metrics['sharpe_ratio']:<8.3f} {metrics['calmar_ratio']:<8.3f} "
                  f"{metrics['win_rate']:<10.2%} {metrics['max_drawdown']:<8.2%} {metrics['total_pnl']:<10.2%}")
    
    print("\n" + "="*60)
    print("KEY ADVANTAGES OF ROBUST OPTIMIZATION")
    print("="*60)
    print("1. Uses composite scoring instead of just total_pnl")
    print("2. Applies minimum thresholds (trades, Sharpe, drawdown)")
    print("3. Provides parameter sensitivity analysis")
    print("4. Shows top multiple results, not just the best")
    print("5. Helps avoid overfitting to outlier trades")
    print("6. Balances return, risk, and consistency")
    
    return results

def get_optimization_parameters(results: Dict, strategy_name: str) -> Dict:
    """
    Extract optimized parameters for a specific strategy.
    
    Args:
        results: Optimization results from run_optimization
        strategy_name: Name of the strategy ('ma', 'rsi', 'donchian')
        
    Returns:
        Dictionary of optimized parameters
    """
    if not results or 'error' in results:
        return {}
    
    if strategy_name not in results:
        print(f"⚠️  Strategy '{strategy_name}' not found in results")
        return {}
    
    result = results[strategy_name]
    if 'error' in result:
        print(f"❌ Error in {strategy_name} optimization: {result['error']}")
        return {}
    
    return result['parameters']

def compare_strategies(results: Dict) -> pd.DataFrame:
    """
    Compare optimization results across strategies.
    
    Args:
        results: Optimization results from run_optimization
        
    Returns:
        DataFrame with strategy comparison
    """
    if not results or 'error' in results:
        return pd.DataFrame()
    
    comparison_data = []
    
    for strategy_name, result in results.items():
        if 'error' in result:
            continue
            
        metrics = result['metrics']
        params = result['parameters']
        
        comparison_data.append({
            'Strategy': strategy_name.upper(),
            'Sharpe_Ratio': metrics['sharpe_ratio'],
            'Total_PnL': metrics['total_pnl'],
            'Win_Rate': metrics['win_rate'],
            'Max_Drawdown': metrics['max_drawdown'],
            'Total_Trades': metrics['total_trades'],
            'Calmar_Ratio': metrics['calmar_ratio'],
            'Profit_Factor': metrics['profit_factor'],
            'Best_Parameters': str(params)
        })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Sort by Sharpe ratio (descending)
        df = df.sort_values('Sharpe_Ratio', ascending=False)
        
        print("\n" + "=" * 80)
        print("📊 STRATEGY COMPARISON")
        print("=" * 80)
        print(df.to_string(index=False))
        
        return df
    else:
        print("❌ No valid results to compare")
        return pd.DataFrame()

def main():
    """
    Command-line interface for optimization runner.
    """
    parser = argparse.ArgumentParser(description='Strategy Optimization Runner')
    parser.add_argument('--symbol', default='BTC-USD', help='Trading symbol')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', default='15m', help='Data interval')
    parser.add_argument('--strategies', nargs='+', default=['ma', 'rsi'], 
                       help='Strategies to optimize')
    parser.add_argument('--trading-fee', type=float, default=0.001,
                       help='Trading fee as decimal (e.g., 0.001 for 0.1%)')
    parser.add_argument('--sharpe-threshold', type=float, default=0.1,
                       help='Minimum Sharpe ratio threshold')
    parser.add_argument('--export', action='store_true',
                       help='Export results to JSON file')
    
    args = parser.parse_args()
    
    # Run optimization
    results = run_optimization(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        enabled_strategies=args.strategies,
        trading_fee=args.trading_fee,
        sharpe_threshold=args.sharpe_threshold
    )
    
    # Export results if requested
    if args.export and 'error' not in results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{args.symbol}_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        export_data = {}
        for strategy_name, result in results.items():
            export_data[strategy_name] = result.copy()
            for key, value in export_data[strategy_name].items():
                if hasattr(value, 'item'):  # numpy scalar
                    export_data[strategy_name][key] = value.item()
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n📁 Results exported to: {filename}")
    
    return results

if __name__ == "__main__":
    main()
