#!/usr/bin/env python3
"""
Simple example showing how to use fetch_data with the robust strategy optimizer.
This demonstrates the same approach as in parallel_run_optimizer.py but with the new robust optimizer.
"""

from data_fetcher import DataFetcher
from strategy_optimizer import (
    optimize_moving_average_crossover,
    optimize_rsi_strategy,
    optimize_donchian_channel
)
import pandas as pd
from datetime import datetime, timedelta

def main():
    """
    Simple example using fetch_data with robust optimization.
    """
    print("ROBUST STRATEGY OPTIMIZATION WITH FETCH_DATA")
    print("="*60)
    
    # Configuration - similar to your parallel_run_optimizer.py
    #symbol = "TATAMOTORS.NS"
    symbol = "BTC-USD"
    #start_date = "2025-07-30"
    start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date=(datetime.now() +timedelta(days=1)).strftime("%Y-%m-%d")
    #end_date = "2025-07-20"
    #end_date = "2025-08-24"
    interval = "15m"
    trading_fee = 0  # 0.1% trading fee

    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    
    # Use your existing DataFetcher
    fetcher = DataFetcher()
    data = fetcher.fetch_data(symbol, start_date, end_date, interval=interval)
    
    if data.empty:
        print(f"No data fetched for {symbol}")
        return
    
    print(f"Successfully fetched {len(data)} data points")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Columns: {list(data.columns)}")
    
    # Save data to CSV (optional)
    filename = f"{symbol}_{start_date}_{end_date}_{interval}.csv"
    data.to_csv(filename)
    print(f"Data saved to {filename}")
    
    # Run robust optimizations
    print("\n" + "="*60)
    print("MOVING AVERAGE CROSSOVER OPTIMIZATION")
    print("="*60)
    
    best_params_ma, best_metrics_ma = optimize_moving_average_crossover(
        data=data,
        short_window_range=[5, 10, 15, 20, 25, 30],
        long_window_range=[20,30, 40, 50, 60, 70],
        risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        trading_fee=trading_fee,
        sharpe_threshold = 0.1
    )
    
    print(f"\nBEST MA PARAMETERS: {best_params_ma}")
    print(f"BEST MA METRICS:")
    print(f"  Total Trades: {best_metrics_ma['total_trades']}")
    print(f"  Win Rate: {best_metrics_ma['win_rate']:.2%}")
    print(f"  Sharpe Ratio: {best_metrics_ma['sharpe_ratio']:.3f}")
    print(f"  Calmar Ratio: {best_metrics_ma['calmar_ratio']:.3f}")
    print(f"  Max Drawdown: {best_metrics_ma['max_drawdown']:.2%}")
    print(f"  Total PnL: {best_metrics_ma['total_pnl']:.2%}")
    
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
        sharpe_threshold=0.1
    )
    
    print(f"\nBEST RSI PARAMETERS: {best_params_rsi}")
    print(f"BEST RSI METRICS:")
    print(f"  Total Trades: {best_metrics_rsi['total_trades']}")
    print(f"  Win Rate: {best_metrics_rsi['win_rate']:.2%}")
    print(f"  Sharpe Ratio: {best_metrics_rsi['sharpe_ratio']:.3f}")
    print(f"  Calmar Ratio: {best_metrics_rsi['calmar_ratio']:.3f}")
    print(f"  Max Drawdown: {best_metrics_rsi['max_drawdown']:.2%}")
    print(f"  Total PnL: {best_metrics_rsi['total_pnl']:.2%}")
    
    print("\n" + "="*60)
    print("DONCHIAN CHANNEL OPTIMIZATION")
    print("="*60)
    
    best_params_donchian, best_metrics_donchian = optimize_donchian_channel(
        data=data,
        channel_period_range=[10, 15, 20, 25, 30],
        risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        trading_fee=trading_fee,
        sharpe_threshold=0.1
    )
    
    print(f"\nBEST DONCHIAN PARAMETERS: {best_params_donchian}")
    print(f"BEST DONCHIAN METRICS:")
    print(f"  Total Trades: {best_metrics_donchian['total_trades']}")
    print(f"  Win Rate: {best_metrics_donchian['win_rate']:.2%}")
    print(f"  Sharpe Ratio: {best_metrics_donchian['sharpe_ratio']:.3f}")
    print(f"  Calmar Ratio: {best_metrics_donchian['calmar_ratio']:.3f}")
    print(f"  Max Drawdown: {best_metrics_donchian['max_drawdown']:.2%}")
    print(f"  Total PnL: {best_metrics_donchian['total_pnl']:.2%}")
    
    # Compare strategies
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    strategies = [
        ("Moving Average", best_metrics_ma),
        ("RSI", best_metrics_rsi),
        ("Donchian Channel", best_metrics_donchian)
    ]
    
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

if __name__ == "__main__":
    main() 