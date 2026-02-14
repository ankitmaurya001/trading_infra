#!/usr/bin/env python3
"""
Get Optimization Results for Live Trading Simulator
Run this script to get optimized parameters and copy them to the simulator
"""

from data_fetcher import DataFetcher
from strategy_optimizer import (
    optimize_moving_average_crossover,
    optimize_rsi_strategy,
    optimize_donchian_channel
)

def main():
    """Run optimization and display results for manual input"""
    print("🔧 Getting Optimization Results for Live Trading Simulator")
    print("=" * 80)
    
    # Configuration - modify these as needed
    symbol = "BTC-USD"
    start_date = "2025-07-30"
    end_date = "2025-08-24"
    interval = "15m"
    trading_fee = 0.001  # 0.1%
    
    print(f"📊 Symbol: {symbol}")
    print(f"📅 Date Range: {start_date} to {end_date}")
    print(f"⏱️  Interval: {interval}")
    print(f"💰 Trading Fee: {trading_fee:.3f}")
    print("-" * 80)
    
    # Fetch data
    print("📥 Fetching historical data...")
    fetcher = DataFetcher()
    data = fetcher.fetch_data(symbol, start_date, end_date, interval=interval)
    
    if data.empty:
        print(f"❌ No data fetched for {symbol}")
        return
    
    print(f"✅ Successfully fetched {len(data)} data points")
    print(f"📈 Data range: {data.index[0]} to {data.index[-1]}")
    print("-" * 80)
    
    # Run optimizations
    print("\n🎯 Running Moving Average Crossover Optimization...")
    best_params_ma, best_metrics_ma = optimize_moving_average_crossover(
        data=data,
        short_window_range=[5, 10, 15, 20, 25, 30],
        long_window_range=[20, 30, 40, 50, 60, 70],
        risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        trading_fee=trading_fee,
        sharpe_threshold=0.1
    )
    
    print("\n🎯 Running RSI Strategy Optimization...")
    best_params_rsi, best_metrics_rsi = optimize_rsi_strategy(
        data=data,
        period_range=[10, 14, 20, 30],
        overbought_range=[65, 70, 75, 80],
        oversold_range=[20, 25, 30, 35],
        risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        trading_fee=trading_fee,
        sharpe_threshold=0.1
    )
    
    print("\n🎯 Running Donchian Channel Optimization...")
    best_params_donchian, best_metrics_donchian = optimize_donchian_channel(
        data=data,
        channel_period_range=[10, 15, 20, 25, 30],
        risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        trading_fee=trading_fee,
        sharpe_threshold=0.1
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("🎯 OPTIMIZATION RESULTS - COPY TO LIVE TRADING SIMULATOR")
    print("=" * 80)
    
    print(f"\n📊 Moving Average Crossover Strategy:")
    print(f"   Short MA Period: {best_params_ma['short_window']}")
    print(f"   Long MA Period: {best_params_ma['long_window']}")
    print(f"   Risk/Reward Ratio: {best_params_ma['risk_reward_ratio']}")
    print(f"   Trading Fee (%): {best_params_ma['trading_fee'] * 100:.2f}")
    print(f"   Performance Metrics:")
    print(f"     - Sharpe Ratio: {best_metrics_ma['sharpe_ratio']:.3f}")
    print(f"     - Total PnL: {best_metrics_ma['total_pnl']:.2%}")
    print(f"     - Win Rate: {best_metrics_ma['win_rate']:.2%}")
    print(f"     - Max Drawdown: {best_metrics_ma['max_drawdown']:.2%}")
    
    print(f"\n📊 RSI Strategy:")
    print(f"   RSI Period: {best_params_rsi['period']}")
    print(f"   Overbought Level: {best_params_rsi['overbought']}")
    print(f"   Oversold Level: {best_params_rsi['oversold']}")
    print(f"   Risk/Reward Ratio: {best_params_rsi['risk_reward_ratio']}")
    print(f"   Trading Fee (%): {best_params_rsi['trading_fee'] * 100:.2f}")
    print(f"   Performance Metrics:")
    print(f"     - Sharpe Ratio: {best_metrics_rsi['sharpe_ratio']:.3f}")
    print(f"     - Total PnL: {best_metrics_rsi['total_pnl']:.2%}")
    print(f"     - Win Rate: {best_metrics_rsi['win_rate']:.2%}")
    print(f"     - Max Drawdown: {best_metrics_rsi['max_drawdown']:.2%}")
    
    print(f"\n📊 Donchian Channel Strategy:")
    print(f"   Channel Period: {best_params_donchian['channel_period']}")
    print(f"   Risk/Reward Ratio: {best_params_donchian['risk_reward_ratio']}")
    print(f"   Trading Fee (%): {best_params_donchian['trading_fee'] * 100:.2f}")
    print(f"   Performance Metrics:")
    print(f"     - Sharpe Ratio: {best_metrics_donchian['sharpe_ratio']:.3f}")
    print(f"     - Total PnL: {best_metrics_donchian['total_pnl']:.2%}")
    print(f"     - Win Rate: {best_metrics_donchian['win_rate']:.2%}")
    print(f"     - Max Drawdown: {best_metrics_donchian['max_drawdown']:.2%}")
    
    print("\n" + "=" * 80)
    print("📋 INSTRUCTIONS FOR LIVE TRADING SIMULATOR")
    print("=" * 80)
    print("1. Launch the Live Trading Simulator:")
    print("   streamlit run live_trading_simulator.py")
    print("\n2. In the sidebar, select the strategies you want to use")
    print("\n3. Copy the parameter values above to the corresponding fields:")
    print("   - Short MA Period → Short MA Period")
    print("   - Long MA Period → Long MA Period")
    print("   - RSI Period → RSI Period")
    print("   - Overbought Level → Overbought Level")
    print("   - Oversold Level → Oversold Level")
    print("   - Channel Period → Channel Period")
    print("   - Risk/Reward Ratio → Risk/Reward Ratio")
    print("   - Trading Fee (%) → Trading Fee (%)")
    print("\n4. Click 'Set Parameters' to initialize the strategies")
    print("\n5. Click 'Start Live Trading' to begin the simulation")
    print("\n" + "=" * 80)
    print("🚀 Happy Trading!")
    print("=" * 80)

if __name__ == "__main__":
    main()
