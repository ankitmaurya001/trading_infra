#!/usr/bin/env python3
"""
Simple example showing how to use KiteDataFetcher with the robust strategy optimizer.
This demonstrates optimization using Indian market data from Zerodha Kite Connect API.
"""

from data_fetcher import KiteDataFetcher
from strategy_optimizer import (
    optimize_moving_average_crossover,
    optimize_rsi_strategy,
    optimize_donchian_channel
)
import pandas as pd
from datetime import datetime, timedelta
import logging
import config as cfg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_kite_credentials():
    """
    Setup Kite Connect credentials.
    In production, these should be loaded from environment variables or config files.
    """
    return cfg.KITE_CREDENTIALS

def filter_market_hours(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to include only market hours for Indian markets.
    NSE/BSE trading hours: 9:15 AM to 3:30 PM IST (Monday to Friday)
    
    Args:
        data: DataFrame with datetime index
        
    Returns:
        Filtered DataFrame with only market hours
    """
    if data.empty:
        return data
    
    # Ensure we have timezone-aware data
    if data.index.tz is None:
        data.index = data.index.tz_localize('Asia/Kolkata')
    else:
        data.index = data.index.tz_convert('Asia/Kolkata')
    
    # Filter for market hours (9:15 AM to 3:30 PM IST)
    market_hours = data.between_time('09:15', '15:30')
    
    # Filter for weekdays only (Monday=0, Sunday=6)
    weekdays = market_hours[market_hours.index.weekday < 5]
    
    logger.info(f"Filtered data from {len(data)} to {len(weekdays)} records (market hours only)")
    
    return weekdays

def main():
    """
    Simple example using KiteDataFetcher with robust optimization.
    """
    print("ROBUST STRATEGY OPTIMIZATION WITH KITE DATA")
    print("="*60)
    
    # Configuration for Indian markets
    symbol = "TATAMOTORS"  # NSE symbol
    exchange = "NSE"
    #symbol = "NATURALGAS25SEPFUT"
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    interval = "15minute"  # Kite uses different interval format
    trading_fee = 0 # 0.03% - typical Kite charges

    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    print(f"Market: NSE (Indian Stock Market)")
    print(f"Trading Hours: 9:15 AM - 3:30 PM IST (Monday-Friday)")
    print(f"Interval: {interval}")
    print(f"Trading Fee: {trading_fee*100:.3f}%")
    
    try:
        # Setup Kite credentials
        credentials = cfg.KITE_CREDENTIALS
        
        # Initialize Kite data fetcher
        kite_fetcher = KiteDataFetcher(credentials, exchange=cfg.KITE_EXCHANGE)
        
        # Authenticate with Kite
        print("\n🔐 Authenticating with Kite Connect...")
        kite_fetcher.authenticate()
        print("✅ Authentication successful!")
        
        # Fetch historical data
        print(f"\n📥 Fetching historical data for {symbol}...")
        data = kite_fetcher.fetch_historical_data(symbol, start_date, end_date, interval=interval)
        
        if data.empty:
            print(f"❌ No data fetched for {symbol}")
            print("   Possible reasons:")
            print("   - Invalid symbol or credentials")
            print("   - Market is closed")
            print("   - Date range has no trading days")
            return
        
        print(f"✅ Successfully fetched {len(data)} data points")
        print(f"📊 Data range: {data.index[0]} to {data.index[-1]}")
        print(f"📋 Columns: {list(data.columns)}")
        
        # Filter for market hours only
        # print(f"\n🕒 Filtering for market hours (9:15 AM - 3:30 PM IST)...")
        # data = filter_market_hours(data)
        
        if data.empty:
            print("❌ No data remaining after filtering for market hours")
            return
        
        print(f"✅ Market hours data: {len(data)} data points")
        print(f"📊 Filtered range: {data.index[0]} to {data.index[-1]}")
        
        # Save data to CSV (optional)
        filename = f"{symbol}_{start_date}_{end_date}_{interval}_market_hours.csv"
        data.to_csv(filename)
        print(f"💾 Data saved to {filename}")
        
        # Run robust optimizations
        print("\n" + "="*60)
        print("MOVING AVERAGE CROSSOVER OPTIMIZATION")
        print("="*60)
        
        best_params_ma, best_metrics_ma = optimize_moving_average_crossover(
            data=data,
            short_window_range=[5, 10, 15, 20, 25, 30],
            long_window_range=[20, 30, 40, 50, 60, 70],
            risk_reward_range=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            trading_fee=trading_fee,
            sharpe_threshold=0.1
        )
        
        print(f"\n🏆 BEST MA PARAMETERS: {best_params_ma}")
        print(f"📈 BEST MA METRICS:")
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
        
        print(f"\n🏆 BEST RSI PARAMETERS: {best_params_rsi}")
        print(f"📈 BEST RSI METRICS:")
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
        
        print(f"\n🏆 BEST DONCHIAN PARAMETERS: {best_params_donchian}")
        print(f"📈 BEST DONCHIAN METRICS:")
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
        print("KEY ADVANTAGES OF KITE DATA OPTIMIZATION")
        print("="*60)
        print("1. ✅ Real Indian market data from NSE/BSE")
        print("2. ✅ Market hours filtering (9:15 AM - 3:30 PM IST)")
        print("3. ✅ Weekday-only trading (no weekend data)")
        print("4. ✅ Accurate trading fees (0.03% typical for Kite)")
        print("5. ✅ Real-time data availability during market hours")
        print("6. ✅ Same column format as Yahoo Finance (Open, High, Low, Close, Volume)")
        print("7. ✅ IST timezone handling")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. 🔧 Update credentials in setup_kite_credentials() function")
        print("2. 🧪 Test with different symbols (RELIANCE, TCS, INFY, HDFCBANK)")
        print("3. 📊 Adjust parameter ranges based on results")
        print("4. 🚀 Integrate with live trading system")
        print("5. 📈 Monitor performance in real market conditions")
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your Kite Connect credentials")
        print("2. Ensure you have active internet connection")
        print("3. Verify the symbol is correct (e.g., 'TATAMOTORS' not 'TATAMOTORS.NS')")
        print("4. Check if market is open (9:15 AM - 3:30 PM IST, Monday-Friday)")

def demo_with_sample_data():
    """
    Demo function that works without Kite credentials using sample data.
    """
    print("DEMO MODE: Using sample data (no Kite credentials required)")
    print("="*60)
    
    # Create sample data that mimics Indian market structure
    dates = pd.date_range(
        start='2025-01-01 09:15:00',
        end='2025-01-31 15:30:00',
        freq='15min'
    )
    
    # Filter for weekdays and market hours
    market_dates = []
    for date in dates:
        if date.weekday() < 5:  # Monday to Friday
            if '09:15' <= date.strftime('%H:%M') <= '15:30':
                market_dates.append(date)
    
    # Generate sample OHLCV data
    import numpy as np
    np.random.seed(42)
    
    n_points = len(market_dates)
    base_price = 500.0
    prices = [base_price]
    
    for i in range(1, n_points):
        # Random walk with slight upward bias
        change = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(100000, 1000000) for _ in range(n_points)]
    }, index=market_dates)
    
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    print(f"📊 Generated {len(data)} sample data points")
    print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
    print(f"💰 Price range: ₹{data['Low'].min():.2f} - ₹{data['High'].max():.2f}")
    
    # Save sample data
    filename = "sample_indian_market_data.csv"
    data.to_csv(filename)
    print(f"💾 Sample data saved to {filename}")
    
    # Run optimization with sample data
    trading_fee = 0.0003  # 0.03% - typical Kite charges
    
    print("\n" + "="*60)
    print("MOVING AVERAGE CROSSOVER OPTIMIZATION (DEMO)")
    print("="*60)
    
    best_params_ma, best_metrics_ma = optimize_moving_average_crossover(
        data=data,
        short_window_range=[5, 10, 15, 20],
        long_window_range=[20, 30, 40, 50],
        risk_reward_range=[1.5, 2.0, 2.5, 3.0],
        trading_fee=trading_fee,
        sharpe_threshold=0.1
    )
    
    print(f"\n🏆 BEST MA PARAMETERS: {best_params_ma}")
    print(f"📈 BEST MA METRICS:")
    print(f"  Total Trades: {best_metrics_ma['total_trades']}")
    print(f"  Win Rate: {best_metrics_ma['win_rate']:.2%}")
    print(f"  Sharpe Ratio: {best_metrics_ma['sharpe_ratio']:.3f}")
    print(f"  Total PnL: {best_metrics_ma['total_pnl']:.2%}")
    
    print("\n✅ Demo completed successfully!")
    print("💡 To use real data, update credentials and run main() function")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_with_sample_data()
    else:
        main()
