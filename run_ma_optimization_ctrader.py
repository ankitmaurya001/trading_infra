#!/usr/bin/env python3
"""
Simple script to run MA optimization with real data from cTrader using the existing visualizer.
Designed for Forex trading (EURUSD, GBPUSD, etc.)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ma_3d_optimization_visualizer import MAOptimization3DVisualizer

# ============================================================================
# GLOBAL CONFIGURATION - Edit these values to set defaults
# ============================================================================
# SYMBOL = "EURUSD"  # Default forex symbol
SYMBOL = "XAUUSD"
DAYS_TO_FETCH = 30  # Default number of days to fetch
INTERVAL = "15m"  # Default interval

# Import cTrader credentials
try:
    from config import (
        CTRADER_CLIENT_ID,
        CTRADER_CLIENT_SECRET,
        CTRADER_ACCESS_TOKEN,
        CTRADER_ACCOUNT_ID,
        CTRADER_DEMO
    )
    CTRADER_AVAILABLE = True
except ImportError:
    CTRADER_AVAILABLE = False
    print("⚠️  cTrader credentials not found in config.py")


def fetch_real_data(symbol="EURUSD", days=30, interval="15m"):
    """Fetch real data from cTrader"""
    print(f"📊 Fetching real data for {symbol} from cTrader...")
    
    if not CTRADER_AVAILABLE:
        print("❌ cTrader credentials not available")
        return generate_sample_data()
    
    try:
        from data_fetcher import CTraderDataFetcher
        
        # Calculate date range
        # end_date = datetime.now()
        # start_date = end_date - timedelta(days=days)
        
        end_date = datetime.now() -timedelta(days=30)
        start_date = end_date - timedelta(days=60)
        
        print(f"   From: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   To: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Interval: {interval}")
        print(f"   Mode: {'Demo' if CTRADER_DEMO else 'Live'}")
        
        # Initialize cTrader fetcher
        fetcher = CTraderDataFetcher(
            client_id=CTRADER_CLIENT_ID,
            client_secret=CTRADER_CLIENT_SECRET,
            access_token=CTRADER_ACCESS_TOKEN,
            account_id=CTRADER_ACCOUNT_ID,
            demo=CTRADER_DEMO
        )
        
        # Fetch data
        data = fetcher.fetch_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d %H:%M:%S'),
            end_date=end_date.strftime('%Y-%m-%d %H:%M:%S'),
            interval=interval
        )
        print(data.head())
        print(data.tail())
        
        if data.empty:
            print(f"❌ No data fetched for {symbol}")
            return None
        
        print(f"✅ Successfully fetched {len(data)} real data points")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Price range: {data['Low'].min():.5f} - {data['High'].max():.5f}")
        return data
        
    except Exception as e:
        print(f"❌ Error fetching real data: {e}")
        import traceback
        traceback.print_exc()
        print("🔄 Falling back to demo data...")
        return generate_sample_data()


def generate_sample_data():
    """Generate sample forex data for demonstration"""
    print("📊 Generating sample forex data...")
    
    # Create sample data (24-hour forex market)
    dates = pd.date_range(
        start='2025-01-01 00:00:00',
        end='2025-01-31 23:45:00',
        freq='15min'
    )
    
    # Filter for weekdays (forex closed on weekends)
    market_dates = [date for date in dates if date.weekday() < 5]
    
    # Generate sample OHLCV data (forex-like prices around 1.10)
    np.random.seed(42)
    n_points = len(market_dates)
    base_price = 1.1000  # EURUSD-like price
    prices = [base_price]
    
    for i in range(1, n_points):
        change = np.random.normal(0.0001, 0.001)  # Smaller moves for forex
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.5))
    
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(1000, 10000) for _ in range(n_points)]
    }, index=market_dates)
    
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    print(f"✅ Generated {len(data)} sample data points")
    return data


def main():
    """Run MA optimization with real cTrader data"""
    print("🚀 MA OPTIMIZATION WITH CTRADER DATA")
    print("=" * 50)
    
    # Ask user for preferences
    print("\n📊 Data Source Options:")
    print(f"1. Real cTrader Data ({SYMBOL}, {DAYS_TO_FETCH} days, {INTERVAL})")
    print("2. Demo Data (Sample)")
    print("3. Real cTrader Data (Custom)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Use default real data
        data = fetch_real_data(symbol=SYMBOL, days=DAYS_TO_FETCH, interval=INTERVAL)
    elif choice == "2":
        # Use demo data
        data = generate_sample_data()
    elif choice == "3":
        # Custom real data
        symbol = input(f"Enter symbol (e.g., EURUSD, GBPUSD, USDJPY, default {SYMBOL}): ").strip().upper() or SYMBOL
        days = input(f"Enter number of days (default {DAYS_TO_FETCH}): ").strip()
        days = int(days) if days.isdigit() else DAYS_TO_FETCH
        interval = input(f"Enter interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, default {INTERVAL}): ").strip() or INTERVAL
        data = fetch_real_data(symbol=symbol, days=days, interval=interval)
    else:
        print("❌ Invalid choice. Using demo data...")
        data = generate_sample_data()
    
    if data is None or data.empty:
        print("❌ No data available. Exiting...")
        return
    
    # Create visualizer (no auto-open, save to plots folder)
    visualizer = MAOptimization3DVisualizer(
        data, 
        trading_fee=0.0,  # Forex typically has spread, not commission
        auto_open=False, 
        output_dir="ma_optimization_plots_ctrader"
    )
    
    # Define parameter ranges
    # short_window_range = [5, 10, 15, 20, 25, 30]
    # long_window_range = [20, 30, 40, 50, 60, 70]

    SHORT_VAL = 10
    LONG_VAL = 40
    RANGE = 9
    
    #Ensure short_window_range max is less than long_window_range min
    short_start = max(SHORT_VAL - RANGE, 4)
    short_end = SHORT_VAL + RANGE
    long_start = max(LONG_VAL - RANGE, SHORT_VAL + RANGE + 1)
    long_end = LONG_VAL + RANGE
    short_window_range = np.arange(short_start, short_end, 1)
    long_window_range = np.arange(long_start, long_end, 1)
    
    risk_reward_ratios = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
    # risk_reward_ratios = [8.0, 8.5, 9.0]
    
    print(f"\n🔍 Running optimization...")
    print(f"   Short windows: {list(short_window_range)}")
    print(f"   Long windows: {list(long_window_range)}")
    print(f"   Risk-reward ratios: {risk_reward_ratios}")
    
    # Run optimization grid
    results = visualizer.run_optimization_grid(
        short_window_range, long_window_range, risk_reward_ratios
    )
    
    # Print summary
    visualizer.print_optimization_summary()
    
    # Generate all plots
    print(f"\n🎨 Generating plots (saving to 'ma_optimization_plots_ctrader' folder)...")
    
    print("   📊 Creating summary plot...")
    visualizer.create_summary_plot(metric='composite_score')
    
    print("   📊 Creating 3D plots...")
    visualizer.create_3d_plots(metric='composite_score')
    
    print("   📊 Creating individual 3D plots...")
    visualizer.create_individual_3d_plots(metric='composite_score')
    
    print("   📊 Creating 2D heatmaps...")
    visualizer.create_2d_heatmaps(metric='composite_score')
    
    print("   📊 Creating distribution contours...")
    visualizer.create_distribution_contour_plots(metric='composite_score')
    
    print("   📊 Creating optimal regions...")
    visualizer.create_optimal_regions_plot(metric='composite_score', percentile_threshold=80.0)
    
    # Neighborhood-aware recommendations
    print("\n   🤖 Calculating neighborhood-aware recommendations...")
    visualizer.print_neighborhood_aware_recommendations(metric='composite_score')
    
    print("\n   📊 Printing top neighborhood-aware points...")
    visualizer.print_top_neighborhood_aware_points(metric='composite_score', top_n=5)
    
    print("\n   🏆 Calculating overall top parameters across all RR ratios...")
    visualizer.print_overall_top_neighborhood_aware_points(metric='composite_score', top_n=5)
    
    print("   📊 Creating 3D Gaussian surfaces...")
    visualizer.create_3d_gaussian_surface_plots(metric='composite_score')
    
    print("   📊 Creating combined 3D Gaussian...")
    visualizer.create_combined_3d_plot(metric='composite_score')
    
    print("   📊 Creating 3D Gaussian bell curves...")
    visualizer.create_3d_gaussian_bell_curves(metric='composite_score', percentile_threshold=80.0)
    
    print("   📊 Creating individual bell curves...")
    visualizer.create_individual_3d_gaussian_bell_curves(metric='composite_score', percentile_threshold=80.0)
    
    print("   📋 Creating parameter selection guide...")
    visualizer.create_parameter_selection_guide(metric='composite_score')
    
    # Create dashboard
    print(f"\n🎨 Creating dashboard...")
    try:
        from create_dashboard import create_dashboard
        create_dashboard(plot_dir="ma_optimization_plots_ctrader")
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        print("   Continuing without dashboard...")
    
    # Open dashboard
    print(f"\n🌐 Opening dashboard...")
    dashboard_path = os.path.join("ma_optimization_plots_ctrader", "dashboard.html")
    
    if os.path.exists(dashboard_path):
        print(f"✅ Dashboard ready!")
        print(f"📁 Location: {os.path.abspath(dashboard_path)}")
        print(f"🌐 Open this file in your browser to view all plots")
        
        # Try to open in browser
        try:
            if sys.platform.startswith('darwin'):  # macOS
                import subprocess
                subprocess.run(['open', dashboard_path])
            elif sys.platform.startswith('win'):  # Windows
                import subprocess
                subprocess.run(['start', dashboard_path], shell=True)
            elif sys.platform.startswith('linux'):  # Linux
                import subprocess
                subprocess.run(['xdg-open', dashboard_path])
            print("🌐 Dashboard opened in your default browser!")
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print(f"   Please manually open: {os.path.abspath(dashboard_path)}")
    else:
        print(f"⚠️  Dashboard not found at {dashboard_path}")
        print(f"   Plots are available in: {os.path.abspath('ma_optimization_plots_ctrader')}")


if __name__ == "__main__":
    main()

