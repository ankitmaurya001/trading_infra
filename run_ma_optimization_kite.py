#!/usr/bin/env python3
"""
Simple script to run MA optimization with real data from Zerodha Kite using the existing visualizer.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ma_3d_optimization_visualizer import MAOptimization3DVisualizer

SYMBOL = "NATURALGAS25DECFUT"  # Default Indian stock symbol
EXCHANGE = "MCX"  # Default exchange (NSE, BSE, MCX, etc.)

def map_interval_to_kite(interval: str) -> str:
    """
    Map common interval formats to Kite Connect format
    
    Args:
        interval (str): Interval in common format (e.g., '15m', '1h', '1d')
        
    Returns:
        str: Kite Connect interval format (e.g., '15minute', 'hour', 'day')
    """
    interval_mapping = {
        '1m': 'minute',
        '3m': '3minute',
        '5m': '5minute',
        '15m': '15minute',
        '30m': '30minute',
        '1h': 'hour',
        '2h': '2hour',
        '4h': '4hour',
        '1d': 'day',
        '1w': 'week',
        '1M': 'month'
    }
    return interval_mapping.get(interval, '15minute')

def fetch_real_data(symbol="TATAMOTORS", days=30, interval="15m", exchange="NSE"):
    """Fetch real data from Zerodha Kite"""
    print(f"📊 Fetching real data for {symbol} from {exchange}...")
    
    try:
        from data_fetcher import KiteDataFetcher
        import config as cfg
        
        # Calculate date range (Kite uses YYYY-MM-DD format)
        # end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')  # Yesterday
        # start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        end_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        print(f"   From: {start_date}")
        print(f"   To: {end_date}")
        print(f"   Interval: {interval} (Kite: {map_interval_to_kite(interval)})")
        print(f"   Exchange: {exchange}")
        
        # Initialize Kite fetcher
        kite_fetcher = KiteDataFetcher(
            credentials=cfg.KITE_CREDENTIALS,
            exchange=exchange
        )
        
        # Authenticate
        print("   🔐 Authenticating with Kite Connect...")
        kite_fetcher.authenticate()
        print("   ✅ Authentication successful!")
        
        # Map interval to Kite format
        kite_interval = map_interval_to_kite(interval)
        
        # Fetch data
        data = kite_fetcher.fetch_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=kite_interval
        )
        
        if data.empty:
            print(f"❌ No data fetched for {symbol}")
            return None
        
        print(f"✅ Successfully fetched {len(data)} real data points")
        return data
        
    except Exception as e:
        print(f"❌ Error fetching real data: {e}")
        import traceback
        traceback.print_exc()
        print("🔄 Falling back to demo data...")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data for demonstration"""
    print("📊 Generating sample data...")
    
    # Create sample data (Indian market hours: 9:15 AM to 3:30 PM IST)
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
    np.random.seed(42)
    n_points = len(market_dates)
    base_price = 500.0
    prices = [base_price]
    
    for i in range(1, n_points):
        change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))
    
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(100000, 1000000) for _ in range(n_points)]
    }, index=market_dates)
    
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    print(f"✅ Generated {len(data)} sample data points")
    return data

def main():
    """Run MA optimization with real Kite data"""
    print("🚀 MA OPTIMIZATION WITH KITE DATA")
    print("=" * 50)
    
    # Ask user for preferences
    print("\n📊 Data Source Options:")
    print(f"1. Real Kite Data ({SYMBOL}, {EXCHANGE}, 30 days, 15m)")
    print("2. Demo Data (Sample)")
    print("3. Real Kite Data (Custom)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Use default real data
        data = fetch_real_data(symbol=SYMBOL, days=30, interval="15m", exchange=EXCHANGE)
    elif choice == "2":
        # Use demo data
        data = generate_sample_data()
    elif choice == "3":
        # Custom real data
        symbol = input(f"Enter symbol (e.g., TATAMOTORS, RELIANCE, TCS): ").strip().upper()
        exchange = input(f"Enter exchange (NSE, BSE, MCX, default {EXCHANGE}): ").strip().upper() or EXCHANGE
        days = input("Enter number of days (default 30): ").strip()
        days = int(days) if days.isdigit() else 30
        interval = input("Enter interval (default 15m): ").strip() or "15m"
        data = fetch_real_data(symbol=symbol, days=days, interval=interval, exchange=exchange)
    else:
        print("❌ Invalid choice. Using demo data...")
        data = generate_sample_data()
    
    if data is None:
        print("❌ No data available. Exiting...")
        return
    
    # Create visualizer (no auto-open, save to plots folder)
    visualizer = MAOptimization3DVisualizer(
        data, 
        trading_fee=0.0, 
        auto_open=False, 
        output_dir="ma_optimization_plots_kite"
    )
    
    # Define parameter ranges
    # short_window_range = [5, 10, 15, 20, 25, 30]
    # long_window_range = [20, 30, 40, 50, 60, 70]
    SHORT_VAL =30
    LONG_VAL = 60
    RANGE = 9
    short_window_range = np.arange(max(SHORT_VAL - RANGE, 5), SHORT_VAL + RANGE, 1)
    long_window_range = np.arange(LONG_VAL - RANGE, LONG_VAL + RANGE, 1)
    risk_reward_ratios = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    
    print(f"\n🔍 Running optimization...")
    print(f"   Short windows: {short_window_range}")
    print(f"   Long windows: {long_window_range}")
    print(f"   Risk-reward ratios: {risk_reward_ratios}")
    
    # Run optimization grid
    results = visualizer.run_optimization_grid(
        short_window_range, long_window_range, risk_reward_ratios
    )
    
    # Print summary
    visualizer.print_optimization_summary()
    
    # Generate all plots using the existing methods
    print(f"\n🎨 Generating plots (saving to 'ma_optimization_plots_kite' folder)...")
    
    # Use the existing demo function from the visualizer
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
        # Pass the correct output directory
        create_dashboard(plot_dir="ma_optimization_plots_kite")
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        print("   Continuing without dashboard...")
    
    # Open dashboard
    print(f"\n🌐 Opening dashboard...")
    dashboard_path = os.path.join("ma_optimization_plots_kite", "dashboard.html")
    
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
        print(f"   Plots are available in: {os.path.abspath('ma_optimization_plots_kite')}")

if __name__ == "__main__":
    main()

