#!/usr/bin/env python3
"""
Simple script to run MA optimization with real data using the existing visualizer.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ma_3d_optimization_visualizer import MAOptimization3DVisualizer

SYMBOL = "BNBUSDT"

def fetch_real_data(symbol="ETHUSDT", days=30, interval="15m"):
    """Fetch real data from Binance"""
    print(f"📊 Fetching real data for {symbol}...")
    
    try:
        from data_fetcher import BinanceDataFetcher
        import config as cfg
        
        # Calculate date range
        # end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')


        end_date = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d %H:%M:%S')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"   From: {start_date}")
        print(f"   To: {end_date}")
        print(f"   Interval: {interval}")
        
        # Fetch data
        binance_fetcher = BinanceDataFetcher(
            api_key=cfg.BINANCE_API_KEY, 
            api_secret=cfg.BINANCE_SECRET_KEY
        )
        data = binance_fetcher.fetch_historical_data(symbol, start_date, end_date, interval=interval)
        
        if data.empty:
            print(f"❌ No data fetched for {symbol}")
            return None
        
        print(f"✅ Successfully fetched {len(data)} real data points")
        return data
        
    except Exception as e:
        print(f"❌ Error fetching real data: {e}")
        print("🔄 Falling back to demo data...")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data for demonstration"""
    print("📊 Generating sample data...")
    
    # Create sample data
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
    """Run MA optimization with real data"""
    print("🚀 MA OPTIMIZATION WITH REAL DATA")
    print("=" * 50)
    
    # Ask user for preferences
    print("\n📊 Data Source Options:")
    print(f"1. Real Binance Data ({SYMBOL}, 30 days, 15m)")
    print("2. Demo Data (Sample)")
    print("3. Real Binance Data (Custom)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Use default real data
        data = fetch_real_data(symbol=SYMBOL, days=30, interval="15m")
    elif choice == "2":
        # Use demo data
        data = generate_sample_data()
    elif choice == "3":
        # Custom real data
        symbol = input(f"Enter symbol (e.g., BTCUSDT, {SYMBOL}): ").strip().upper()
        days = input("Enter number of days (default 30): ").strip()
        days = int(days) if days.isdigit() else 30
        interval = input("Enter interval (default 15m): ").strip() or "15m"
        data = fetch_real_data(symbol=symbol, days=days, interval=interval)
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
        output_dir="ma_optimization_plots"
    )
    
    # Define parameter ranges
    # short_window_range = [5, 10, 15, 20, 25, 30]
    # long_window_range = [20, 30, 40, 50, 60, 70]
    short_window_range = np.arange(15, 26, 1)
    long_window_range = np.arange(25, 36, 1)
    risk_reward_ratios = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
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
    print(f"\n🎨 Generating plots (saving to 'ma_optimization_plots' folder)...")
    
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
        create_dashboard()
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        return
    
    # Open dashboard
    print(f"\n🌐 Opening dashboard...")
    dashboard_path = os.path.join("ma_optimization_plots", "dashboard.html")
    
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
        print(f"❌ Dashboard not found at {dashboard_path}")

if __name__ == "__main__":
    main()
