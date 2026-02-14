#!/usr/bin/env python3
"""
Simple script to generate all MA optimization plots without auto-opening them.
All plots will be saved in the 'ma_optimization_plots' folder.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ma_3d_optimization_visualizer import MAOptimization3DVisualizer

def fetch_real_data(symbol="ETHUSDT", days=30, interval="15m"):
    """Fetch real data from Binance"""
    print(f"📊 Fetching real data for {symbol}...")
    
    try:
        from data_fetcher import BinanceDataFetcher
        import config as cfg
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        
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
    """Generate all plots"""
    print("🎨 MA OPTIMIZATION PLOT GENERATOR")
    print("=" * 50)
    
    # Try to fetch real data first, fallback to sample data
    data = fetch_real_data(symbol="ETHUSDT", days=30, interval="15m")
    if data is None:
        data = generate_sample_data()
    
    # Create visualizer (no auto-open, save to plots folder)
    visualizer = MAOptimization3DVisualizer(
        data, 
        trading_fee=0.0, 
        auto_open=False, 
        output_dir="ma_optimization_plots"
    )
    
    # Define parameter ranges (smaller for demo)
    short_window_range = [5, 10, 15, 20, 25]
    long_window_range = [20, 30, 40, 50, 60]
    risk_reward_ratios = [1.5, 2.0, 2.5, 3.0, 3.5]
    
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
    
    # Generate all plots
    print(f"\n🎨 Generating plots (saving to 'ma_optimization_plots' folder)...")
    
    plot_functions = [
        ("Summary Plot", lambda: visualizer.create_summary_plot(metric='composite_score')),
        ("3D Grid Plots", lambda: visualizer.create_3d_plots(metric='composite_score')),
        ("Individual 3D Plots", lambda: visualizer.create_individual_3d_plots(metric='composite_score')),
        ("2D Heatmaps", lambda: visualizer.create_2d_heatmaps(metric='composite_score')),
        ("Distribution Contours", lambda: visualizer.create_distribution_contour_plots(metric='composite_score')),
        ("Optimal Regions", lambda: visualizer.create_optimal_regions_plot(metric='composite_score', percentile_threshold=80.0)),
        ("3D Gaussian Surfaces", lambda: visualizer.create_3d_gaussian_surface_plots(metric='composite_score')),
        ("Combined 3D Gaussian", lambda: visualizer.create_combined_3d_plot(metric='composite_score')),
        ("3D Gaussian Bell Curves", lambda: visualizer.create_3d_gaussian_bell_curves(metric='composite_score', percentile_threshold=80.0)),
        ("Individual Bell Curves", lambda: visualizer.create_individual_3d_gaussian_bell_curves(metric='composite_score', percentile_threshold=80.0)),
    ]
    
    for i, (plot_name, plot_func) in enumerate(plot_functions):
        try:
            print(f"   {i+1:2d}. Creating {plot_name}...")
            plot_func()
        except Exception as e:
            print(f"   ⚠️  Error creating {plot_name}: {e}")
            continue
    
    # Create parameter selection guide
    print(f"   📋 Creating parameter selection guide...")
    visualizer.create_parameter_selection_guide(metric='composite_score')
    
    print(f"\n✅ All plots generated successfully!")
    print(f"📁 Check the 'ma_optimization_plots' folder for all HTML files")
    print(f"🌐 Open 'ma_optimization_plots/dashboard.html' to view all plots in one place")

if __name__ == "__main__":
    main()
