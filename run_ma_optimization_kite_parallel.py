#!/usr/bin/env python3
"""
Parallel version of MA optimization script for Zerodha Kite.
Uses multiprocessing to speed up optimization by running independent parameter
combinations in parallel.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

# Try to import tqdm for progress bar, but continue without it if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Create a dummy tqdm that does nothing
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, *args):
            pass

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ma_3d_optimization_visualizer import MAOptimization3DVisualizer
from strategies import MovingAverageCrossover

# ============================================================================
# GLOBAL CONFIGURATION - Edit these values to set defaults
# ============================================================================
SYMBOL = "NATGASMINI26FEBFUT"  # Default Indian stock symbol
#SYMBOL = "CRUDEOIL26MARFUT"  # Default Indian stock symbol
EXCHANGE = "MCX"  # Default exchange (NSE, BSE, MCX, etc.)
DAYS_TO_FETCH = 30  # Default number of days to fetch
INTERVAL = "15m"  # Default interval
MAX_WORKERS = None  # None = use all available CPUs, or set to specific number


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
        '1h': '60minute',
        '2h': '2hour',
        '4h': '4hour',
        '1d': 'day',
        '1w': 'week',
        '1M': 'month'
    }
    result = interval_mapping.get(interval, '15minute')
    return result


# ============================================================================
# PARALLEL WORKER FUNCTION
# ============================================================================
def evaluate_ma_combo_worker(args):
    """
    Worker function to evaluate a single MA combination.
    Must be at module level for pickling with ProcessPoolExecutor.
    
    Args:
        args: Tuple of (data_dict, short_window, long_window, risk_reward_ratio, trading_fee)
              data_dict is a dict representation of the DataFrame (to avoid pickling issues)
    
    Returns:
        Dictionary with results for this combination
    """
    data_dict, short_window, long_window, risk_reward_ratio, trading_fee = args
    
    # Reconstruct DataFrame from dict
    data = pd.DataFrame(data_dict['data'], columns=data_dict['columns'])
    data.index = pd.to_datetime(data_dict['index'])
    
    try:
        # Validate parameters
        if short_window >= long_window:
            return {
                'risk_reward_ratio': risk_reward_ratio,
                'short_window': short_window,
                'long_window': long_window,
                'composite_score': -999,
                'sharpe_ratio': -999,
                'calmar_ratio': -999,
                'profit_factor': -999,
                'win_rate': -999,
                'total_pnl': -999,
                'total_trades': 0,
                'max_drawdown': -999,
                'geometric_mean_return': -999
            }
        
        # Create strategy instance
        strategy = MovingAverageCrossover(
            short_window=short_window,
            long_window=long_window,
            risk_reward_ratio=risk_reward_ratio,
            trading_fee=trading_fee
        )
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Get strategy metrics
        metrics = strategy.get_strategy_metrics()
        
        # Calculate composite score (same as in MAOptimization3DVisualizer)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        calmar_ratio = metrics.get('calmar_ratio', 0)
        profit_factor = metrics.get('profit_factor', 0)
        win_rate = metrics.get('win_rate', 0)
        geometric_mean_return = metrics.get('geometric_mean_return', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        
        # Handle NaN and infinite values
        if np.isinf(profit_factor) or np.isnan(profit_factor):
            profit_factor = 0
        if np.isinf(calmar_ratio) or np.isnan(calmar_ratio):
            calmar_ratio = 0
        if np.isinf(sharpe_ratio) or np.isnan(sharpe_ratio):
            sharpe_ratio = 0
        
        # Normalize metrics to comparable scales (0-1 range)
        sharpe_normalized = np.clip((sharpe_ratio + 3) / 6, 0, 1)
        calmar_normalized = np.clip(calmar_ratio / 10, 0, 1)
        profit_factor_normalized = np.clip(profit_factor / 3, 0, 1)
        win_rate_normalized = np.clip(win_rate, 0, 1)
        gmr_normalized = np.clip((geometric_mean_return + 0.05) / 0.10, 0, 1)
        drawdown_score = np.clip(1 - max_drawdown, 0, 1)
        
        # Calculate composite score with normalized metrics
        composite_score = (
            0.25 * sharpe_normalized +
            0.20 * calmar_normalized +
            0.15 * profit_factor_normalized +
            0.15 * win_rate_normalized +
            0.15 * gmr_normalized +
            0.10 * drawdown_score
        )
        
        return {
            'risk_reward_ratio': risk_reward_ratio,
            'short_window': short_window,
            'long_window': long_window,
            'composite_score': composite_score,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'total_pnl': metrics.get('total_pnl', 0),
            'total_trades': metrics.get('total_trades', 0),
            'max_drawdown': max_drawdown,
            'geometric_mean_return': geometric_mean_return
        }
        
    except Exception as e:
        return {
            'risk_reward_ratio': risk_reward_ratio,
            'short_window': short_window,
            'long_window': long_window,
            'composite_score': -999,
            'sharpe_ratio': -999,
            'calmar_ratio': -999,
            'profit_factor': -999,
            'win_rate': -999,
            'total_pnl': -999,
            'total_trades': 0,
            'max_drawdown': -999,
            'geometric_mean_return': -999
        }


def run_parallel_optimization_grid(data, short_window_range, long_window_range, 
                                   risk_reward_ratios, trading_fee=0.0, 
                                   max_workers=None):
    """
    Run optimization grid in parallel using multiprocessing.
    
    Args:
        data: OHLCV DataFrame
        short_window_range: List/array of short window values
        long_window_range: List/array of long window values
        risk_reward_ratios: List of risk-reward ratios
        trading_fee: Trading fee per trade
        max_workers: Number of parallel workers (None = use all CPUs)
    
    Returns:
        Dictionary in the same format as MAOptimization3DVisualizer.run_optimization_grid()
    """
    print("🚀 Running PARALLEL optimization grid...")
    print(f"   Short windows: {list(short_window_range)}")
    print(f"   Long windows: {list(long_window_range)}")
    print(f"   Risk-reward ratios: {risk_reward_ratios}")
    
    # Convert ranges to lists if they're numpy arrays
    short_window_range = list(short_window_range)
    long_window_range = list(long_window_range)
    
    # Generate all combinations
    all_combinations = []
    for rr in risk_reward_ratios:
        for short_w in short_window_range:
            for long_w in long_window_range:
                if short_w < long_w:  # Valid combination
                    all_combinations.append((short_w, long_w, rr))
    
    total_combinations = len(all_combinations)
    print(f"   Total combinations to test: {total_combinations}")
    
    if max_workers is None:
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
    print(f"   Using {max_workers} parallel workers")
    
    # Convert DataFrame to dict for pickling (avoids issues with complex DataFrame objects)
    # Store as records format which is more reliable for multiprocessing
    data_dict = {
        'data': data.to_dict('records'),
        'index': data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'columns': list(data.columns)
    }
    
    # Prepare arguments for workers
    worker_args = [
        (data_dict, short_w, long_w, rr, trading_fee)
        for short_w, long_w, rr in all_combinations
    ]
    
    # Run parallel optimization
    results_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(evaluate_ma_combo_worker, args): args 
                   for args in worker_args}
        
        # Process completed tasks with progress bar
        completed = 0
        if HAS_TQDM:
            pbar = tqdm(total=total_combinations, desc="Optimizing", unit="combo")
        else:
            pbar = None
        
        try:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results_list.append(result)
                    completed += 1
                    if pbar:
                        pbar.update(1)
                    elif completed % max(1, total_combinations // 20) == 0:
                        # Print progress every 5% if no tqdm
                        print(f"   Progress: {completed}/{total_combinations} ({100*completed/total_combinations:.1f}%)")
                except Exception as e:
                    print(f"\n⚠️  Error processing combination: {e}")
                    completed += 1
                    if pbar:
                        pbar.update(1)
        finally:
            if pbar:
                pbar.close()
    
    # Organize results into the format expected by visualizer
    results = {}
    for rr in risk_reward_ratios:
        rr_results = {
            'short_windows': [],
            'long_windows': [],
            'composite_score': [],
            'sharpe_ratio': [],
            'total_pnl': [],
            'total_trades': [],
            'win_rate': [],
            'max_drawdown': []
        }
        
        # Filter results for this risk_reward_ratio
        rr_data = [r for r in results_list if r['risk_reward_ratio'] == rr]
        
        # Sort by short_window, then long_window for consistency
        rr_data.sort(key=lambda x: (x['short_window'], x['long_window']))
        
        for r in rr_data:
            rr_results['short_windows'].append(r['short_window'])
            rr_results['long_windows'].append(r['long_window'])
            rr_results['composite_score'].append(r['composite_score'])
            rr_results['sharpe_ratio'].append(r['sharpe_ratio'])
            rr_results['total_pnl'].append(r['total_pnl'])
            rr_results['total_trades'].append(r['total_trades'])
            rr_results['win_rate'].append(r['win_rate'])
            rr_results['max_drawdown'].append(r['max_drawdown'])
        
        results[rr] = rr_results
    
    print(f"\n✅ Parallel optimization complete!")
    print(f"   Processed {len(results_list)} combinations")
    
    return results


def fetch_real_data(symbol="TATAMOTORS", days=30, interval="15m", exchange="NSE"):
    """Fetch real data from Zerodha Kite"""
    print(f"📊 Fetching real data for {symbol} from {exchange}...")
    
    try:
        from data_fetcher import KiteDataFetcher
        import config as cfg
        
        # Calculate date range (Kite uses YYYY-MM-DD format)
        # Kite API's to_date needs to be tomorrow to get today's data
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # end_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        # start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
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
            # Use time comparison instead of string comparison for robustness
            time_str = date.strftime('%H:%M')
            if '09:15' <= time_str <= '15:30':
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
    """Run parallel MA optimization with real Kite data"""
    print("🚀 PARALLEL MA OPTIMIZATION WITH KITE DATA")
    print("=" * 50)
    
    # Ask user for preferences
    print("\n📊 Data Source Options:")
    print(f"1. Real Kite Data ({SYMBOL}, {EXCHANGE}, {DAYS_TO_FETCH} days, {INTERVAL})")
    print("2. Demo Data (Sample)")
    print("3. Real Kite Data (Custom)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        data = fetch_real_data(symbol=SYMBOL, days=DAYS_TO_FETCH, interval=INTERVAL, exchange=EXCHANGE)
    elif choice == "2":
        data = generate_sample_data()
    elif choice == "3":
        symbol = input(f"Enter symbol (e.g., TATAMOTORS, RELIANCE, TCS): ").strip().upper()
        exchange = input(f"Enter exchange (NSE, BSE, MCX, default {EXCHANGE}): ").strip().upper() or EXCHANGE
        days = input(f"Enter number of days (default {DAYS_TO_FETCH}): ").strip()
        days = int(days) if days.isdigit() else DAYS_TO_FETCH
        interval = input(f"Enter interval (default {INTERVAL}): ").strip() or INTERVAL
        data = fetch_real_data(symbol=symbol, days=days, interval=interval, exchange=exchange)
    else:
        print("❌ Invalid choice. Using demo data...")
        data = generate_sample_data()
    
    if data is None or data.empty:
        print("❌ No data available. Exiting...")
        return
    
    # Ask for number of workers
    print(f"\n⚙️  Parallel Processing Configuration:")
    print(f"   Available CPUs: {os.cpu_count()}")
    workers_input = input(f"   Number of workers (default: all CPUs, or enter number): ").strip()
    max_workers = int(workers_input) if workers_input.isdigit() else MAX_WORKERS
    
    # Create visualizer
    visualizer = MAOptimization3DVisualizer(
        data, 
        trading_fee=0.0, 
        auto_open=False, 
        output_dir="ma_optimization_plots_kite"
    )
    
    # short_window_range = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
    # long_window_range = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    
    # Define parameter ranges (same as original script)
    SHORT_VAL = 5
    LONG_VAL = 180
    RANGE = 9
    
    # Ensure short_window_range max is less than long_window_range min to avoid invalid combinations
    short_start = max(SHORT_VAL - RANGE, 4)
    short_end = SHORT_VAL + RANGE
    long_start = max(LONG_VAL - RANGE, SHORT_VAL + RANGE + 1)
    long_end = LONG_VAL + RANGE
    short_window_range = np.arange(short_start, short_end, 1)
    long_window_range = np.arange(long_start, long_end, 1)
    
    risk_reward_ratios = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
    
    print(f"\n🔍 Running parallel optimization...")
    print(f"   Short windows: {list(short_window_range)}")
    print(f"   Long windows: {list(long_window_range)}")
    print(f"   Risk-reward ratios: {risk_reward_ratios}")
    
    # Run parallel optimization grid
    results = run_parallel_optimization_grid(
        data=data,
        short_window_range=short_window_range,
        long_window_range=long_window_range,
        risk_reward_ratios=risk_reward_ratios,
        trading_fee=0.0,
        max_workers=max_workers
    )
    
    # Set results in visualizer (so we can use all its plotting methods)
    visualizer.results = results
    
    # Print summary
    visualizer.print_optimization_summary()
    
    # Generate all plots
    print(f"\n🎨 Generating plots (saving to 'ma_optimization_plots_kite' folder)...")
    
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

