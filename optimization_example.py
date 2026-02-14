import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from strategy_optimizer import (
    StrategyOptimizer, 
    optimize_moving_average_crossover,
    optimize_rsi_strategy,
    optimize_donchian_channel
)
from strategies import MovingAverageCrossover, RSIStrategy, DonchianChannelBreakout

def fetch_sample_data(symbol: str = "AAPL", start_date: str = "2023-01-01", end_date: str = "2024-01-01", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch sample data using the existing DataFetcher class.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "BTC-USD", "TATAMOTORS.NS")
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval (e.g., "1d", "5m", "15m", "1h")
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching {interval} data for {symbol} from {start_date} to {end_date}...")
    
    try:
        fetcher = DataFetcher()
        data = fetcher.fetch_data(symbol, start_date, end_date, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data fetched for {symbol}")
        
        print(f"Successfully fetched {len(data)} data points")
        print(f"Data range: {data.index[0]} to {data.index[-1]}")
        print(f"Columns: {list(data.columns)}")
        
        return data
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Using synthetic data instead...")
        # Create synthetic data for demonstration
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame({
            "Open": np.random.rand(len(dates)) * 100 + 100,
            "High": np.random.rand(len(dates)) * 100 + 100,
            "Low": np.random.rand(len(dates)) * 100 + 100,
            "Close": np.random.rand(len(dates)) * 100 + 100,
            "Volume": np.random.randint(1000, 10000, len(dates)),
            "Returns": np.random.randn(len(dates)) * 0.02,
            "Avg_Daily_Return": np.random.randn(len(dates)) * 0.01,
            "Volatility": np.random.rand(len(dates)) * 0.03,
            "Key_Point": np.random.choice([True, False], len(dates), p=[0.1, 0.9])
        }, index=dates)
        return data

def demonstrate_optimization_metrics():
    """
    Demonstrate why using multiple metrics is better than just total_pnl.
    """
    print("="*80)
    print("WHY MULTIPLE METRICS ARE BETTER THAN JUST TOTAL_PNL")
    print("="*80)
    
    print("\n1. PROBLEMS WITH TOTAL_PNL OPTIMIZATION:")
    print("   - Can be skewed by a single outlier trade")
    print("   - Doesn't consider risk (drawdown)")
    print("   - Doesn't account for consistency")
    print("   - May lead to overfitting to specific market conditions")
    
    print("\n2. BETTER METRICS FOR HYPERPARAMETER SELECTION:")
    print("   - Sharpe Ratio: Risk-adjusted returns")
    print("   - Calmar Ratio: Return vs maximum drawdown")
    print("   - Profit Factor: Total profit / total loss")
    print("   - Win Rate: Consistency of winning trades")
    print("   - Max Drawdown: Worst peak-to-trough decline")
    print("   - Geometric Mean Return: Compound growth rate")
    
    print("\n3. COMPOSITE SCORE APPROACH:")
    print("   - Combines multiple metrics with weights")
    print("   - Penalizes insufficient data (min_trades)")
    print("   - Penalizes excessive risk (max_drawdown_threshold)")
    print("   - Penalizes poor risk-adjusted returns (sharpe_threshold)")
    print("   - Balances return, risk, and consistency")

def run_moving_average_optimization(data: pd.DataFrame):
    """
    Demonstrate Moving Average Crossover optimization.
    """
    print("\n" + "="*80)
    print("MOVING AVERAGE CROSSOVER OPTIMIZATION")
    print("="*80)
    
    # Define parameter ranges
    short_window_range = [5, 10, 15, 20, 25]
    long_window_range = [30, 40, 50, 60, 70]
    risk_reward_range = [1.5, 2.0, 2.5, 3.0]
    
    print(f"\nParameter ranges:")
    print(f"  Short window: {short_window_range}")
    print(f"  Long window: {long_window_range}")
    print(f"  Risk-reward ratio: {risk_reward_range}")
    print(f"  Total combinations: {len(short_window_range) * len(long_window_range) * len(risk_reward_range)}")
    
    # Run optimization
    best_params, best_metrics = optimize_moving_average_crossover(
        data=data,
        short_window_range=short_window_range,
        long_window_range=long_window_range,
        risk_reward_range=risk_reward_range,
        trading_fee=0.001
    )
    
    print(f"\nBEST PARAMETERS FOUND:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nBEST METRICS:")
    print(f"  Total Trades: {best_metrics['total_trades']}")
    print(f"  Win Rate: {best_metrics['win_rate']:.2%}")
    print(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.3f}")
    print(f"  Calmar Ratio: {best_metrics['calmar_ratio']:.3f}")
    print(f"  Profit Factor: {best_metrics['profit_factor']:.3f}")
    print(f"  Max Drawdown: {best_metrics['max_drawdown']:.2%}")
    print(f"  Total PnL: {best_metrics['total_pnl']:.2%}")
    print(f"  Geometric Mean Return: {best_metrics['geometric_mean_return']:.4f}")
    
    return best_params, best_metrics

def run_rsi_optimization(data: pd.DataFrame):
    """
    Demonstrate RSI strategy optimization.
    """
    print("\n" + "="*80)
    print("RSI STRATEGY OPTIMIZATION")
    print("="*80)
    
    # Define parameter ranges
    period_range = [10, 14, 20, 30]
    overbought_range = [65, 70, 75, 80]
    oversold_range = [20, 25, 30, 35]
    risk_reward_range = [1.5, 2.0, 2.5, 3.0]
    
    print(f"\nParameter ranges:")
    print(f"  Period: {period_range}")
    print(f"  Overbought: {overbought_range}")
    print(f"  Oversold: {oversold_range}")
    print(f"  Risk-reward ratio: {risk_reward_range}")
    
    # Run optimization
    best_params, best_metrics = optimize_rsi_strategy(
        data=data,
        period_range=period_range,
        overbought_range=overbought_range,
        oversold_range=oversold_range,
        risk_reward_range=risk_reward_range,
        trading_fee=0.001
    )
    
    print(f"\nBEST PARAMETERS FOUND:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nBEST METRICS:")
    print(f"  Total Trades: {best_metrics['total_trades']}")
    print(f"  Win Rate: {best_metrics['win_rate']:.2%}")
    print(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.3f}")
    print(f"  Calmar Ratio: {best_metrics['calmar_ratio']:.3f}")
    print(f"  Profit Factor: {best_metrics['profit_factor']:.3f}")
    print(f"  Max Drawdown: {best_metrics['max_drawdown']:.2%}")
    
    return best_params, best_metrics

def compare_optimization_approaches(data: pd.DataFrame):
    """
    Compare different optimization approaches to show the benefits of composite scoring.
    """
    print("\n" + "="*80)
    print("COMPARING OPTIMIZATION APPROACHES")
    print("="*80)
    
    # Test different optimization metrics
    metrics_to_test = ["total_pnl", "sharpe_ratio", "calmar_ratio", "composite_score"]
    
    param_ranges = {
        'short_window': [10, 15, 20],
        'long_window': [40, 50, 60],
        'risk_reward_ratio': [2.0, 2.5, 3.0],
        'trading_fee': [0.001]
    }
    
    results = {}
    
    for metric in metrics_to_test:
        print(f"\nOptimizing for: {metric}")
        
        optimizer = StrategyOptimizer(
            data=data,
            strategy_class=MovingAverageCrossover,
            param_ranges=param_ranges,
            optimization_metric=metric,
            min_trades=10,
            max_drawdown_threshold=0.5,
            sharpe_threshold=0.3
        )
        
        best_params, best_metrics = optimizer.optimize()
        
        results[metric] = {
            'params': best_params,
            'metrics': best_metrics
        }
        
        print(f"  Best {metric}: {best_metrics.get(metric, 0):.4f}")
        print(f"  Parameters: {best_params}")
    
    print(f"\nCOMPARISON SUMMARY:")
    print("-" * 60)
    for metric, result in results.items():
        metrics = result['metrics']
        print(f"\n{metric.upper()} OPTIMIZATION:")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
        print(f"  Calmar: {metrics['calmar_ratio']:.3f}")
        print(f"  Max DD: {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Total PnL: {metrics['total_pnl']:.2%}")

def demonstrate_robustness_analysis(data: pd.DataFrame):
    """
    Demonstrate how to analyze the robustness of optimized parameters.
    """
    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS")
    print("="*80)
    
    # Use a smaller parameter range for demonstration
    param_ranges = {
        'short_window': [10, 15, 20],
        'long_window': [40, 50, 60],
        'risk_reward_ratio': [2.0, 2.5, 3.0],
        'trading_fee': [0.001]
    }
    
    optimizer = StrategyOptimizer(
        data=data,
        strategy_class=MovingAverageCrossover,
        param_ranges=param_ranges,
        optimization_metric="composite_score",
        min_trades=10,
        max_drawdown_threshold=0.5,
        sharpe_threshold=0.3
    )
    
    best_params, best_metrics = optimizer.optimize()
    
    print(f"\nPARAMETER SENSITIVITY ANALYSIS:")
    print("-" * 60)
    
    for param_name in param_ranges.keys():
        sensitivity = optimizer.analyze_parameter_sensitivity(param_name)
        print(f"\n{param_name.upper()}:")
        for value, avg_score in sorted(sensitivity.items()):
            print(f"  {value}: {avg_score:.4f}")
    
    print(f"\nTOP 10 RESULTS:")
    print("-" * 60)
    top_results = optimizer.get_top_results(10)
    for i, result in enumerate(top_results, 1):
        metrics = result['metrics']
        print(f"{i:2d}. Score: {result['score']:.4f} | "
              f"Sharpe: {metrics['sharpe_ratio']:.3f} | "
              f"Calmar: {metrics['calmar_ratio']:.3f} | "
              f"Win Rate: {metrics['win_rate']:.2%} | "
              f"Params: {result['parameters']}")

def main():
    """
    Main function to demonstrate the robust strategy optimizer.
    """
    print("ROBUST STRATEGY OPTIMIZER DEMONSTRATION")
    print("="*80)
    
    # Example 1: Stock data (daily)
    print("\nEXAMPLE 1: Stock Data (Daily)")
    print("-" * 40)
    try:
        data = fetch_sample_data("AAPL", "2022-01-01", "2023-01-01", "1d")
    except Exception as e:
        print(f"Error with AAPL data: {e}")
        data = None
    
    # Example 2: Crypto data (15-minute intervals)
    print("\nEXAMPLE 2: Crypto Data (15-minute intervals)")
    print("-" * 40)
    try:
        crypto_data = fetch_sample_data("BTC-USD", "2023-06-15", "2023-07-15", "15m")
        if crypto_data is not None and not crypto_data.empty:
            data = crypto_data  # Use crypto data for demonstration
            print("Using BTC-USD data for optimization demonstration")
    except Exception as e:
        print(f"Error with BTC-USD data: {e}")
    
    # Example 3: Indian stock data (5-minute intervals)
    print("\nEXAMPLE 3: Indian Stock Data (5-minute intervals)")
    print("-" * 40)
    try:
        indian_data = fetch_sample_data("TATAMOTORS.NS", "2023-06-01", "2023-07-20", "5m")
        if indian_data is not None and not indian_data.empty:
            data = indian_data  # Use Indian stock data for demonstration
            print("Using TATAMOTORS.NS data for optimization demonstration")
    except Exception as e:
        print(f"Error with TATAMOTORS.NS data: {e}")
    
    # If all data fetching failed, use synthetic data
    if data is None or data.empty:
        print("\nUsing synthetic data for demonstration...")
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=500, freq='D')
        data = pd.DataFrame({
            "Open": np.random.rand(500) * 100 + 100,
            "High": np.random.rand(500) * 100 + 100,
            "Low": np.random.rand(500) * 100 + 100,
            "Close": np.random.rand(500) * 100 + 100,
            "Volume": np.random.randint(1000, 10000, 500),
            "Returns": np.random.randn(500) * 0.02,
            "Avg_Daily_Return": np.random.randn(500) * 0.01,
            "Volatility": np.random.rand(500) * 0.03,
            "Key_Point": np.random.choice([True, False], 500, p=[0.1, 0.9])
        }, index=dates)
    
    # Demonstrate the concepts
    demonstrate_optimization_metrics()
    
    # Run optimizations
    ma_params, ma_metrics = run_moving_average_optimization(data)
    rsi_params, rsi_metrics = run_rsi_optimization(data)
    
    # Compare approaches
    compare_optimization_approaches(data)
    
    # Demonstrate robustness analysis
    demonstrate_robustness_analysis(data)
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("1. Use composite scoring instead of single metrics")
    print("2. Set minimum thresholds for trades, Sharpe ratio, and max drawdown")
    print("3. Analyze parameter sensitivity to understand robustness")
    print("4. Consider multiple top results, not just the best one")
    print("5. Validate parameters on out-of-sample data")
    print("6. Monitor for overfitting by checking parameter stability")
    
    print("\n" + "="*80)
    print("DATA FETCHING EXAMPLES")
    print("="*80)
    print("You can use different symbols and intervals:")
    print("- Stocks: 'AAPL', 'MSFT', 'GOOGL'")
    print("- Crypto: 'BTC-USD', 'ETH-USD', 'ADA-USD'")
    print("- Indian Stocks: 'TATAMOTORS.NS', 'RELIANCE.NS', 'INFY.NS'")
    print("- Intervals: '1d', '5m', '15m', '1h', '4h'")
    print("\nExample usage:")
    print("data = fetch_sample_data('BTC-USD', '2023-06-15', '2023-07-15', '15m')")
    print("data = fetch_sample_data('TATAMOTORS.NS', '2023-06-01', '2023-07-20', '5m')")

if __name__ == "__main__":
    main() 