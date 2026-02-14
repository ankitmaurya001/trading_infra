#!/usr/bin/env python3
"""
Example: Modular Trading System Usage
Demonstrates how to use the new modular components for optimization and trading
"""

from strategy_manager import StrategyManager
from trading_engine import TradingEngine
from optimization_runner import run_optimization, get_optimization_parameters
from data_fetcher import DataFetcher
import pandas as pd
from datetime import datetime, timedelta

def example_1_optimization():
    """
    Example 1: Run optimization using the same approach as simple_optimization_example.py
    """
    print("=" * 60)
    print("EXAMPLE 1: Optimization")
    print("=" * 60)
    
    # Run optimization using the same approach as simple_optimization_example.py
    results = run_optimization(
        symbol="BTC-USD",
        start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d"),
        interval="15m",
        enabled_strategies=['ma', 'rsi'],
        trading_fee=0,
        sharpe_threshold=0.1
    )
    
    if 'error' in results:
        print(f"❌ Optimization failed: {results['error']}")
        return results
    
    # Extract parameters for each strategy
    ma_params = get_optimization_parameters(results, 'ma')
    rsi_params = get_optimization_parameters(results, 'rsi')
    
    print(f"\n📊 Moving Average Parameters:")
    for key, value in ma_params.items():
        print(f"   {key}: {value}")
    
    print(f"\n📊 RSI Parameters:")
    for key, value in rsi_params.items():
        print(f"   {key}: {value}")
    
    return results

def example_2_strategy_manager_usage():
    """
    Example 2: Using StrategyManager for strategy setup
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: StrategyManager Usage")
    print("=" * 60)
    
    # Initialize strategy manager
    strategy_manager = StrategyManager()
    
    # Set manual parameters (from optimization results)
    strategy_manager.set_manual_parameters(
        ma_params={'short_window': 20, 'long_window': 50, 'risk_reward_ratio': 2.0, 'trading_fee': 0.001},
        rsi_params={'period': 14, 'overbought': 70, 'oversold': 30, 'risk_reward_ratio': 2.0, 'trading_fee': 0.001}
    )
    
    # Initialize strategies
    strategies = strategy_manager.initialize_strategies(['ma', 'rsi'])
    print(f"✅ Initialized {len(strategies)} strategies:")
    for strategy in strategies:
        print(f"   - {strategy.name}")
    
    # Get strategy parameters
    params = strategy_manager.get_all_parameters()
    print(f"\n📋 Strategy Parameters:")
    for strategy_name, strategy_params in params.items():
        print(f"   {strategy_name.upper()}: {strategy_params}")
    
    return strategy_manager

def example_3_trading_engine_usage():
    """
    Example 3: Using TradingEngine for trade execution and performance tracking
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: TradingEngine Usage")
    print("=" * 60)
    
    # Initialize trading engine
    trading_engine = TradingEngine(initial_balance=10000)
    
    # Setup logging
    session_id = f"example_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trading_engine.setup_logging(session_id, "BTC-USD")
    
    # Get current status
    status = trading_engine.get_current_status()
    print(f"💰 Initial Balance: ${status['current_balance']:,.2f}")
    print(f"📊 Can Trade: {status['can_trade']}")
    
    # Get performance metrics
    metrics = trading_engine.calculate_performance_metrics()
    print(f"\n📊 Performance Metrics:")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.2%}")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    
    return trading_engine

def example_4_integrated_workflow():
    """
    Example 4: Complete integrated workflow from optimization to trading
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Integrated Workflow")
    print("=" * 60)
    
    # Step 1: Run optimization
    print("Step 1: Running optimization...")
    results = run_optimization(
        symbol="BTC-USD",
        start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d"),
        interval="15m",
        enabled_strategies=['ma', 'rsi'],
        trading_fee=0.001,
        sharpe_threshold=0.1
    )
    
    if 'error' in results:
        print(f"❌ Optimization failed: {results['error']}")
        return
    
    # Step 2: Setup strategy manager with optimized parameters
    print("\nStep 2: Setting up strategy manager...")
    strategy_manager = StrategyManager()
    
    # Extract and set parameters
    ma_params = get_optimization_parameters(results, 'ma')
    rsi_params = get_optimization_parameters(results, 'rsi')
    
    if ma_params:
        strategy_manager.set_manual_parameters(ma_params=ma_params)
    if rsi_params:
        strategy_manager.set_manual_parameters(rsi_params=rsi_params)
    
    # Initialize strategies
    strategies = strategy_manager.initialize_strategies(['ma', 'rsi'])
    print(f"✅ Initialized {len(strategies)} strategies")
    
    # Step 3: Setup trading engine
    print("\nStep 3: Setting up trading engine...")
    trading_engine = TradingEngine(initial_balance=10000)
    
    session_id = f"integrated_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trading_engine.setup_logging(session_id, "BTC-USD")
    
    # Step 4: Fetch some data for demonstration
    print("\nStep 4: Fetching sample data...")
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_data(
        symbol="BTC-USD",
        start_date=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d"),
        interval="15m"
    )
    
    if not data.empty:
        print(f"✅ Fetched {len(data)} data points")
        print(f"📊 Data range: {data.index[0]} to {data.index[-1]}")
        
        # Step 5: Process strategies on sample data
        print("\nStep 5: Processing strategies on sample data...")
        current_time = datetime.now()
        
        for strategy in strategies:
            print(f"\n🎯 Processing {strategy.name}...")
            result = trading_engine.process_strategy_signals(strategy, data, current_time)
            
            if result['signal'] is not None:
                print(f"   Signal: {result['signal_name']}")
                print(f"   Action: {result['action']}")
                print(f"   Trades Executed: {len(result['trades_executed'])}")
        
        # Step 6: Get final status and metrics
        print("\nStep 6: Final status and metrics...")
        status = trading_engine.get_current_status(data)
        metrics = trading_engine.calculate_performance_metrics()
        
        print(f"💰 Current Balance: ${status['current_balance']:,.2f}")
        print(f"📈 Total PnL: ${status['total_pnl']:,.2f}")
        print(f"📊 Total Trades: {status['total_trades']}")
        print(f"🎯 Win Rate: {metrics['win_rate']:.2%}")
        print(f"📈 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        
        # Step 7: Export results
        print("\nStep 7: Exporting results...")
        strategy_manager.export_optimization_results("example_optimization_results.json")
        
        print("\n✅ Integrated workflow completed successfully!")
        
    else:
        print("❌ Failed to fetch data")
    
    return {
        'strategy_manager': strategy_manager,
        'trading_engine': trading_engine,
        'optimization_results': results
    }

def example_5_simplified_live_trading():
    """
    Example 5: Using the simplified live trading simulator
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Simplified Live Trading")
    print("=" * 60)
    
    try:
        from simplified_live_trading import SimplifiedLiveTradingSimulator
        
        # Initialize simulator
        simulator = SimplifiedLiveTradingSimulator(initial_balance=10000)
        
        # Setup strategies with optimization
        print("Setting up strategies with optimization...")
        success = simulator.optimize_and_setup_strategies(
            symbol="BTC-USD",
            enabled_strategies=['ma', 'rsi'],
            auto_optimize=True
        )
        
        if success:
            print("✅ Strategies setup completed!")
            
            # Get strategy parameters
            params = simulator.get_strategy_parameters()
            print(f"📋 Strategy Parameters: {list(params.keys())}")
            
            # Get current status
            status = simulator.get_current_status()
            print(f"💰 Initial Balance: ${status['current_balance']:,.2f}")
            
            print("✅ Simplified live trading simulator is ready!")
            print("💡 To start trading, use: simulator.start_live_trading(...)")
            
        else:
            print("❌ Strategy setup failed!")
            
        return simulator
        
    except ImportError:
        print("⚠️  Simplified live trading simulator not available")
        return None

def main():
    """
    Run all examples
    """
    print("🚀 MODULAR TRADING SYSTEM EXAMPLES")
    print("=" * 80)
    
    try:
        # Example 1: Optimization
        example_1_optimization()
        
        # Example 2: Strategy manager usage
        example_2_strategy_manager_usage()
        
        # Example 3: Trading engine usage
        example_3_trading_engine_usage()
        
        # Example 4: Integrated workflow
        example_4_integrated_workflow()
        
        # Example 5: Simplified live trading
        example_5_simplified_live_trading()
        
        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📚 Key Benefits of the Modular System:")
        print("   1. 🎯 StrategyManager: Centralized strategy optimization and management")
        print("   2. 🔄 TradingEngine: Clean separation of trading logic")
        print("   3. ⚡ OptimizationRunner: Simple optimization interface (matches simple_optimization_example.py)")
        print("   4. 📊 Modular Design: Easy to extend and maintain")
        print("   5. 🔧 Reusable Components: Use components independently")
        print("\n🚀 Next Steps:")
        print("   1. Run optimization: python optimization_runner.py --symbol BTC-USD --strategies ma rsi")
        print("   2. Start trading: streamlit run simplified_live_trading.py")
        print("   3. Use components: from strategy_manager import StrategyManager")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
