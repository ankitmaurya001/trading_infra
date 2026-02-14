#!/usr/bin/env python3
"""
Simple example demonstrating Kite comprehensive strategy validation.
This shows how to use the Kite validation pipeline for Indian markets.
"""

from kite_comprehensive_strategy_validation import KiteComprehensiveStrategyValidator, create_kite_validator_for_exchange
from datetime import datetime, timedelta
import sys


def example_nse_validation():
    """
    Example: Validate strategies for NSE stock (TATAMOTORS).
    """
    print("📈 NSE STOCK VALIDATION EXAMPLE")
    print("="*50)
    
    # Create NSE validator
    validator = create_kite_validator_for_exchange("NSE")
    
    # Configuration
    symbol = "TATAMOTORS"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        # Fetch and split data
        print(f"📥 Fetching data for {symbol}...")
        train_data, test_data = validator.fetch_and_split_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="15minute",
            train_ratio=0.7
        )
        
        # Optimize strategies
        print(f"\n🔍 Optimizing strategies...")
        optimization_results = validator.optimize_strategies_on_train_data(
            strategies_to_optimize=['ma', 'rsi']
        )
        
        # Validate on test data
        print(f"\n🧪 Validating on test data...")
        validation_results = validator.validate_strategies_on_test_data(
            strategies_to_validate=['ma', 'rsi'],
            mock_trading_delay=0,
            top_n_params=3
        )
        
        # Compare strategies
        print(f"\n📊 Comparing strategies...")
        recommendations = validator.compare_strategies()
        
        # Save results
        validator.save_results("nse_example_results")
        
        print("✅ NSE validation completed!")
        return True
        
    except Exception as e:
        print(f"❌ NSE validation failed: {e}")
        return False


def example_mcx_validation():
    """
    Example: Validate strategies for MCX commodity (COPPER).
    """
    print("\n🥇 MCX COMMODITY VALIDATION EXAMPLE")
    print("="*50)
    
    # Create MCX validator
    validator = create_kite_validator_for_exchange("MCX")
    
    # Configuration
    symbol = "COPPER25OCTFUT"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        # Fetch and split data
        print(f"📥 Fetching data for {symbol}...")
        train_data, test_data = validator.fetch_and_split_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="15minute",
            train_ratio=0.7
        )
        
        # Optimize strategies
        print(f"\n🔍 Optimizing strategies...")
        optimization_results = validator.optimize_strategies_on_train_data(
            strategies_to_optimize=['ma']
        )
        
        # Validate on test data
        print(f"\n🧪 Validating on test data...")
        validation_results = validator.validate_strategies_on_test_data(
            strategies_to_validate=['ma'],
            mock_trading_delay=0,
            top_n_params=3
        )
        
        # Compare strategies
        print(f"\n📊 Comparing strategies...")
        recommendations = validator.compare_strategies()
        
        # Save results
        validator.save_results("mcx_example_results")
        
        print("✅ MCX validation completed!")
        return True
        
    except Exception as e:
        print(f"❌ MCX validation failed: {e}")
        return False


def example_custom_validation():
    """
    Example: Custom validation with specific parameters.
    """
    print("\n⚙️ CUSTOM VALIDATION EXAMPLE")
    print("="*50)
    
    # Create custom validator
    validator = KiteComprehensiveStrategyValidator(
        initial_balance=50000,  # Higher balance
        max_leverage=5.0,       # Conservative leverage
        max_loss_percent=1.0,   # Strict risk management
        trading_fee=0.0003,     # Kite fees
        exchange="NSE"
    )
    
    # Show market info
    market_info = validator.get_market_info()
    print(f"Market Configuration: {market_info}")
    
    # Configuration
    symbol = "RELIANCE"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')  # Longer period
    
    try:
        # Fetch and split data with custom parameters
        print(f"📥 Fetching data for {symbol}...")
        train_data, test_data = validator.fetch_and_split_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1hour",  # Hourly data
            train_ratio=0.8,   # More training data
            random_shift_test_percentage=1.1  # Slight overlap
        )
        
        # Optimize only MA strategy
        print(f"\n🔍 Optimizing MA strategy...")
        optimization_results = validator.optimize_strategies_on_train_data(
            strategies_to_optimize=['ma']
        )
        
        # Validate with top 5 parameters
        print(f"\n🧪 Validating with top 5 parameters...")
        validation_results = validator.validate_strategies_on_test_data(
            strategies_to_validate=['ma'],
            mock_trading_delay=0,
            top_n_params=5
        )
        
        # Get live trading setup
        print(f"\n🚀 Getting live trading setup...")
        live_setup = validator.get_live_trading_setup()
        
        # Save results
        validator.save_results("custom_example_results")
        
        print("✅ Custom validation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Custom validation failed: {e}")
        return False


def main():
    """
    Main function to run examples.
    """
    print("🚀 KITE COMPREHENSIVE VALIDATION EXAMPLES")
    print("="*60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--nse":
            success = example_nse_validation()
        elif sys.argv[1] == "--mcx":
            success = example_mcx_validation()
        elif sys.argv[1] == "--custom":
            success = example_custom_validation()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python kite_validation_example.py --nse     # NSE stock validation")
            print("  python kite_validation_example.py --mcx     # MCX commodity validation")
            print("  python kite_validation_example.py --custom  # Custom validation")
            print("  python kite_validation_example.py --help    # Show this help")
            print("\nPrerequisites:")
            print("  1. Update Kite credentials in config.py")
            print("  2. Ensure Kite Connect API access")
            return
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    else:
        # Run all examples
        print("Running all examples...")
        success1 = example_nse_validation()
        success2 = example_mcx_validation()
        success3 = example_custom_validation()
        success = success1 and success2 and success3
    
    if success:
        print("\n🎯 All examples completed successfully!")
        print("📁 Check the result directories for detailed outputs")
    else:
        print("\n❌ Some examples failed. Check the error messages above.")


if __name__ == "__main__":
    main()
