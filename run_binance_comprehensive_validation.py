#!/usr/bin/env python3
"""
Example script demonstrating the comprehensive strategy validation pipeline.
This script shows how to use the ComprehensiveStrategyValidator for robust strategy testing.
"""

from comprehensive_strategy_validation import ComprehensiveStrategyValidator
from datetime import datetime, timedelta
import sys


def run_validation_example():
    """
    Run a complete validation example with different configurations.
    """
    print("🚀 COMPREHENSIVE STRATEGY VALIDATION EXAMPLE")
    print("="*80)
    
    # Configuration
    symbol = "SOLUSDT"
    end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    random_shift_test_percentage = 1.2
    interval = "15m"
    train_ratio = 0.7
    
    print(f"📊 Configuration:")
    print(f"  Symbol: {symbol}")
    print(f"  Date Range: {start_date} to {end_date}")
    print(f"  Interval: {interval}")
    print(f"  Train/Test Split: {train_ratio:.1%}/{1-train_ratio:.1%}")
    
    # Initialize validator
    validator = ComprehensiveStrategyValidator(
        initial_balance=10000,
        max_leverage=10.0,
        max_loss_percent=2.0,
        trading_fee=0.0
    )
    
    try:
        # Step 1: Fetch and split data
        print(f"\n📥 Step 1: Fetching and splitting data...")
        train_data, test_data = validator.fetch_and_split_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            train_ratio=train_ratio,
            random_shift_test_percentage=random_shift_test_percentage
        )
        
        # Step 2: Optimize strategies
        print(f"\n🔍 Step 2: Optimizing strategies on training data...")
        # optimization_results = validator.optimize_strategies_on_train_data(
        #     strategies_to_optimize=['ma', 'rsi', 'donchian']
        # )
        # optimization_results = validator.optimize_strategies_on_train_data(
        #     strategies_to_optimize=['ma', 'rsi', 'donchian']
        # )
        optimization_results = validator.optimize_strategies_on_train_data(
            strategies_to_optimize=['ma']
        )
        
        # Step 3: Validate on test data
        print(f"\n🧪 Step 3: Validating strategies on test data...")
        # validation_results = validator.validate_strategies_on_test_data(
        #     strategies_to_validate=['ma', 'rsi', 'donchian'],
        #     mock_trading_delay=0   # Fast simulation
        # )
        # validation_results = validator.validate_strategies_on_test_data(
        #     strategies_to_validate=['ma', 'rsi', 'donchian'],
        #     mock_trading_delay=0,   # Fast simulation
        #     top_n_params=5  # Test top 5 parameter sets for each strategy
        # )
        validation_results = validator.validate_strategies_on_test_data(
            strategies_to_validate=['ma'],
            mock_trading_delay=0,   # Fast simulation
            top_n_params=5  # Test top 5 parameter sets for each strategy
        )
        
        
        # Step 4: Compare and get recommendations
        print(f"\n📊 Step 4: Comparing strategies...")
        recommendations = validator.compare_strategies()
        
        # Step 5: Get live trading setup
        print(f"\n🚀 Step 5: Preparing live trading setup...")
        live_setup = validator.get_live_trading_setup()
        
        # Step 6: Save results
        print(f"\n💾 Step 6: Saving results...")
        validator.save_results("validation_results")
        
        print(f"\n✅ VALIDATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("📁 Results saved in 'validation_results' directory")
        print("🚀 Ready for live trading with the best strategy!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False


def run_quick_validation():
    """
    Run a quick validation with minimal data for testing.
    """
    print("⚡ QUICK VALIDATION (Testing Mode)")
    print("="*50)
    
    # Shorter time period for quick testing
    symbol = "ETHUSDT"
    end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    interval = "15m"
    train_ratio = 0.7
    
    validator = ComprehensiveStrategyValidator(
        initial_balance=5000,  # Smaller balance for testing
        max_leverage=5.0,
        max_loss_percent=1.0,
        trading_fee=0.001
    )
    
    try:
        # Fetch data
        train_data, test_data = validator.fetch_and_split_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            train_ratio=train_ratio
        )
        
        # Optimize only MA strategy for speed
        optimization_results = validator.optimize_strategies_on_train_data(
            strategies_to_optimize=['ma']
        )
        
        # Validate on test data
        validation_results = validator.validate_strategies_on_test_data(
            strategies_to_validate=['ma'],
            mock_trading_delay=0.001  # Very fast simulation
        )
        
        # Get recommendations
        recommendations = validator.compare_strategies()
        
        # Save results
        validator.save_results("quick_validation_results")
        
        print("✅ Quick validation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick validation failed: {e}")
        return False


def main():
    """
    Main function with command line options.
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            success = run_quick_validation()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python run_comprehensive_validation.py          # Full validation")
            print("  python run_comprehensive_validation.py --quick  # Quick validation")
            print("  python run_comprehensive_validation.py --help  # Show this help")
            return
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    else:
        success = run_validation_example()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Review the validation results")
        print("2. Use the best strategy for live trading")
        print("3. Monitor performance and adjust as needed")
    else:
        print("\n❌ Validation failed. Check the error messages above.")


if __name__ == "__main__":
    main()
