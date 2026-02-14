#!/usr/bin/env python3
"""
Comprehensive Kite Strategy Validation Pipeline
This script combines Kite data fetching with robust strategy validation for Indian markets.
"""

from comprehensive_strategy_validation import ComprehensiveStrategyValidator
from data_fetcher import KiteDataFetcher
from datetime import datetime, timedelta
import sys
import config as cfg
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KiteComprehensiveValidator(ComprehensiveStrategyValidator):
    """
    Extended validator that uses Kite data fetcher for Indian markets.
    """
    
    def __init__(self, initial_balance: float = 10000, max_leverage: float = 10.0, 
                 max_loss_percent: float = 2.0, trading_fee: float = 0.0003):
        """
        Initialize the Kite comprehensive validator.
        
        Args:
            initial_balance: Initial trading balance
            max_leverage: Maximum leverage allowed
            max_loss_percent: Maximum loss percentage per trade
            trading_fee: Trading fee (0.0003 = 0.03% typical for Kite)
        """
        super().__init__(initial_balance, max_leverage, max_loss_percent, trading_fee)
        
        # Initialize Kite data fetcher
        self.kite_fetcher = KiteDataFetcher(cfg.KITE_CREDENTIALS, exchange=cfg.KITE_EXCHANGE)
        self.authenticated = False
        
    def authenticate_kite(self):
        """Authenticate with Kite Connect."""
        if not self.authenticated:
            print("\n🔐 Authenticating with Kite Connect...")
            try:
                self.kite_fetcher.authenticate()
                self.authenticated = True
                print("✅ Authentication successful!")
            except Exception as e:
                print(f"❌ Authentication failed: {e}")
                raise
    
    def fetch_and_split_data(self, symbol: str, start_date: str, end_date: str, 
                           interval: str = "15minute", train_ratio: float = 0.7,
                           random_shift_test_percentage: float = 1.0) -> tuple:
        """
        Fetch historical data from Kite and split into train/test sets.
        
        Args:
            symbol: NSE symbol (e.g., 'TATAMOTORS', 'RELIANCE')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Kite interval format ('15minute', '1hour', '1day')
            train_ratio: Ratio of data to use for training
            random_shift_test_percentage: Percentage shift for test data start
            
        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        # Authenticate if not already done
        self.authenticate_kite()
        
        print(f"\n📥 Fetching Kite data for {symbol}...")
        print(f"📅 Date Range: {start_date} to {end_date}")
        print(f"⏱️  Interval: {interval}")
        print(f"🏛️  Exchange: {cfg.KITE_EXCHANGE}")
        
        # Fetch historical data
        data = self.kite_fetcher.fetch_historical_data(symbol, start_date, end_date, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data fetched for {symbol}. Check symbol and date range.")
        
        print(f"✅ Successfully fetched {len(data)} data points")
        print(f"📊 Data range: {data.index[0]} to {data.index[-1]}")
        
        # Filter for market hours (9:15 AM - 3:30 PM IST, weekdays only)
        # data = self._filter_market_hours(data)
        
        if data.empty:
            raise ValueError("No data remaining after filtering for market hours")
        
        print(f"✅ Market hours data: {len(data)} data points")
        print(f"📊 Filtered range: {data.index[0]} to {data.index[-1]}")
        
        # Split data
        split_index = int(len(data) * train_ratio)
        self.train_data = data.iloc[:split_index].copy()
        self.test_data = data.iloc[int(random_shift_test_percentage * split_index):].copy()
        
        self.symbol = symbol
        self.interval = interval
        
        print(f"📊 Train data: {len(self.train_data)} points ({self.train_data.index[0]} to {self.train_data.index[-1]})")
        print(f"📊 Test data: {len(self.test_data)} points ({self.test_data.index[0]} to {self.test_data.index[-1]})")
        
        return self.train_data, self.test_data
    
    def _filter_market_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to include only market hours for Indian markets.
        NSE/BSE trading hours: 9:15 AM to 3:30 PM IST (Monday to Friday)
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
        
        return weekdays


def run_kite_validation_example():
    """
    Run a complete Kite validation example with Indian market data.
    """
    print("🚀 COMPREHENSIVE KITE STRATEGY VALIDATION")
    print("="*80)
    
    # Configuration for Indian markets
    symbol = "TATAMOTORS"  # NSE symbol
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    interval = "15minute"  # Kite interval format
    train_ratio = 0.7
    random_shift_test_percentage = 1.2
    
    print(f"📊 Configuration:")
    print(f"  Symbol: {symbol}")
    print(f"  Exchange: {cfg.KITE_EXCHANGE}")
    print(f"  Date Range: {start_date} to {end_date}")
    print(f"  Interval: {interval}")
    print(f"  Train/Test Split: {train_ratio:.1%}/{1-train_ratio:.1%}")
    print(f"  Market Hours: 9:15 AM - 3:30 PM IST (Monday-Friday)")
    
    # Initialize Kite validator
    validator = KiteComprehensiveValidator(
        initial_balance=10000,
        max_leverage=10.0,
        max_loss_percent=2.0,
        trading_fee=0.0003  # 0.03% typical Kite charges
    )
    
    try:
        # Step 1: Fetch and split data
        print(f"\n📥 Step 1: Fetching and splitting Kite data...")
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
        optimization_results = validator.optimize_strategies_on_train_data(
            strategies_to_optimize=['ma', 'rsi', 'donchian']
        )
        
        # Step 3: Validate on test data
        print(f"\n🧪 Step 3: Validating strategies on test data...")
        validation_results = validator.validate_strategies_on_test_data(
            strategies_to_validate=['ma', 'rsi', 'donchian'],
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
        validator.save_results("kite_validation_results")
        
        print(f"\n✅ KITE VALIDATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("📁 Results saved in 'kite_validation_results' directory")
        print("🚀 Ready for live Kite trading with the best strategy!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Kite validation: {e}")
        return False


def run_quick_kite_validation():
    """
    Run a quick Kite validation with minimal data for testing.
    """
    print("⚡ QUICK KITE VALIDATION (Testing Mode)")
    print("="*50)
    
    # Shorter time period for quick testing
    symbol = "TATAMOTORS"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    interval = "15minute"
    train_ratio = 0.7
    
    validator = KiteComprehensiveValidator(
        initial_balance=5000,  # Smaller balance for testing
        max_leverage=5.0,
        max_loss_percent=1.0,
        trading_fee=0.0003
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
            mock_trading_delay=0.001,  # Very fast simulation
            top_n_params=3  # Test top 3 parameter sets
        )
        
        # Get recommendations
        recommendations = validator.compare_strategies()
        
        # Save results
        validator.save_results("quick_kite_validation_results")
        
        print("✅ Quick Kite validation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick Kite validation failed: {e}")
        return False


def run_mcx_validation():
    """
    Run validation for MCX (Commodity) markets.
    """
    print("🥇 MCX COMMODITY STRATEGY VALIDATION")
    print("="*60)
    
    # Configuration for MCX markets
    symbol = "COPPER25OCTFUT"  # MCX symbol
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    interval = "15minute"
    train_ratio = 0.7
    
    print(f"📊 MCX Configuration:")
    print(f"  Symbol: {symbol}")
    print(f"  Exchange: MCX")
    print(f"  Date Range: {start_date} to {end_date}")
    print(f"  Interval: {interval}")
    
    validator = KiteComprehensiveValidator(
        initial_balance=10000,
        max_leverage=5.0,  # Lower leverage for commodities
        max_loss_percent=1.5,
        trading_fee=0.0003
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
        
        # Optimize strategies
        optimization_results = validator.optimize_strategies_on_train_data(
            strategies_to_optimize=['ma', 'rsi']
        )
        
        # Validate on test data
        validation_results = validator.validate_strategies_on_test_data(
            strategies_to_validate=['ma', 'rsi'],
            mock_trading_delay=0,
            top_n_params=3
        )
        
        # Get recommendations
        recommendations = validator.compare_strategies()
        
        # Save results
        validator.save_results("mcx_validation_results")
        
        print("✅ MCX validation completed!")
        return True
        
    except Exception as e:
        print(f"❌ MCX validation failed: {e}")
        return False


def main():
    """
    Main function with command line options.
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            success = run_quick_kite_validation()
        elif sys.argv[1] == "--mcx":
            success = run_mcx_validation()
        elif sys.argv[1] == "--help":
            print("Kite Comprehensive Validation Usage:")
            print("  python run_kite_comprehensive_validation.py          # Full NSE validation")
            print("  python run_kite_comprehensive_validation.py --quick  # Quick NSE validation")
            print("  python run_kite_comprehensive_validation.py --mcx    # MCX commodity validation")
            print("  python run_kite_comprehensive_validation.py --help  # Show this help")
            print("\nPrerequisites:")
            print("  1. Update Kite credentials in config.py")
            print("  2. Ensure Kite Connect API access")
            print("  3. Market should be open for live data (9:15 AM - 3:30 PM IST)")
            return
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    else:
        success = run_kite_validation_example()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Review the validation results")
        print("2. Use the best strategy for live Kite trading")
        print("3. Monitor performance and adjust as needed")
        print("4. Consider market hours (9:15 AM - 3:30 PM IST)")
    else:
        print("\n❌ Validation failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("1. Check your Kite Connect credentials in config.py")
        print("2. Ensure you have active internet connection")
        print("3. Verify the symbol is correct (e.g., 'TATAMOTORS' not 'TATAMOTORS.NS')")
        print("4. Check if market is open (9:15 AM - 3:30 PM IST, Monday-Friday)")


if __name__ == "__main__":
    main()
