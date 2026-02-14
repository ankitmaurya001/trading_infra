#!/usr/bin/env python3
"""
Kite-specific comprehensive strategy validation with Indian market features.
Extends the base comprehensive validation with Kite Connect integration.
"""

from comprehensive_strategy_validation import ComprehensiveStrategyValidator
from data_fetcher import KiteDataFetcher
import pandas as pd
import config as cfg
import logging

logger = logging.getLogger(__name__)


class KiteComprehensiveStrategyValidator(ComprehensiveStrategyValidator):
    """
    Comprehensive strategy validator specifically designed for Kite Connect (Indian markets).
    
    Features:
    - Kite Connect authentication
    - Indian market hours filtering (9:15 AM - 3:30 PM IST)
    - Weekday-only trading
    - MCX commodity support
    - NSE/BSE equity support
    """
    
    def __init__(self, initial_balance: float = 10000, max_leverage: float = 10.0, 
                 max_loss_percent: float = 2.0, trading_fee: float = 0.0003,
                 exchange: str = "NSE"):
        """
        Initialize the Kite comprehensive validator.
        
        Args:
            initial_balance: Initial trading balance
            max_leverage: Maximum leverage allowed
            max_loss_percent: Maximum loss percentage per trade
            trading_fee: Trading fee (0.0003 = 0.03% typical for Kite)
            exchange: Exchange to use ("NSE", "BSE", "MCX")
        """
        super().__init__(initial_balance, max_leverage, max_loss_percent, trading_fee)
        
        # Initialize Kite data fetcher
        self.exchange = exchange
        self.kite_fetcher = KiteDataFetcher(cfg.KITE_CREDENTIALS, exchange=exchange)
        self.authenticated = False
        
        # Indian market specific settings
        self.market_hours_start = "09:15"
        self.market_hours_end = "15:30"
        self.timezone = "Asia/Kolkata"
        
    def authenticate_kite(self):
        """Authenticate with Kite Connect."""
        if not self.authenticated:
            print(f"\n🔐 Authenticating with Kite Connect ({self.exchange})...")
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
            symbol: Exchange symbol (e.g., 'TATAMOTORS', 'RELIANCE', 'COPPER25OCTFUT')
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
        print(f"🏛️  Exchange: {self.exchange}")
        
        # Fetch historical data
        data = self.kite_fetcher.fetch_historical_data(symbol, start_date, end_date, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data fetched for {symbol}. Check symbol and date range.")
        
        print(f"✅ Successfully fetched {len(data)} data points")
        print(f"📊 Data range: {data.index[0]} to {data.index[-1]}")
        
        # Filter for market hours based on exchange
        if self.exchange in ["NSE", "BSE"]:
            data = self._filter_equity_market_hours(data)
        elif self.exchange == "MCX":
            data = self._filter_mcx_market_hours(data)
        else:
            print(f"⚠️  Unknown exchange {self.exchange}, skipping market hours filter")
        
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
    
    def _filter_equity_market_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data for NSE/BSE equity market hours.
        Trading hours: 9:15 AM to 3:30 PM IST (Monday to Friday)
        """
        if data.empty:
            return data
        
        # Ensure we have timezone-aware data
        if data.index.tz is None:
            data.index = data.index.tz_localize(self.timezone)
        else:
            data.index = data.index.tz_convert(self.timezone)
        
        # Filter for market hours (9:15 AM to 3:30 PM IST)
        market_hours = data.between_time(self.market_hours_start, self.market_hours_end)
        
        # Filter for weekdays only (Monday=0, Sunday=6)
        weekdays = market_hours[market_hours.index.weekday < 5]
        
        print(f"🕒 Filtered for NSE/BSE market hours: {len(data)} → {len(weekdays)} points")
        
        return weekdays
    
    def _filter_mcx_market_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data for MCX commodity market hours.
        MCX trading hours: 9:00 AM to 11:30 PM IST (Monday to Friday)
        """
        if data.empty:
            return data
        
        # Ensure we have timezone-aware data
        if data.index.tz is None:
            data.index = data.index.tz_localize(self.timezone)
        else:
            data.index = data.index.tz_convert(self.timezone)
        
        # MCX has extended hours: 9:00 AM to 11:30 PM IST
        market_hours = data.between_time("09:00", "23:30")
        
        # Filter for weekdays only (Monday=0, Sunday=6)
        weekdays = market_hours[market_hours.index.weekday < 5]
        
        print(f"🕒 Filtered for MCX market hours: {len(data)} → {len(weekdays)} points")
        
        return weekdays
    
    def get_market_info(self) -> dict:
        """
        Get information about the current market configuration.
        
        Returns:
            Dictionary with market information
        """
        return {
            "exchange": self.exchange,
            "market_hours": f"{self.market_hours_start} - {self.market_hours_end} IST" if self.exchange in ["NSE", "BSE"] else "09:00 - 23:30 IST",
            "timezone": self.timezone,
            "trading_days": "Monday - Friday",
            "trading_fee": f"{self.trading_fee * 100:.3f}%",
            "authenticated": self.authenticated
        }
    
    def save_results(self, output_dir: str = "kite_validation_results"):
        """
        Save validation results with Kite-specific information.
        
        Args:
            output_dir: Directory to save results
        """
        # Add market info to results
        market_info = self.get_market_info()
        
        # Call parent save method
        super().save_results(output_dir)
        
        # Save additional Kite-specific information
        import json
        import os
        
        kite_info_file = os.path.join(output_dir, "kite_market_info.json")
        with open(kite_info_file, 'w') as f:
            json.dump(market_info, f, indent=2, default=str)
        
        print(f"💾 Kite market info saved to {kite_info_file}")


def create_kite_validator_for_exchange(exchange: str, **kwargs) -> KiteComprehensiveStrategyValidator:
    """
    Factory function to create a Kite validator for a specific exchange.
    
    Args:
        exchange: Exchange name ("NSE", "BSE", "MCX")
        **kwargs: Additional arguments for validator initialization
        
    Returns:
        Configured KiteComprehensiveStrategyValidator instance
    """
    # Set default parameters based on exchange
    defaults = {
        "NSE": {
            "max_leverage": 10.0,
            "max_loss_percent": 2.0,
            "trading_fee": 0.0003
        },
        "BSE": {
            "max_leverage": 10.0,
            "max_loss_percent": 2.0,
            "trading_fee": 0.0003
        },
        "MCX": {
            "max_leverage": 5.0,  # Lower leverage for commodities
            "max_loss_percent": 1.5,
            "trading_fee": 0.0003
        }
    }
    
    # Merge defaults with provided kwargs
    config = defaults.get(exchange, defaults["NSE"]).copy()
    config.update(kwargs)
    
    return KiteComprehensiveStrategyValidator(exchange=exchange, **config)


# Example usage functions
def validate_nse_stock(symbol: str = "TATAMOTORS", days_back: int = 30):
    """Validate strategies for NSE stock."""
    validator = create_kite_validator_for_exchange("NSE")
    
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    return validator.fetch_and_split_data(symbol, start_date, end_date)


def validate_mcx_commodity(symbol: str = "COPPER25OCTFUT", days_back: int = 30):
    """Validate strategies for MCX commodity."""
    validator = create_kite_validator_for_exchange("MCX")
    
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    return validator.fetch_and_split_data(symbol, start_date, end_date)


if __name__ == "__main__":
    # Example usage
    print("Kite Comprehensive Strategy Validator")
    print("="*50)
    
    # Test with NSE
    try:
        print("\n📈 Testing NSE validation...")
        validator = create_kite_validator_for_exchange("NSE")
        market_info = validator.get_market_info()
        print(f"Market Info: {market_info}")
    except Exception as e:
        print(f"NSE test failed: {e}")
    
    # Test with MCX
    try:
        print("\n🥇 Testing MCX validation...")
        validator = create_kite_validator_for_exchange("MCX")
        market_info = validator.get_market_info()
        print(f"Market Info: {market_info}")
    except Exception as e:
        print(f"MCX test failed: {e}")
