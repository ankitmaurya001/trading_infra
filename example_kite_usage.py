#!/usr/bin/env python3
"""
Example usage of KiteDataFetcher for Zerodha Kite Connect
This script demonstrates how to fetch data from Indian markets using Kite Connect API
"""

import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from data_fetcher import KiteDataFetcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your Kite Connect API credentials
API_KEY = "wu80p2aelj2d73v5"
API_SECRET = "xorznn9fcocx1xww4uflrqprsorrij4t"

# Your Zerodha login credentials (for automated authentication)
USERNAME = "KMX177"  # Replace with your actual username
PASSWORD = "Jack298!"  # Replace with your actual password
TOTP_KEY = "N6RGEW7E5VBDTGBLFOONFU3KOQZGR27G"  # Replace with your actual TOTP key

credentials={"username":USERNAME, 
             "password" : PASSWORD,
            "api_key":API_KEY,
             "api_secret": API_SECRET,
            "totp_key": TOTP_KEY}

def main():
    """
    Main function demonstrating Kite Connect usage
    """
    
    # Initialize the Kite Data Fetcher
    # try:
    kite_fetcher = KiteDataFetcher(credentials)
    kite_fetcher.authenticate()
    logger.info("KiteDataFetcher initialized successfully")
   
    # except Exception as e:
    #     logger.error(f"Error initializing KiteDataFetcher: {e}")
    #     return
    
    
    # Step 2: Fetch TATAMOTORS data
    # Set date range for the last 30 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    logger.info(f"Fetching TATAMOTORS data from {start_date} to {end_date}")
    
    # Fetch TATAMOTORS data with 15-minute intervals
    tatamotors_data = kite_fetcher.fetch_historical_data(
        symbol="TATAMOTORS",
        start_date=start_date,
        end_date=end_date,
        interval="15minute"
    )
    
    if not tatamotors_data.empty:
        logger.info(f"Successfully fetched {len(tatamotors_data)} records for TATAMOTORS")
        logger.info(f"Data shape: {tatamotors_data.shape}")
        logger.info(f"Columns: {tatamotors_data.columns.tolist()}")
        logger.info(f"Date range: {tatamotors_data.index.min()} to {tatamotors_data.index.max()}")
        
        # Display first few rows
        print("\nFirst 5 rows of TATAMOTORS data:")
        print(tatamotors_data.head())
        
        # Display last few rows
        print("\nLast 5 rows of TATAMOTORS data:")
        print(tatamotors_data.tail())
        
        # Basic statistics
        print("\nBasic statistics:")
        print(tatamotors_data.describe())
        
        # Save data to CSV
        filename = f"TATAMOTORS_{start_date}_to_{end_date}.csv"
        tatamotors_data.to_csv(filename)
        logger.info(f"Data saved to {filename}")
    else:
        logger.warning("No data received for TATAMOTORS")
    
    # Step 3: Example of fetching data for other symbols
    other_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']
    
    for symbol in other_symbols:
        logger.info(f"Fetching data for {symbol}")
        try:
            data = kite_fetcher.fetch_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="day"
            )
            
            if not data.empty:
                logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                print(f"\n{symbol} - Latest Close: {data['Close'].iloc[-1]:.2f}")
                
                # Save individual symbol data
                symbol_filename = f"{symbol}_{start_date}_to_{end_date}.csv"
                data.to_csv(symbol_filename)
                logger.info(f"{symbol} data saved to {symbol_filename}")
            else:
                logger.warning(f"No data received for {symbol}")
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    # Step 4: Example of getting real-time quotes
    logger.info("Fetching real-time quotes...")
    for symbol in ['TATAMOTORS', 'RELIANCE']:
        try:
            quote = kite_fetcher.get_quote(symbol)
            if quote:
                logger.info(f"Quote for {symbol}: {quote}")
            
            ltp = kite_fetcher.get_ltp(symbol)
            if ltp:
                logger.info(f"LTP for {symbol}: {ltp}")
                
        except Exception as e:
            logger.error(f"Error getting quote/LTP for {symbol}: {e}")
    
    logger.info("Data fetching completed successfully!")

def demo_with_mock_data():
    """
    Demo function showing the expected data structure
    """
    logger.info("Demonstrating expected data structure with mock data...")
    
    # Create sample data similar to what Kite Connect would return
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2025-01-01', periods=100, freq='15min'),
        'Open': [100 + i * 0.1 for i in range(100)],
        'High': [101 + i * 0.1 for i in range(100)],
        'Low': [99 + i * 0.1 for i in range(100)],
        'Close': [100.5 + i * 0.1 for i in range(100)],
        'Volume': [1000 + i * 10 for i in range(100)]
    })
    
    sample_data = sample_data.set_index('Date')
    
    logger.info(f"Sample data shape: {sample_data.shape}")
    logger.info(f"Sample data columns: {sample_data.columns.tolist()}")
    
    print("\nSample TATAMOTORS data structure:")
    print(sample_data.head(10))
    
    # Calculate some basic metrics
    print(f"\nPrice range: {sample_data['Low'].min():.2f} - {sample_data['High'].max():.2f}")
    print(f"Average volume: {sample_data['Volume'].mean():.0f}")
    print(f"Total volume: {sample_data['Volume'].sum():,.0f}")

def setup_credentials():
    """
    Interactive setup for credentials
    """
    print("\n" + "=" * 60)
    print("Credential Setup")
    print("=" * 60)
    
    global USERNAME, PASSWORD, TOTP_KEY
    
    print("Please enter your Zerodha credentials:")
    USERNAME = input("Username: ").strip()
    PASSWORD = input("Password: ").strip()
    TOTP_KEY = input("TOTP Secret Key: ").strip()
    
    print("\nCredentials set successfully!")
    print(f"Username: {USERNAME}")
    print(f"TOTP enabled: {'Yes' if TOTP_KEY else 'No'}")

if __name__ == "__main__":
    print("=" * 60)
    print("Kite Connect Data Fetcher Example")
    print("=" * 60)
    
    # Check if credentials are set
    # if USERNAME == "KMX177":  # Default placeholder
    #     print("Default credentials detected. Setting up credentials...")
    #     setup_credentials()
    
    # Run the main example
    main()
    
    print("\n" + "=" * 60)
    print("Mock Data Demo")
    print("=" * 60)
    
    # Run the mock data demo
    demo_with_mock_data()
    
    print("\n" + "=" * 60)
    print("Setup Instructions")
    print("=" * 60)
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set your Zerodha credentials in the script or use interactive setup")
    print("3. Run the script for automated authentication and data fetching")
    print("4. Data will be automatically saved to CSV files")
