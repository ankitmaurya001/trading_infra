#!/usr/bin/env python3
"""
Test script to fetch EURUSD data from cTrader Open API
"""

import os
import sys
from datetime import datetime, timedelta
from data_fetcher import CTraderDataFetcher
from config import CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, CTRADER_ACCOUNT_ID, CTRADER_DEMO

def main():
    """
    Test cTrader data fetching for EURUSD
    """
    print("=" * 60)
    print("cTrader Data Fetcher Test")
    print("=" * 60)
    
    # Get credentials from environment variables or config
    # You'll need to set these after setting up your cTrader Open API application
    client_id = CTRADER_CLIENT_ID
    client_secret = CTRADER_CLIENT_SECRET
    access_token = CTRADER_ACCESS_TOKEN
    account_id = CTRADER_ACCOUNT_ID
    demo = CTRADER_DEMO
    
    # Account ID should be numeric - if it's a string, try to convert or warn
    if isinstance(account_id, str) and not account_id.isdigit():
        print(f"\n⚠️  Warning: Account ID '{account_id}' looks like a username, not a numeric account ID")
        print(f"   cTrader Account ID should be a number (e.g., 12345678)")
        print(f"   Check your FTMO dashboard or cTrader account settings for the numeric Account ID")
        print(f"   For now, trying with the provided value...")
    
    # Check if credentials are provided
    if not all([client_id, client_secret, access_token, account_id]):
        print("\n❌ Missing cTrader credentials!")
        print("\nTo use this script, you need to:")
        print("1. Create a cTrader Open API application at: https://openapi.ctrader.com/")
        print("2. Get your client_id, client_secret, and access_token")
        print("3. Get your account_id (ctidTraderAccountId)")
        print("\nThen set environment variables:")
        print("  export CTRADER_CLIENT_ID='your_client_id'")
        print("  export CTRADER_CLIENT_SECRET='your_client_secret'")
        print("  export CTRADER_ACCESS_TOKEN='your_access_token'")
        print("  export CTRADER_ACCOUNT_ID='your_account_id'")
        print("  export CTRADER_DEMO='true'  # or 'false' for live")
        print("\nOr modify this script to hardcode your credentials (not recommended)")
        return
    
    print(f"\n📋 Configuration:")
    print(f"   Client ID: {client_id[:10]}...")
    print(f"   Account ID: {account_id}")
    print(f"   Mode: {'Demo' if demo else 'Live'}")
    
    try:
        # Initialize data fetcher
        print("\n🔌 Initializing cTrader data fetcher...")
        fetcher = CTraderDataFetcher(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            account_id=account_id,
            demo=demo
        )
        print("✅ Data fetcher initialized")
        
        # Test parameters
        symbol = "EURUSD"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 7 days
        interval = "15m"
        
        print(f"\n📥 Fetching {interval} data for {symbol}")
        print(f"   From: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   To: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Fetch data (use full datetime with time for latest data)
        data = fetcher.fetch_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d %H:%M:%S'),
            end_date=end_date.strftime('%Y-%m-%d %H:%M:%S'),
            interval=interval
        )
        
        if data.empty:
            print("\n❌ No data received!")
            print("   Possible issues:")
            print("   - Invalid credentials")
            print("   - Account not authorized")
            print("   - Symbol not available")
            print("   - Network/connection issues")
            return
        
        print(f"\n✅ Successfully fetched {len(data)} data points")
        print(f"\n📊 Data Summary:")
        print(f"   First timestamp: {data.index[0]}")
        print(f"   Last timestamp: {data.index[-1]}")
        print(f"   Price range: {data['Close'].min():.5f} - {data['Close'].max():.5f}")
        print(f"   Latest close: {data['Close'].iloc[-1]:.5f}")
        
        print(f"\n📈 Sample data (last 5 rows):")
        print(data[['Open', 'High', 'Low', 'Close']].tail())
        
        # Test current price
        print(f"\n💰 Fetching current price for {symbol}...")
        current_price = fetcher.get_current_price(symbol)
        if current_price:
            print(f"   Current price: {current_price:.5f}")
        else:
            print("   Could not fetch current price")
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nInstall cTrader Open API SDK:")
        print("  pip install ctrader-open-api")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

