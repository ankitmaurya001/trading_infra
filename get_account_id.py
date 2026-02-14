#!/usr/bin/env python3
"""
Simple script to get cTrader Account ID using access token
Run this after you have CTRADER_ACCESS_TOKEN in config.py
"""

import sys
import config as cfg
from data_fetcher import CTraderDataFetcher

def main():
    """Get account ID from cTrader"""
    print("=" * 70)
    print("cTrader Account ID Fetcher")
    print("=" * 70)
    
    # Check if we have required credentials
    if not cfg.CTRADER_CLIENT_ID or not cfg.CTRADER_CLIENT_SECRET:
        print("\n❌ Missing CLIENT_ID or CLIENT_SECRET in config.py")
        return
    
    if not cfg.CTRADER_ACCESS_TOKEN:
        print("\n❌ Missing CTRADER_ACCESS_TOKEN in config.py")
        print("   Run: python get_ctrader_token.py first")
        return
    
    print(f"\n📋 Using credentials from config.py")
    print(f"   Client ID: {cfg.CTRADER_CLIENT_ID[:20]}...")
    print(f"   Access Token: {cfg.CTRADER_ACCESS_TOKEN[:30]}...")
    
    # Try to get account ID by attempting to connect
    # We'll use a dummy account ID first, then the SDK will tell us the real one
    print(f"\n🔌 Attempting to connect to cTrader...")
    
    # Try demo first, then live
    for demo_mode in [True, False]:
        mode_name = "DEMO" if demo_mode else "LIVE"
        print(f"\n   Trying {mode_name} server...")
        
        try:
            # Initialize fetcher (we'll use a dummy account ID)
            fetcher = CTraderDataFetcher(
                client_id=cfg.CTRADER_CLIENT_ID,
                client_secret=cfg.CTRADER_CLIENT_SECRET,
                access_token=cfg.CTRADER_ACCESS_TOKEN,
                account_id=0,  # Dummy - we'll discover the real one
                demo=demo_mode
            )
            
            # Try to get symbol list (this will authenticate and reveal account info)
            print(f"   Fetching symbol list (this will authenticate)...")
            
            # This is a workaround - we'll try to fetch a small amount of data
            # which will trigger authentication and reveal account info
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            # Try fetching EURUSD data (this will authenticate)
            data = fetcher.fetch_historical_data(
                symbol='EURUSD',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='1h'
            )
            
            if not data.empty:
                print(f"\n✅ Successfully connected to {mode_name} server!")
                print(f"   This means your account is accessible")
                print(f"\n📝 To find your Account ID:")
                print(f"   1. Log into your cTrader account")
                print(f"   2. Go to Account Settings")
                print(f"   3. Look for 'Account ID' or 'ctidTraderAccountId'")
                print(f"   4. It's usually a 6-8 digit number")
                print(f"\n   Or check your FTMO dashboard for the account number")
                return
            
        except Exception as e:
            print(f"   Error: {e}")
            continue
    
    print(f"\n⚠️  Could not automatically determine account ID")
    print(f"\n📝 Manual steps to find Account ID:")
    print(f"   1. Log into your cTrader account (web or desktop)")
    print(f"   2. Go to Account Settings / Profile")
    print(f"   3. Look for 'Account ID', 'Account Number', or 'ctidTraderAccountId'")
    print(f"   4. It's usually a 6-8 digit number like: 12345678")
    print(f"\n   Or:")
    print(f"   - Check your FTMO dashboard")
    print(f"   - Check your account statement")
    print(f"   - Contact FTMO support if you can't find it")

if __name__ == "__main__":
    main()

