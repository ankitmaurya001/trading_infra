#!/usr/bin/env python3
"""
Simple test script to verify Binance testnet balance retrieval
"""
import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brokers.binance_spot import BinanceSpotBroker
import config as cfg


def test_balance_retrieval():
    """Test basic balance retrieval from Binance testnet"""
    print("🔍 Testing Binance Testnet Balance Retrieval")
    print("=" * 50)
    
    # Initialize broker
    api_key = os.environ.get("BINANCE_API_KEY") or cfg.BINANCE_API_KEY
    api_secret = os.environ.get("BINANCE_API_SECRET") or cfg.BINANCE_SECRET_KEY
    
    print(f"📡 Using API Key: {api_key[:10]}...")
    print(f"🌐 Environment: TESTNET")
    
    try:
        broker = BinanceSpotBroker(api_key=api_key, api_secret=api_secret, testnet=True)
        
        # Test 1: Ping
        print("\n1️⃣ Testing connection ping...")
        ping_result = broker.ping()
        print(f"   Ping result: {'✅ SUCCESS' if ping_result else '❌ FAILED'}")
        
        if not ping_result:
            print("❌ Cannot proceed - connection failed")
            return False
            
        # Test 2: Get account info
        print("\n2️⃣ Testing account info retrieval...")
        account = broker.get_account()
        print(f"   Account permissions: {account.get('permissions', [])}")
        print(f"   Account type: {account.get('accountType', 'Unknown')}")
        print(f"   Can trade: {account.get('canTrade', False)}")
        
        # Test 3: Get balances
        print("\n3️⃣ Testing balance retrieval...")
        balances = broker.get_balances()
        
        if not balances:
            print("   ⚠️  No balances found (this is normal for new testnet accounts)")
        else:
            print(f"   Found {len(balances)} assets with balances:")
            for asset, balance in sorted(balances.items()):
                print(f"     {asset}: {balance}")
        
        # Test 4: Get specific symbol info
        print("\n4️⃣ Testing symbol info retrieval...")
        try:
            symbol_info = broker.client.get_symbol_info('BTCUSDT')
            print(f"   BTCUSDT symbol info retrieved successfully")
            print(f"   Status: {symbol_info.get('status')}")
            print(f"   Base asset: {symbol_info.get('baseAsset')}")
            print(f"   Quote asset: {symbol_info.get('quoteAsset')}")
        except Exception as e:
            print(f"   ❌ Failed to get symbol info: {e}")
            
        print("\n✅ Balance retrieval test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print(f"🚀 Binance Testnet API Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    success = test_balance_retrieval()
    
    if success:
        print("\n🎉 All balance tests passed!")
    else:
        print("\n💥 Balance tests failed!")
        sys.exit(1)
