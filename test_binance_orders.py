#!/usr/bin/env python3
"""
Test script to verify Binance testnet order placement and management
"""
import os
import sys
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brokers.binance_spot import BinanceSpotBroker
import config as cfg


def test_order_operations():
    """Test order placement, confirmation, and management"""
    print("🔍 Testing Binance Testnet Order Operations")
    print("=" * 50)
    
    # Initialize broker
    api_key = os.environ.get("BINANCE_API_KEY") or cfg.BINANCE_API_KEY
    api_secret = os.environ.get("BINANCE_API_SECRET") or cfg.BINANCE_SECRET_KEY
    
    print(f"📡 Using API Key: {api_key[:10]}...")
    print(f"🌐 Environment: TESTNET")
    
    try:
        broker = BinanceSpotBroker(api_key=api_key, api_secret=api_secret, testnet=True)
        
        # Test 1: Connection
        print("\n1️⃣ Testing connection...")
        if not broker.ping():
            print("❌ Connection failed")
            return False
        print("✅ Connection successful")
        
        # Test 2: Get current price
        print("\n2️⃣ Getting current BTCUSDT price...")
        try:
            current_price = broker.get_price('BTCUSDT')
            print(f"   Current BTCUSDT price: ${current_price:,.2f}")
        except Exception as e:
            print(f"   ❌ Failed to get price: {e}")
            return False
            
        # Test 3: Get symbol filters
        print("\n3️⃣ Getting symbol filters...")
        try:
            filters = broker.get_symbol_filters('BTCUSDT')
            print(f"   Symbol filters retrieved: {len(filters)} filters")
            for filter_type, filter_data in filters.items():
                print(f"     {filter_type}: {filter_data}")
        except Exception as e:
            print(f"   ❌ Failed to get filters: {e}")
            return False
            
        # Test 4: Place a small test order (BUY)
        print("\n4️⃣ Testing order placement...")
        try:
            # Calculate a small quantity (worth about $10)
            test_quantity = 0.0002  # Very small amount for testing
            test_price = current_price * 0.99  # 1% below current price to ensure it doesn't fill immediately
            
            print(f"   Placing BUY order: {test_quantity} BTC at ${test_price:.2f}")
            order_result = broker.place_order(
                symbol='BTCUSDT',
                side='BUY',
                order_type='LIMIT',
                quantity=test_quantity,
                price=test_price,
                time_in_force='GTC'
            )
            
            order_id = order_result.get('orderId')
            print(f"   ✅ Order placed successfully! Order ID: {order_id}")
            print(f"   Order status: {order_result.get('status')}")
            
        except Exception as e:
            print(f"   ❌ Failed to place order: {e}")
            return False
            
        # Test 5: Check open orders
        print("\n5️⃣ Checking open orders...")
        try:
            open_orders = broker.get_open_orders('BTCUSDT')
            print(f"   Found {len(open_orders)} open orders for BTCUSDT")
            for order in open_orders:
                print(f"     Order {order.get('orderId')}: {order.get('side')} {order.get('origQty')} @ {order.get('price')} - {order.get('status')}")
        except Exception as e:
            print(f"   ❌ Failed to get open orders: {e}")
            
        # Test 6: Cancel the test order
        print("\n6️⃣ Canceling test order...")
        try:
            if order_id:
                cancel_result = broker.cancel_order('BTCUSDT', str(order_id))
                print(f"   ✅ Order canceled successfully!")
                print(f"   Cancel status: {cancel_result.get('status')}")
            else:
                print("   ⚠️  No order ID to cancel")
        except Exception as e:
            print(f"   ❌ Failed to cancel order: {e}")
            print(f"   Error details: {e}")
            
        # Test 7: Verify order was canceled
        print("\n7️⃣ Verifying order cancellation...")
        try:
            time.sleep(1)  # Wait a moment for the cancellation to process
            open_orders = broker.get_open_orders('BTCUSDT')
            remaining_orders = [o for o in open_orders if str(o.get('orderId')) == str(order_id)]
            if not remaining_orders:
                print("   ✅ Order successfully canceled and removed from open orders")
            else:
                print(f"   ⚠️  Order still appears in open orders: {remaining_orders}")
        except Exception as e:
            print(f"   ❌ Failed to verify cancellation: {e}")
            
        print("\n✅ Order operations test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print(f"🚀 Binance Testnet Order Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    success = test_order_operations()
    
    if success:
        print("\n🎉 All order tests passed!")
    else:
        print("\n💥 Order tests failed!")
        sys.exit(1)
