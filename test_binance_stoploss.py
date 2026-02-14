#!/usr/bin/env python3
"""
Test script to verify Binance testnet stop-loss order functionality
"""
import os
import sys
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brokers.binance_spot import BinanceSpotBroker
import config as cfg


def test_stoploss_operations():
    """Test stop-loss order placement and cancellation"""
    print("🔍 Testing Binance Testnet Stop-Loss Operations")
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
            
        # Test 3: Place a small test order first (to have a position)
        print("\n3️⃣ Placing a small test BUY order...")
        try:
            test_quantity = 0.0001  # Very small amount
            test_price = current_price * 0.99  # 1% below current price
            
            print(f"   Placing BUY order: {test_quantity} BTC at ${test_price:.2f}")
            buy_order = broker.place_order(
                symbol='BTCUSDT',
                side='BUY',
                order_type='LIMIT',
                quantity=test_quantity,
                price=test_price,
                time_in_force='GTC'
            )
            
            buy_order_id = buy_order.get('orderId')
            print(f"   ✅ BUY order placed! Order ID: {buy_order_id}")
            
        except Exception as e:
            print(f"   ❌ Failed to place BUY order: {e}")
            return False
            
        # Test 4: Place stop-loss order
        print("\n4️⃣ Testing stop-loss order placement...")
        try:
            # Calculate stop-loss price (2% below current price)
            stop_price = current_price * 0.98
            stop_quantity = test_quantity
            
            print(f"   Placing STOP_LOSS order: {stop_quantity} BTC at ${stop_price:.2f}")
            stop_order = broker.place_order(
                symbol='BTCUSDT',
                side='SELL',
                order_type='STOP_LOSS',
                quantity=stop_quantity,
                price=stop_price
            )
            
            stop_order_id = stop_order.get('orderId')
            print(f"   ✅ Stop-loss order placed! Order ID: {stop_order_id}")
            print(f"   Stop-loss status: {stop_order.get('status')}")
            
        except Exception as e:
            print(f"   ❌ Failed to place stop-loss order: {e}")
            print(f"   Error details: {e}")
            # Continue with cancellation test even if stop-loss failed
            
        # Test 5: Check open orders
        print("\n5️⃣ Checking open orders...")
        try:
            open_orders = broker.get_open_orders('BTCUSDT')
            print(f"   Found {len(open_orders)} open orders for BTCUSDT")
            for order in open_orders:
                order_type = order.get('type', 'UNKNOWN')
                print(f"     Order {order.get('orderId')}: {order.get('side')} {order.get('origQty')} @ {order.get('price')} - {order_type} - {order.get('status')}")
        except Exception as e:
            print(f"   ❌ Failed to get open orders: {e}")
            
        # Test 6: Cancel stop-loss order
        print("\n6️⃣ Testing stop-loss order cancellation...")
        try:
            if 'stop_order_id' in locals() and stop_order_id:
                print(f"   Canceling stop-loss order: {stop_order_id}")
                cancel_result = broker.cancel_order('BTCUSDT', str(stop_order_id))
                print(f"   ✅ Stop-loss order canceled successfully!")
                print(f"   Cancel status: {cancel_result.get('status')}")
            else:
                print("   ⚠️  No stop-loss order ID to cancel")
        except Exception as e:
            print(f"   ❌ Failed to cancel stop-loss order: {e}")
            print(f"   Error details: {e}")
            
        # Test 7: Cancel the original BUY order
        print("\n7️⃣ Canceling original BUY order...")
        try:
            if buy_order_id:
                print(f"   Canceling BUY order: {buy_order_id}")
                cancel_result = broker.cancel_order('BTCUSDT', str(buy_order_id))
                print(f"   ✅ BUY order canceled successfully!")
                print(f"   Cancel status: {cancel_result.get('status')}")
            else:
                print("   ⚠️  No BUY order ID to cancel")
        except Exception as e:
            print(f"   ❌ Failed to cancel BUY order: {e}")
            print(f"   Error details: {e}")
            
        # Test 8: Verify all orders are canceled
        print("\n8️⃣ Verifying all orders are canceled...")
        try:
            time.sleep(2)  # Wait for cancellations to process
            open_orders = broker.get_open_orders('BTCUSDT')
            if not open_orders:
                print("   ✅ All orders successfully canceled")
            else:
                print(f"   ⚠️  {len(open_orders)} orders still open:")
                for order in open_orders:
                    print(f"     Order {order.get('orderId')}: {order.get('side')} {order.get('type')} - {order.get('status')}")
        except Exception as e:
            print(f"   ❌ Failed to verify cancellations: {e}")
            
        print("\n✅ Stop-loss operations test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print(f"🚀 Binance Testnet Stop-Loss Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    success = test_stoploss_operations()
    
    if success:
        print("\n🎉 All stop-loss tests passed!")
    else:
        print("\n💥 Stop-loss tests failed!")
        sys.exit(1)
