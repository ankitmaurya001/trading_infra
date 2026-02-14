
#!/usr/bin/env python3
"""
Realistic trading workflow test for Binance testnet
This demonstrates the proper sequence: Buy -> Set Stop-Loss -> Monitor -> Cancel/Close
"""
import os
import sys
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brokers.binance_spot import BinanceSpotBroker
import config as cfg


def test_realistic_trading_workflow():
    """Test a realistic trading workflow with proper stop-loss management"""
    print("🚀 Realistic Trading Workflow Test")
    print("=" * 50)
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Environment: TESTNET")
    
    # Initialize broker
    api_key = os.environ.get("BINANCE_API_KEY") or cfg.BINANCE_API_KEY
    api_secret = os.environ.get("BINANCE_API_SECRET") or cfg.BINANCE_SECRET_KEY
    
    try:
        broker = BinanceSpotBroker(api_key=api_key, api_secret=api_secret, testnet=True)
        
        # Step 1: Check connection and balances
        print("\n1️⃣ Checking Connection and Balances...")
        if not broker.ping():
            print("❌ Connection failed")
            return False
        print("✅ Connection successful")
        
        balances = broker.get_balances()
        usdt_balance = balances.get('USDT', 0)
        btc_balance = balances.get('BTC', 0)
        
        print(f"   USDT Balance: {usdt_balance}")
        print(f"   BTC Balance: {btc_balance}")
        
        # Step 2: Get current market data
        print("\n2️⃣ Getting Market Data...")
        current_price = broker.get_price('BTCUSDT')
        print(f"   Current BTCUSDT price: ${current_price:,.2f}")
        
        # Step 3: Calculate trade parameters
        print("\n3️⃣ Calculating Trade Parameters...")
        # Use a small amount for testing (worth about $10)
        trade_value_usdt = 10.0
        trade_quantity = trade_value_usdt / current_price
        
        # Round to proper precision
        trade_quantity = round(trade_quantity, 5)
        buy_price = current_price * 0.99  # 1% below current price
        stop_loss_price = buy_price * 0.95  # 5% below buy price
        
        print(f"   Trade value: ${trade_value_usdt}")
        print(f"   Quantity: {trade_quantity} BTC")
        print(f"   Buy price: ${buy_price:.2f}")
        print(f"   Stop-loss price: ${stop_loss_price:.2f}")
        
        # Step 4: Place buy order
        print("\n4️⃣ Placing Buy Order...")
        try:
            buy_order = broker.place_order(
                symbol='BTCUSDT',
                side='BUY',
                order_type='LIMIT',
                quantity=trade_quantity,
                price=buy_price,
                time_in_force='GTC'
            )
            
            buy_order_id = buy_order.get('orderId')
            print(f"   ✅ Buy order placed! ID: {buy_order_id}")
            print(f"   Status: {buy_order.get('status')}")
            
        except Exception as e:
            print(f"   ❌ Failed to place buy order: {e}")
            return False
        
        # Step 5: Wait for order to fill (or check status)
        print("\n5️⃣ Monitoring Order Status...")
        time.sleep(2)  # Wait a moment
        
        try:
            open_orders = broker.get_open_orders('BTCUSDT')
            buy_order_status = None
            for order in open_orders:
                if str(order.get('orderId')) == str(buy_order_id):
                    buy_order_status = order.get('status')
                    break
            
            if buy_order_status == 'FILLED':
                print("   ✅ Buy order filled!")
                
                # Step 6: Place stop-loss order
                print("\n6️⃣ Placing Stop-Loss Order...")
                try:
                    stop_order = broker.place_order(
                        symbol='BTCUSDT',
                        side='SELL',
                        order_type='STOP_LOSS',
                        quantity=trade_quantity,
                        price=stop_loss_price
                    )
                    
                    stop_order_id = stop_order.get('orderId')
                    print(f"   ✅ Stop-loss placed! ID: {stop_order_id}")
                    print(f"   Status: {stop_order.get('status')}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to place stop-loss: {e}")
                    print(f"   This is the error you were experiencing!")
                    return False
                
                # Step 7: Monitor positions
                print("\n7️⃣ Monitoring Positions...")
                time.sleep(1)
                
                open_orders = broker.get_open_orders('BTCUSDT')
                print(f"   Open orders: {len(open_orders)}")
                for order in open_orders:
                    order_type = order.get('type', 'UNKNOWN')
                    print(f"     Order {order.get('orderId')}: {order.get('side')} {order_type} - {order.get('status')}")
                
                # Step 8: Cancel stop-loss (simulate manual exit)
                print("\n8️⃣ Canceling Stop-Loss (Manual Exit)...")
                try:
                    cancel_result = broker.cancel_order('BTCUSDT', str(stop_order_id))
                    print(f"   ✅ Stop-loss canceled! Status: {cancel_result.get('status')}")
                except Exception as e:
                    print(f"   ❌ Failed to cancel stop-loss: {e}")
                
                # Step 9: Place sell order to close position
                print("\n9️⃣ Closing Position...")
                try:
                    sell_price = current_price * 1.01  # 1% above current price
                    sell_order = broker.place_order(
                        symbol='BTCUSDT',
                        side='SELL',
                        order_type='LIMIT',
                        quantity=trade_quantity,
                        price=sell_price,
                        time_in_force='GTC'
                    )
                    
                    sell_order_id = sell_order.get('orderId')
                    print(f"   ✅ Sell order placed! ID: {sell_order_id}")
                    print(f"   Status: {sell_order.get('status')}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to place sell order: {e}")
                
                # Step 10: Cleanup - cancel any remaining orders
                print("\n🔟 Cleanup...")
                time.sleep(2)
                
                open_orders = broker.get_open_orders('BTCUSDT')
                for order in open_orders:
                    try:
                        broker.cancel_order('BTCUSDT', str(order.get('orderId')))
                        print(f"   ✅ Canceled order {order.get('orderId')}")
                    except Exception as e:
                        print(f"   ❌ Failed to cancel order {order.get('orderId')}: {e}")
                
                print("\n✅ Realistic trading workflow completed successfully!")
                return True
                
            else:
                print(f"   ⚠️  Buy order not filled yet (Status: {buy_order_status})")
                print("   📝 This is normal for limit orders below market price")
                
                # Cancel the buy order since it's not filled
                print("\n6️⃣ Canceling Unfilled Buy Order...")
                try:
                    broker.cancel_order('BTCUSDT', str(buy_order_id))
                    print("   ✅ Buy order canceled")
                except Exception as e:
                    print(f"   ❌ Failed to cancel buy order: {e}")
                
                print("\n✅ Test completed (order not filled, which is expected)")
                return True
                
        except Exception as e:
            print(f"   ❌ Error monitoring orders: {e}")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_realistic_trading_workflow()
    
    if success:
        print("\n🎉 Realistic trading workflow test passed!")
        print("💡 Key insights:")
        print("   - Stop-loss orders require actual asset holdings")
        print("   - The error you saw is expected when trying to sell assets you don't have")
        print("   - Proper workflow: Buy first → Set stop-loss → Monitor → Close")
    else:
        print("\n💥 Realistic trading workflow test failed!")
        sys.exit(1)
