#!/usr/bin/env python3
"""
Enhanced test script that uses market orders to get BTC, then tests stop-loss functionality
"""
import os
import sys
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brokers.binance_spot import BinanceSpotBroker
import config as cfg


def test_with_market_orders():
    """Test using market orders to get BTC, then test stop-loss"""
    print("🚀 Binance Testnet Test with Market Orders")
    print("=" * 50)
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Environment: TESTNET")
    
    # Initialize broker
    api_key = os.environ.get("BINANCE_API_KEY") or cfg.BINANCE_API_KEY
    api_secret = os.environ.get("BINANCE_API_SECRET") or cfg.BINANCE_SECRET_KEY
    
    try:
        broker = BinanceSpotBroker(api_key=api_key, api_secret=api_secret, testnet=True)
        
        # Step 1: Check connection
        print("\n1️⃣ Testing Connection...")
        if not broker.ping():
            print("❌ Connection failed")
            return False
        print("✅ Connection successful")
        
        # Step 2: Check balances
        print("\n2️⃣ Checking Initial Balances...")
        balances = broker.get_balances()
        usdt_balance = balances.get('USDT', 0)
        btc_balance = balances.get('BTC', 0)
        print(f"   USDT Balance: {usdt_balance}")
        print(f"   BTC Balance: {btc_balance}")
        
        # Step 3: Get current price
        print("\n3️⃣ Getting Current Price...")
        current_price = broker.get_price('BTCUSDT')
        print(f"   Current BTCUSDT price: ${current_price:,.2f}")
        
        # Step 4: Place small market buy order to get some BTC
        print("\n4️⃣ Placing Market Buy Order...")
        try:
            # Use a fixed small quantity that meets minimum requirements
            buy_quantity = 0.0001  # Small fixed amount
            
            print(f"   Buying {buy_quantity} BTC (worth ~${buy_quantity * current_price:.2f})")
            
            market_buy = broker.place_order(
                symbol='BTCUSDT',
                side='BUY',
                order_type='MARKET',
                quantity=buy_quantity
            )
            
            buy_order_id = market_buy.get('orderId')
            print(f"   ✅ Market buy order placed! ID: {buy_order_id}")
            print(f"   Status: {market_buy.get('status')}")
            
        except Exception as e:
            print(f"   ❌ Failed to place market buy order: {e}")
            return False
        
        # Step 5: Wait for order to fill and check new balances
        print("\n5️⃣ Waiting for Order to Fill...")
        time.sleep(3)  # Wait for market order to fill
        
        balances = broker.get_balances()
        new_btc_balance = balances.get('BTC', 0)
        print(f"   New BTC Balance: {new_btc_balance}")
        
        if new_btc_balance <= btc_balance:
            print("   ⚠️  BTC balance didn't increase - order may not have filled")
            return False
        
        # Step 6: Now test stop-loss with actual BTC holdings
        print("\n6️⃣ Testing Stop-Loss with Real BTC Holdings...")
        try:
            # Use a quantity that won't have scientific notation issues
            stop_quantity = 0.0001  # Use the same quantity we bought
            stop_price = current_price * 0.95  # 5% below current price
            
            print(f"   Placing STOP_LOSS order: {stop_quantity} BTC at ${stop_price:.2f}")
            
            stop_order = broker.place_order(
                symbol='BTCUSDT',
                side='SELL',
                order_type='STOP_LOSS',
                quantity=stop_quantity,
                price=stop_price
            )
            
            stop_order_id = stop_order.get('orderId')
            print(f"   ✅ Stop-loss placed! ID: {stop_order_id}")
            print(f"   Status: {stop_order.get('status')}")
            
        except Exception as e:
            print(f"   ❌ Failed to place stop-loss: {e}")
            print(f"   Error details: {e}")
            return False
        
        # Step 7: Check open orders
        print("\n7️⃣ Checking Open Orders...")
        open_orders = broker.get_open_orders('BTCUSDT')
        print(f"   Found {len(open_orders)} open orders:")
        for order in open_orders:
            order_type = order.get('type', 'UNKNOWN')
            print(f"     Order {order.get('orderId')}: {order.get('side')} {order_type} - {order.get('status')}")
        
        # Step 8: Cancel stop-loss order
        print("\n8️⃣ Canceling Stop-Loss Order...")
        try:
            cancel_result = broker.cancel_order('BTCUSDT', str(stop_order_id))
            print(f"   ✅ Stop-loss canceled! Status: {cancel_result.get('status')}")
        except Exception as e:
            print(f"   ❌ Failed to cancel stop-loss: {e}")
            print(f"   Error details: {e}")
        
        # Step 9: Place market sell order to close position
        print("\n9️⃣ Closing Position with Market Sell...")
        try:
            # Sell the BTC we bought (use the same quantity we bought)
            sell_quantity = 0.0001  # Same quantity we bought
            
            # Format quantity to ensure no scientific notation
            sell_quantity = float(f"{sell_quantity:.6f}")
            
            print(f"   Selling {sell_quantity} BTC")
            
            market_sell = broker.place_order(
                symbol='BTCUSDT',
                side='SELL',
                order_type='MARKET',
                quantity=sell_quantity
            )
            
            sell_order_id = market_sell.get('orderId')
            print(f"   ✅ Market sell order placed! ID: {sell_order_id}")
            print(f"   Status: {market_sell.get('status')}")
            
        except Exception as e:
            print(f"   ❌ Failed to place market sell order: {e}")
            return False
        
        # Step 10: Final cleanup and verification
        print("\n🔟 Final Cleanup...")
        time.sleep(2)
        
        # Check final balances
        final_balances = broker.get_balances()
        final_btc_balance = final_balances.get('BTC', 0)
        final_usdt_balance = final_balances.get('USDT', 0)
        
        print(f"   Final BTC Balance: {final_btc_balance}")
        print(f"   Final USDT Balance: {final_usdt_balance}")
        
        # Check for any remaining open orders
        open_orders = broker.get_open_orders('BTCUSDT')
        if open_orders:
            print(f"   ⚠️  {len(open_orders)} orders still open - canceling...")
            for order in open_orders:
                try:
                    broker.cancel_order('BTCUSDT', str(order.get('orderId')))
                    print(f"   ✅ Canceled order {order.get('orderId')}")
                except Exception as e:
                    print(f"   ❌ Failed to cancel order {order.get('orderId')}: {e}")
        else:
            print("   ✅ No remaining open orders")
        
        print("\n✅ Market order test completed successfully!")
        print("💡 Key insights:")
        print("   - Market orders fill immediately")
        print("   - Stop-loss orders work when you have the asset to sell")
        print("   - The error you experienced was due to trying to sell BTC you didn't have")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_market_orders()
    
    if success:
        print("\n🎉 Market order test passed!")
    else:
        print("\n💥 Market order test failed!")
        sys.exit(1)
