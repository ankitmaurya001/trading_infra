#!/usr/bin/env python3
"""
Comprehensive test script for Binance testnet API functionality
Tests all operations in sequence: balance, orders, stop-loss, and cleanup
"""
import os
import sys
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brokers.binance_spot import BinanceSpotBroker
import config as cfg


class BinanceAPITester:
    def __init__(self):
        self.api_key = os.environ.get("BINANCE_API_KEY") or cfg.BINANCE_API_KEY
        self.api_secret = os.environ.get("BINANCE_API_SECRET") or cfg.BINANCE_SECRET_KEY
        self.broker = None
        self.test_orders = []  # Track orders for cleanup
        
    def initialize(self):
        """Initialize the broker connection"""
        print("🔧 Initializing Binance Testnet Broker...")
        try:
            self.broker = BinanceSpotBroker(
                api_key=self.api_key, 
                api_secret=self.api_secret, 
                testnet=True
            )
            return True
        except Exception as e:
            print(f"❌ Failed to initialize broker: {e}")
            return False
    
    def test_connection(self):
        """Test basic connection and ping"""
        print("\n1️⃣ Testing Connection...")
        try:
            if not self.broker.ping():
                print("❌ Ping failed")
                return False
            print("✅ Connection successful")
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def test_balance_retrieval(self):
        """Test balance and account info retrieval"""
        print("\n2️⃣ Testing Balance Retrieval...")
        try:
            # Get account info
            account = self.broker.get_account()
            print(f"   Account type: {account.get('accountType')}")
            print(f"   Can trade: {account.get('canTrade')}")
            
            # Get balances
            balances = self.broker.get_balances()
            if balances:
                print(f"   Found {len(balances)} assets with balances:")
                for asset, balance in sorted(balances.items()):
                    print(f"     {asset}: {balance}")
            else:
                print("   No balances found (normal for new testnet accounts)")
            
            print("✅ Balance retrieval successful")
            return True
        except Exception as e:
            print(f"❌ Balance test failed: {e}")
            return False
    
    def test_price_and_filters(self):
        """Test price retrieval and symbol filters"""
        print("\n3️⃣ Testing Price and Symbol Filters...")
        try:
            # Get current price
            current_price = self.broker.get_price('BTCUSDT')
            print(f"   Current BTCUSDT price: ${current_price:,.2f}")
            
            # Get symbol filters
            filters = self.broker.get_symbol_filters('BTCUSDT')
            print(f"   Symbol filters: {len(filters)} filters found")
            
            # Display key filters
            for filter_type in ['PRICE_FILTER', 'LOT_SIZE', 'MIN_NOTIONAL']:
                if filter_type in filters:
                    print(f"     {filter_type}: {filters[filter_type]}")
            
            print("✅ Price and filters test successful")
            return True, current_price
        except Exception as e:
            print(f"❌ Price/filters test failed: {e}")
            return False, None
    
    def test_order_placement(self, current_price):
        """Test order placement and management"""
        print("\n4️⃣ Testing Order Placement...")
        try:
            # Calculate test parameters
            test_quantity = 0.0001  # Very small amount
            test_price = current_price * 0.99  # 1% below current price
            
            print(f"   Placing BUY order: {test_quantity} BTC at ${test_price:.2f}")
            
            # Place order
            order_result = self.broker.place_order(
                symbol='BTCUSDT',
                side='BUY',
                order_type='LIMIT',
                quantity=test_quantity,
                price=test_price,
                time_in_force='GTC'
            )
            
            order_id = order_result.get('orderId')
            self.test_orders.append(('BTCUSDT', order_id, 'BUY'))
            
            print(f"   ✅ Order placed! ID: {order_id}, Status: {order_result.get('status')}")
            
            # Check open orders
            open_orders = self.broker.get_open_orders('BTCUSDT')
            print(f"   Open orders: {len(open_orders)}")
            
            print("✅ Order placement test successful")
            return True
        except Exception as e:
            print(f"❌ Order placement test failed: {e}")
            return False
    
    def test_stoploss_placement(self, current_price):
        """Test stop-loss order placement"""
        print("\n5️⃣ Testing Stop-Loss Order Placement...")
        try:
            # First, check if we have BTC to sell
            balances = self.broker.get_balances()
            btc_balance = balances.get('BTC', 0)
            
            if btc_balance < 0.0001:
                print("   ⚠️  Insufficient BTC balance for stop-loss test")
                print("   📝 Note: Stop-loss orders require actual asset holdings")
                print("   💡 This is expected behavior - stop-loss protects existing positions")
                print("✅ Stop-loss test skipped (no BTC to protect)")
                return True
            
            # Calculate stop-loss parameters
            stop_quantity = min(btc_balance * 0.5, 0.0001)  # Use half of available BTC
            stop_price = current_price * 0.98  # 2% below current price
            
            print(f"   Placing STOP_LOSS order: {stop_quantity} BTC at ${stop_price:.2f}")
            
            # Place stop-loss order
            stop_order = self.broker.place_order(
                symbol='BTCUSDT',
                side='SELL',
                order_type='STOP_LOSS',
                quantity=stop_quantity,
                price=stop_price
            )
            
            stop_order_id = stop_order.get('orderId')
            self.test_orders.append(('BTCUSDT', stop_order_id, 'STOP_LOSS'))
            
            print(f"   ✅ Stop-loss placed! ID: {stop_order_id}, Status: {stop_order.get('status')}")
            
            # Check open orders again
            open_orders = self.broker.get_open_orders('BTCUSDT')
            print(f"   Total open orders: {len(open_orders)}")
            
            print("✅ Stop-loss placement test successful")
            return True
        except Exception as e:
            print(f"❌ Stop-loss placement test failed: {e}")
            print(f"   Error details: {e}")
            return False
    
    def test_order_cancellation(self):
        """Test order cancellation"""
        print("\n6️⃣ Testing Order Cancellation...")
        try:
            success_count = 0
            total_orders = len(self.test_orders)
            
            for symbol, order_id, order_type in self.test_orders:
                try:
                    print(f"   Canceling {order_type} order: {order_id}")
                    cancel_result = self.broker.cancel_order(symbol, str(order_id))
                    print(f"   ✅ {order_type} order canceled! Status: {cancel_result.get('status')}")
                    success_count += 1
                except Exception as e:
                    print(f"   ❌ Failed to cancel {order_type} order {order_id}: {e}")
            
            print(f"   Cancellation summary: {success_count}/{total_orders} orders canceled")
            
            # Verify all orders are canceled
            time.sleep(2)
            open_orders = self.broker.get_open_orders('BTCUSDT')
            remaining_orders = len(open_orders)
            
            if remaining_orders == 0:
                print("✅ All orders successfully canceled")
                return True
            else:
                print(f"⚠️  {remaining_orders} orders still open")
                return False
                
        except Exception as e:
            print(f"❌ Order cancellation test failed: {e}")
            return False
    
    def test_trade_history(self):
        """Test trade history retrieval"""
        print("\n7️⃣ Testing Trade History...")
        try:
            # Get recent trades
            trades = self.broker.get_account_trades(
                symbol='BTCUSDT',
                limit=10
            )
            
            print(f"   Found {len(trades)} recent trades")
            if trades:
                for trade in trades[-3:]:  # Show last 3 trades
                    trade_time = datetime.fromtimestamp(trade.get('time', 0) / 1000)
                    print(f"     {trade_time}: {trade.get('side')} {trade.get('qty')} @ {trade.get('price')}")
            
            print("✅ Trade history test successful")
            return True
        except Exception as e:
            print(f"❌ Trade history test failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup any remaining test orders"""
        print("\n🧹 Cleanup...")
        try:
            open_orders = self.broker.get_open_orders('BTCUSDT')
            if open_orders:
                print(f"   Found {len(open_orders)} remaining orders to cancel")
                for order in open_orders:
                    try:
                        self.broker.cancel_order('BTCUSDT', str(order.get('orderId')))
                        print(f"   ✅ Canceled order {order.get('orderId')}")
                    except Exception as e:
                        print(f"   ❌ Failed to cancel order {order.get('orderId')}: {e}")
            else:
                print("   No remaining orders to clean up")
        except Exception as e:
            print(f"   ❌ Cleanup failed: {e}")
    
    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 Binance Testnet Comprehensive API Test")
        print("=" * 60)
        print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Environment: TESTNET")
        print(f"📡 API Key: {self.api_key[:10]}...")
        
        test_results = []
        
        try:
            # Initialize
            if not self.initialize():
                return False
            
            # Run all tests
            test_results.append(("Connection", self.test_connection()))
            test_results.append(("Balance Retrieval", self.test_balance_retrieval()))
            
            price_result, current_price = self.test_price_and_filters()
            test_results.append(("Price & Filters", price_result))
            
            if current_price:
                test_results.append(("Order Placement", self.test_order_placement(current_price)))
                test_results.append(("Stop-Loss Placement", self.test_stoploss_placement(current_price)))
            
            test_results.append(("Order Cancellation", self.test_order_cancellation()))
            test_results.append(("Trade History", self.test_trade_history()))
            
            # Cleanup
            self.cleanup()
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:20} {status}")
            if result:
                passed += 1
        
        print("-" * 60)
        print(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Your Binance testnet API is working correctly.")
            return True
        else:
            print("💥 Some tests failed. Check the errors above.")
            return False


def main():
    tester = BinanceAPITester()
    success = tester.run_comprehensive_test()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
