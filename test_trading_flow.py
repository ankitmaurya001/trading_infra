#!/usr/bin/env python3
"""
Test script to verify the complete trading flow for Kite MCX.
Tests both LONG and SHORT trades with stop-loss GTT and take-profit handling.

Usage:
    python test_trading_flow.py              # Dry run (no real orders)
    python test_trading_flow.py --live       # Real orders (USE WITH CAUTION!)
    python test_trading_flow.py --long       # Test LONG trade only
    python test_trading_flow.py --short      # Test SHORT trade only
"""

import sys
import time
import argparse
from datetime import datetime
from data_fetcher import KiteDataFetcher
from brokers import KiteCommodityBroker
import config as cfg

# Test configuration
TEST_SYMBOL = "NATGASMINI26FEBFUT"
TEST_EXCHANGE = "MCX"
STOP_LOSS_PERCENT = 2.0
TAKE_PROFIT_PERCENT = 4.0


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step: int, total: int, description: str):
    """Print a step indicator."""
    print(f"\n[{step}/{total}] {description}")


class TradingFlowTester:
    """Test the complete trading flow."""
    
    def __init__(self, live_mode: bool = False):
        self.live_mode = live_mode
        self.data_fetcher = None
        self.broker = None
        self.test_results = {}
        
    def setup(self) -> bool:
        """Setup authentication and broker."""
        print_section("SETUP")
        
        try:
            print_step(1, 2, "Authenticating with Kite Connect...")
            self.data_fetcher = KiteDataFetcher(cfg.KITE_CREDENTIALS, TEST_EXCHANGE)
            self.data_fetcher.authenticate()
            print("✅ Authentication successful!")
            
            print_step(2, 2, "Initializing broker...")
            self.broker = KiteCommodityBroker(
                kite=self.data_fetcher.kite,
                exchange=TEST_EXCHANGE
            )
            print("✅ Broker initialized!")
            
            return True
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check margin and trading availability."""
        print_section("PREREQUISITES CHECK")
        
        try:
            # Check margin
            margins = self.broker.check_margins()
            available = margins.get('available', 0)
            actually_enabled = margins.get('actually_enabled', False)
            
            print(f"💰 Available Margin: ₹{available:,.2f}")
            print(f"📊 Trading Enabled: {'✅ YES' if actually_enabled else '❌ NO'}")
            
            if not actually_enabled:
                print("❌ Commodity trading not enabled!")
                return False
            
            # Get price
            price = self.broker.get_price(TEST_SYMBOL)
            print(f"💰 Current Price ({TEST_SYMBOL}): ₹{price:,.2f}")
            
            # Get margin requirement
            order_margins = self.broker.get_order_margins(
                symbol=TEST_SYMBOL,
                transaction_type='BUY',
                quantity=1,
                price=price,
                order_type='MARKET'
            )
            margin_required = order_margins.get('total', 0)
            print(f"📊 Margin Required: ₹{margin_required:,.2f}")
            
            if available < margin_required * 1.2:
                print(f"⚠️  Insufficient margin for testing!")
                print(f"   Need: ₹{margin_required * 1.2:,.2f}, Have: ₹{available:,.2f}")
                if self.live_mode:
                    return False
            
            return True
        except Exception as e:
            print(f"❌ Prerequisites check failed: {e}")
            return False
    
    def test_long_trade(self) -> dict:
        """Test LONG trade flow."""
        print_section("TEST: LONG TRADE FLOW")
        result = {
            'success': False,
            'order_id': None,
            'gtt_id': None,
            'exit_order_id': None,
            'errors': []
        }
        
        try:
            # Get current price
            price = self.broker.get_price(TEST_SYMBOL)
            stop_loss = round(price * (1 - STOP_LOSS_PERCENT / 100), 2)
            take_profit = round(price * (1 + TAKE_PROFIT_PERCENT / 100), 2)
            
            print(f"\n📊 Trade Setup:")
            print(f"   Symbol: {TEST_SYMBOL}")
            print(f"   Side: BUY (LONG)")
            print(f"   Entry Price: ₹{price:,.2f}")
            print(f"   Stop Loss: ₹{stop_loss:,.2f} ({STOP_LOSS_PERCENT}% below)")
            print(f"   Take Profit: ₹{take_profit:,.2f} ({TAKE_PROFIT_PERCENT}% above)")
            
            if not self.live_mode:
                print("\n⚠️  DRY RUN - Orders will NOT be placed")
                print("   ✅ LONG order would be placed")
                print("   ✅ GTT stop-loss would be placed (SELL @ ₹{:.2f})".format(stop_loss))
                result['success'] = True
                return result
            
            # Step 1: Place entry order
            print_step(1, 4, "Placing LONG entry order...")
            order = self.broker.place_order(
                symbol=TEST_SYMBOL,
                side='BUY',
                order_type='MARKET',
                quantity=1
            )
            result['order_id'] = order.get('orderId') or order.get('order_id')
            print(f"   ✅ Entry order placed: {result['order_id']}")
            
            # Wait for order to be filled
            time.sleep(2)
            
            # Step 2: Place GTT stop-loss (SELL for LONG)
            print_step(2, 4, "Placing GTT stop-loss order...")
            gtt_order = self.broker.place_gtt_order(
                symbol=TEST_SYMBOL,
                trigger_price=stop_loss,
                last_price=price,
                transaction_type='SELL',  # SELL to exit LONG
                quantity=1,
                order_price=stop_loss
            )
            # Extract GTT ID (handle both string and dict formats)
            gtt_id = gtt_order.get('gtt_id') or gtt_order.get('trigger_id')
            if isinstance(gtt_id, dict):
                gtt_id = gtt_id.get('trigger_id')
            result['gtt_id'] = str(gtt_id) if gtt_id else None
            print(f"   ✅ GTT stop-loss placed: {result['gtt_id']}")
            
            # Step 3: Verify position
            print_step(3, 4, "Verifying position...")
            positions = self.broker.get_positions()
            mcx_positions = [p for p in positions if p.get('tradingsymbol') == TEST_SYMBOL]
            if mcx_positions:
                pos = mcx_positions[0]
                print(f"   ✅ Position verified: {pos.get('quantity')} @ ₹{pos.get('average_price', 0):,.2f}")
            else:
                print(f"   ⚠️  Position not found (may be delayed)")
            
            # Step 4: Close position (simulate take-profit)
            print_step(4, 4, "Closing position (simulating take-profit)...")
            
            # First delete GTT
            try:
                self.data_fetcher.kite.delete_gtt(result['gtt_id'])
                print(f"   ✅ GTT deleted: {result['gtt_id']}")
            except Exception as e:
                print(f"   ⚠️  GTT deletion: {e}")
            
            # Then place exit order
            exit_order = self.broker.place_order(
                symbol=TEST_SYMBOL,
                side='SELL',  # SELL to exit LONG
                order_type='MARKET',
                quantity=1
            )
            result['exit_order_id'] = exit_order.get('orderId') or exit_order.get('order_id')
            print(f"   ✅ Exit order placed: {result['exit_order_id']}")
            
            result['success'] = True
            print("\n✅ LONG trade flow completed successfully!")
            
        except Exception as e:
            result['errors'].append(str(e))
            print(f"\n❌ LONG trade flow failed: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def test_short_trade(self) -> dict:
        """Test SHORT trade flow."""
        print_section("TEST: SHORT TRADE FLOW")
        result = {
            'success': False,
            'order_id': None,
            'gtt_id': None,
            'exit_order_id': None,
            'errors': []
        }
        
        try:
            # Get current price
            price = self.broker.get_price(TEST_SYMBOL)
            stop_loss = round(price * (1 + STOP_LOSS_PERCENT / 100), 2)  # Above for SHORT
            take_profit = round(price * (1 - TAKE_PROFIT_PERCENT / 100), 2)  # Below for SHORT
            
            print(f"\n📊 Trade Setup:")
            print(f"   Symbol: {TEST_SYMBOL}")
            print(f"   Side: SELL (SHORT)")
            print(f"   Entry Price: ₹{price:,.2f}")
            print(f"   Stop Loss: ₹{stop_loss:,.2f} ({STOP_LOSS_PERCENT}% above)")
            print(f"   Take Profit: ₹{take_profit:,.2f} ({TAKE_PROFIT_PERCENT}% below)")
            
            if not self.live_mode:
                print("\n⚠️  DRY RUN - Orders will NOT be placed")
                print("   ✅ SHORT order would be placed")
                print("   ✅ GTT stop-loss would be placed (BUY @ ₹{:.2f})".format(stop_loss))
                result['success'] = True
                return result
            
            # Step 1: Place entry order
            print_step(1, 4, "Placing SHORT entry order...")
            order = self.broker.place_order(
                symbol=TEST_SYMBOL,
                side='SELL',
                order_type='MARKET',
                quantity=1
            )
            result['order_id'] = order.get('orderId') or order.get('order_id')
            print(f"   ✅ Entry order placed: {result['order_id']}")
            
            # Wait for order to be filled
            time.sleep(2)
            
            # Step 2: Place GTT stop-loss (BUY for SHORT)
            print_step(2, 4, "Placing GTT stop-loss order...")
            gtt_order = self.broker.place_gtt_order(
                symbol=TEST_SYMBOL,
                trigger_price=stop_loss,
                last_price=price,
                transaction_type='BUY',  # BUY to exit SHORT
                quantity=1,
                order_price=stop_loss
            )
            # Extract GTT ID (handle both string and dict formats)
            gtt_id = gtt_order.get('gtt_id') or gtt_order.get('trigger_id')
            if isinstance(gtt_id, dict):
                gtt_id = gtt_id.get('trigger_id')
            result['gtt_id'] = str(gtt_id) if gtt_id else None
            print(f"   ✅ GTT stop-loss placed: {result['gtt_id']}")
            
            # Step 3: Verify position
            print_step(3, 4, "Verifying position...")
            positions = self.broker.get_positions()
            mcx_positions = [p for p in positions if p.get('tradingsymbol') == TEST_SYMBOL]
            if mcx_positions:
                pos = mcx_positions[0]
                print(f"   ✅ Position verified: {pos.get('quantity')} @ ₹{pos.get('average_price', 0):,.2f}")
            else:
                print(f"   ⚠️  Position not found (may be delayed)")
            
            # Step 4: Close position (simulate take-profit)
            print_step(4, 4, "Closing position (simulating take-profit)...")
            
            # First delete GTT
            try:
                self.data_fetcher.kite.delete_gtt(result['gtt_id'])
                print(f"   ✅ GTT deleted: {result['gtt_id']}")
            except Exception as e:
                print(f"   ⚠️  GTT deletion: {e}")
            
            # Then place exit order
            exit_order = self.broker.place_order(
                symbol=TEST_SYMBOL,
                side='BUY',  # BUY to exit SHORT
                order_type='MARKET',
                quantity=1
            )
            result['exit_order_id'] = exit_order.get('orderId') or exit_order.get('order_id')
            print(f"   ✅ Exit order placed: {result['exit_order_id']}")
            
            result['success'] = True
            print("\n✅ SHORT trade flow completed successfully!")
            
        except Exception as e:
            result['errors'].append(str(e))
            print(f"\n❌ SHORT trade flow failed: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def test_gtt_detection(self) -> dict:
        """Test that GTT trigger detection works."""
        print_section("TEST: GTT DETECTION")
        result = {'success': False, 'gtts': [], 'errors': []}
        
        try:
            print("Checking for active GTT orders...")
            gtts = self.data_fetcher.kite.get_gtts()
            active_gtts = [g for g in gtts if g.get('status') == 'active']
            
            print(f"📋 Total GTT orders: {len(gtts)}")
            print(f"📋 Active GTT orders: {len(active_gtts)}")
            
            for gtt in active_gtts:
                condition = gtt.get('condition', {})
                symbol = condition.get('tradingsymbol', 'Unknown')
                trigger_values = condition.get('trigger_values', [])
                trigger = trigger_values[0] if trigger_values else 'Unknown'
                print(f"   GTT {gtt.get('id')}: {symbol} @ ₹{trigger}")
            
            result['gtts'] = active_gtts
            result['success'] = True
            print("\n✅ GTT detection test passed!")
            
        except Exception as e:
            result['errors'].append(str(e))
            print(f"\n❌ GTT detection failed: {e}")
        
        return result
    
    def test_position_sync(self) -> dict:
        """Test position synchronization with broker."""
        print_section("TEST: POSITION SYNC")
        result = {'success': False, 'positions': [], 'errors': []}
        
        try:
            print("Fetching positions from broker...")
            positions = self.broker.get_positions()
            mcx_positions = [p for p in positions if p.get('exchange') == 'MCX' and p.get('quantity', 0) != 0]
            
            print(f"📊 MCX positions: {len(mcx_positions)}")
            
            for pos in mcx_positions:
                symbol = pos.get('tradingsymbol', 'Unknown')
                qty = pos.get('quantity', 0)
                avg_price = pos.get('average_price', 0)
                pnl = pos.get('pnl', 0)
                position_type = "LONG 📈" if qty > 0 else "SHORT 📉"
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                
                print(f"   {symbol}: {position_type} {abs(qty)} lot(s) @ ₹{avg_price:,.2f} | P&L: {pnl_emoji} ₹{pnl:,.2f}")
            
            result['positions'] = mcx_positions
            result['success'] = True
            print("\n✅ Position sync test passed!")
            
        except Exception as e:
            result['errors'].append(str(e))
            print(f"\n❌ Position sync failed: {e}")
        
        return result
    
    def run_all_tests(self, test_long: bool = True, test_short: bool = True):
        """Run all tests."""
        print("\n" + "=" * 70)
        print("  KITE MCX TRADING FLOW TEST")
        print("=" * 70)
        print(f"\n📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Symbol: {TEST_SYMBOL}")
        print(f"🏛️  Exchange: {TEST_EXCHANGE}")
        print(f"💰 Mode: {'🔴 LIVE (Real Orders)' if self.live_mode else '🟢 DRY RUN'}")
        
        if self.live_mode:
            print("\n⚠️  WARNING: LIVE MODE - Real orders will be placed!")
            confirm = input("Are you sure you want to continue? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("❌ Cancelled by user.")
                return
        
        # Setup
        if not self.setup():
            return
        
        # Prerequisites
        if not self.check_prerequisites():
            if self.live_mode:
                return
        
        # Run tests
        results = {}
        
        # Test position sync
        results['position_sync'] = self.test_position_sync()
        
        # Test GTT detection
        results['gtt_detection'] = self.test_gtt_detection()
        
        # Test LONG trade
        if test_long:
            results['long_trade'] = self.test_long_trade()
        
        # Test SHORT trade
        if test_short:
            results['short_trade'] = self.test_short_trade()
        
        # Summary
        print_section("TEST SUMMARY")
        
        all_passed = True
        for test_name, result in results.items():
            status = "✅ PASSED" if result.get('success') else "❌ FAILED"
            print(f"   {test_name}: {status}")
            if not result.get('success'):
                all_passed = False
                for error in result.get('errors', []):
                    print(f"      Error: {error}")
        
        print("\n" + "=" * 70)
        if all_passed:
            print("  ✅ ALL TESTS PASSED!")
        else:
            print("  ❌ SOME TESTS FAILED")
        print("=" * 70)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Test Kite MCX trading flow")
    parser.add_argument('--live', action='store_true', help="Enable live mode (real orders)")
    parser.add_argument('--long', action='store_true', help="Test LONG trade only")
    parser.add_argument('--short', action='store_true', help="Test SHORT trade only")
    args = parser.parse_args()
    
    # Determine which tests to run
    test_long = True
    test_short = True
    if args.long and not args.short:
        test_short = False
    elif args.short and not args.long:
        test_long = False
    
    # Run tests
    tester = TradingFlowTester(live_mode=args.live)
    try:
        tester.run_all_tests(test_long=test_long, test_short=test_short)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

