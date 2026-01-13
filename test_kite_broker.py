#!/usr/bin/env python3
"""
Test script for Kite Commodity Broker APIs
Tests all broker functionality: margins, lot sizes, orders, GTT, positions, etc.
"""

import sys
import time
from datetime import datetime
import pytz
from data_fetcher import KiteDataFetcher
from brokers import KiteCommodityBroker
import config as cfg

# Test configuration
TEST_SYMBOL = "NATGASMINI26FEBFUT"  # Change to your test symbol
TEST_EXCHANGE = "MCX"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_authentication():
    """Test Kite authentication."""
    print_section("1. Testing Authentication")
    try:
        data_fetcher = KiteDataFetcher(cfg.KITE_CREDENTIALS, cfg.KITE_EXCHANGE)
        print("🔐 Authenticating with Kite Connect...")
        data_fetcher.authenticate()
        print("✅ Authentication successful!")
        return data_fetcher
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return None


def test_broker_initialization(data_fetcher):
    """Test broker initialization."""
    print_section("2. Testing Broker Initialization")
    try:
        broker = KiteCommodityBroker(
            kite=data_fetcher.kite,
            exchange=TEST_EXCHANGE
        )
        print(f"✅ Broker initialized for {TEST_EXCHANGE}")
        
        # Test ping
        if broker.ping():
            print("✅ Broker ping successful")
        else:
            print("❌ Broker ping failed")
        
        return broker
    except Exception as e:
        print(f"❌ Broker initialization failed: {e}")
        return None


def test_margins(broker):
    """Test margin checking."""
    print_section("3. Testing Margin Checking")
    try:
        margins = broker.check_margins()
        enabled = margins.get('enabled', False)
        actually_enabled = margins.get('actually_enabled', enabled)
        
        print(f"💰 Commodity Trading (API flag): {'✅ YES' if enabled else '❌ NO'}")
        print(f"💰 Commodity Trading (Actual): {'✅ YES' if actually_enabled else '❌ NO'}")
        print(f"💰 Available Margin: ₹{margins.get('available', 0):,.2f}")
        print(f"💰 Net Margin: ₹{margins.get('net', 0):,.2f}")
        print(f"💰 Utilised Margin: ₹{margins.get('utilised', 0):,.2f}")
        print(f"💰 Total Margin: ₹{margins.get('total', 0):,.2f}")
        
        using_single_ledger = margins.get('using_single_ledger', False)
        
        if using_single_ledger:
            print(f"\n💡 SINGLE LEDGER FACILITY ACTIVE:")
            print(f"   Zerodha allows using equity funds for commodity trading!")
            print(f"   Your equity funds (₹{margins.get('available', 0):,.2f}) are available for MCX trading.")
            print(f"   No need to transfer funds manually.")
            print(f"   Ref: https://support.zerodha.com/category/funds/adding-funds/other-fund-related-queries/articles/can-i-use-the-same-funds-for-trading-on-equity-as-well-as-commodity")
        elif not actually_enabled:
            print(f"\n⚠️  WARNING: Commodity trading appears to be disabled!")
            print(f"   You need to:")
            print(f"   1. Activate commodity segment from Console")
            print(f"   2. Your equity funds will then be usable for commodity trading (single ledger)")
        elif not enabled and actually_enabled:
            print(f"\n💡 NOTE: API shows 'enabled: false' but trading is enabled.")
            print(f"   MCX found in your exchanges - commodity trading works.")
        
        # Check equity margins to see if funds are there
        equity_margins = broker.check_equity_margins()
        if equity_margins.get('available', 0) > 0:
            print(f"\n💡 Found funds in Equity segment:")
            print(f"   Equity Available: ₹{equity_margins.get('available', 0):,.2f}")
            print(f"   Equity Net: ₹{equity_margins.get('net', 0):,.2f}")
            print(f"   → Transfer these funds to Commodity segment to trade MCX")
        
        # Print raw structure for debugging
        raw = margins.get('raw', {})
        print(f"\n📋 Raw commodity margin structure (for debugging):")
        if isinstance(raw, dict):
            print(f"   Top-level keys: {list(raw.keys())}")
            for key, value in raw.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"      {sub_key}: {sub_value}")
                else:
                    print(f"   {key}: {value}")
        
        print("✅ Margin check successful")
        return margins
    except Exception as e:
        print(f"❌ Margin check failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_lot_size(broker, symbol: str):
    """Test lot size retrieval."""
    print_section("4. Testing Lot Size Retrieval")
    try:
        lot_size = broker.get_lot_size(symbol)
        if lot_size:
            print(f"📦 Lot size for {symbol}: {lot_size}")
            print("✅ Lot size retrieval successful")
            return lot_size
        else:
            print(f"⚠️  Lot size not found for {symbol}")
            return None
    except Exception as e:
        print(f"❌ Lot size retrieval failed: {e}")
        return None


def test_get_price(broker, symbol: str):
    """Test price retrieval."""
    print_section("5. Testing Price Retrieval")
    try:
        price = broker.get_price(symbol)
        if price > 0:
            print(f"💰 Current price for {symbol}: ₹{price:,.2f}")
            print("✅ Price retrieval successful")
            return price
        else:
            print(f"⚠️  Invalid price returned: {price}")
            return None
    except Exception as e:
        print(f"❌ Price retrieval failed: {e}")
        return None


def test_symbol_info(broker, symbol: str):
    """Test symbol information retrieval."""
    print_section("6. Testing Symbol Information")
    try:
        symbol_info = broker.get_symbol_filters(symbol)
        if symbol_info:
            print(f"📊 Symbol Info for {symbol}:")
            for key, value in symbol_info.items():
                print(f"   {key}: {value}")
            print("✅ Symbol info retrieval successful")
            return symbol_info
        else:
            print(f"⚠️  Symbol info not found for {symbol}")
            return None
    except Exception as e:
        print(f"❌ Symbol info retrieval failed: {e}")
        return None


def test_get_positions(broker):
    """Test position retrieval."""
    print_section("7. Testing Position Retrieval")
    try:
        positions = broker.get_positions()
        print(f"📊 Current positions: {len(positions)}")
        for pos in positions:
            if pos.get('quantity', 0) != 0:  # Only show non-zero positions
                print(f"   {pos.get('tradingsymbol')}: {pos.get('quantity')} @ ₹{pos.get('average_price', 0):,.2f}")
        print("✅ Position retrieval successful")
        return positions
    except Exception as e:
        print(f"❌ Position retrieval failed: {e}")
        return None


def test_get_open_orders(broker, symbol: str = None):
    """Test open orders retrieval."""
    print_section("8. Testing Open Orders Retrieval")
    try:
        orders = broker.get_open_orders(symbol)
        print(f"📋 Open orders: {len(orders)}")
        for order in orders:
            print(f"   Order ID: {order.get('order_id')}, "
                  f"Symbol: {order.get('tradingsymbol')}, "
                  f"Type: {order.get('order_type')}, "
                  f"Status: {order.get('status')}")
        print("✅ Open orders retrieval successful")
        return orders
    except Exception as e:
        print(f"❌ Open orders retrieval failed: {e}")
        return None


def test_get_gtts(broker):
    """Test GTT orders retrieval."""
    print_section("9. Testing GTT Orders Retrieval")
    try:
        gtts = broker.get_gtts()
        print(f"🔔 Active GTT orders: {len(gtts)}")
        for gtt in gtts:
            print(f"   GTT ID: {gtt.get('id')}, "
                  f"Symbol: {gtt.get('tradingsymbol')}, "
                  f"Status: {gtt.get('status')}")
        print("✅ GTT orders retrieval successful")
        return gtts
    except Exception as e:
        print(f"❌ GTT orders retrieval failed: {e}")
        return None


def test_place_order_dry_run(broker, symbol: str, price: float):
    """Test order placement (dry run - commented out for safety)."""
    print_section("10. Testing Order Placement (DRY RUN)")
    print("⚠️  This is a DRY RUN - orders will NOT be placed")
    print(f"   Would place: BUY 1 lot of {symbol} @ MARKET")
    print(f"   Current price: ₹{price:,.2f}")
    print("✅ Order placement test (dry run) completed")
    print("\n💡 To actually place an order, uncomment the code in test_place_order()")
    
    # Uncomment below to actually place an order (USE WITH CAUTION!)
    """
    try:
        order = broker.place_order(
            symbol=symbol,
            side='BUY',
            order_type='MARKET',
            quantity=1
        )
        print(f"✅ Order placed: {order.get('orderId')}")
        return order
    except Exception as e:
        print(f"❌ Order placement failed: {e}")
        return None
    """
    return None


def test_place_gtt_dry_run(broker, symbol: str, current_price: float, stop_loss_price: float):
    """Test GTT order placement (dry run)."""
    print_section("11. Testing GTT Order Placement (DRY RUN)")
    print("⚠️  This is a DRY RUN - GTT orders will NOT be placed")
    print(f"   Would place GTT: SELL 1 lot of {symbol}")
    print(f"   Trigger price: ₹{stop_loss_price:,.2f}")
    print(f"   Current price: ₹{current_price:,.2f}")
    print("✅ GTT order placement test (dry run) completed")
    print("\n💡 To actually place a GTT order, uncomment the code in test_place_gtt()")
    
    # Uncomment below to actually place a GTT order (USE WITH CAUTION!)
    """
    try:
        gtt_order = broker.place_gtt_order(
            symbol=symbol,
            trigger_price=stop_loss_price,
            last_price=current_price,
            transaction_type='SELL',
            quantity=1,
            order_price=stop_loss_price
        )
        print(f"✅ GTT order placed: {gtt_order.get('gtt_id')}")
        return gtt_order
    except Exception as e:
        print(f"❌ GTT order placement failed: {e}")
        return None
    """
    return None


def test_margin_calculation(broker, symbol: str, price: float, lot_size: int):
    """Test margin calculation using Kite's order_margins API."""
    print_section("12. Testing Margin Calculation")
    try:
        margins = broker.check_margins()
        available = margins.get('available', 0)
        
        # Use Kite's order_margins API to get ACTUAL margin requirement
        print(f"📊 Getting ACTUAL margin requirement from Kite API...")
        order_margins = broker.get_order_margins(
            symbol=symbol,
            transaction_type='BUY',
            quantity=1,  # 1 lot
            price=price,
            order_type='MARKET'
        )
        
        actual_margin_required = order_margins.get('total', 0.0)
        available_after = order_margins.get('available_after', 0.0)
        
        print(f"\n📊 Margin Calculation for {symbol}:")
        print(f"   Lot size: {lot_size}")
        print(f"   Price: ₹{price:,.2f}")
        print(f"   Contract value: ₹{lot_size * price:,.2f}")
        print(f"   ACTUAL margin required (from Kite): ₹{actual_margin_required:,.2f}")
        print(f"   Available margin: ₹{available:,.2f}")
        print(f"   Available after order: ₹{available_after:,.2f}")
        
        # Add 20% buffer
        required_with_buffer = actual_margin_required * 1.2
        
        if available >= required_with_buffer:
            print(f"\n✅ Sufficient margin available (with 20% buffer)")
            print(f"   Need: ₹{required_with_buffer:,.2f}, Have: ₹{available:,.2f}")
        else:
            print(f"\n⚠️  Insufficient margin (with 20% buffer)")
            print(f"   Need: ₹{required_with_buffer:,.2f}, Have: ₹{available:,.2f}")
            print(f"   Shortfall: ₹{required_with_buffer - available:,.2f}")
        
        # Show raw margin data for debugging
        raw_margin = order_margins.get('raw', {})
        if raw_margin:
            print(f"\n📋 Raw order margin data:")
            for key, value in raw_margin.items():
                print(f"   {key}: {value}")
        
        return actual_margin_required
    except Exception as e:
        print(f"❌ Margin calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  KITE COMMODITY BROKER API TEST SUITE")
    print("=" * 60)
    print(f"\n📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Test Symbol: {TEST_SYMBOL}")
    print(f"🏛️  Exchange: {TEST_EXCHANGE}")
    print("\n⚠️  NOTE: Order placement tests are in DRY RUN mode")
    print("   Uncomment code in test functions to place real orders\n")
    
    # Step 1: Authentication
    data_fetcher = test_authentication()
    if not data_fetcher:
        print("\n❌ Authentication failed. Cannot proceed with tests.")
        return
    
    # Step 2: Broker initialization
    broker = test_broker_initialization(data_fetcher)
    if not broker:
        print("\n❌ Broker initialization failed. Cannot proceed with tests.")
        return
    
    # Step 3: Test margins
    margins = test_margins(broker)
    
    # Step 4: Test lot size
    lot_size = test_lot_size(broker, TEST_SYMBOL)
    if not lot_size:
        print(f"\n⚠️  Warning: Could not get lot size for {TEST_SYMBOL}")
        print("   Using default lot size: 1")
        lot_size = 1
    
    # Step 5: Test price
    price = test_get_price(broker, TEST_SYMBOL)
    if not price or price <= 0:
        print(f"\n❌ Could not get price for {TEST_SYMBOL}")
        print("   Cannot proceed with order tests")
        return
    
    # Step 6: Test symbol info
    symbol_info = test_symbol_info(broker, TEST_SYMBOL)
    
    # Step 7: Test positions
    positions = test_get_positions(broker)
    
    # Step 8: Test open orders
    orders = test_get_open_orders(broker, TEST_SYMBOL)
    
    # Step 9: Test GTT orders
    gtts = test_get_gtts(broker)
    
    # Step 10: Test margin calculation (using actual order_margins API)
    actual_margin = test_margin_calculation(broker, TEST_SYMBOL, price, lot_size)
    
    # Step 11: Test order placement (dry run)
    test_place_order_dry_run(broker, TEST_SYMBOL, price)
    
    # Step 12: Test GTT placement (dry run)
    if price > 0:
        # Calculate a stop-loss price (5% below current price for LONG)
        stop_loss_price = price * 0.95
        test_place_gtt_dry_run(broker, TEST_SYMBOL, price, stop_loss_price)
    
    # Summary
    print_section("TEST SUMMARY")
    print("✅ All API tests completed!")
    print(f"\n📊 Test Results:")
    print(f"   - Authentication: {'✅' if data_fetcher else '❌'}")
    print(f"   - Broker Init: {'✅' if broker else '❌'}")
    print(f"   - Margins: {'✅' if margins else '❌'}")
    print(f"   - Lot Size: {'✅' if lot_size else '❌'}")
    print(f"   - Price: {'✅' if price else '❌'}")
    print(f"   - Positions: {'✅' if positions is not None else '❌'}")
    print(f"   - Orders: {'✅' if orders is not None else '❌'}")
    print(f"   - GTTs: {'✅' if gtts is not None else '❌'}")
    print("\n💡 To enable live order placement, uncomment code in test functions")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

