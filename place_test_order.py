#!/usr/bin/env python3
"""
Script to place a REAL test order on MCX.
⚠️  WARNING: This will use REAL MONEY!

Usage:
    python place_test_order.py

This script will:
1. Authenticate with Kite
2. Show current margin status
3. Show the order details
4. Ask for confirmation
5. Place a BUY order for 1 lot
"""

import sys
from datetime import datetime
from data_fetcher import KiteDataFetcher
from brokers import KiteCommodityBroker
import config as cfg

# Configuration
SYMBOL = "NATGASMINI26FEBFUT"  # Symbol to trade
EXCHANGE = "MCX"
QUANTITY = 1  # 1 lot (minimum)
SIDE = "BUY"  # BUY or SELL
ORDER_TYPE = "MARKET"  # MARKET or LIMIT
STOP_LOSS_PERCENT = 2.0  # Stop loss at 2% below entry price


def main():
    print("\n" + "=" * 60)
    print("  🚨 REAL ORDER PLACEMENT TEST 🚨")
    print("=" * 60)
    print(f"\n📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Symbol: {SYMBOL}")
    print(f"🏛️  Exchange: {EXCHANGE}")
    print(f"📈 Side: {SIDE}")
    print(f"📦 Quantity: {QUANTITY} lot(s)")
    print(f"💰 Order Type: {ORDER_TYPE}")
    
    print("\n⚠️  WARNING: This will place a REAL order with REAL money!")
    print("=" * 60)
    
    # Step 1: Authenticate
    print("\n[1/5] Authenticating with Kite Connect...")
    try:
        data_fetcher = KiteDataFetcher(cfg.KITE_CREDENTIALS, EXCHANGE)
        data_fetcher.authenticate()
        print("✅ Authentication successful!")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return
    
    # Step 2: Initialize broker
    print("\n[2/5] Initializing broker...")
    try:
        broker = KiteCommodityBroker(
            kite=data_fetcher.kite,
            exchange=EXCHANGE
        )
        print("✅ Broker initialized!")
    except Exception as e:
        print(f"❌ Broker initialization failed: {e}")
        return
    
    # Step 3: Check margin
    print("\n[3/5] Checking margin...")
    try:
        margins = broker.check_margins()
        available = margins.get('available', 0)
        actually_enabled = margins.get('actually_enabled', False)
        using_single_ledger = margins.get('using_single_ledger', False)
        
        print(f"   💰 Available Margin: ₹{available:,.2f}")
        print(f"   📊 Trading Enabled: {'✅ YES' if actually_enabled else '❌ NO'}")
        if using_single_ledger:
            print(f"   📋 Using Single Ledger (Equity funds for commodity)")
        
        if not actually_enabled:
            print("\n❌ Commodity trading not enabled. Cannot proceed.")
            return
    except Exception as e:
        print(f"❌ Margin check failed: {e}")
        return
    
    # Step 4: Get current price and margin requirement
    print("\n[4/5] Getting price and margin requirement...")
    try:
        price = broker.get_price(SYMBOL)
        print(f"   💰 Current Price: ₹{price:,.2f}")
        
        # Get actual margin required
        order_margins = broker.get_order_margins(
            symbol=SYMBOL,
            transaction_type=SIDE,
            quantity=QUANTITY,
            price=price,
            order_type=ORDER_TYPE
        )
        margin_required = order_margins.get('total', 0)
        print(f"   📊 Margin Required: ₹{margin_required:,.2f}")
        print(f"   📊 Available After Order: ₹{available - margin_required:,.2f}")
        
        if available < margin_required:
            print(f"\n⚠️  WARNING: Insufficient margin!")
            print(f"   Need: ₹{margin_required:,.2f}, Have: ₹{available:,.2f}")
            proceed = input("\nProceed anyway? (may fail) [y/N]: ").strip().lower()
            if proceed != 'y':
                print("❌ Order cancelled by user.")
                return
    except Exception as e:
        print(f"⚠️  Could not get margin requirement: {e}")
        print("   Proceeding anyway...")
    
    # Calculate stop-loss price
    if SIDE == "BUY":
        stop_loss_price = round(price * (1 - STOP_LOSS_PERCENT / 100), 2)
    else:
        stop_loss_price = round(price * (1 + STOP_LOSS_PERCENT / 100), 2)
    
    # Step 5: Confirmation
    print("\n" + "=" * 60)
    print("  📋 ORDER SUMMARY")
    print("=" * 60)
    print(f"   Symbol: {SYMBOL}")
    print(f"   Side: {SIDE}")
    print(f"   Quantity: {QUANTITY} lot(s)")
    print(f"   Order Type: {ORDER_TYPE}")
    print(f"   Entry Price: ₹{price:,.2f} (Market)")
    print(f"   Stop Loss: ₹{stop_loss_price:,.2f} ({STOP_LOSS_PERCENT}% {'below' if SIDE == 'BUY' else 'above'})")
    print(f"   Max Loss: ₹{abs(price - stop_loss_price) * QUANTITY:,.2f}")
    print("=" * 60)
    
    confirm = input("\n🚨 CONFIRM: Place this REAL order with stop-loss? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("❌ Order cancelled by user.")
        return
    
    # Place the order!
    print("\n[5/5] Placing order...")
    try:
        order = broker.place_order(
            symbol=SYMBOL,
            side=SIDE,
            order_type=ORDER_TYPE,
            quantity=QUANTITY
        )
        order_id = order.get('orderId') or order.get('order_id')
        print(f"✅ Main order placed: {order_id}")
    except Exception as e:
        print(f"\n❌ Order placement failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Place stop-loss GTT order
    print("\n[6/6] Placing stop-loss GTT order...")
    try:
        # For LONG position (BUY), stop-loss is a SELL order
        # For SHORT position (SELL), stop-loss is a BUY order
        sl_transaction_type = "SELL" if SIDE == "BUY" else "BUY"
        
        gtt_order = broker.place_gtt_order(
            symbol=SYMBOL,
            trigger_price=stop_loss_price,
            last_price=price,  # Current market price
            transaction_type=sl_transaction_type,
            quantity=QUANTITY,
            order_price=stop_loss_price  # Same as trigger for market-like execution
        )
        gtt_id = gtt_order.get('trigger_id') or gtt_order.get('gtt_id')
        print(f"✅ Stop-loss GTT placed: {gtt_id}")
        
        print("\n" + "=" * 60)
        print("  ✅ ORDER + STOP-LOSS PLACED SUCCESSFULLY!")
        print("=" * 60)
        print(f"   Main Order ID: {order_id}")
        print(f"   Stop-Loss GTT ID: {gtt_id}")
        print(f"   Entry Price: ₹{price:,.2f}")
        print(f"   Stop-Loss Trigger: ₹{stop_loss_price:,.2f}")
        print("\n💡 Check your Kite app/web to verify.")
        print("💡 The GTT will automatically trigger if price hits stop-loss.")
        print("💡 Run 'python close_position.py' to manually close.")
    except Exception as e:
        print(f"\n⚠️  Stop-loss GTT placement failed: {e}")
        print(f"   Main order was placed successfully (ID: {order_id})")
        print(f"   You should manually set a stop-loss in Kite app!")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

