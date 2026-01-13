#!/usr/bin/env python3
"""
Script to check open positions and close them.
⚠️  WARNING: This will place REAL SELL orders!

Usage:
    python close_position.py

This script will:
1. Authenticate with Kite
2. Show all open MCX positions
3. Let you select which position to close
4. Ask for confirmation
5. Place a SELL order to close the position
"""

import sys
from datetime import datetime
from data_fetcher import KiteDataFetcher
from brokers import KiteCommodityBroker
import config as cfg

EXCHANGE = "MCX"


def main():
    print("\n" + "=" * 60)
    print("  📊 POSITION CHECKER & CLOSER")
    print("=" * 60)
    print(f"\n📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Authenticate
    print("\n[1/4] Authenticating with Kite Connect...")
    try:
        data_fetcher = KiteDataFetcher(cfg.KITE_CREDENTIALS, EXCHANGE)
        data_fetcher.authenticate()
        print("✅ Authentication successful!")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return
    
    # Step 2: Initialize broker
    print("\n[2/4] Initializing broker...")
    try:
        broker = KiteCommodityBroker(
            kite=data_fetcher.kite,
            exchange=EXCHANGE
        )
        print("✅ Broker initialized!")
    except Exception as e:
        print(f"❌ Broker initialization failed: {e}")
        return
    
    # Step 3: Get positions
    print("\n[3/4] Fetching positions...")
    try:
        positions = data_fetcher.kite.positions()
        
        # Filter for MCX positions with non-zero quantity
        net_positions = positions.get('net', [])
        mcx_positions = [
            p for p in net_positions 
            if p.get('exchange') == 'MCX' and p.get('quantity', 0) != 0
        ]
        
        if not mcx_positions:
            print("\n📭 No open MCX positions found.")
            print("   Nothing to close!")
            return
        
        print(f"\n📊 Found {len(mcx_positions)} open MCX position(s):")
        print("-" * 60)
        
        for i, pos in enumerate(mcx_positions, 1):
            symbol = pos.get('tradingsymbol', 'Unknown')
            qty = pos.get('quantity', 0)
            avg_price = pos.get('average_price', 0)
            ltp = pos.get('last_price', 0)
            pnl = pos.get('pnl', 0)
            product = pos.get('product', 'Unknown')
            
            # Determine position type
            position_type = "LONG 📈" if qty > 0 else "SHORT 📉"
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            
            print(f"\n  [{i}] {symbol}")
            print(f"      Position: {position_type}")
            print(f"      Quantity: {abs(qty)} lot(s)")
            print(f"      Avg Price: ₹{avg_price:,.2f}")
            print(f"      Current Price: ₹{ltp:,.2f}")
            print(f"      P&L: {pnl_emoji} ₹{pnl:,.2f}")
            print(f"      Product: {product}")
        
        print("\n" + "-" * 60)
        
    except Exception as e:
        print(f"❌ Failed to fetch positions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Select position to close
    print("\n[4/4] Close a position...")
    
    if len(mcx_positions) == 1:
        selection = input(f"\nClose position [{mcx_positions[0]['tradingsymbol']}]? [y/N]: ").strip().lower()
        if selection != 'y':
            print("❌ Cancelled by user.")
            return
        selected_pos = mcx_positions[0]
    else:
        selection = input(f"\nEnter position number to close (1-{len(mcx_positions)}) or 'q' to quit: ").strip()
        if selection.lower() == 'q':
            print("❌ Cancelled by user.")
            return
        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(mcx_positions):
                print("❌ Invalid selection.")
                return
            selected_pos = mcx_positions[idx]
        except ValueError:
            print("❌ Invalid input.")
            return
    
    # Get position details
    symbol = selected_pos.get('tradingsymbol')
    qty = selected_pos.get('quantity', 0)
    avg_price = selected_pos.get('average_price', 0)
    ltp = selected_pos.get('last_price', 0)
    pnl = selected_pos.get('pnl', 0)
    
    # Determine order side (opposite of position)
    if qty > 0:
        # Long position - need to SELL
        close_side = "SELL"
        close_qty = abs(qty)
    else:
        # Short position - need to BUY
        close_side = "BUY"
        close_qty = abs(qty)
    
    # Get fresh price
    try:
        current_price = broker.get_price(symbol)
    except:
        current_price = ltp
    
    # Calculate expected P&L
    if qty > 0:  # Long
        expected_pnl = (current_price - avg_price) * abs(qty)
    else:  # Short
        expected_pnl = (avg_price - current_price) * abs(qty)
    
    print("\n" + "=" * 60)
    print("  📋 CLOSE ORDER SUMMARY")
    print("=" * 60)
    print(f"   Symbol: {symbol}")
    print(f"   Action: {close_side} (close position)")
    print(f"   Quantity: {close_qty} lot(s)")
    print(f"   Order Type: MARKET")
    print(f"   Entry Price: ₹{avg_price:,.2f}")
    print(f"   Current Price: ₹{current_price:,.2f}")
    print(f"   Expected P&L: {'🟢' if expected_pnl >= 0 else '🔴'} ₹{expected_pnl:,.2f}")
    print("=" * 60)
    
    # Check for GTT orders on this symbol
    print("\n📋 Checking for GTT orders on this symbol...")
    gtts_to_cancel = []
    try:
        gtts = data_fetcher.kite.get_gtts()
        for gtt in gtts:
            if gtt.get('condition', {}).get('tradingsymbol') == symbol and gtt.get('status') == 'active':
                gtts_to_cancel.append(gtt)
        
        if gtts_to_cancel:
            print(f"   ⚠️  Found {len(gtts_to_cancel)} active GTT order(s) on {symbol}:")
            for gtt in gtts_to_cancel:
                trigger_values = gtt.get('condition', {}).get('trigger_values', [])
                trigger_price = trigger_values[0] if trigger_values else 'Unknown'
                print(f"      GTT ID: {gtt.get('id')} @ ₹{trigger_price}")
            print(f"   ⚠️  These will be CANCELLED along with closing the position.")
        else:
            print("   ✅ No active GTT orders found.")
    except Exception as e:
        print(f"   ⚠️  Could not check GTT orders: {e}")
    
    confirm = input("\n🚨 CONFIRM: Close position and cancel GTTs? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled by user.")
        return
    
    # Cancel GTT orders first
    if gtts_to_cancel:
        print("\n🗑️  Cancelling GTT orders...")
        for gtt in gtts_to_cancel:
            try:
                gtt_id = gtt.get('id')
                data_fetcher.kite.delete_gtt(gtt_id)
                print(f"   ✅ Cancelled GTT: {gtt_id}")
            except Exception as e:
                print(f"   ⚠️  Failed to cancel GTT {gtt.get('id')}: {e}")
    
    # Place the close order
    print(f"\n📤 Placing {close_side} order...")
    try:
        order = broker.place_order(
            symbol=symbol,
            side=close_side,
            order_type='MARKET',
            quantity=close_qty
        )
        print("\n" + "=" * 60)
        print("  ✅ POSITION CLOSED SUCCESSFULLY!")
        print("=" * 60)
        print(f"   Order ID: {order.get('orderId') or order.get('order_id')}")
        print(f"   Status: {order.get('status', 'placed')}")
        if gtts_to_cancel:
            print(f"   GTTs Cancelled: {len(gtts_to_cancel)}")
        print(f"\n💰 Realized P&L: ₹{expected_pnl:,.2f} (approx)")
        print("\n💡 Check your Kite app/web to verify.")
    except Exception as e:
        print(f"\n❌ Order placement failed: {e}")
        import traceback
        traceback.print_exc()
        return


def show_orders():
    """Show recent orders."""
    print("\n" + "=" * 60)
    print("  📋 RECENT ORDERS")
    print("=" * 60)
    
    # Authenticate
    try:
        data_fetcher = KiteDataFetcher(cfg.KITE_CREDENTIALS, EXCHANGE)
        data_fetcher.authenticate()
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return
    
    try:
        orders = data_fetcher.kite.orders()
        
        # Filter for today's MCX orders
        mcx_orders = [o for o in orders if o.get('exchange') == 'MCX']
        
        if not mcx_orders:
            print("\n📭 No MCX orders found today.")
            return
        
        print(f"\n📋 Found {len(mcx_orders)} MCX order(s) today:")
        print("-" * 60)
        
        for order in mcx_orders[-10:]:  # Last 10 orders
            order_id = order.get('order_id', 'Unknown')
            symbol = order.get('tradingsymbol', 'Unknown')
            side = order.get('transaction_type', 'Unknown')
            qty = order.get('quantity', 0)
            price = order.get('average_price', 0) or order.get('price', 0)
            status = order.get('status', 'Unknown')
            order_time = order.get('order_timestamp', '')
            
            status_emoji = "✅" if status == "COMPLETE" else "⏳" if status == "OPEN" else "❌"
            
            print(f"\n  {status_emoji} Order ID: {order_id}")
            print(f"     Symbol: {symbol}")
            print(f"     Side: {side}")
            print(f"     Quantity: {qty}")
            print(f"     Price: ₹{price:,.2f}")
            print(f"     Status: {status}")
            print(f"     Time: {order_time}")
        
        print("\n" + "-" * 60)
        
    except Exception as e:
        print(f"❌ Failed to fetch orders: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check and close MCX positions")
    parser.add_argument('--orders', '-o', action='store_true', help="Show recent orders only")
    args = parser.parse_args()
    
    try:
        if args.orders:
            show_orders()
        else:
            main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

