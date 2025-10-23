#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import config as cfg
from brokers import BinanceSpotBroker


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def main():
    parser = argparse.ArgumentParser(description="Print broker balances, open orders, recent trades, and order history")
    parser.add_argument("--symbol", help="Trading symbol for fetching trades and orders (e.g., BTCUSDT)")
    parser.add_argument("--days", type=int, default=7, help="Lookback days for trades and orders (default: 7)")
    parser.add_argument("--limit", type=int, default=1000, help="Max trades/orders to fetch (default: 1000)")
    parser.add_argument("--orders", action="store_true", help="Include order history (requires --symbol)")
    parser.add_argument("--all-orders", action="store_true", help="Include order history from all symbols (overrides --symbol)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--testnet", action="store_true", help="Use Binance Spot testnet")
    mode.add_argument("--live", action="store_true", help="Use Binance Spot live")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output JSON instead of pretty text")

    args = parser.parse_args()

    api_key = os.environ.get("BINANCE_API_KEY") or cfg.BINANCE_API_KEY
    api_secret = os.environ.get("BINANCE_API_SECRET") or cfg.BINANCE_SECRET_KEY

    testnet = True if args.testnet else (False if args.live else True)

    broker = BinanceSpotBroker(api_key=api_key, api_secret=api_secret, testnet=testnet)

    # Ping
    ping_ok = broker.ping()

    # Balances
    balances = broker.get_balances()

    # Open orders
    open_orders = broker.get_open_orders()

    # Trades
    trades = []
    if args.symbol:
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=args.days)
        try:
            trades = broker.get_account_trades(
                symbol=args.symbol,
                start_time=to_ms(start_dt),
                end_time=to_ms(end_dt),
                limit=args.limit,
            )
        except Exception as e:
            # Try without time constraints if time range fails
            try:
                trades = broker.get_account_trades(symbol=args.symbol, limit=args.limit)
            except Exception as e2:
                trades = []

    # Order History
    order_history = []
    if args.all_orders:
        # Get orders from all symbols
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=args.days)
        try:
            order_history = broker.get_all_orders_all_symbols(
                start_time=to_ms(start_dt),
                end_time=to_ms(end_dt),
                limit=args.limit,
            )
        except Exception as e:
            # Try without time constraints if time range fails
            try:
                order_history = broker.get_all_orders_all_symbols(limit=args.limit)
            except Exception as e2:
                order_history = []
    elif args.orders and args.symbol:
        # Get orders from specific symbol
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=args.days)
        try:
            order_history = broker.get_all_orders(
                symbol=args.symbol,
                start_time=to_ms(start_dt),
                end_time=to_ms(end_dt),
                limit=args.limit,
            )
        except Exception as e:
            # Try without time constraints if time range fails
            try:
                order_history = broker.get_all_orders(symbol=args.symbol, limit=args.limit)
            except Exception as e2:
                order_history = []

    output: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "environment": "testnet" if testnet else "live",
        "ping": ping_ok,
        "balances": balances,
        "open_orders_count": len(open_orders) if isinstance(open_orders, list) else None,
        "open_orders": open_orders,
        "symbol": args.symbol,
        "trades_count": len(trades) if isinstance(trades, list) else None,
        "trades": trades,
        "order_history_count": len(order_history) if isinstance(order_history, list) else None,
        "order_history": order_history,
    }

    if args.as_json:
        print(json.dumps(output, indent=2, default=str))
        return

    # Pretty print
    print(f"📡 Broker ping: {'OK' if ping_ok else 'FAILED'}")
    print(f"🌐 Environment: {'TESTNET' if testnet else 'LIVE'}")
    print("💼 Balances:")
    if not balances:
        print("  (none)")
    else:
        HIGHLIGHT_ASSETS = ["USDT", "ETH", "BTC", "BNB", "SOL"]
        highlighted_assets = []
        other_assets = []
        for asset, qty in sorted(balances.items()):
            if asset in HIGHLIGHT_ASSETS:
                highlighted_assets.append((asset, qty))
            else:
                other_assets.append((asset, qty))
        for asset, qty in other_assets:
            print(f"  {asset}: {qty}")
        # Print highlighted assets after all others
        for asset, qty in highlighted_assets:
            print(f"  \033[1;33m{asset}: {qty}\033[0m  <-- HIGHLIGHTED")

    print(f"\n📑 Open orders (currently active): {len(open_orders) if isinstance(open_orders, list) else 'N/A'}")
    if isinstance(open_orders, list):
        for o in open_orders:
            try:
                print(f"  {o.get('symbol')} #{o.get('orderId')} {o.get('side')} {o.get('origQty')} @ {o.get('price')} - {o.get('status')}")
            except Exception:
                print(f"  {o}")
    else:
        print("  (none)")

    if args.symbol:
        print(f"\n🧾 Trades for {args.symbol} (last {args.days} days): {len(trades)}")
        for t in trades:
            try:
                time_str = datetime.utcfromtimestamp(t.get('time', 0)/1000.0).isoformat()
                print(f"  {time_str} id={t.get('id')} {t.get('isBuyer') and 'BUY' or 'SELL'} qty={t.get('qty')} price={t.get('price')} commission={t.get('commission')} {t.get('commissionAsset')}")
            except Exception:
                print(f"  {t}")

    if args.all_orders:
        print(f"\n📋 Order History from ALL SYMBOLS (completed/canceled orders, last {args.days} days): {len(order_history)}")
        if order_history:
            for o in order_history:
                try:
                    time_str = datetime.utcfromtimestamp(o.get('time', 0)/1000.0).isoformat()
                    status = o.get('status', 'UNKNOWN')
                    side = o.get('side', 'UNKNOWN')
                    order_type = o.get('type', 'UNKNOWN')
                    qty = o.get('origQty', '0')
                    price = o.get('price', '0')
                    filled_qty = o.get('executedQty', '0')
                    symbol = o.get('symbol', 'UNKNOWN')
                    
                    # Color code status
                    if status == 'FILLED':
                        status_color = '\033[1;32m'  # Green
                    elif status == 'CANCELED':
                        status_color = '\033[1;31m'  # Red
                    elif status == 'REJECTED':
                        status_color = '\033[1;31m'  # Red
                    else:
                        status_color = '\033[1;33m'  # Yellow
                    
                    print(f"  {time_str} {symbol} #{o.get('orderId')} {side} {order_type} {qty} @ {price} (filled: {filled_qty}) - {status_color}{status}\033[0m")
                except Exception:
                    print(f"  {o}")
        else:
            print("  (no completed/canceled orders found)")
            print("  Note: Open orders are shown above, not in order history")
    elif args.orders and args.symbol:
        print(f"\n📋 Order History for {args.symbol} (completed/canceled orders, last {args.days} days): {len(order_history)}")
        if order_history:
            for o in order_history:
                try:
                    time_str = datetime.utcfromtimestamp(o.get('time', 0)/1000.0).isoformat()
                    status = o.get('status', 'UNKNOWN')
                    side = o.get('side', 'UNKNOWN')
                    order_type = o.get('type', 'UNKNOWN')
                    qty = o.get('origQty', '0')
                    price = o.get('price', '0')
                    filled_qty = o.get('executedQty', '0')
                    
                    # Color code status
                    if status == 'FILLED':
                        status_color = '\033[1;32m'  # Green
                    elif status == 'CANCELED':
                        status_color = '\033[1;31m'  # Red
                    elif status == 'REJECTED':
                        status_color = '\033[1;31m'  # Red
                    else:
                        status_color = '\033[1;33m'  # Yellow
                    
                    print(f"  {time_str} #{o.get('orderId')} {side} {order_type} {qty} @ {price} (filled: {filled_qty}) - {status_color}{status}\033[0m")
                except Exception:
                    print(f"  {o}")
        else:
            print("  (no completed/canceled orders found)")
            print("  Note: Open orders are shown above, not in order history")


if __name__ == "__main__":
    main()


