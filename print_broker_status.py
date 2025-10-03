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
    parser = argparse.ArgumentParser(description="Print broker balances, open orders, and recent trades")
    parser.add_argument("--symbol", help="Trading symbol for fetching trades (e.g., BTCUSDT)")
    parser.add_argument("--days", type=int, default=7, help="Lookback days for trades (default: 7)")
    parser.add_argument("--limit", type=int, default=1000, help="Max trades to fetch (default: 1000)")
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
        trades = broker.get_account_trades(
            symbol=args.symbol,
            start_time=to_ms(start_dt),
            end_time=to_ms(end_dt),
            limit=args.limit,
        )

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
        HIGHLIGHT_ASSETS = ["USDT", "ETH", "BTC"]
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

    print(f"\n📑 Open orders: {len(open_orders) if isinstance(open_orders, list) else 'N/A'}")
    if isinstance(open_orders, list):
        for o in open_orders:
            try:
                print(f"  {o.get('symbol')} #{o.get('orderId')} {o.get('side')} {o.get('origQty')} @ {o.get('price')} - {o.get('status')}")
            except Exception:
                print(f"  {o}")

    if args.symbol:
        print(f"\n🧾 Trades for {args.symbol} (last {args.days} days): {len(trades)}")
        for t in trades:
            try:
                time_str = datetime.utcfromtimestamp(t.get('time', 0)/1000.0).isoformat()
                print(f"  {time_str} id={t.get('id')} {t.get('isBuyer') and 'BUY' or 'SELL'} qty={t.get('qty')} price={t.get('price')} commission={t.get('commission')} {t.get('commissionAsset')}")
            except Exception:
                print(f"  {t}")


if __name__ == "__main__":
    main()


