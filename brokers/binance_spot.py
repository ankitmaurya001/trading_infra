from typing import Dict, Any, Optional

from binance.client import Client
from .filters import BinanceSymbolFilters


class BinanceSpotBroker:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client = Client(api_key, api_secret, testnet=testnet)
        if testnet:
            # Ensure base endpoint for testnet spot
            self.client.API_URL = 'https://testnet.binance.vision/api'

    def ping(self) -> bool:
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def get_account(self) -> Dict[str, Any]:
        return self.client.get_account()

    def get_balances(self) -> Dict[str, float]:
        account = self.client.get_account()
        balances = {}
        for b in account.get('balances', []):
            free = float(b.get('free', '0'))
            locked = float(b.get('locked', '0'))
            total = free + locked
            if total > 0:
                balances[b['asset']] = total
        return balances

    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        info = self.client.get_symbol_info(symbol)
        return {f['filterType']: f for f in info.get('filters', [])}

    def get_price(self, symbol: str) -> float:
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Apply exchange filters
        symbol_info = self.client.get_symbol_info(symbol)
        filt = BinanceSymbolFilters(symbol_info)
        fmt_price = filt.format_price(price) if price is not None else None
        fmt_qty = filt.format_qty(quantity)
        ok, msg = filt.validate(fmt_price, fmt_qty)
        if not ok:
            raise ValueError(f"Order validation failed: {msg}")

        params: Dict[str, Any] = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': fmt_qty,
        }
        
        # Handle different order types
        if order_type in ['LIMIT', 'STOP_LOSS_LIMIT']:
            params['price'] = fmt_price
            params['timeInForce'] = time_in_force or 'GTC'
        elif order_type in ['STOP_MARKET', 'STOP_LOSS']:
            params['stopPrice'] = fmt_price
        elif order_type == 'LIMIT' and fmt_price is not None:
            params['price'] = fmt_price
            if time_in_force is not None:
                params['timeInForce'] = time_in_force
                
        return self.client.create_order(**params)

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        return self.client.cancel_order(symbol=symbol, orderId=order_id)

    def get_open_orders(self, symbol: Optional[str] = None):
        if symbol:
            return self.client.get_open_orders(symbol=symbol)
        return self.client.get_open_orders()

    def get_account_trades(self, symbol: Optional[str] = None, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: Optional[int] = None):
        """
        Fetch account trade history. If symbol is provided, fetch trades for that symbol; otherwise,
        aggregate trades across symbols using Binance's my_trades endpoint per symbol.

        Note: Binance Spot API requires symbol for my_trades. If symbol is None, this returns an empty list.
        """
        if symbol is None:
            return []
        params: Dict[str, Any] = {"symbol": symbol}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if limit is not None:
            params["limit"] = limit
        return self.client.get_my_trades(**params)


