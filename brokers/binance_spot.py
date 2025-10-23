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
        Handles Binance's 24-hour limitation by chunking requests when needed.
        """
        if symbol is None:
            return []
        
        # If no time range specified, get recent trades
        if start_time is None and end_time is None:
            params: Dict[str, Any] = {"symbol": symbol}
            if limit is not None:
                params["limit"] = limit
            return self.client.get_my_trades(**params)
        
        # Handle 24-hour limitation by chunking requests
        all_trades = []
        current_start = start_time
        chunk_size_ms = 24 * 60 * 60 * 1000  # 24 hours in milliseconds
        
        while current_start is not None and (end_time is None or current_start < end_time):
            current_end = min(current_start + chunk_size_ms, end_time) if end_time else current_start + chunk_size_ms
            
            params: Dict[str, Any] = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": current_end
            }
            if limit is not None:
                params["limit"] = limit
            
            try:
                chunk_trades = self.client.get_my_trades(**params)
                all_trades.extend(chunk_trades)
                
                # If we got fewer trades than requested, we've reached the end
                if len(chunk_trades) < (limit or 1000):
                    break
                    
            except Exception as e:
                # If we hit an error, try without time constraints
                if "More than 24 hours" in str(e):
                    params.pop("startTime", None)
                    params.pop("endTime", None)
                    chunk_trades = self.client.get_my_trades(**params)
                    all_trades.extend(chunk_trades)
                    break
                else:
                    raise e
            
            current_start = current_end + 1
            
            # Safety check to prevent infinite loops
            if len(all_trades) > (limit or 10000):
                break
        
        # Sort by time and apply limit if needed
        all_trades.sort(key=lambda x: x.get('time', 0))
        if limit is not None and len(all_trades) > limit:
            all_trades = all_trades[-limit:]
            
        return all_trades

    def get_all_orders(self, symbol: Optional[str] = None, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: Optional[int] = None):
        """
        Fetch all orders (historical and current) for a symbol.
        Handles Binance's 24-hour limitation by chunking requests when needed.
        
        Args:
            symbol: Trading symbol (required by Binance API)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds  
            limit: Maximum number of orders to return (default 500, max 1000)
            
        Returns:
            List of order dictionaries
        """
        if symbol is None:
            return []
        
        # If no time range specified, get recent orders
        if start_time is None and end_time is None:
            params: Dict[str, Any] = {"symbol": symbol}
            if limit is not None:
                params["limit"] = limit
            return self.client.get_all_orders(**params)
        
        # Handle 24-hour limitation by chunking requests
        all_orders = []
        current_start = start_time
        chunk_size_ms = 24 * 60 * 60 * 1000  # 24 hours in milliseconds
        
        while current_start is not None and (end_time is None or current_start < end_time):
            current_end = min(current_start + chunk_size_ms, end_time) if end_time else current_start + chunk_size_ms
            
            params: Dict[str, Any] = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": current_end
            }
            if limit is not None:
                params["limit"] = limit
            
            try:
                chunk_orders = self.client.get_all_orders(**params)
                all_orders.extend(chunk_orders)
                
                # If we got fewer orders than requested, we've reached the end
                if len(chunk_orders) < (limit or 1000):
                    break
                    
            except Exception as e:
                # If we hit an error, try without time constraints
                if "More than 24 hours" in str(e):
                    params.pop("startTime", None)
                    params.pop("endTime", None)
                    chunk_orders = self.client.get_all_orders(**params)
                    all_orders.extend(chunk_orders)
                    break
                else:
                    raise e
            
            current_start = current_end + 1
            
            # Safety check to prevent infinite loops
            if len(all_orders) > (limit or 10000):
                break
        
        # Sort by time and apply limit if needed
        all_orders.sort(key=lambda x: x.get('time', 0))
        if limit is not None and len(all_orders) > limit:
            all_orders = all_orders[-limit:]
            
        return all_orders

    def get_all_orders_all_symbols(self, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: Optional[int] = None, max_symbols: int = 50):
        """
        Fetch all orders across multiple symbols.
        This method gets orders from the most active trading pairs.
        
        Args:
            start_time: Start time in milliseconds
            end_time: End time in milliseconds  
            limit: Maximum number of orders per symbol
            max_symbols: Maximum number of symbols to check
            
        Returns:
            List of order dictionaries from all symbols
        """
        all_orders = []
        
        # Get list of active trading pairs
        try:
            exchange_info = self.client.get_exchange_info()
            symbols = [symbol['symbol'] for symbol in exchange_info.get('symbols', [])]
            
            # Filter for USDT pairs and limit the number
            usdt_symbols = [s for s in symbols if s.endswith('USDT')][:max_symbols]
            
            for symbol in usdt_symbols:
                try:
                    symbol_orders = self.get_all_orders(
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                        limit=limit
                    )
                    all_orders.extend(symbol_orders)
                    
                    # Stop if we have enough orders
                    if limit and len(all_orders) >= limit * 2:  # Get a bit more than needed
                        break
                        
                except Exception as e:
                    # Skip symbols that fail (might not have trading history)
                    continue
                    
        except Exception as e:
            # Fallback: try common symbols
            common_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']
            for symbol in common_symbols:
                try:
                    symbol_orders = self.get_all_orders(
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                        limit=limit
                    )
                    all_orders.extend(symbol_orders)
                except Exception:
                    continue
        
        # Sort by time and apply limit if needed
        all_orders.sort(key=lambda x: x.get('time', 0), reverse=True)
        if limit and len(all_orders) > limit:
            all_orders = all_orders[:limit]
            
        return all_orders


