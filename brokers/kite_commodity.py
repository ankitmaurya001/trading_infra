#!/usr/bin/env python3
"""
Kite Connect Broker Implementation for MCX Commodity Trading
Handles lot-based trading, margin checking, GTT orders for overnight positions
"""

from typing import Dict, Any, Optional, List
import logging
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)


class KiteCommodityBroker:
    """
    Broker implementation for Kite Connect MCX commodity trading.
    Handles lot-based positions, margin management, and GTT orders.
    """
    
    def __init__(self, kite: KiteConnect, exchange: str = "MCX"):
        """
        Initialize Kite Commodity Broker.
        
        Args:
            kite: Authenticated KiteConnect instance
            exchange: Exchange name (default: MCX)
        """
        self.kite = kite
        self.exchange = exchange
        self.instruments_cache = {}
        self.lot_sizes_cache = {}
        
    def ping(self) -> bool:
        """Check if broker connection is alive."""
        try:
            self.kite.profile()
            return True
        except Exception as e:
            logger.error(f"Broker ping failed: {e}")
            return False
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            return self.kite.profile()
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {}
    
    def get_balances(self) -> Dict[str, float]:
        """Get account balances."""
        try:
            margins = self.kite.margins()
            balances = {}
            if 'commodity' in margins:
                commodity_margin = margins['commodity']
                balances['available'] = commodity_margin.get('available', {}).get('cash', 0.0)
                balances['utilised'] = commodity_margin.get('utilised', {}).get('debits', 0.0)
                balances['total'] = commodity_margin.get('available', {}).get('cash', 0.0) + \
                                  commodity_margin.get('utilised', {}).get('debits', 0.0)
            return balances
        except Exception as e:
            logger.error(f"Error getting balances: {e}")
            return {}
    
    def check_margins(self) -> Dict[str, Any]:
        """
        Check available commodity margins.
        
        Returns:
            Dictionary with margin information:
            {
                'available': float,
                'utilised': float,
                'total': float,
                'enabled': bool,  # Whether commodity trading is enabled
                'raw': dict  # Full margin response
            }
        """
        try:
            margins = self.kite.margins()
            print(f"[INFO] margins = {margins}")
            commodity_margin = margins.get('commodity', {})
            
            # Check if commodity trading is enabled (API flag - may be inaccurate)
            enabled = commodity_margin.get('enabled', False)
            
            # Actually check if commodity trading works by checking for positions
            # If user has positions, trading is actually enabled regardless of API flag
            actually_enabled = enabled
            if not enabled:
                try:
                    # Check if there are any commodity positions
                    # Use 'net' positions only to avoid double-counting (day positions are included in net)
                    positions = self.kite.positions()
                    net_positions = positions.get('net', [])
                    
                    # Filter for MCX positions with non-zero quantity (actual open positions)
                    mcx_positions_with_qty = [
                        p for p in net_positions 
                        if p.get('exchange') == 'MCX' and p.get('quantity', 0) != 0
                    ]
                    # Also check for any MCX trading activity (even closed positions)
                    mcx_positions_any = [
                        p for p in net_positions 
                        if p.get('exchange') == 'MCX'
                    ]
                    
                    if mcx_positions_with_qty:
                        actually_enabled = True
                        logger.info(f"Commodity trading enabled (found {len(mcx_positions_with_qty)} open MCX positions)")
                    elif mcx_positions_any:
                        # Even if quantity is 0, having MCX positions means trading is enabled
                        actually_enabled = True
                        logger.info(f"Commodity trading enabled (found {len(mcx_positions_any)} MCX positions, some may be closed)")
                    
                    # Also check if user profile shows MCX in exchanges
                    if not actually_enabled:
                        try:
                            profile = self.kite.profile()
                            exchanges = profile.get('exchanges', [])
                            if 'MCX' in exchanges:
                                actually_enabled = True
                                logger.info(f"Commodity trading enabled (MCX found in user exchanges: {exchanges})")
                        except Exception as profile_e:
                            logger.debug(f"Could not check profile for MCX access: {profile_e}")
                    
                except Exception as e:
                    logger.debug(f"Could not check positions to verify commodity trading: {e}")
            
            # Try different possible field structures
            available = 0.0
            utilised = 0.0
            
            # Check available margin - try multiple possible paths
            available_dict = commodity_margin.get('available', {})
            if isinstance(available_dict, dict):
                # Try different field names - cash is the main one
                available = available_dict.get('cash', 0.0)
                # Also check live_balance (current balance)
                if available == 0.0:
                    available = available_dict.get('live_balance', available_dict.get('opening_balance', 0.0))
            else:
                available = float(available_dict) if available_dict else 0.0
            
            # Check utilised margin
            utilised_dict = commodity_margin.get('utilised', {})
            if isinstance(utilised_dict, dict):
                utilised = utilised_dict.get('debits', utilised_dict.get('span', 0.0))
            else:
                utilised = float(utilised_dict) if utilised_dict else 0.0
            
            # Use 'net' field if available (net = available - utilised)
            net = commodity_margin.get('net', None)
            if net is not None:
                # If net is available, use it as the available margin
                available = float(net) if net > 0 else available
            
            # SINGLE LEDGER FACILITY: If commodity margin is 0 but equity margin is available,
            # Zerodha allows using equity funds for commodity trading through single ledger
            # See: https://support.zerodha.com/category/funds/adding-funds/other-fund-related-queries/articles/can-i-use-the-same-funds-for-trading-on-equity-as-well-as-commodity
            equity_margin = margins.get('equity', {})
            equity_available = 0.0
            equity_net = equity_margin.get('net', 0.0)
            if equity_net > 0:
                equity_available = float(equity_net)
            else:
                equity_available_dict = equity_margin.get('available', {})
                if isinstance(equity_available_dict, dict):
                    equity_available = equity_available_dict.get('live_balance', equity_available_dict.get('cash', 0.0))
            
            # If commodity shows 0 but equity has funds, use equity (single ledger)
            using_single_ledger = False
            if available == 0.0 and equity_available > 0 and actually_enabled:
                available = equity_available
                using_single_ledger = True
                logger.info(f"Using single ledger: Equity funds (₹{equity_available:.2f}) available for commodity trading")
            
            total = available + utilised
            
            # Log the raw structure for debugging
            logger.debug(f"Raw commodity margin structure: {commodity_margin}")
            
            return {
                'available': float(available),
                'utilised': float(utilised),
                'total': float(total),
                'enabled': enabled,
                'actually_enabled': actually_enabled,
                'using_single_ledger': using_single_ledger,
                'net': float(net) if net is not None else 0.0,
                'raw': commodity_margin
            }
        except Exception as e:
            logger.error(f"Error checking margins: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'available': 0.0,
                'utilised': 0.0,
                'total': 0.0,
                'enabled': False,
                'actually_enabled': False,
                'using_single_ledger': False,
                'net': 0.0,
                'raw': {}
            }
    
    def check_equity_margins(self) -> Dict[str, Any]:
        """
        Check equity margins (in case funds are in equity segment).
        
        Returns:
            Dictionary with equity margin information
        """
        try:
            margins = self.kite.margins()
            equity_margin = margins.get('equity', {})
            
            available_dict = equity_margin.get('available', {})
            if isinstance(available_dict, dict):
                available = available_dict.get('cash', available_dict.get('live_balance', 0.0))
            else:
                available = float(available_dict) if available_dict else 0.0
            
            net = equity_margin.get('net', 0.0)
            
            return {
                'available': float(available),
                'net': float(net),
                'enabled': equity_margin.get('enabled', False),
                'raw': equity_margin
            }
        except Exception as e:
            logger.error(f"Error checking equity margins: {e}")
            return {
                'available': 0.0,
                'net': 0.0,
                'enabled': False,
                'raw': {}
            }
    
    def get_order_margins(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int = 1,
        price: Optional[float] = None,
        order_type: str = "MARKET"
    ) -> Dict[str, Any]:
        """
        Get actual margin requirement for a specific order using Kite's order_margins API.
        This gives the REAL margin requirement, not an estimate.
        
        Args:
            symbol: Trading symbol
            transaction_type: 'BUY' or 'SELL'
            quantity: Number of lots (default: 1)
            price: Price for limit orders (optional for market orders)
            order_type: 'MARKET' or 'LIMIT'
            
        Returns:
            Dictionary with margin information:
            {
                'total': float,  # Total margin required
                'available': float,  # Available margin after this order
                'raw': dict  # Full order margins response
            }
        """
        try:
            # Map transaction type
            if transaction_type.upper() == 'BUY':
                kite_transaction_type = self.kite.TRANSACTION_TYPE_BUY
            elif transaction_type.upper() == 'SELL':
                kite_transaction_type = self.kite.TRANSACTION_TYPE_SELL
            else:
                raise ValueError(f"Invalid transaction_type: {transaction_type}")
            
            # Map order type
            order_type_map = {
                'MARKET': self.kite.ORDER_TYPE_MARKET,
                'LIMIT': self.kite.ORDER_TYPE_LIMIT,
            }
            kite_order_type = order_type_map.get(order_type.upper(), self.kite.ORDER_TYPE_MARKET)
            
            # Build order parameters for margin calculation
            order_params = {
                'exchange': self.kite.EXCHANGE_MCX,
                'tradingsymbol': symbol,
                'transaction_type': kite_transaction_type,
                'quantity': quantity,
                'order_type': kite_order_type,
                'product': self.kite.PRODUCT_NRML,
            }
            
            # Add price for limit orders
            if kite_order_type == self.kite.ORDER_TYPE_LIMIT:
                if price is None:
                    # Get current price for limit order margin calculation
                    price = self.get_price(symbol)
                order_params['price'] = float(price)
            
            # Get order margins
            order_margins = self.kite.order_margins([order_params])
            
            if order_margins and len(order_margins) > 0:
                margin_data = order_margins[0]
                total_margin = float(margin_data.get('total', 0.0))
                
                # Get available margin after this order
                current_margins = self.check_margins()
                available_after = current_margins.get('available', 0.0) - total_margin
                
                return {
                    'total': total_margin,
                    'available_after': available_after,
                    'raw': margin_data
                }
            else:
                logger.warning("No margin data returned from order_margins API")
                return {
                    'total': 0.0,
                    'available_after': 0.0,
                    'raw': {}
                }
                
        except Exception as e:
            logger.error(f"Error getting order margins: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'total': 0.0,
                'available_after': 0.0,
                'raw': {}
            }
    
    def get_lot_size(self, symbol: str) -> Optional[int]:
        """
        Get lot size for a commodity symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'GOLDM25DECFUT')
            
        Returns:
            Lot size (quantity per lot) or None if not found
        """
        # Check cache first
        if symbol in self.lot_sizes_cache:
            return self.lot_sizes_cache[symbol]
        
        try:
            # Load instruments if not cached
            if self.exchange not in self.instruments_cache:
                self.instruments_cache[self.exchange] = self.kite.instruments(self.exchange)
            
            instruments = self.instruments_cache[self.exchange]
            
            # Find the instrument
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    lot_size = instrument.get('lot_size', 1)
                    self.lot_sizes_cache[symbol] = lot_size
                    logger.info(f"Lot size for {symbol}: {lot_size}")
                    return lot_size
            
            logger.warning(f"Lot size not found for symbol: {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting lot size for {symbol}: {e}")
            return None
    
    def get_price(self, symbol: str) -> float:
        """
        Get current price (LTP) for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Last traded price
        """
        try:
            ltp_data = self.kite.ltp(f"{self.exchange}:{symbol}")
            price = ltp_data[f"{self.exchange}:{symbol}"]["last_price"]
            return float(price)
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float = 1,
        price: Optional[float] = None,
        time_in_force: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Place an order on Kite Connect.
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            order_type: 'MARKET', 'LIMIT', etc.
            quantity: Number of lots (default: 1)
            price: Price for limit orders
            time_in_force: Time in force (not used for Kite, but kept for interface compatibility)
            
        Returns:
            Order response dictionary with order_id
        """
        try:
            # Map side to Kite transaction type
            if side.upper() == 'BUY':
                transaction_type = self.kite.TRANSACTION_TYPE_BUY
            elif side.upper() == 'SELL':
                transaction_type = self.kite.TRANSACTION_TYPE_SELL
            else:
                raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'")
            
            # Map order type to Kite order type
            # Note: SL-M is ORDER_TYPE_SLM (no hyphen) in KiteConnect
            order_type_map = {
                'MARKET': self.kite.ORDER_TYPE_MARKET,
                'LIMIT': self.kite.ORDER_TYPE_LIMIT,
                'SL': self.kite.ORDER_TYPE_SL,
                'SL-M': self.kite.ORDER_TYPE_SLM,
                'SLM': self.kite.ORDER_TYPE_SLM,
            }
            kite_order_type = order_type_map.get(order_type.upper(), self.kite.ORDER_TYPE_MARKET)
            
            # Build order parameters
            order_params = {
                'variety': self.kite.VARIETY_REGULAR,
                'exchange': self.kite.EXCHANGE_MCX,
                'tradingsymbol': symbol,
                'transaction_type': transaction_type,
                'quantity': int(quantity),  # Quantity in lots
                'order_type': kite_order_type,
                'product': self.kite.PRODUCT_NRML,  # For overnight positions
                'validity': self.kite.VALIDITY_DAY,
            }
            
            # Add price for limit orders
            if kite_order_type in [self.kite.ORDER_TYPE_LIMIT, self.kite.ORDER_TYPE_SL]:
                if price is None:
                    raise ValueError(f"Price required for {order_type} order")
                order_params['price'] = float(price)
            
            # Place the order
            order_id = self.kite.place_order(**order_params)
            
            logger.info(f"Order placed: {order_id} - {side} {quantity} lots of {symbol} @ {price if price else 'MARKET'}")
            
            return {
                'orderId': str(order_id),
                'order_id': str(order_id),
                'status': 'placed'
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    def place_gtt_order(
        self,
        symbol: str,
        trigger_price: float,
        last_price: float,
        transaction_type: str,
        quantity: int = 1,
        order_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a Good Till Triggered (GTT) order for stop-loss.
        GTT orders persist overnight and across trading sessions.
        
        Args:
            symbol: Trading symbol
            trigger_price: Price at which GTT should trigger
            last_price: Last traded price (for GTT validation)
            transaction_type: 'SELL' for LONG exit, 'BUY' for SHORT exit
            quantity: Number of lots (default: 1)
            order_price: Price for the limit order when triggered (default: trigger_price)
            
        Returns:
            GTT order response with gtt_id
        """
        try:
            # Map transaction type
            if transaction_type.upper() == 'SELL':
                kite_transaction_type = self.kite.TRANSACTION_TYPE_SELL
            elif transaction_type.upper() == 'BUY':
                kite_transaction_type = self.kite.TRANSACTION_TYPE_BUY
            else:
                raise ValueError(f"Invalid transaction_type: {transaction_type}")
            
            # Use trigger_price as order_price if not specified
            if order_price is None:
                order_price = trigger_price
            
            # Create GTT order
            gtt_response = self.kite.place_gtt(
                trigger_type=self.kite.GTT_TYPE_SINGLE,
                tradingsymbol=symbol,
                exchange=self.kite.EXCHANGE_MCX,
                trigger_values=[trigger_price],
                last_price=last_price,
                orders=[{
                    "transaction_type": kite_transaction_type,
                    "quantity": quantity,
                    "price": order_price,
                    "order_type": self.kite.ORDER_TYPE_LIMIT,
                    "product": self.kite.PRODUCT_NRML
                }]
            )
            
            # Extract trigger_id from response (API returns {'trigger_id': 123456})
            if isinstance(gtt_response, dict):
                gtt_id = gtt_response.get('trigger_id')
            else:
                gtt_id = gtt_response
            
            logger.info(f"GTT order placed: {gtt_id} - {symbol} @ trigger {trigger_price}, order {order_price}")
            
            return {
                'gtt_id': str(gtt_id),
                'trigger_id': str(gtt_id),
                'status': 'placed'
            }
            
        except Exception as e:
            logger.error(f"Error placing GTT order: {e}")
            raise
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions from Kite.
        
        Returns:
            List of position dictionaries
        """
        try:
            positions = self.kite.positions()
            # Return net positions (day + net)
            net_positions = positions.get('net', [])
            return net_positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders from Kite.
        
        Args:
            symbol: Optional symbol to filter orders
            
        Returns:
            List of open order dictionaries
        """
        try:
            orders = self.kite.orders()
            # Filter for open orders
            open_orders = [o for o in orders if o['status'] in ['OPEN', 'TRIGGER PENDING']]
            
            if symbol:
                open_orders = [o for o in open_orders if o.get('tradingsymbol') == symbol]
            
            return open_orders
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def get_gtts(self) -> List[Dict[str, Any]]:
        """
        Get all GTT orders.
        
        Returns:
            List of GTT order dictionaries
        """
        try:
            gtts = self.kite.get_gtts()
            return gtts
        except Exception as e:
            logger.error(f"Error getting GTTs: {e}")
            return []
    
    def get_gtt(self, gtt_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific GTT order by ID.
        
        Args:
            gtt_id: GTT order ID
            
        Returns:
            GTT order dictionary or None if not found
        """
        try:
            gtt = self.kite.get_gtt(gtt_id)
            return gtt
        except Exception as e:
            logger.error(f"Error getting GTT {gtt_id}: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
            
        Returns:
            Cancellation response
        """
        try:
            result = self.kite.cancel_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=order_id
            )
            logger.info(f"Order cancelled: {order_id}")
            return result
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            raise
    
    def delete_gtt(self, gtt_id: str) -> Dict[str, Any]:
        """
        Delete a GTT order.
        
        Args:
            gtt_id: GTT order ID to delete
            
        Returns:
            Deletion response
        """
        try:
            result = self.kite.delete_gtt(gtt_id)
            logger.info(f"GTT deleted: {gtt_id}")
            return result
        except Exception as e:
            logger.error(f"Error deleting GTT {gtt_id}: {e}")
            raise
    
    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information (lot size, tick size, etc.).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Symbol information dictionary
        """
        try:
            if self.exchange not in self.instruments_cache:
                self.instruments_cache[self.exchange] = self.kite.instruments(self.exchange)
            
            instruments = self.instruments_cache[self.exchange]
            
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    return {
                        'lot_size': instrument.get('lot_size', 1),
                        'tick_size': instrument.get('tick_size', 0.01),
                        'instrument_token': instrument.get('instrument_token'),
                        'exchange': instrument.get('exchange'),
                        'segment': instrument.get('segment'),
                    }
            
            return {}
        except Exception as e:
            logger.error(f"Error getting symbol filters for {symbol}: {e}")
            return {}

