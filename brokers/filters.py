from typing import Dict, Any, Optional, Tuple
import math


def _decimal_places(step: float) -> int:
    if step <= 0:
        return 0
    s = f"{step:.16f}".rstrip('0')
    if '.' in s:
        return len(s.split('.')[1])
    return 0


class BinanceSymbolFilters:
    def __init__(self, symbol_info: Dict[str, Any]):
        self.symbol_info = symbol_info or {}
        self.filters = {f['filterType']: f for f in self.symbol_info.get('filters', [])}

        # PRICE_FILTER
        price_filter = self.filters.get('PRICE_FILTER', {})
        self.tick_size = float(price_filter.get('tickSize', '0')) if price_filter else 0.0
        self.min_price = float(price_filter.get('minPrice', '0')) if price_filter else 0.0
        self.max_price = float(price_filter.get('maxPrice', '0')) if price_filter else float('inf')
        self.price_decimals = _decimal_places(self.tick_size) if self.tick_size else 8

        # LOT_SIZE
        lot_filter = self.filters.get('LOT_SIZE', {})
        self.step_size = float(lot_filter.get('stepSize', '0')) if lot_filter else 0.0
        self.min_qty = float(lot_filter.get('minQty', '0')) if lot_filter else 0.0
        self.max_qty = float(lot_filter.get('maxQty', '0')) if lot_filter else float('inf')
        self.qty_decimals = _decimal_places(self.step_size) if self.step_size else 8

        # MIN_NOTIONAL
        notional_filter = self.filters.get('MIN_NOTIONAL', {})
        self.min_notional = float(notional_filter.get('minNotional', '0')) if notional_filter else 0.0

    def format_price(self, price: float) -> float:
        if self.tick_size and self.tick_size > 0:
            steps = math.floor(price / self.tick_size)
            return round(steps * self.tick_size, self.price_decimals)
        return round(price, self.price_decimals)

    def format_qty(self, quantity: float) -> float:
        if self.step_size and self.step_size > 0:
            steps = math.floor(quantity / self.step_size)
            return round(steps * self.step_size, self.qty_decimals)
        return round(quantity, self.qty_decimals)

    def validate(self, price: Optional[float], quantity: float) -> Tuple[bool, Optional[str]]:
        if quantity < self.min_qty:
            return False, f"Quantity {quantity} < minQty {self.min_qty}"
        if quantity > self.max_qty:
            return False, f"Quantity {quantity} > maxQty {self.max_qty}"
        if price is not None:
            if price < self.min_price:
                return False, f"Price {price} < minPrice {self.min_price}"
            if price > self.max_price:
                return False, f"Price {price} > maxPrice {self.max_price}"
            notional = price * quantity
            if notional < self.min_notional:
                return False, f"Notional {notional} < minNotional {self.min_notional}"
        return True, None


