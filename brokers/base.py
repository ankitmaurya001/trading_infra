from typing import Protocol, Dict, Any, Optional


class Broker(Protocol):
    def ping(self) -> bool:
        ...

    def get_account(self) -> Dict[str, Any]:
        ...

    def get_balances(self) -> Dict[str, float]:
        ...

    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        ...

    def get_price(self, symbol: str) -> float:
        ...

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        ...

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
        ...

    def get_account_trades(self, symbol: Optional[str] = None, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: Optional[int] = None) -> Any:
        ...

    def get_all_orders(self, symbol: Optional[str] = None, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: Optional[int] = None) -> Any:
        ...

    def get_all_orders_all_symbols(self, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: Optional[int] = None, max_symbols: int = 50) -> Any:
        ...


