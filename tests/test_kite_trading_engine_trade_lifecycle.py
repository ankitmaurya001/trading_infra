from datetime import datetime

import pytest

pd = pytest.importorskip("pandas")

from strategies import BaseStrategy, PositionType, Signal, Trade
from trading_engine import TradingEngine


class DummyStrategy(BaseStrategy):
    """Minimal deterministic strategy test-double for TradingEngine tests."""

    def __init__(self):
        super().__init__(name="DummyStrategy", risk_reward_ratio=2.0, atr_period=3)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        frame["Signal"] = Signal.HOLD.value
        frame["Take_Profit"] = pd.NA
        frame["Stop_Loss"] = pd.NA
        return frame


class DataBrokerStub:
    """Deterministic broker fake with predefined margin and order responses."""

    def __init__(
        self,
        available_margin: float = 1000.0,
        lot_margin: float = 200.0,
        lot_size: int = 10,
    ):
        self.available_margin = available_margin
        self.lot_margin = lot_margin
        self.lot_size = lot_size
        self.placed_orders = []
        self.placed_gtts = []
        self.deleted_gtts = []

    def check_margins(self):
        return {"available": self.available_margin, "utilised": 0.0}

    def get_order_margins(self, symbol, transaction_type, quantity, price, order_type):
        return {"total": self.lot_margin}

    def place_order(self, symbol, side, order_type, quantity, price=None):
        order_id = f"order-{len(self.placed_orders) + 1}"
        payload = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
        }
        self.placed_orders.append(payload)
        return {"order_id": order_id}

    def place_gtt_order(
        self, symbol, trigger_price, last_price, transaction_type, quantity, order_price
    ):
        gtt_id = f"gtt-{len(self.placed_gtts) + 1}"
        payload = {
            "trigger_id": gtt_id,
            "symbol": symbol,
            "trigger_price": trigger_price,
            "last_price": last_price,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_price": order_price,
        }
        self.placed_gtts.append(payload)
        return {"trigger_id": gtt_id}

    def get_symbol_filters(self, symbol):
        return {"lot_size": self.lot_size}

    def delete_gtt(self, trigger_id):
        self.deleted_gtts.append(trigger_id)

    def get_orders(self):
        return []

    def cancel_order(self, symbol, order_id):
        return {"cancelled": True, "order_id": order_id}


def _sample_ohlc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "High": [101, 102, 103, 104],
            "Low": [99, 100, 101, 102],
            "Close": [100, 101, 102, 103],
        }
    )


def _append_closed_strategy_trade(
    strategy: DummyStrategy,
    status: str,
    entry_price: float,
    exit_price: float,
    ts: datetime,
):
    strategy.trades.append(
        Trade(
            entry_date=pd.Timestamp(ts),
            entry_price=entry_price,
            exit_date=pd.Timestamp(ts),
            exit_price=exit_price,
            position_type=PositionType.LONG,
            status=status,
            pnl=0.0,
        )
    )


def _build_engine_with_broker():
    engine = TradingEngine(
        initial_balance=1000.0, max_leverage=2.0, max_loss_percent=1.0
    )
    engine.symbol = "NATGASMINI26FEBFUT"
    broker = DataBrokerStub(available_margin=1000.0, lot_margin=200.0, lot_size=10)
    engine.broker = broker
    engine.use_broker = True
    return engine, broker


def test_execute_trade_places_entry_and_gtt_stoploss_for_kite_style_broker():
    engine, broker = _build_engine_with_broker()
    strategy = DummyStrategy()

    trade = engine.execute_trade(
        strategy=strategy,
        action="BUY",
        price=100.0,
        timestamp=datetime(2024, 1, 1, 9, 15),
        data=_sample_ohlc(),
    )

    assert trade["status"] == "open"
    assert trade["broker_order_id"] == "order-1"
    assert trade["gtt_id"] == "gtt-1"
    assert trade["quantity"] == 1
    assert trade["lot_size"] == 10
    assert len(engine.active_trades) == 1
    assert len(broker.placed_orders) == 1
    assert len(broker.placed_gtts) == 1
    assert engine.current_balance == 800.0


def test_stop_loss_close_uses_gtt_path_without_sending_duplicate_exit_order():
    engine, broker = _build_engine_with_broker()
    strategy = DummyStrategy()
    ts = datetime(2024, 1, 1, 9, 15)

    engine.execute_trade(strategy, "BUY", 100.0, ts, _sample_ohlc())
    _append_closed_strategy_trade(
        strategy, status="sl_hit", entry_price=100.0, exit_price=95.0, ts=ts
    )

    closed = engine.close_trades(
        strategy=strategy,
        position_type="LONG",
        price=95.0,
        timestamp=datetime(2024, 1, 1, 9, 30),
        exit_type="sl_hit",
    )

    assert len(closed) == 1
    assert closed[0]["status"] == "sl_hit"
    assert len(engine.active_trades) == 0
    assert len(broker.placed_orders) == 1  # entry only, no extra exit order


def test_take_profit_close_places_market_exit_order():
    engine, broker = _build_engine_with_broker()
    strategy = DummyStrategy()
    ts = datetime(2024, 1, 1, 9, 15)

    engine.execute_trade(strategy, "BUY", 100.0, ts, _sample_ohlc())
    _append_closed_strategy_trade(
        strategy, status="tp_hit", entry_price=100.0, exit_price=110.0, ts=ts
    )

    closed = engine.close_trades(
        strategy=strategy,
        position_type="LONG",
        price=110.0,
        timestamp=datetime(2024, 1, 1, 9, 30),
        exit_type="tp_hit",
    )

    assert len(closed) == 1
    assert closed[0]["status"] == "tp_hit"
    assert len(engine.active_trades) == 0
    assert len(broker.placed_orders) == 2
    assert broker.placed_orders[-1]["side"] == "SELL"
    assert broker.placed_orders[-1]["order_type"] == "MARKET"
