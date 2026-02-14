#!/usr/bin/env python3
"""
Unit tests for KiteTradingEngine / TradingEngine stop-loss handling.

These tests verify that:
1. When use_gtt_for_stop_loss is False, TradingEngine.execute_trade() does NOT
   place a broker-side GTT stop-loss order (no gtt_id stored on the trade).
2. When close_trades() is called with exit_type='sl_hit' for a Kite-style broker
   trade that has no gtt_id (engine-managed SL), a MARKET exit order is sent.
"""

from datetime import datetime

import pandas as pd

from trading_engine import TradingEngine


class _DummyStrategy:
    """Minimal strategy stub."""

    def __init__(self, name: str = "dummy"):
        self.name = name

    def calculate_atr(self, data: pd.DataFrame):
        # Simple constant ATR for tests
        return pd.Series([1.0], index=data.index)

    def calculate_trade_levels(self, price, position_type, atr):
        # Simple symmetric SL/TP around price for tests
        return price + 10.0, price - 10.0

    def calculate_leverage_position_size(
        self,
        entry_price,
        position_type,
        atr,
        available_balance,
        max_leverage,
        max_loss_percent,
    ):
        # Return fixed leverage, position_size, quantity
        leverage = 2.0
        position_size = available_balance * 0.5
        quantity = position_size / entry_price
        return leverage, position_size, quantity

    def generate_signals(self, data: pd.DataFrame):
        return pd.DataFrame()

    def get_trade_history(self) -> pd.DataFrame:
        return pd.DataFrame()


class _FakeKiteBroker:
    """
    Minimal fake broker that looks like a Kite broker for the purposes
    of these tests (has place_gtt_order and delete_gtt attributes).
    """

    def __init__(self):
        self.orders = []
        self.gtt_orders = []

    def check_margins(self):
        return {"available": 1_000_000.0, "utilised": 0.0}

    def get_order_margins(self, symbol, transaction_type, quantity, price, order_type):
        return {"total": 10000.0}

    def get_symbol_filters(self, symbol):
        # Return a dummy lot_size so SL/TP validation runs
        return {"lot_size": 250}

    def place_order(self, symbol, side, order_type, quantity, price=None):
        order = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "orderId": f"ORD-{len(self.orders)+1}",
        }
        self.orders.append(order)
        return order

    def place_gtt_order(
        self, symbol, trigger_price, last_price, transaction_type, quantity, order_price
    ):
        gtt = {
            "symbol": symbol,
            "trigger_price": trigger_price,
            "last_price": last_price,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_price": order_price,
            "gtt_id": f"GTT-{len(self.gtt_orders)+1}",
        }
        self.gtt_orders.append(gtt)
        return gtt

    def delete_gtt(self, gtt_id):
        # Simulate successful delete
        self.gtt_orders = [g for g in self.gtt_orders if g.get("gtt_id") != gtt_id]


def test_execute_trade_does_not_place_gtt_when_disabled():
    """
    When TradingEngine.use_gtt_for_stop_loss is False, execute_trade() should
    NOT place a GTT stop-loss order, even for a Kite-style broker.
    """
    engine = TradingEngine(initial_balance=100000.0, max_leverage=5.0, max_loss_percent=2.0)
    engine.symbol = "NATGASMINI26FEBFUT"
    engine.broker = _FakeKiteBroker()
    engine.use_broker = True
    engine.use_gtt_for_stop_loss = False

    strategy = _DummyStrategy(name="dummy")
    # Minimal data to drive ATR calculation
    data = pd.DataFrame({"Close": [250.0]}, index=[pd.Timestamp("2026-01-01 09:15:00")])

    trade = engine.execute_trade(
        strategy=strategy,
        action="BUY",
        price=250.0,
        timestamp=datetime(2026, 1, 1, 9, 15, 0),
        data=data,
    )

    # Entry order should have been placed
    assert engine.broker.orders, "Expected at least one entry order to be placed"
    # No GTT orders should be placed when use_gtt_for_stop_loss is False
    assert not engine.broker.gtt_orders, "GTT stop-loss should NOT be placed when disabled"
    # Trade record should not have any gtt_id
    assert trade.get("gtt_id") in (None, ""), "Trade should not store a GTT id when disabled"


def test_close_trades_places_market_exit_for_engine_managed_sl():
    """
    If a trade was created without a GTT (engine-managed SL) and close_trades()
    is called with exit_type='sl_hit', the engine should send a MARKET exit
    order to the broker.
    """
    engine = TradingEngine(initial_balance=100000.0, max_leverage=5.0, max_loss_percent=2.0)
    engine.symbol = "NATGASMINI26FEBFUT"
    engine.broker = _FakeKiteBroker()
    engine.use_broker = True
    engine.use_gtt_for_stop_loss = False

    strategy = _DummyStrategy(name="dummy")

    # Simulate an open trade that has no gtt_id (engine-managed SL)
    trade = {
        "id": 1,
        "strategy": strategy.name,
        "action": "BUY",
        "entry_price": 250.0,
        "entry_time": datetime(2026, 1, 1, 9, 15, 0),
        "quantity": 1,
        "leverage": 2.0,
        "position_size": 250.0,
        "atr": 1.0,
        "take_profit": 260.0,
        "stop_loss": 240.0,
        "status": "open",
        "pnl": 0.0,
        "broker_order_id": "ORD-1",
        "stop_loss_order_id": None,
        "gtt_id": None,
        "lot_size": 250,
        "margin_used": 10000.0,
        "effective_leverage": 2.0,
    }

    engine.active_trades.append(trade)
    engine.trade_history.append(trade)

    # Balance before close is margin-removed account (typical live behaviour)
    engine.current_balance = engine.initial_balance - trade["margin_used"]

    exit_price = 240.0
    closed = engine.close_trades(
        strategy=strategy,
        position_type="LONG",
        price=exit_price,
        timestamp=datetime(2026, 1, 1, 10, 0, 0),
        exit_type="sl_hit",
    )

    assert len(closed) == 1
    # For engine-managed SL (no GTT id), we expect a MARKET exit order to be sent
    assert any(
        o["order_type"] == "MARKET" and o["side"] == "SELL" for o in engine.broker.orders
    ), "Expected a MARKET SELL exit order for engine-managed stop-loss"


