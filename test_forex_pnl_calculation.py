import pandas as pd
from datetime import datetime

from trading_engine import TradingEngine


class _DummyStrategy:
    """
    Minimal strategy stub for testing TradingEngine.close_trades().
    """

    def __init__(self, name: str = "dummy"):
        self.name = name

    def get_trade_history(self) -> pd.DataFrame:
        # Engine uses this only to try and infer detailed status; returning empty is fine.
        return pd.DataFrame()


def test_close_trades_uses_quantity_only_for_fx_pnl():
    """
    For FX/spot-style trades (no exchange lot_size), TradingEngine.close_trades()
    should calculate dollar PnL using quantity only:
        PnL_$ = (exit_price - entry_price) * quantity     (for LONG)
    and then convert that to percentage based on margin_used.
    """
    engine = TradingEngine(initial_balance=10000.0, max_leverage=10.0, max_loss_percent=0.2)
    strategy = _DummyStrategy()

    entry_price = 184.14
    exit_price = 183.99
    quantity = 167.49428492392414
    margin_used = 10000.0

    # Simulate an open FX trade in the engine (no lot_size stored)
    trade = {
        "id": 1,
        "strategy": strategy.name,
        "action": "BUY",
        "entry_price": entry_price,
        "entry_time": datetime(2026, 1, 2, 1, 0, 0),
        "quantity": quantity,
        "leverage": 3.0,
        "position_size": entry_price * quantity,
        "atr": 0.12,
        "take_profit": None,
        "stop_loss": None,
        "status": "open",
        "pnl": 0.0,
        "lot_size": None,  # Critical: FX/spot trades should not rely on lot_size
        "margin_used": margin_used,
        "effective_leverage": 3.0,
    }

    engine.active_trades.append(trade)
    engine.trade_history.append(trade)

    # Balance before close is margin-removed account (as in live engine)
    engine.current_balance = 0.0

    closed = engine.close_trades(strategy, "LONG", exit_price, datetime(2026, 1, 2, 8, 30, 0))
    assert len(closed) == 1

    closed_trade = closed[0]

    # Expected raw dollar PnL using quantity only (FX/Spot logic)
    expected_dollar_pnl = (exit_price - entry_price) * quantity

    # Engine stores trade['pnl'] as return-on-margin
    assert pytest.approx(closed_trade["pnl"], rel=1e-9) == expected_dollar_pnl / margin_used

    # Balance should be previous balance + margin_used + dollar_pnl
    assert pytest.approx(engine.current_balance, rel=1e-9) == margin_used + expected_dollar_pnl


