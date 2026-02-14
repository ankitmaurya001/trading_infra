import pandas as pd


def test_dashboard_process_trade_data_includes_atr():
    """
    Ensure TradingDashboard.process_trade_data carries ATR from the entry record into the combined trade row.
    """
    # `trading_dashboard.py` imports streamlit at module import time, but streamlit is not required
    # for `process_trade_data()`. Stub it so this unit test can run in minimal environments.
    import sys
    import types

    if "streamlit" not in sys.modules:
        sys.modules["streamlit"] = types.SimpleNamespace()

    from trading_dashboard import TradingDashboard

    dash = TradingDashboard(log_folder="logs")

    # Minimal synthetic trade history with entry+exit rows (same trade_id)
    dash.trade_history_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01 09:15:00",
                "symbol": "NATGASMINI26FEBFUT",
                "strategy": "ma",
                "action": "BUY",
                "price": 264.50,
                "quantity": 1,
                "leverage": 4.0,
                "position_size": 0.0,
                "atr": 1.72,
                "balance": 9000.0,
                "pnl": 0.0,
                "trade_id": 1,
                "status": "open",
            },
            {
                "timestamp": "2026-01-01 10:15:00",
                "symbol": "NATGASMINI26FEBFUT",
                "strategy": "ma",
                "action": "EXIT",
                "price": 276.80,
                "quantity": 1,
                "leverage": 4.0,
                "position_size": 0.0,
                "atr": 1.72,
                "balance": 12400.0,
                "pnl": 0.2353,
                "trade_id": 1,
                "status": "closed",
            },
        ]
    )

    processed = dash.process_trade_data()
    assert not processed.empty
    assert "atr" in processed.columns
    assert processed.loc[0, "atr"] == 1.72


