from pathlib import Path

import pandas as pd
import pytest

from analyze_kite_live_logs import (
    build_closed_trades,
    calculate_metrics,
    discover_sessions,
    load_decisions,
    load_statuses,
    load_trades,
    write_outputs,
)


def _write_sample_session(root: Path) -> Path:
    session = root / "NATGASMINI_20260511_101500_live"
    session.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp": "2026-05-11 10:15:00+05:30",
                "symbol": "NATGASMINI",
                "strategy": "MA_Crossover",
                "action": "BUY",
                "price": 100.0,
                "quantity": 1,
                "leverage": 2.0,
                "position_size": 200.0,
                "atr": 5.0,
                "balance": 10000.0,
                "pnl": 0.0,
                "trade_id": 1,
                "status": "open",
                "reject_reason": "",
            },
            {
                "timestamp": "2026-05-11 10:45:00+05:30",
                "symbol": "NATGASMINI",
                "strategy": "MA_Crossover",
                "action": "EXIT",
                "price": 110.0,
                "quantity": 1,
                "leverage": 2.0,
                "position_size": 200.0,
                "atr": 5.0,
                "balance": 10100.0,
                "pnl": 0.01,
                "trade_id": 1,
                "status": "tp_hit",
                "reject_reason": "",
            },
            {
                "timestamp": "2026-05-11 11:00:00+05:30",
                "symbol": "NATGASMINI",
                "strategy": "MA_Crossover",
                "action": "SELL",
                "price": 111.0,
                "quantity": 0,
                "leverage": 0,
                "position_size": 0,
                "atr": 5.0,
                "balance": 10100.0,
                "pnl": 0.0,
                "trade_id": 2,
                "status": "rejected",
                "reject_reason": "Insufficient margin",
            },
        ]
    ).to_csv(session / "trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "timestamp": "2026-05-11 10:15:00+05:30",
                "symbol": "NATGASMINI",
                "strategy": "MA_Crossover",
                "signal_name": "LONG_ENTRY",
                "trade_status": "NONE",
                "current_price": 100.0,
            }
        ]
    ).to_csv(session / "decisions.csv", index=False)
    (session / "status.json").write_text('{"current_balance": 10100, "total_trades": 2}', encoding="utf-8")
    return session


def test_analyzer_builds_closed_trades_and_outputs(tmp_path):
    logs_dir = tmp_path / "kite_logs"
    _write_sample_session(logs_dir)

    sessions = discover_sessions(logs_dir)
    trades = load_trades(sessions)
    decisions = load_decisions(sessions)
    statuses = load_statuses(sessions)
    closed = build_closed_trades(trades)
    metrics = calculate_metrics(trades, closed, decisions)

    assert len(sessions) == 1
    assert len(trades) == 3
    assert len(closed) == 1
    assert closed.iloc[0]["pnl"] == 0.01
    assert closed.iloc[0]["r_multiple"] == 2.0
    assert metrics["rejections"] == 1
    assert metrics["closed_trades"] == 1
    assert not statuses.empty

    output_dir = tmp_path / "analysis"
    write_outputs(output_dir, sessions, trades, closed, decisions, statuses, metrics)

    assert (output_dir / "kite_live_log_analysis.md").exists()
    assert (output_dir / "closed_trades_enriched.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "equity_curve.png").exists()
    assert (output_dir / "cumulative_pnl_rupees.png").exists()


def test_analyzer_pairs_kite_tradebook_and_estimates_brokerage(tmp_path):
    logs_dir = tmp_path / "kite_logs"
    logs_dir.mkdir()
    pd.DataFrame(
        [
            {
                "symbol": "NATGASMINI26FEBFUT",
                "trade_date": "2026-01-14",
                "exchange": "MCX",
                "segment": "COM",
                "trade_type": "sell",
                "quantity": 250,
                "price": 256.8,
                "trade_id": 190146287,
                "order_id": 601419324588170,
                "order_execution_time": "2026-01-14T18:00:04",
                "expiry_date": "2026-02-24",
            },
            {
                "symbol": "NATGASMINI26FEBFUT",
                "trade_date": "2026-01-14",
                "exchange": "MCX",
                "segment": "COM",
                "trade_type": "buy",
                "quantity": 250,
                "price": 256.4,
                "trade_id": 190148398,
                "order_id": 601419324596819,
                "order_execution_time": "2026-01-14T18:05:01",
                "expiry_date": "2026-02-24",
            },
        ]
    ).to_csv(logs_dir / "trades.csv", index=False)

    sessions = discover_sessions(logs_dir)
    trades = load_trades(sessions)
    closed = build_closed_trades(trades, brokerage_rate=0.0003, brokerage_cap=20)
    metrics = calculate_metrics(trades, closed, pd.DataFrame())

    assert len(closed) == 1
    assert closed.iloc[0]["direction"] == "SELL"
    assert closed.iloc[0]["gross_pnl_rupees"] == pytest.approx(100)
    assert round(closed.iloc[0]["brokerage_estimate"], 2) == 38.49
    assert round(closed.iloc[0]["net_pnl_rupees"], 2) == 61.51
    assert metrics["win_rate"] == 1.0
    assert metrics["net_win_rate"] == 1.0
