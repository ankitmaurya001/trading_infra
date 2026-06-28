#!/usr/bin/env python3
"""
In-depth Kite live trading log analyzer.

This script scans Kite live-trading session folders, combines trades/decisions/status
files, calculates execution, P&L, risk, drawdown, timing, and data-quality metrics,
and writes a self-contained report plus CSV/PNG artifacts.

Expected session layout (also supports a root folder containing these files directly):
    kite_logs/{symbol}_{timestamp}_{mode}/trades.csv
    kite_logs/{symbol}_{timestamp}_{mode}/decisions.csv
    kite_logs/{symbol}_{timestamp}_{mode}/status.json
    kite_logs/{symbol}_{timestamp}_{mode}/market_data.csv

Usage:
    python analyze_kite_live_logs.py --logs-dir kite_logs
    python analyze_kite_live_logs.py --logs-dir kite_logs --output-dir kite_log_analysis
    python analyze_kite_live_logs.py --logs-dir logs --session NATGASMINI_20260511_101500_live
    python analyze_kite_live_logs.py --logs-dir kite_logs --brokerage-rate 0.0003 --brokerage-cap 20
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLOSED_STATUSES = {"closed", "tp_hit", "sl_hit", "reversed"}
TRADE_REQUIRED_COLUMNS = [
    "timestamp",
    "symbol",
    "strategy",
    "action",
    "price",
    "quantity",
    "leverage",
    "position_size",
    "atr",
    "balance",
    "pnl",
    "trade_id",
    "status",
    "reject_reason",
]
DEFAULT_BROKERAGE_RATE = 0.0003
DEFAULT_BROKERAGE_CAP = 20.0


@dataclass(frozen=True)
class SessionFiles:
    """Files that belong to one trading session."""

    session_name: str
    session_path: Path
    trades_path: Path
    decisions_path: Optional[Path]
    status_path: Optional[Path]
    market_data_path: Optional[Path]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float while tolerating blanks, NaN, and infinities."""
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _fmt_number(value: Any, digits: int = 2) -> str:
    number = _safe_float(value, np.nan)
    if math.isnan(number):
        return "n/a"
    return f"{number:,.{digits}f}"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    number = _safe_float(value, np.nan)
    if math.isnan(number):
        return "n/a"
    return f"{number * 100:,.{digits}f}%"


def _fmt_rupees(value: Any, digits: int = 2) -> str:
    number = _safe_float(value, np.nan)
    if math.isnan(number):
        return "n/a"
    return f"Rs {number:,.{digits}f}"


def _series_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _series_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV while keeping the analyzer tolerant of empty files."""
    if path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _coerce_trade_frame(df: pd.DataFrame, session: SessionFiles) -> pd.DataFrame:
    """Normalize a trades.csv frame and annotate it with session metadata."""
    df = df.copy()
    is_kite_tradebook = {"trade_type", "order_execution_time"}.issubset(df.columns)

    if is_kite_tradebook:
        if "timestamp" not in df.columns:
            df["timestamp"] = df["order_execution_time"]
        if "action" not in df.columns:
            df["action"] = df["trade_type"]
        if "status" not in df.columns:
            df["status"] = "executed"
        if "strategy" not in df.columns:
            df["strategy"] = "KiteTradebook"
        if "position_size" not in df.columns and {"price", "quantity"}.issubset(df.columns):
            df["position_size"] = pd.to_numeric(df["price"], errors="coerce") * pd.to_numeric(df["quantity"], errors="coerce")
        if "balance" not in df.columns:
            df["balance"] = np.nan
        if "pnl" not in df.columns:
            df["pnl"] = np.nan
        if "reject_reason" not in df.columns:
            df["reject_reason"] = ""

    for column in TRADE_REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df["session"] = session.session_name
    df["session_path"] = str(session.session_path)
    df["source_format"] = "kite_tradebook" if is_kite_tradebook else "live_session"
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    numeric_columns = ["price", "quantity", "leverage", "position_size", "atr", "balance", "pnl"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["trade_id"] = df["trade_id"].astype(str)
    df["status"] = df["status"].fillna("").astype(str).str.strip().str.lower()
    df["action"] = df["action"].fillna("").astype(str).str.strip().str.upper()
    df["action"] = df["action"].replace({"BUY": "BUY", "SELL": "SELL", "EXIT": "EXIT"})
    df["symbol"] = df["symbol"].fillna("UNKNOWN").astype(str)
    df["strategy"] = df["strategy"].fillna("UNKNOWN").astype(str)
    df["reject_reason"] = df["reject_reason"].fillna("").astype(str)
    df["is_exit"] = df["action"].eq("EXIT") | df["status"].isin(CLOSED_STATUSES)
    df["is_rejected"] = df["status"].eq("rejected")
    return df.sort_values(["timestamp", "session", "trade_id"], na_position="last")


def discover_sessions(logs_dir: Path, session_filters: Optional[Sequence[str]] = None) -> List[SessionFiles]:
    """Find session folders containing trades.csv under a Kite log directory."""
    if not logs_dir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {logs_dir}")

    filters = set(session_filters or [])
    candidates: List[Path] = []
    root_trade_file = logs_dir / "trades.csv"
    if root_trade_file.exists():
        candidates.append(logs_dir)
    candidates.extend(path.parent for path in logs_dir.rglob("trades.csv") if path.parent != logs_dir)

    sessions: List[SessionFiles] = []
    for session_path in sorted(set(candidates)):
        session_name = session_path.name
        if filters and session_name not in filters:
            continue
        trades_path = session_path / "trades.csv"
        sessions.append(
            SessionFiles(
                session_name=session_name,
                session_path=session_path,
                trades_path=trades_path,
                decisions_path=(session_path / "decisions.csv") if (session_path / "decisions.csv").exists() else None,
                status_path=(session_path / "status.json") if (session_path / "status.json").exists() else None,
                market_data_path=(session_path / "market_data.csv") if (session_path / "market_data.csv").exists() else None,
            )
        )
    return sessions


def load_trades(sessions: Sequence[SessionFiles]) -> pd.DataFrame:
    """Load and combine all trade logs."""
    frames: List[pd.DataFrame] = []
    for session in sessions:
        df = _read_csv(session.trades_path)
        if not df.empty:
            frames.append(_coerce_trade_frame(df, session))
    if not frames:
        return pd.DataFrame(columns=TRADE_REQUIRED_COLUMNS + ["session", "session_path", "source_format", "is_exit", "is_rejected"])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp", na_position="last")


def load_decisions(sessions: Sequence[SessionFiles]) -> pd.DataFrame:
    """Load and combine decision logs when present."""
    frames: List[pd.DataFrame] = []
    for session in sessions:
        if not session.decisions_path:
            continue
        df = _read_csv(session.decisions_path)
        if df.empty:
            continue
        df = df.copy()
        df["session"] = session.session_name
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        for column in ["current_price", "current_balance", "take_profit", "stop_loss"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp", na_position="last")


def load_statuses(sessions: Sequence[SessionFiles]) -> pd.DataFrame:
    """Load terminal status.json snapshots when present."""
    rows: List[Dict[str, Any]] = []
    for session in sessions:
        if not session.status_path:
            continue
        with session.status_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["session"] = session.session_name
        payload["session_path"] = str(session.session_path)
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def _order_brokerage(notional: float, brokerage_rate: float, brokerage_cap: float) -> float:
    if math.isnan(notional) or notional <= 0:
        return 0.0
    percentage_brokerage = notional * brokerage_rate
    if brokerage_cap <= 0:
        return percentage_brokerage
    return min(percentage_brokerage, brokerage_cap)


def _brokerage_for_closed_lot(
    entry_notional: float,
    exit_notional: float,
    entry_fraction: float,
    exit_fraction: float,
    brokerage_rate: float,
    brokerage_cap: float,
) -> float:
    entry_charge = _order_brokerage(entry_notional, brokerage_rate, brokerage_cap) * entry_fraction
    exit_charge = _order_brokerage(exit_notional, brokerage_rate, brokerage_cap) * exit_fraction
    return entry_charge + exit_charge


def _add_money_columns(
    row: Dict[str, Any],
    brokerage_rate: float,
    brokerage_cap: float,
    entry_fraction: float = 1.0,
    exit_fraction: float = 1.0,
) -> Dict[str, Any]:
    entry_price = _safe_float(row.get("entry_price"), np.nan)
    exit_price = _safe_float(row.get("exit_price"), np.nan)
    quantity = _safe_float(row.get("quantity"), 0.0)
    direction = str(row.get("direction", "")).upper()
    direction_sign = 1 if direction == "BUY" else -1 if direction == "SELL" else 0
    entry_notional = entry_price * quantity if not math.isnan(entry_price) else np.nan
    exit_notional = exit_price * quantity if not math.isnan(exit_price) else np.nan
    turnover = _safe_float(entry_notional, 0.0) + _safe_float(exit_notional, 0.0)

    gross_pnl = row.get("gross_pnl_rupees")
    if gross_pnl is None or pd.isna(gross_pnl):
        price_pnl = (exit_price - entry_price) * quantity * direction_sign if direction_sign and not math.isnan(entry_price) and not math.isnan(exit_price) else np.nan
        balance_pnl = np.nan
        entry_balance = _safe_float(row.get("entry_balance"), np.nan)
        exit_balance = _safe_float(row.get("exit_balance"), np.nan)
        if not math.isnan(entry_balance) and not math.isnan(exit_balance):
            balance_pnl = exit_balance - entry_balance
        fractional_pnl = _safe_float(row.get("pnl"), np.nan)
        if not math.isnan(balance_pnl):
            gross_pnl = balance_pnl
        elif not math.isnan(fractional_pnl) and not math.isnan(entry_balance):
            gross_pnl = fractional_pnl * entry_balance
        else:
            gross_pnl = price_pnl

    brokerage = _brokerage_for_closed_lot(
        _safe_float(entry_notional, 0.0),
        _safe_float(exit_notional, 0.0),
        entry_fraction,
        exit_fraction,
        brokerage_rate,
        brokerage_cap,
    )
    net_pnl = _safe_float(gross_pnl, 0.0) - brokerage
    pnl_base = _safe_float(entry_notional, 0.0)
    row.update(
        {
            "entry_notional": entry_notional,
            "exit_notional": exit_notional,
            "turnover": turnover,
            "gross_pnl_rupees": gross_pnl,
            "brokerage_estimate": brokerage,
            "net_pnl_rupees": net_pnl,
            "pnl_rupees": gross_pnl,
            "pnl_pct": _safe_float(gross_pnl, np.nan) / pnl_base if pnl_base else np.nan,
            "net_pnl_pct": net_pnl / pnl_base if pnl_base else np.nan,
        }
    )
    if pd.isna(row.get("pnl")):
        row["pnl"] = row["pnl_pct"]
    return row


def _build_closed_trades_from_tradebook(
    trades: pd.DataFrame,
    brokerage_rate: float,
    brokerage_cap: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sortable = trades[~trades["is_rejected"]].sort_values("timestamp", na_position="last")

    for (session_name, symbol), symbol_trades in sortable.groupby(["session", "symbol"], dropna=False):
        open_lots: List[Dict[str, Any]] = []
        for _, execution in symbol_trades.iterrows():
            action = str(execution.get("action", "")).upper()
            if action not in {"BUY", "SELL"}:
                continue
            remaining = _safe_float(execution.get("quantity"), 0.0)
            if remaining <= 0:
                continue

            while remaining > 0 and open_lots and open_lots[0]["direction"] != action:
                entry = open_lots[0]
                closed_qty = min(remaining, entry["remaining_quantity"])
                entry_price = _safe_float(entry["price"], np.nan)
                exit_price = _safe_float(execution.get("price"), np.nan)
                direction = entry["direction"]
                direction_sign = 1 if direction == "BUY" else -1
                gross_pnl = (exit_price - entry_price) * closed_qty * direction_sign
                price_move = (exit_price - entry_price) * direction_sign
                duration_minutes = np.nan
                entry_time = entry["timestamp"]
                exit_time = execution.get("timestamp")
                if pd.notna(entry_time) and pd.notna(exit_time):
                    duration_minutes = (exit_time - entry_time).total_seconds() / 60.0

                row = {
                    "session": session_name,
                    "trade_id": f"{entry['trade_id']}->{execution.get('trade_id')}",
                    "entry_trade_id": entry["trade_id"],
                    "exit_trade_id": execution.get("trade_id"),
                    "entry_order_id": entry.get("order_id"),
                    "exit_order_id": execution.get("order_id"),
                    "symbol": symbol,
                    "strategy": execution.get("strategy", "KiteTradebook"),
                    "direction": direction,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "duration_minutes": duration_minutes,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "quantity": closed_qty,
                    "leverage": np.nan,
                    "position_size": entry_price * closed_qty if not math.isnan(entry_price) else np.nan,
                    "atr": np.nan,
                    "price_move": price_move,
                    "price_move_pct": price_move / entry_price if entry_price else np.nan,
                    "r_multiple": np.nan,
                    "pnl": np.nan,
                    "entry_balance": np.nan,
                    "exit_balance": np.nan,
                    "gross_pnl_rupees": gross_pnl,
                    "status": "closed",
                    "source_format": "kite_tradebook",
                }
                rows.append(
                    _add_money_columns(
                        row,
                        brokerage_rate,
                        brokerage_cap,
                        entry_fraction=closed_qty / entry["quantity"] if entry["quantity"] else 1.0,
                        exit_fraction=closed_qty / _safe_float(execution.get("quantity"), closed_qty),
                    )
                )

                remaining -= closed_qty
                entry["remaining_quantity"] -= closed_qty
                if entry["remaining_quantity"] <= 1e-9:
                    open_lots.pop(0)

            if remaining > 1e-9:
                open_lots.append(
                    {
                        "direction": action,
                        "timestamp": execution.get("timestamp"),
                        "price": _safe_float(execution.get("price"), np.nan),
                        "quantity": _safe_float(execution.get("quantity"), remaining),
                        "remaining_quantity": remaining,
                        "trade_id": execution.get("trade_id"),
                        "order_id": execution.get("order_id", np.nan),
                    }
                )

    if not rows:
        return pd.DataFrame()
    closed = pd.DataFrame(rows).sort_values("exit_time", na_position="last")
    closed["is_win"] = closed["gross_pnl_rupees"] > 0
    closed["is_net_win"] = closed["net_pnl_rupees"] > 0
    closed["exit_date"] = closed["exit_time"].dt.date
    closed["exit_hour"] = closed["exit_time"].dt.hour
    closed["exit_weekday"] = closed["exit_time"].dt.day_name()
    return closed


def build_closed_trades(
    trades: pd.DataFrame,
    brokerage_rate: float = DEFAULT_BROKERAGE_RATE,
    brokerage_cap: float = DEFAULT_BROKERAGE_CAP,
) -> pd.DataFrame:
    """Pair entry and exit rows into one enriched closed-trade table."""
    if trades.empty:
        return pd.DataFrame()

    if "source_format" in trades.columns and trades["source_format"].eq("kite_tradebook").any():
        tradebook = trades[trades["source_format"].eq("kite_tradebook")]
        live_session = trades[~trades["source_format"].eq("kite_tradebook")]
        frames = [_build_closed_trades_from_tradebook(tradebook, brokerage_rate, brokerage_cap)]
        if not live_session.empty:
            frames.append(build_closed_trades(live_session, brokerage_rate, brokerage_cap))
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("exit_time", na_position="last")

    rows: List[Dict[str, Any]] = []
    grouped = trades[~trades["is_rejected"]].groupby(["session", "trade_id"], dropna=False)
    for (session_name, trade_id), group in grouped:
        group = group.sort_values("timestamp", na_position="last")
        entries = group[~group["is_exit"]]
        exits = group[group["is_exit"]]
        if exits.empty:
            continue

        entry = entries.iloc[0] if not entries.empty else group.iloc[0]
        exit_row = exits.iloc[-1]
        entry_price = _safe_float(entry.get("price"), np.nan)
        exit_price = _safe_float(exit_row.get("price"), np.nan)
        action = str(entry.get("action", "")).upper()
        direction = 1 if action == "BUY" else -1 if action == "SELL" else 0
        quantity = _safe_float(entry.get("quantity"), _safe_float(exit_row.get("quantity"), 0.0))
        price_move = (exit_price - entry_price) * direction if direction and not math.isnan(entry_price) and not math.isnan(exit_price) else np.nan
        price_move_pct = price_move / entry_price if entry_price and not math.isnan(price_move) else np.nan
        atr = _safe_float(entry.get("atr"), np.nan)
        r_multiple = price_move / atr if atr and not math.isnan(price_move) else np.nan
        entry_time = entry.get("timestamp")
        exit_time = exit_row.get("timestamp")
        duration_minutes = np.nan
        if pd.notna(entry_time) and pd.notna(exit_time):
            duration_minutes = (exit_time - entry_time).total_seconds() / 60.0

        row = {
                "session": session_name,
                "trade_id": trade_id,
                "symbol": entry.get("symbol", exit_row.get("symbol")),
                "strategy": entry.get("strategy", exit_row.get("strategy")),
                "direction": action,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "duration_minutes": duration_minutes,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "leverage": _safe_float(entry.get("leverage"), np.nan),
                "position_size": _safe_float(entry.get("position_size"), np.nan),
                "atr": atr,
                "price_move": price_move,
                "price_move_pct": price_move_pct,
                "r_multiple": r_multiple,
                "pnl": _safe_float(exit_row.get("pnl"), 0.0),
                "entry_balance": _safe_float(entry.get("balance"), np.nan),
                "exit_balance": _safe_float(exit_row.get("balance"), np.nan),
                "status": exit_row.get("status", ""),
                "source_format": "live_session",
            }
        rows.append(_add_money_columns(row, brokerage_rate, brokerage_cap))
    if not rows:
        return pd.DataFrame()
    closed = pd.DataFrame(rows).sort_values("exit_time", na_position="last")
    closed["is_win"] = closed["pnl"] > 0
    closed["is_net_win"] = closed["net_pnl_rupees"] > 0
    closed["exit_date"] = closed["exit_time"].dt.date
    closed["exit_hour"] = closed["exit_time"].dt.hour
    closed["exit_weekday"] = closed["exit_time"].dt.day_name()
    return closed


def _max_streak(values: Iterable[bool], target: bool) -> int:
    best = 0
    current = 0
    for value in values:
        if bool(value) is target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _equity_curve_from_pnl(closed: pd.DataFrame) -> pd.DataFrame:
    curve = closed[["exit_time", "pnl"]].copy().sort_values("exit_time", na_position="last")
    if curve.empty:
        return curve
    curve["equity_index"] = (1.0 + curve["pnl"].fillna(0.0)).cumprod()
    curve["running_peak"] = curve["equity_index"].cummax()
    curve["drawdown"] = curve["equity_index"] / curve["running_peak"] - 1.0
    return curve


def calculate_metrics(trades: pd.DataFrame, closed: pd.DataFrame, decisions: pd.DataFrame) -> Dict[str, Any]:
    """Calculate portfolio, execution, risk, and decision quality metrics."""
    metrics: Dict[str, Any] = {
        "sessions": int(trades["session"].nunique()) if not trades.empty else 0,
        "symbols": sorted(trades["symbol"].dropna().unique().tolist()) if not trades.empty else [],
        "strategies": sorted(trades["strategy"].dropna().unique().tolist()) if not trades.empty else [],
        "first_timestamp": trades["timestamp"].min() if not trades.empty else pd.NaT,
        "last_timestamp": trades["timestamp"].max() if not trades.empty else pd.NaT,
        "trade_log_rows": int(len(trades)),
        "buy_executions": int(trades["action"].eq("BUY").sum()) if not trades.empty else 0,
        "sell_executions": int(trades["action"].eq("SELL").sum()) if not trades.empty else 0,
        "entries": int((~trades["is_exit"] & ~trades["is_rejected"]).sum()) if not trades.empty else 0,
        "exits": int(trades["is_exit"].sum()) if not trades.empty else 0,
        "rejections": int(trades["is_rejected"].sum()) if not trades.empty else 0,
        "decision_rows": int(len(decisions)),
    }

    if closed.empty:
        metrics.update(
            {
                "closed_trades": 0,
                "win_rate": np.nan,
                "total_return_simple": 0.0,
                "total_return_compounded": 0.0,
                "total_turnover": 0.0,
                "total_gross_pnl_rupees": 0.0,
                "total_brokerage_estimate": 0.0,
                "total_net_pnl_rupees": 0.0,
                "gross_pnl_pct_on_turnover": np.nan,
                "net_pnl_pct_on_turnover": np.nan,
                "net_win_rate": np.nan,
                "profit_factor": np.nan,
                "net_profit_factor": np.nan,
                "expectancy": np.nan,
                "net_expectancy_rupees": np.nan,
                "max_drawdown": np.nan,
                "best_trade": np.nan,
                "worst_trade": np.nan,
                "best_trade_rupees": np.nan,
                "worst_trade_rupees": np.nan,
                "avg_win": np.nan,
                "avg_loss": np.nan,
                "avg_duration_minutes": np.nan,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "sharpe_per_trade": np.nan,
            }
        )
        return metrics

    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] < 0]
    net_wins = closed[closed["net_pnl_rupees"] > 0] if "net_pnl_rupees" in closed.columns else pd.DataFrame()
    net_losses = closed[closed["net_pnl_rupees"] < 0] if "net_pnl_rupees" in closed.columns else pd.DataFrame()
    gross_profit = wins["pnl"].sum()
    gross_loss = losses["pnl"].sum()
    gross_profit_rupees = closed.loc[closed["gross_pnl_rupees"] > 0, "gross_pnl_rupees"].sum() if "gross_pnl_rupees" in closed.columns else 0.0
    gross_loss_rupees = closed.loc[closed["gross_pnl_rupees"] < 0, "gross_pnl_rupees"].sum() if "gross_pnl_rupees" in closed.columns else 0.0
    net_profit_rupees = net_wins["net_pnl_rupees"].sum() if not net_wins.empty else 0.0
    net_loss_rupees = net_losses["net_pnl_rupees"].sum() if not net_losses.empty else 0.0
    total_turnover = closed["turnover"].sum() if "turnover" in closed.columns else np.nan
    curve = _equity_curve_from_pnl(closed)
    pnl_std = closed["pnl"].std(ddof=1)

    metrics.update(
        {
            "closed_trades": int(len(closed)),
            "win_rate": float(len(wins) / len(closed)),
            "net_win_rate": float(len(net_wins) / len(closed)) if "net_pnl_rupees" in closed.columns else np.nan,
            "total_return_simple": float(closed["pnl"].sum()),
            "total_return_compounded": float((1.0 + closed["pnl"].fillna(0.0)).prod() - 1.0),
            "total_turnover": float(total_turnover),
            "total_gross_pnl_rupees": float(closed["gross_pnl_rupees"].sum()) if "gross_pnl_rupees" in closed.columns else np.nan,
            "total_brokerage_estimate": float(closed["brokerage_estimate"].sum()) if "brokerage_estimate" in closed.columns else np.nan,
            "total_net_pnl_rupees": float(closed["net_pnl_rupees"].sum()) if "net_pnl_rupees" in closed.columns else np.nan,
            "gross_pnl_pct_on_turnover": float(closed["gross_pnl_rupees"].sum() / total_turnover) if total_turnover else np.nan,
            "net_pnl_pct_on_turnover": float(closed["net_pnl_rupees"].sum() / total_turnover) if total_turnover else np.nan,
            "profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss < 0 else np.inf if gross_profit > 0 else np.nan,
            "gross_profit_factor_rupees": float(gross_profit_rupees / abs(gross_loss_rupees)) if gross_loss_rupees < 0 else np.inf if gross_profit_rupees > 0 else np.nan,
            "net_profit_factor": float(net_profit_rupees / abs(net_loss_rupees)) if net_loss_rupees < 0 else np.inf if net_profit_rupees > 0 else np.nan,
            "expectancy": _series_mean(closed["pnl"]),
            "expectancy_rupees": _series_mean(closed["gross_pnl_rupees"]) if "gross_pnl_rupees" in closed.columns else np.nan,
            "net_expectancy_rupees": _series_mean(closed["net_pnl_rupees"]) if "net_pnl_rupees" in closed.columns else np.nan,
            "max_drawdown": float(curve["drawdown"].min()) if not curve.empty else np.nan,
            "best_trade": float(closed["pnl"].max()),
            "worst_trade": float(closed["pnl"].min()),
            "best_trade_rupees": float(closed["gross_pnl_rupees"].max()) if "gross_pnl_rupees" in closed.columns else np.nan,
            "worst_trade_rupees": float(closed["gross_pnl_rupees"].min()) if "gross_pnl_rupees" in closed.columns else np.nan,
            "avg_win": _series_mean(wins["pnl"]) if not wins.empty else np.nan,
            "avg_loss": _series_mean(losses["pnl"]) if not losses.empty else np.nan,
            "avg_duration_minutes": _series_mean(closed["duration_minutes"]),
            "median_duration_minutes": _series_median(closed["duration_minutes"]),
            "max_consecutive_wins": _max_streak(closed["is_win"], True),
            "max_consecutive_losses": _max_streak(closed["is_win"], False),
            "sharpe_per_trade": float(_series_mean(closed["pnl"]) / pnl_std * math.sqrt(len(closed))) if pnl_std and not math.isnan(pnl_std) else np.nan,
            "avg_r_multiple": _series_mean(closed["r_multiple"]),
            "median_r_multiple": _series_median(closed["r_multiple"]),
            "avg_leverage": _series_mean(closed["leverage"]),
            "max_leverage": float(pd.to_numeric(closed["leverage"], errors="coerce").max()) if pd.to_numeric(closed["leverage"], errors="coerce").notna().any() else np.nan,
        }
    )
    return metrics


def summarize_by_dimensions(closed: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Produce grouped summary tables for key dimensions."""
    if closed.empty:
        return {}

    def grouped_summary(group_columns: List[str]) -> pd.DataFrame:
        summary = (
            closed.groupby(group_columns, dropna=False)
            .agg(
                trades=("pnl", "size"),
                win_rate=("is_win", "mean"),
                net_win_rate=("is_net_win", "mean"),
                total_pnl=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
                gross_pnl_rupees=("gross_pnl_rupees", "sum"),
                brokerage_estimate=("brokerage_estimate", "sum"),
                net_pnl_rupees=("net_pnl_rupees", "sum"),
                turnover=("turnover", "sum"),
                best_trade=("pnl", "max"),
                worst_trade=("pnl", "min"),
                best_trade_rupees=("gross_pnl_rupees", "max"),
                worst_trade_rupees=("gross_pnl_rupees", "min"),
                avg_r=("r_multiple", "mean"),
                avg_duration_minutes=("duration_minutes", "mean"),
            )
            .reset_index()
            .sort_values("net_pnl_rupees", ascending=False)
        )
        return summary

    return {
        "by_symbol": grouped_summary(["symbol"]),
        "by_strategy": grouped_summary(["strategy"]),
        "by_symbol_strategy": grouped_summary(["symbol", "strategy"]),
        "by_status": grouped_summary(["status"]),
        "by_hour": grouped_summary(["exit_hour"]),
        "by_weekday": grouped_summary(["exit_weekday"]),
        "daily_summary": grouped_summary(["exit_date"]),
    }


def analyze_rejections(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "is_rejected" not in trades.columns:
        return pd.DataFrame()
    rejected = trades[trades["is_rejected"]]
    if rejected.empty:
        return pd.DataFrame()
    return (
        rejected.groupby(["symbol", "strategy", "reject_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def analyze_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    group_cols = [column for column in ["symbol", "strategy", "signal_name", "trade_status"] if column in decisions.columns]
    if not group_cols:
        return pd.DataFrame()
    return decisions.groupby(group_cols, dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)


def write_plots(output_dir: Path, closed: pd.DataFrame, trades: pd.DataFrame) -> List[Path]:
    """Generate PNG plots for equity, drawdown, rupee/percent P&L, and statuses."""
    plot_paths: List[Path] = []
    if not closed.empty:
        curve = _equity_curve_from_pnl(closed)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(curve["exit_time"], curve["equity_index"], marker="o", linewidth=1.8)
        ax.set_title("Compounded Equity Index by Closed Trade")
        ax.set_ylabel("Equity Index")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        path = output_dir / "equity_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(curve["exit_time"], curve["drawdown"] * 100, 0, alpha=0.35)
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown %")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        path = output_dir / "drawdown.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)

        money_curve = closed.sort_values("exit_time", na_position="last").copy()
        money_curve["cumulative_gross_pnl_rupees"] = money_curve["gross_pnl_rupees"].fillna(0.0).cumsum()
        money_curve["cumulative_net_pnl_rupees"] = money_curve["net_pnl_rupees"].fillna(0.0).cumsum()
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(money_curve["exit_time"], money_curve["cumulative_gross_pnl_rupees"], marker="o", linewidth=1.8, label="Gross P&L")
        ax.plot(money_curve["exit_time"], money_curve["cumulative_net_pnl_rupees"], marker="o", linewidth=1.8, label="Net after brokerage")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title("Cumulative Realized P&L in Rupees")
        ax.set_ylabel("P&L (Rs)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        path = output_dir / "cumulative_pnl_rupees.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)

        daily = closed.groupby("exit_date")["pnl"].sum()
        fig, ax = plt.subplots(figsize=(11, 4))
        colors = ["#2ca02c" if value >= 0 else "#d62728" for value in daily]
        ax.bar([str(index) for index in daily.index], daily.values * 100, color=colors)
        ax.set_title("Daily Realized P&L")
        ax.set_ylabel("P&L %")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        path = output_dir / "daily_pnl.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)

        daily_money = closed.groupby("exit_date").agg(gross_pnl_rupees=("gross_pnl_rupees", "sum"), net_pnl_rupees=("net_pnl_rupees", "sum"), brokerage_estimate=("brokerage_estimate", "sum"))
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(daily_money))
        colors = ["#2ca02c" if value >= 0 else "#d62728" for value in daily_money["net_pnl_rupees"]]
        ax.bar(x, daily_money["net_pnl_rupees"], color=colors, label="Net P&L")
        ax.plot(x, daily_money["gross_pnl_rupees"], color="#1f77b4", marker="o", linewidth=1.4, label="Gross P&L")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title("Daily Realized P&L in Rupees")
        ax.set_ylabel("P&L (Rs)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(index) for index in daily_money.index], rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        path = output_dir / "daily_pnl_rupees.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar([str(index) for index in daily_money.index], daily_money["brokerage_estimate"], color="#9467bd")
        ax.set_title("Estimated Brokerage by Day")
        ax.set_ylabel("Brokerage (Rs)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        path = output_dir / "daily_brokerage.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)

    if not trades.empty:
        status_counts = trades["status"].replace("", "unknown").value_counts()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(status_counts.index.astype(str), status_counts.values)
        ax.set_title("Trade Log Status Counts")
        ax.set_ylabel("Rows")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        path = output_dir / "status_counts.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(path)
    return plot_paths


def _markdown_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    """Render a small Markdown table without relying on optional tabulate."""
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: _fmt_number(value, 4))
        else:
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else str(value))

    headers = [str(column) for column in view.columns]
    rows = view.astype(str).values.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        escaped = [cell.replace("\n", " ").replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    sessions: Sequence[SessionFiles],
    trades: pd.DataFrame,
    closed: pd.DataFrame,
    decisions: pd.DataFrame,
    statuses: pd.DataFrame,
    metrics: Dict[str, Any],
    summaries: Dict[str, pd.DataFrame],
    rejections: pd.DataFrame,
    decision_summary: pd.DataFrame,
    plot_paths: Sequence[Path],
) -> Path:
    """Write a Markdown report with all major findings."""
    report_path = output_dir / "kite_live_log_analysis.md"
    lines: List[str] = []
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    has_live_session_rows = not trades.empty and "source_format" in trades.columns and trades["source_format"].eq("live_session").any()
    execution_lines = [
        f"- Buy / sell executions / rejections: **{metrics.get('buy_executions', 0)} / {metrics.get('sell_executions', 0)} / {metrics['rejections']}**"
    ]
    if has_live_session_rows:
        execution_lines.append(f"- Live-session entries / exits: **{metrics['entries']} / {metrics['exits']}**")
    lines.extend(
        [
            "# Kite Live Trading Log Analysis",
            "",
            f"Generated at: **{generated_at}**",
            "",
            "## Scope",
            "",
            f"- Sessions analyzed: **{len(sessions)}**",
            f"- Trade log rows: **{metrics['trade_log_rows']}**",
            f"- Decision log rows: **{metrics['decision_rows']}**",
            f"- Symbols: **{', '.join(metrics['symbols']) if metrics['symbols'] else 'n/a'}**",
            f"- Strategies: **{', '.join(metrics['strategies']) if metrics['strategies'] else 'n/a'}**",
            f"- Time range: **{metrics['first_timestamp']}** to **{metrics['last_timestamp']}**",
            "",
            "## Executive Summary",
            "",
            *execution_lines,
            f"- Closed trades: **{metrics['closed_trades']}**",
            f"- Win rate gross / net after brokerage: **{_fmt_pct(metrics['win_rate'])} / {_fmt_pct(metrics.get('net_win_rate'))}**",
            f"- Gross P&L: **{_fmt_rupees(metrics.get('total_gross_pnl_rupees'))}**",
            f"- Estimated brokerage: **{_fmt_rupees(metrics.get('total_brokerage_estimate'))}**",
            f"- Net P&L after brokerage: **{_fmt_rupees(metrics.get('total_net_pnl_rupees'))}**",
            f"- Gross / net P&L on turnover: **{_fmt_pct(metrics.get('gross_pnl_pct_on_turnover'))} / {_fmt_pct(metrics.get('net_pnl_pct_on_turnover'))}**",
            f"- Total turnover: **{_fmt_rupees(metrics.get('total_turnover'))}**",
            f"- Total return, simple sum: **{_fmt_pct(metrics['total_return_simple'])}**",
            f"- Total return, compounded: **{_fmt_pct(metrics['total_return_compounded'])}**",
            f"- Profit factor percent / rupee net: **{_fmt_number(metrics['profit_factor'], 3)} / {_fmt_number(metrics.get('net_profit_factor'), 3)}**",
            f"- Expectancy per closed trade: **{_fmt_pct(metrics['expectancy'])} / {_fmt_rupees(metrics.get('net_expectancy_rupees'))} net**",
            f"- Max drawdown: **{_fmt_pct(metrics['max_drawdown'])}**",
            f"- Best / worst trade: **{_fmt_pct(metrics['best_trade'])} / {_fmt_pct(metrics['worst_trade'])}**, **{_fmt_rupees(metrics.get('best_trade_rupees'))} / {_fmt_rupees(metrics.get('worst_trade_rupees'))}**",
            f"- Average win / loss: **{_fmt_pct(metrics['avg_win'])} / {_fmt_pct(metrics['avg_loss'])}**",
            f"- Max consecutive wins / losses: **{metrics['max_consecutive_wins']} / {metrics['max_consecutive_losses']}**",
            f"- Average duration: **{_fmt_number(metrics.get('avg_duration_minutes'), 1)} minutes**",
            f"- Average / max leverage: **{_fmt_number(metrics.get('avg_leverage'), 2)} / {_fmt_number(metrics.get('max_leverage'), 2)}**",
            "",
            "## Generated Artifacts",
            "",
        ]
    )
    artifacts = ["closed_trades_enriched.csv", "all_trades_normalized.csv", "metrics.json"]
    artifacts.extend(path.name for path in plot_paths)
    for artifact in artifacts:
        lines.append(f"- `{artifact}`")
    lines.append("")

    lines.extend(["## Session Inventory", "", _markdown_table(pd.DataFrame([s.__dict__ for s in sessions])), ""])

    if not statuses.empty:
        lines.extend(["## Latest Status Snapshots", "", _markdown_table(statuses), ""])

    for name, summary in summaries.items():
        title = name.replace("_", " ").title()
        lines.extend([f"## {title}", "", _markdown_table(summary), ""])

    lines.extend(["## Rejection Analysis", "", _markdown_table(rejections), ""])
    lines.extend(["## Decision Log Signal Mix", "", _markdown_table(decision_summary), ""])

    if not closed.empty:
        tail_columns = [
            "session",
            "trade_id",
            "symbol",
            "strategy",
            "direction",
            "entry_time",
            "exit_time",
            "pnl",
            "gross_pnl_rupees",
            "brokerage_estimate",
            "net_pnl_rupees",
            "r_multiple",
            "status",
        ]
        lines.extend(["## Most Recent Closed Trades", "", _markdown_table(closed[tail_columns].tail(20), max_rows=20), ""])

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- `pnl` is treated as the fractional return stored by the trading engine. For example, `0.01` is reported as `1.00%`.",
            "- Kite tradebook exports are paired FIFO by symbol to reconstruct closed trades and calculate realized rupee P&L.",
            "- `brokerage_estimate` uses the configured brokerage rate and per-order cap. It is an estimate, not a broker contract note or tax ledger.",
            "- `r_multiple` is estimated from entry/exit price movement divided by entry ATR. It is only as accurate as the ATR recorded in `trades.csv`.",
            "- Compounded return is calculated from closed-trade `pnl` rows, not from deposits/withdrawals or broker ledger adjustments.",
            "- Drawdown is calculated on the compounded closed-trade equity index.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_outputs(
    output_dir: Path,
    sessions: Sequence[SessionFiles],
    trades: pd.DataFrame,
    closed: pd.DataFrame,
    decisions: pd.DataFrame,
    statuses: pd.DataFrame,
    metrics: Dict[str, Any],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, List[Path], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_dir / "all_trades_normalized.csv", index=False)
    closed.to_csv(output_dir / "closed_trades_enriched.csv", index=False)
    decisions.to_csv(output_dir / "all_decisions_normalized.csv", index=False)
    statuses.to_csv(output_dir / "status_snapshots.csv", index=False)

    summaries = summarize_by_dimensions(closed)
    for name, summary in summaries.items():
        summary.to_csv(output_dir / f"{name}.csv", index=False)

    rejections = analyze_rejections(trades)
    rejections.to_csv(output_dir / "rejections_summary.csv", index=False)
    decision_summary = analyze_decisions(decisions)
    decision_summary.to_csv(output_dir / "decision_signal_summary.csv", index=False)

    metrics_json = {key: str(value) if isinstance(value, (pd.Timestamp, np.generic)) else value for key, value in metrics.items()}
    (output_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2, default=str), encoding="utf-8")

    plot_paths = write_plots(output_dir, closed, trades)
    report_path = write_report(
        output_dir,
        sessions,
        trades,
        closed,
        decisions,
        statuses,
        metrics,
        summaries,
        rejections,
        decision_summary,
        plot_paths,
    )
    return summaries, rejections, decision_summary, plot_paths, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Kite live trading logs in depth.")
    parser.add_argument("--logs-dir", default="kite_logs", type=Path, help="Directory containing Kite session folders. Default: kite_logs")
    parser.add_argument("--output-dir", default=None, type=Path, help="Directory for generated analysis artifacts.")
    parser.add_argument("--session", action="append", help="Specific session folder name to include. Can be repeated.")
    parser.add_argument("--fail-on-empty", action="store_true", help="Exit with an error if no closed trades are found.")
    parser.add_argument(
        "--brokerage-rate",
        type=float,
        default=DEFAULT_BROKERAGE_RATE,
        help="Brokerage rate per executed order before cap. Default: 0.0003 (0.03%).",
    )
    parser.add_argument(
        "--brokerage-cap",
        type=float,
        default=DEFAULT_BROKERAGE_CAP,
        help="Maximum brokerage per executed order in rupees. Default: 20.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or Path(f"kite_log_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    sessions = discover_sessions(args.logs_dir, args.session)
    if not sessions:
        print(f"No sessions with trades.csv found under {args.logs_dir}")
        return 2

    trades = load_trades(sessions)
    decisions = load_decisions(sessions)
    statuses = load_statuses(sessions)
    closed = build_closed_trades(trades, brokerage_rate=args.brokerage_rate, brokerage_cap=args.brokerage_cap)
    metrics = calculate_metrics(trades, closed, decisions)

    if args.fail_on_empty and closed.empty:
        print("No closed trades found; refusing to generate an empty performance report because --fail-on-empty was set.")
        return 3

    _, _, _, _, report_path = write_outputs(output_dir, sessions, trades, closed, decisions, statuses, metrics)

    print(f"Analyzed {len(sessions)} session(s) from {args.logs_dir}")
    print(f"Trade rows: {len(trades)} | Closed trades: {len(closed)} | Rejections: {metrics['rejections']}")
    print(f"Report: {report_path}")
    print(f"Artifacts directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
