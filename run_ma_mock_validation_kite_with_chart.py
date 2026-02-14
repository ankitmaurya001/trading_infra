#!/usr/bin/env python3
"""
Run MA mock validation using Kite data, then generate an interactive OHLC chart with:
- Candles (OHLC)
- Short/Long moving averages
- Trade entry/exit markers (long and short)
- Trade details on hover (ATR, PnL, status, etc.)

This script reuses the core simulation flow from run_ma_mock_validation_kite.py.
"""

import argparse
import os
import webbrowser
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go

from kite_comprehensive_strategy_validation import KiteComprehensiveStrategyValidator
from run_ma_mock_validation_kite import (
    _load_params_list,
    convert_date_format,
    map_interval_to_kite,
    run_ma_mock,
)

# ============================================================================
# GLOBAL CONFIGURATION - Edit these values to set defaults
# ============================================================================
# You can override these via command-line arguments if needed
DEFAULT_SYMBOL = "NATGASMINI26FEBFUT"
# DEFAULT_SYMBOL = "CRUDEOIL26MARFUT"
DEFAULT_EXCHANGE = "MCX"
DAYS_TO_VALIDATE = 30
DEFAULT_START_DATE = (datetime.now() - timedelta(days=DAYS_TO_VALIDATE)).strftime("%Y-%m-%d")
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_INTERVAL = "15m"
DEFAULT_PARAMS = [
    {"short_window": 4, "long_window": 184, "risk_reward_ratio": 7.0}
]
# ============================================================================


def _to_naive_ts(value):
    if value is None or value == "":
        return pd.NaT
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_localize(None)
    return ts


def _normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLC columns to lowercase names expected by plotting logic.
    Supports common variants like Open/High/Low/Close.
    """
    if df is None or df.empty:
        return df

    normalized = df.copy()
    lower_to_actual = {str(c).strip().lower(): c for c in normalized.columns}
    rename_map = {}
    for target in ["open", "high", "low", "close"]:
        actual = lower_to_actual.get(target)
        if actual is not None and actual != target:
            rename_map[actual] = target
    if rename_map:
        normalized = normalized.rename(columns=rename_map)
    return normalized


def _get_col_series(df: pd.DataFrame, col: str, default=None) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _first_available_series(df: pd.DataFrame, cols, default=None) -> pd.Series:
    for col in cols:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _calc_realized_pnl_rupees(row: pd.Series):
    entry = row.get("entry_price")
    exit_price = row.get("exit_price")
    action = str(row.get("action", "")).upper()
    qty = row.get("quantity", 1)
    lot_size = row.get("lot_size")
    pnl_pct = row.get("pnl")
    margin_used = row.get("margin_used")
    position_size = row.get("position_size")
    leverage = row.get("leverage")

    if pd.isna(entry) or pd.isna(exit_price):
        return None

    try:
        qty = float(qty) if pd.notna(qty) else 1.0
    except Exception:
        qty = 1.0

    # Match trading_engine commodity formula:
    # BUY: (exit - entry) * lot_size * qty, SELL: (entry - exit) * lot_size * qty
    if pd.notna(lot_size) and float(lot_size) > 0:
        lot_size = float(lot_size)
        price_change = (exit_price - entry) if action == "BUY" else (entry - exit_price)
        return price_change * lot_size * qty

    # Match non-commodity branch in trading_engine
    price_change = (exit_price - entry) if action == "BUY" else (entry - exit_price)
    pnl_direct = price_change * qty
    if pd.notna(pnl_direct):
        return pnl_direct

    # Fallback to pct pnl on margin when direct computation isn't available
    if pd.notna(pnl_pct):
        if pd.isna(margin_used) and pd.notna(position_size) and pd.notna(leverage) and float(leverage) > 0:
            margin_used = float(position_size) / float(leverage)
        if pd.notna(margin_used):
            return float(margin_used) * float(pnl_pct)

    return None


def create_ohlc_ma_trade_chart(
    ohlc: pd.DataFrame,
    trade_history: pd.DataFrame,
    params: Dict,
    symbol: str,
    exchange: str,
    output_path: str,
) -> Optional[str]:
    if ohlc is None or ohlc.empty:
        print("⚠️  No OHLC data to plot")
        return None

    chart_df = _normalize_ohlc_columns(ohlc)
    chart_df = chart_df.sort_index()

    required_cols = {"open", "high", "low", "close"}
    missing = sorted(required_cols - set(chart_df.columns))
    if missing:
        print(
            f"⚠️  Missing OHLC columns for chart: {missing}. "
            f"Available columns: {list(chart_df.columns)}"
        )
        return None

    short_window = int(params.get("short_window", 9))
    long_window = int(params.get("long_window", 21))

    chart_df["short_ma"] = chart_df["close"].rolling(short_window).mean()
    chart_df["long_ma"] = chart_df["close"].rolling(long_window).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="OHLC",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["short_ma"],
            mode="lines",
            name=f"Short MA ({short_window})",
            line=dict(width=1.8, color="#1f77b4"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["long_ma"],
            mode="lines",
            name=f"Long MA ({long_window})",
            line=dict(width=1.8, color="#ff7f0e"),
        )
    )

    if trade_history is not None and not trade_history.empty:
        trades = trade_history.copy()
        trades["entry_time"] = _get_col_series(trades, "entry_time", pd.NaT).apply(_to_naive_ts)
        trades["exit_time"] = _get_col_series(trades, "exit_time", pd.NaT).apply(_to_naive_ts)
        status_series = _get_col_series(trades, "status", "")
        action_series = _get_col_series(trades, "action", "")

        non_rejected = trades[status_series.astype(str).str.lower() != "rejected"].copy()

        if not non_rejected.empty:
            entry_action = _get_col_series(non_rejected, "action", "").astype(str).str.upper()
            long_entries = non_rejected[entry_action == "BUY"]
            short_entries = non_rejected[entry_action == "SELL"]

            for subset, name, color, symbol_marker in [
                (long_entries, "Long Entry", "#2ca02c", "triangle-up"),
                (short_entries, "Short Entry", "#d62728", "triangle-down"),
            ]:
                if subset.empty:
                    continue

                hover_text = (
                    "<b>" + name + "</b><br>"
                    + "Time: " + _get_col_series(subset, "entry_time", "").astype(str)
                    + "<br>Price: " + _get_col_series(subset, "entry_price", "").map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                    + "<br>Trade ID: " + _first_available_series(subset, ["trade_id", "id"], "N/A").astype(str)
                    + "<br>ATR: " + _get_col_series(subset, "atr", "").map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                    + "<br>TP: " + _get_col_series(subset, "take_profit", "").map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                    + "<br>SL: " + _get_col_series(subset, "stop_loss", "").map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                    + "<br>Status: " + _get_col_series(subset, "status", "N/A").astype(str)
                )

                fig.add_trace(
                    go.Scatter(
                        x=subset["entry_time"],
                        y=subset["entry_price"],
                        mode="markers",
                        name=name,
                        marker=dict(symbol=symbol_marker, size=10, color=color, line=dict(color="white", width=1)),
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

            exit_rows = non_rejected[non_rejected["exit_time"].notna()].copy()
            if not exit_rows.empty:
                exit_rows["pnl_rupees"] = exit_rows.apply(_calc_realized_pnl_rupees, axis=1)
                hover_text = (
                    "<b>Trade Exit</b><br>"
                    + "Time: " + _get_col_series(exit_rows, "exit_time", "").astype(str)
                    + "<br>Price: " + _get_col_series(exit_rows, "exit_price", "").map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                    + "<br>Trade ID: " + _first_available_series(exit_rows, ["trade_id", "id"], "N/A").astype(str)
                    + "<br>Status: " + _get_col_series(exit_rows, "status", "N/A").astype(str)
                    + "<br>PnL (%): " + _get_col_series(exit_rows, "pnl", "").map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A")
                    + "<br>PnL (Rs): " + _get_col_series(exit_rows, "pnl_rupees", "").map(lambda x: f"{x:+,.2f}" if pd.notna(x) else "N/A")
                    + "<br>ATR: " + _get_col_series(exit_rows, "atr", "").map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                    + "<br>Side: " + _get_col_series(exit_rows, "action", "N/A").astype(str)
                )
                fig.add_trace(
                    go.Scatter(
                        x=exit_rows["exit_time"],
                        y=exit_rows["exit_price"],
                        mode="markers",
                        name="Trade Exit",
                        marker=dict(symbol="x", size=10, color="#9467bd", line=dict(width=1, color="white")),
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

    title = (
        f"{symbol} ({exchange}) MA Mock Validation"
        f"<br><sup>Short MA={short_window}, Long MA={long_window}, "
        f"RR={params.get('risk_reward_ratio')}</sup>"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        dragmode="pan",
        height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)

    fig.write_html(
        output_path,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
        },
    )
    print(f"📊 OHLC+MA+Trade chart saved: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MA mock validation (Kite) + OHLC/MA trade chart")
    p.add_argument("--symbol", default=None, help=f"Symbol (default: {DEFAULT_SYMBOL})")
    p.add_argument("--exchange", default=None, help=f"Exchange (default: {DEFAULT_EXCHANGE})")
    p.add_argument("--start", default=None, help=f"Start date YYYY-MM-DD (default: {DEFAULT_START_DATE})")
    p.add_argument("--end", default=None, help=f"End date YYYY-MM-DD (default: {DEFAULT_END_DATE})")
    p.add_argument("--interval", default=None, help=f"Interval like 15m (default: {DEFAULT_INTERVAL})")
    p.add_argument("--params", help="JSON array string of MA params")
    p.add_argument("--params-file", help="Path to JSON file with MA params array")
    p.add_argument("--initial-balance", type=float, default=10000.0)
    p.add_argument("--max-leverage", type=float, default=10.0)
    p.add_argument("--max-loss-percent", type=float, default=2.0)
    p.add_argument("--trading-fee", type=float, default=0.0)
    p.add_argument("--out", default="ma_mock_results_kite", help="Output directory")
    p.add_argument("--no-parameter-validation", action="store_true")
    p.add_argument("--validation-window-days", type=int, default=31)
    p.add_argument("--days-to-validate", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()

    symbol = (args.symbol or DEFAULT_SYMBOL).upper()
    exchange = (args.exchange or DEFAULT_EXCHANGE).upper()
    start_date = args.start or DEFAULT_START_DATE
    end_date = args.end or DEFAULT_END_DATE
    interval = args.interval or DEFAULT_INTERVAL
    params_list = _load_params_list(args.params, args.params_file, DEFAULT_PARAMS)

    summary = run_ma_mock(
        symbol=symbol,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        params_list=params_list,
        initial_balance=args.initial_balance,
        max_leverage=args.max_leverage,
        max_loss_percent=args.max_loss_percent,
        trading_fee=args.trading_fee,
        output_dir=args.out,
        enable_parameter_validation=not args.no_parameter_validation,
        validation_data_window_days=args.validation_window_days,
        days_to_validate=args.days_to_validate,
    )

    results = summary.get("results", [])
    if not results:
        print("⚠️  No successful results returned; skipping OHLC chart generation")
        return

    # Reuse already-fetched backtest data from run_ma_mock to avoid extra Kite fetch/auth.
    ohlc = summary.get("backtest_data")
    if ohlc is None or (hasattr(ohlc, "empty") and ohlc.empty):
        validator = KiteComprehensiveStrategyValidator(exchange=exchange)
        validator.authenticate_kite()
        ohlc = validator.kite_fetcher.fetch_historical_data(
            symbol,
            convert_date_format(start_date),
            convert_date_format(end_date),
            interval=map_interval_to_kite(interval),
        )
        if exchange in ["NSE", "BSE"]:
            ohlc = validator._filter_equity_market_hours(ohlc)
        elif exchange == "MCX":
            ohlc = validator._filter_mcx_market_hours(ohlc)

    os.makedirs(args.out, exist_ok=True)
    generated_chart_paths = []

    for result in results:
        trades = result.get("trade_history")
        params = result.get("parameters", {})
        session_id = result.get("session_id", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        chart_path = os.path.join(args.out, f"{session_id}_ohlc_ma_trades.html")
        out_path = create_ohlc_ma_trade_chart(
            ohlc=ohlc,
            trade_history=trades,
            params=params,
            symbol=symbol,
            exchange=exchange,
            output_path=chart_path,
        )
        if out_path:
            generated_chart_paths.append(out_path)

    if generated_chart_paths:
        first_chart = os.path.abspath(generated_chart_paths[0])
        print(f"🌐 Opening chart: {first_chart}")
        try:
            webbrowser.open(f"file://{first_chart}")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"   Please open manually: {first_chart}")


if __name__ == "__main__":
    main()
