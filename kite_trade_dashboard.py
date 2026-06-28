#!/usr/bin/env python3
"""
Interactive Streamlit dashboard for Kite trade CSV analysis.

Run:
    streamlit run kite_trade_dashboard.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analyze_kite_live_logs import (
    DEFAULT_BROKERAGE_CAP,
    DEFAULT_BROKERAGE_RATE,
    SessionFiles,
    _coerce_trade_frame,
    _fmt_pct,
    _fmt_rupees,
    build_closed_trades,
    calculate_metrics,
    summarize_by_dimensions,
)


SAMPLE_TRADEBOOK_PATH = Path("kite_logs/trades.csv")


def _metric_value(metrics: Dict[str, object], key: str, formatter) -> str:
    return formatter(metrics.get(key, np.nan))


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _to_json_bytes(payload: Dict[str, object]) -> bytes:
    clean_payload = {}
    for key, value in payload.items():
        if isinstance(value, (pd.Timestamp, np.generic)):
            clean_payload[key] = str(value)
        else:
            clean_payload[key] = value
    return json.dumps(clean_payload, indent=2, default=str).encode("utf-8")


def _load_uploaded_trades(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def _load_sample_trades() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_TRADEBOOK_PATH)


@st.cache_data(show_spinner=False)
def analyze_frame(
    raw_trades: pd.DataFrame,
    session_name: str,
    brokerage_rate: float,
    brokerage_cap: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    session = SessionFiles(
        session_name=session_name,
        session_path=Path("."),
        trades_path=Path("uploaded_trades.csv"),
        decisions_path=None,
        status_path=None,
        market_data_path=None,
    )
    trades = _coerce_trade_frame(raw_trades.copy(), session)
    closed = build_closed_trades(trades, brokerage_rate=brokerage_rate, brokerage_cap=brokerage_cap)
    metrics = calculate_metrics(trades, closed, pd.DataFrame())
    return trades, closed, metrics


def apply_filters(
    trades: pd.DataFrame,
    closed: pd.DataFrame,
    selected_symbols,
    date_range,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    filtered_trades = trades.copy()
    filtered_closed = closed.copy()

    if selected_symbols:
        filtered_trades = filtered_trades[filtered_trades["symbol"].isin(selected_symbols)]
        filtered_closed = filtered_closed[filtered_closed["symbol"].isin(selected_symbols)]

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        trade_dates = filtered_trades["timestamp"].dt.date
        filtered_trades = filtered_trades[(trade_dates >= start_date) & (trade_dates <= end_date)]
        if not filtered_closed.empty:
            exit_dates = filtered_closed["exit_time"].dt.date
            filtered_closed = filtered_closed[(exit_dates >= start_date) & (exit_dates <= end_date)]

    return filtered_trades, filtered_closed


def render_metric_cards(metrics: Dict[str, object]) -> None:
    row1 = st.columns(5)
    row1[0].metric("Closed trades", f"{int(metrics.get('closed_trades', 0))}")
    row1[1].metric("Gross win rate", _metric_value(metrics, "win_rate", _fmt_pct))
    row1[2].metric("Net win rate", _metric_value(metrics, "net_win_rate", _fmt_pct))
    row1[3].metric("Gross P&L", _metric_value(metrics, "total_gross_pnl_rupees", _fmt_rupees))
    row1[4].metric("Net P&L", _metric_value(metrics, "total_net_pnl_rupees", _fmt_rupees))

    row2 = st.columns(5)
    row2[0].metric("Brokerage", _metric_value(metrics, "total_brokerage_estimate", _fmt_rupees))
    row2[1].metric("Turnover", _metric_value(metrics, "total_turnover", _fmt_rupees))
    row2[2].metric("Net P&L / turnover", _metric_value(metrics, "net_pnl_pct_on_turnover", _fmt_pct))
    row2[3].metric("Profit factor", f"{float(metrics.get('net_profit_factor', np.nan)):,.3f}" if pd.notna(metrics.get("net_profit_factor")) else "n/a")
    row2[4].metric("Max drawdown", _metric_value(metrics, "max_drawdown", _fmt_pct))


def render_charts(closed: pd.DataFrame) -> None:
    if closed.empty:
        st.info("No closed trades available for charts.")
        return

    closed = closed.sort_values("exit_time").copy()
    closed["cumulative_gross_pnl_rupees"] = closed["gross_pnl_rupees"].fillna(0).cumsum()
    closed["cumulative_net_pnl_rupees"] = closed["net_pnl_rupees"].fillna(0).cumsum()
    closed["equity_index"] = (1.0 + closed["pnl"].fillna(0)).cumprod()
    closed["drawdown"] = closed["equity_index"] / closed["equity_index"].cummax() - 1.0

    pnl_fig = go.Figure()
    pnl_fig.add_trace(go.Scatter(x=closed["exit_time"], y=closed["cumulative_gross_pnl_rupees"], mode="lines+markers", name="Gross P&L"))
    pnl_fig.add_trace(go.Scatter(x=closed["exit_time"], y=closed["cumulative_net_pnl_rupees"], mode="lines+markers", name="Net P&L"))
    pnl_fig.add_hline(y=0, line_width=1, line_color="gray")
    pnl_fig.update_layout(title="Cumulative realized P&L", xaxis_title="Exit time", yaxis_title="P&L (Rs)", hovermode="x unified")
    st.plotly_chart(pnl_fig, use_container_width=True)

    daily = (
        closed.groupby("exit_date")
        .agg(
            gross_pnl_rupees=("gross_pnl_rupees", "sum"),
            net_pnl_rupees=("net_pnl_rupees", "sum"),
            brokerage_estimate=("brokerage_estimate", "sum"),
            trades=("trade_id", "count"),
        )
        .reset_index()
    )
    daily_fig = go.Figure()
    daily_fig.add_trace(go.Bar(x=daily["exit_date"], y=daily["net_pnl_rupees"], name="Net P&L"))
    daily_fig.add_trace(go.Scatter(x=daily["exit_date"], y=daily["gross_pnl_rupees"], mode="lines+markers", name="Gross P&L"))
    daily_fig.add_hline(y=0, line_width=1, line_color="gray")
    daily_fig.update_layout(title="Daily P&L", xaxis_title="Exit date", yaxis_title="P&L (Rs)", hovermode="x unified")
    st.plotly_chart(daily_fig, use_container_width=True)

    drawdown_fig = px.area(closed, x="exit_time", y="drawdown", title="Drawdown")
    drawdown_fig.update_yaxes(tickformat=".2%")
    st.plotly_chart(drawdown_fig, use_container_width=True)

    trade_fig = px.histogram(
        closed,
        x="net_pnl_rupees",
        nbins=30,
        title="Net P&L per closed trade",
        labels={"net_pnl_rupees": "Net P&L (Rs)"},
    )
    st.plotly_chart(trade_fig, use_container_width=True)


def render_summary_tables(closed: pd.DataFrame) -> None:
    if closed.empty:
        st.info("No closed trades available for summary tables.")
        return

    summaries = summarize_by_dimensions(closed)
    tabs = st.tabs(["By symbol", "By weekday", "By hour", "Daily", "Closed trades"])
    table_map = [
        ("by_symbol", tabs[0]),
        ("by_weekday", tabs[1]),
        ("by_hour", tabs[2]),
        ("daily_summary", tabs[3]),
    ]

    for name, tab in table_map:
        with tab:
            st.dataframe(summaries.get(name, pd.DataFrame()), use_container_width=True, hide_index=True)

    with tabs[4]:
        display_cols = [
            "symbol",
            "direction",
            "entry_time",
            "exit_time",
            "quantity",
            "entry_price",
            "exit_price",
            "gross_pnl_rupees",
            "brokerage_estimate",
            "net_pnl_rupees",
            "pnl_pct",
            "status",
        ]
        available_cols = [col for col in display_cols if col in closed.columns]
        st.dataframe(closed[available_cols].sort_values("exit_time", ascending=False), use_container_width=True, hide_index=True)


def render_position_check(trades: pd.DataFrame) -> None:
    if trades.empty or "action" not in trades.columns:
        return

    signed = trades.copy()
    signed["signed_quantity"] = np.where(signed["action"].eq("BUY"), signed["quantity"], -signed["quantity"])
    balances = signed.groupby("symbol", dropna=False)["signed_quantity"].sum().reset_index(name="open_quantity")
    balances = balances[balances["open_quantity"].abs() > 1e-9]
    if balances.empty:
        st.success("All uploaded buy/sell quantities are fully paired by symbol.")
    else:
        st.warning("Some symbols have unpaired quantity after FIFO matching.")
        st.dataframe(balances, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Kite Trade Analyzer", layout="wide")
    st.title("Kite Trade Analyzer")

    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader("Upload Kite trades CSV", type=["csv"])
        use_sample = st.checkbox("Use kite_logs/trades.csv", value=uploaded_file is None and SAMPLE_TRADEBOOK_PATH.exists())

        st.header("Brokerage")
        brokerage_rate_pct = st.number_input("Brokerage rate per order (%)", min_value=0.0, max_value=5.0, value=DEFAULT_BROKERAGE_RATE * 100, step=0.005, format="%.4f")
        brokerage_cap = st.number_input("Brokerage cap per order (Rs)", min_value=0.0, value=float(DEFAULT_BROKERAGE_CAP), step=1.0)

    if uploaded_file is None and not use_sample:
        st.info("Upload a Kite trades CSV to start.")
        return

    try:
        if uploaded_file is not None:
            raw_trades = _load_uploaded_trades(uploaded_file)
            session_name = Path(uploaded_file.name).stem
        else:
            raw_trades = _load_sample_trades()
            session_name = SAMPLE_TRADEBOOK_PATH.parent.name
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    if raw_trades.empty:
        st.warning("The CSV is empty.")
        return

    brokerage_rate = brokerage_rate_pct / 100
    trades, closed, all_metrics = analyze_frame(raw_trades, session_name, brokerage_rate, brokerage_cap)

    with st.sidebar:
        st.header("Filters")
        symbols = sorted(trades["symbol"].dropna().unique().tolist())
        selected_symbols = st.multiselect("Symbols", options=symbols, default=symbols)
        min_date = trades["timestamp"].dt.date.min()
        max_date = trades["timestamp"].dt.date.max()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    filtered_trades, filtered_closed = apply_filters(trades, closed, selected_symbols, date_range)
    metrics = calculate_metrics(filtered_trades, filtered_closed, pd.DataFrame())

    st.caption(f"Loaded {len(trades):,} execution rows and reconstructed {len(closed):,} closed trades.")
    render_metric_cards(metrics)

    st.divider()
    render_position_check(filtered_trades)

    chart_tab, table_tab, download_tab, raw_tab = st.tabs(["Charts", "Analysis tables", "Downloads", "Raw data"])
    with chart_tab:
        render_charts(filtered_closed)

    with table_tab:
        render_summary_tables(filtered_closed)

    with download_tab:
        st.download_button("Download closed trades CSV", _to_csv_bytes(filtered_closed), "closed_trades_enriched.csv", "text/csv")
        st.download_button("Download normalized trades CSV", _to_csv_bytes(filtered_trades), "all_trades_normalized.csv", "text/csv")
        st.download_button("Download metrics JSON", _to_json_bytes(metrics), "metrics.json", "application/json")
        st.json(metrics)

    with raw_tab:
        st.subheader("Uploaded columns")
        st.json({"columns": raw_trades.columns.tolist(), "session": asdict(SessionFiles(session_name, Path("."), Path("uploaded_trades.csv"), None, None, None))})
        st.dataframe(raw_trades, use_container_width=True, hide_index=True)

    with st.expander("Brokerage assumptions"):
        st.write(
            "Brokerage is estimated as min(order turnover * brokerage rate, brokerage cap) for each entry and exit order. "
            "Taxes, exchange charges, STT/CTT, stamp duty, and GST are not included."
        )
        st.write({"brokerage_rate": brokerage_rate, "brokerage_cap": brokerage_cap, "unfiltered_metrics": all_metrics})


if __name__ == "__main__":
    main()
