#!/usr/bin/env python3
"""
Run mock tests for the Moving Average (MA) strategy over a specified date range
with user-specified parameter sets using Zerodha Kite data.
This script bypasses optimization and directly simulates provided parameter configurations.

QUICK START:
  Edit the global variables at the top of this script (DEFAULT_SYMBOL, 
  DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_INTERVAL, DEFAULT_PARAMS)
  and simply run: python run_ma_mock_validation_kite.py

Usage examples:

  - Using global defaults (edit DEFAULT_* variables in script):
    python run_ma_mock_validation_kite.py

  - Override symbol only:
    python run_ma_mock_validation_kite.py --symbol TCS

  - Single param set via JSON string:
    python run_ma_mock_validation_kite.py \
      --symbol RELIANCE \
      --start "2025-10-15" \
      --end "2025-11-15" \
      --interval 15m \
      --params '[{"short_window":10,"long_window":40,"risk_reward_ratio":3.0}]'

  - Multiple param sets via JSON file:
    python run_ma_mock_validation_kite.py \
      --symbol TCS \
      --start "2025-10-01" \
      --end "2025-11-01" \
      --interval 15m \
      --params-file params_ma.json

The params must be a list of objects with keys:
  short_window (int), long_window (int), risk_reward_ratio (float)
Optional: trading_fee will be taken from the validator config if omitted.

Note: Dates should be in YYYY-MM-DD format (Kite uses date-only format)
Interval will be automatically converted to Kite format (e.g., 15m -> 15minute)
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kite_comprehensive_strategy_validation import KiteComprehensiveStrategyValidator
from trading_engine import TradingEngine
from parameter_validator import ParameterValidator
import config as cfg


# ============================================================================
# GLOBAL CONFIGURATION - Edit these values to set defaults
# ============================================================================
# You can override these via command-line arguments if needed
DEFAULT_SYMBOL = "NATGASMINI26FEBFUT"  # Default Indian stock symbol
#DEFAULT_SYMBOL = "CRUDEOIL26MARFUT"  # Default Indian stock symbol
DEFAULT_EXCHANGE = "MCX"  # Default exchange (NSE, BSE, MCX)
DAYS_TO_VALIDATE = 30
# use 30 days ago (Kite uses YYYY-MM-DD format)
DEFAULT_START_DATE = (datetime.now() - timedelta(days=DAYS_TO_VALIDATE)).strftime("%Y-%m-%d")
# use today's date
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

# # use 30 days ago (Kite uses YYYY-MM-DD format)
# DEFAULT_START_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
# # use today's date
# DEFAULT_END_DATE = (datetime.now() - timedelta(days=0)).strftime("%Y-%m-%d")
DEFAULT_INTERVAL = "15m"  # Will be converted to "15minute" for Kite

DEFAULT_VALIDATION_WINDOW_DAYS = 31  # How often to run validation (every N days)

# Parameter ranges for validation optimization (matching run_ma_optimization_kite.py format)
VALIDATION_SHORT_VAL = 4# Center value for short window range
VALIDATION_LONG_VAL = 184  # Center value for long window range
VALIDATION_RR_VAL = 7.0
VALIDATION_RANGE = 4  # Range around center values
VALIDATION_SHORT_START = max(VALIDATION_SHORT_VAL - VALIDATION_RANGE, 4)
VALIDATION_SHORT_END = VALIDATION_SHORT_VAL + VALIDATION_RANGE
VALIDATION_LONG_START = max(VALIDATION_LONG_VAL - VALIDATION_RANGE, VALIDATION_SHORT_VAL + VALIDATION_RANGE + 1)
VALIDATION_LONG_END = VALIDATION_LONG_VAL + VALIDATION_RANGE
DEFAULT_VALIDATION_SHORT_RANGE = list(range(VALIDATION_SHORT_START, VALIDATION_SHORT_END + 1))
DEFAULT_VALIDATION_LONG_RANGE = list(range(VALIDATION_LONG_START, VALIDATION_LONG_END + 1))
DEFAULT_VALIDATION_RR_RANGE = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]  # Risk-reward ratios to test
# ============================================================================

DEFAULT_PARAMS = [
    {"short_window": VALIDATION_SHORT_VAL, "long_window": VALIDATION_LONG_VAL, "risk_reward_ratio": VALIDATION_RR_VAL}
]


def map_interval_to_kite(interval: str) -> str:
    """
    Map common interval formats to Kite Connect format
    
    Args:
        interval (str): Interval in common format (e.g., '15m', '1h', '1d')
        
    Returns:
        str: Kite Connect interval format (e.g., '15minute', 'hour', 'day')
    """
    interval_mapping = {
        '1m': 'minute',
        '3m': '3minute',
        '5m': '5minute',
        '15m': '15minute',
        '30m': '30minute',
        '1h': 'hour',
        '2h': '2hour',
        '4h': '4hour',
        '1d': 'day',
        '1w': 'week',
        '1M': 'month'
    }
    return interval_mapping.get(interval, '15minute')


def convert_date_format(date_str: str) -> str:
    """
    Convert date from YYYY-MM-DD HH:MM:SS to YYYY-MM-DD (Kite format)
    or return as-is if already in YYYY-MM-DD format
    
    Args:
        date_str (str): Date string in either format
        
    Returns:
        str: Date string in YYYY-MM-DD format
    """
    try:
        # Try parsing as full datetime
        dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return dt.strftime('%Y-%m-%d')
    except ValueError:
        try:
            # Try parsing as date only
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            # If neither works, return as-is (will fail later with better error)
            return date_str


def create_cumulative_pnl_chart(
    trade_history: pd.DataFrame,
    params: Dict,
    symbol: str,
    initial_balance: float,
    output_path: str
) -> str:
    """
    Create a cumulative PnL chart based on closed trades.
    
    Shows how realized PnL accumulates over time after each trade closes.
    
    Args:
        trade_history: DataFrame with trade records
        params: Parameter dict for the strategy
        symbol: Trading symbol
        initial_balance: Starting balance
        output_path: Path to save the HTML chart
        
    Returns:
        Path to saved chart
    """
    if trade_history.empty:
        print("⚠️  No trade history to plot")
        return None
    
    # Filter closed trades only
    closed_statuses = ['closed', 'tp_hit', 'sl_hit', 'reversed']
    closed_trades = trade_history[trade_history['status'].isin(closed_statuses)].copy()
    
    if closed_trades.empty:
        print("⚠️  No closed trades to plot")
        return None
    
    # Sort by exit time
    if 'exit_time' not in closed_trades.columns:
        print("⚠️  No exit_time in trade history")
        return None
    
    closed_trades = closed_trades.sort_values('exit_time').reset_index(drop=True)
    
    # Calculate cumulative PnL
    # Each trade's pnl is already a percentage return
    cumulative_pnl = []
    cumulative_pnl_rupees = []  # Cumulative PnL in rupees
    cumulative_balance = []
    trade_pnl_rupees = []  # Per-trade PnL in rupees
    running_balance = initial_balance
    
    for idx, trade in closed_trades.iterrows():
        # Get the PnL percentage for this trade
        trade_pnl_pct = trade['pnl']  # This is already a decimal (e.g., 0.05 for 5%)
        
        # Calculate rupee profit/loss based on position size at entry
        # First try to use stored margin_used (most accurate)
        margin_used = trade.get('margin_used')
        
        if margin_used and margin_used > 0:
            # Use stored margin_used if available (from Kite API for commodity futures)
            trade_profit = margin_used * trade_pnl_pct
        else:
            # Fallback: calculate margin from position size and leverage
            leverage = trade.get('leverage', 1.0)
            position_size = trade.get('position_size', trade.get('quantity', 0) * trade.get('entry_price', 0))
            
            if position_size > 0 and leverage > 0:
                margin_used = position_size / leverage
                trade_profit = margin_used * trade_pnl_pct
            else:
                # Final fallback: use running balance with pnl percentage
                trade_profit = running_balance * trade_pnl_pct
        
        running_balance += trade_profit
        cumulative_balance.append(running_balance)
        cumulative_pnl.append((running_balance - initial_balance) / initial_balance * 100)
        cumulative_pnl_rupees.append(running_balance - initial_balance)
        trade_pnl_rupees.append(trade_profit)
    
    # Create the chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
        subplot_titles=('📈 Cumulative PnL (%)', '📊 Per-Trade PnL (%)')
    )
    
    exit_times = closed_trades['exit_time'].tolist()
    trade_pnls = (closed_trades['pnl'] * 100).tolist()  # Convert to percentage
    
    # Add starting point (0% at first trade entry or start)
    first_entry = closed_trades['entry_time'].iloc[0] if 'entry_time' in closed_trades.columns else exit_times[0]
    
    # 1. Cumulative PnL line
    plot_times = [first_entry] + exit_times
    plot_pnls = [0] + cumulative_pnl
    plot_pnls_rupees = [0] + cumulative_pnl_rupees
    
    # Color based on positive/negative
    colors_line = ['green' if p >= 0 else 'red' for p in plot_pnls]
    
    fig.add_trace(
        go.Scatter(
            x=plot_times,
            y=plot_pnls,
            mode='lines+markers',
            name='Cumulative PnL',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color=colors_line, line=dict(width=1, color='white')),
            hovertemplate='<b>Time:</b> %{x}<br><b>Cumulative PnL:</b> %{y:.2f}% (₹%{customdata:.2f})<extra></extra>',
            customdata=plot_pnls_rupees,
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.2)'
        ),
        row=1, col=1
    )
    
    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
    
    # 2. Per-trade PnL bar chart
    bar_colors = ['green' if p >= 0 else 'red' for p in trade_pnls]
    
    fig.add_trace(
        go.Bar(
            x=exit_times,
            y=trade_pnls,
            name='Trade PnL',
            marker_color=bar_colors,
            hovertemplate='<b>Exit:</b> %{x}<br><b>PnL:</b> %{y:.2f}% (₹%{customdata:.2f})<extra></extra>',
            customdata=trade_pnl_rupees,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add zero line for bar chart
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)
    
    # Calculate summary stats
    final_pnl = cumulative_pnl[-1] if cumulative_pnl else 0
    final_pnl_rupees = cumulative_pnl_rupees[-1] if cumulative_pnl_rupees else 0
    final_balance = cumulative_balance[-1] if cumulative_balance else initial_balance
    total_trades = len(closed_trades)
    winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
    losing_trades = len(closed_trades[closed_trades['pnl'] < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # Calculate max drawdown from cumulative PnL
    peak = 0
    max_dd = 0
    for pnl in cumulative_pnl:
        if pnl > peak:
            peak = pnl
        dd = peak - pnl
        if dd > max_dd:
            max_dd = dd
    
    # Format params for title
    params_str = f"Short={params.get('short_window')}, Long={params.get('long_window')}, RR={params.get('risk_reward_ratio')}"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"📈 {symbol} MA Strategy - Cumulative PnL<br>"
                 f"<sup>{params_str} | Final PnL: {final_pnl:+.2f}% (₹{final_pnl_rupees:+,.2f}) | "
                 f"Trades: {total_trades} (W:{winning_trades}/L:{losing_trades}) | "
                 f"Win Rate: {win_rate:.1f}% | Max DD: {max_dd:.2f}%</sup>",
            font=dict(size=16)
        ),
        xaxis2_title="Time",
        yaxis_title="Cumulative PnL (%)",
        yaxis2_title="Trade PnL (%)",
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    # Save chart
    fig.write_html(output_path)
    print(f"📊 Cumulative PnL chart saved: {output_path}")
    
    return output_path


def create_combined_pnl_chart(
    all_results: List[Dict],
    symbol: str,
    initial_balance: float,
    output_path: str
) -> str:
    """
    Create a combined cumulative PnL chart comparing all parameter sets.
    
    Args:
        all_results: List of results with trade_history for each param set
        symbol: Trading symbol
        initial_balance: Starting balance
        output_path: Path to save the HTML chart
        
    Returns:
        Path to saved chart
    """
    if not all_results:
        print("⚠️  No results to plot")
        return None
    
    fig = go.Figure()
    
    colors = ['#2E86AB', '#E74C3C', '#27AE60', '#9B59B6', '#F39C12', '#1ABC9C', '#E91E63', '#3F51B5']
    
    for idx, result in enumerate(all_results):
        trade_history = result.get('trade_history')
        if trade_history is None or trade_history.empty:
            continue
        
        # Filter closed trades
        closed_statuses = ['closed', 'tp_hit', 'sl_hit', 'reversed']
        closed_trades = trade_history[trade_history['status'].isin(closed_statuses)].copy()
        
        if closed_trades.empty or 'exit_time' not in closed_trades.columns:
            continue
        
        closed_trades = closed_trades.sort_values('exit_time').reset_index(drop=True)
        
        # Calculate cumulative PnL
        cumulative_pnl = []
        running_balance = initial_balance
        
        for _, trade in closed_trades.iterrows():
            trade_pnl_pct = trade['pnl']
            leverage = trade.get('leverage', 1.0)
            position_size = trade.get('position_size', trade.get('quantity', 0) * trade.get('entry_price', 0))
            
            if position_size > 0 and leverage > 0:
                margin_used = position_size / leverage
                trade_profit = margin_used * trade_pnl_pct
            else:
                trade_profit = running_balance * trade_pnl_pct
            
            running_balance += trade_profit
            cumulative_pnl.append((running_balance - initial_balance) / initial_balance * 100)
        
        params = result['parameters']
        params_label = f"S={params.get('short_window')}, L={params.get('long_window')}, RR={params.get('risk_reward_ratio')}"
        
        exit_times = closed_trades['exit_time'].tolist()
        first_entry = closed_trades['entry_time'].iloc[0] if 'entry_time' in closed_trades.columns else exit_times[0]
        
        plot_times = [first_entry] + exit_times
        plot_pnls = [0] + cumulative_pnl
        
        color = colors[idx % len(colors)]
        final_pnl = cumulative_pnl[-1] if cumulative_pnl else 0
        
        fig.add_trace(
            go.Scatter(
                x=plot_times,
                y=plot_pnls,
                mode='lines+markers',
                name=f"{params_label} ({final_pnl:+.1f}%)",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{params_label}</b><br>Time: %{{x}}<br>PnL: %{{y:.2f}}%<extra></extra>'
            )
        )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=dict(
            text=f"📊 {symbol} - Parameter Comparison (Cumulative PnL %)",
            font=dict(size=16)
        ),
        xaxis_title="Time",
        yaxis_title="Cumulative PnL (%)",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    fig.write_html(output_path)
    print(f"📊 Combined PnL chart saved: {output_path}")
    
    return output_path


def _load_params_list(params_str: str, params_file: str, default_params: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if params_str and params_file:
        raise ValueError("Provide either --params or --params-file, not both.")
    if params_str:
        parsed = json.loads(params_str)
        if not isinstance(parsed, list):
            raise ValueError("--params must be a JSON array of parameter objects")
        return parsed
    if params_file:
        with open(params_file, "r") as f:
            parsed = json.load(f)
        if not isinstance(parsed, list):
            raise ValueError("--params-file must contain a JSON array of parameter objects")
        return parsed
    # Use default params if provided
    if default_params is not None:
        return default_params
    raise ValueError("You must provide parameter sets via --params, --params-file, or set DEFAULT_PARAMS in the script")


def run_ma_mock(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    params_list: List[Dict[str, Any]],
    exchange: str = "NSE",
    initial_balance: float = 10000.0,
    max_leverage: float = 10.0,
    max_loss_percent: float = 2.0,
    trading_fee: float = 0.0003,
    mock_trading_delay: float = 0.0,
    output_dir: str = "ma_mock_results_kite",
    enable_parameter_validation: bool = True,
    validation_data_window_days: int = 7,  # How often to run validation
    days_to_validate: int = 30,  # How much past data to use for optimization during validation
    validation_short_range: List[int] = None,  # Short window range for validation optimization
    validation_long_range: List[int] = None,  # Long window range for validation optimization
    validation_rr_range: List[float] = None  # Risk-reward range for validation optimization
) -> Dict[str, Any]:
    # Use defaults if not provided
    if validation_short_range is None:
        validation_short_range = DEFAULT_VALIDATION_SHORT_RANGE
    if validation_long_range is None:
        validation_long_range = DEFAULT_VALIDATION_LONG_RANGE
    if validation_rr_range is None:
        validation_rr_range = DEFAULT_VALIDATION_RR_RANGE
    print("🚀 MA STRATEGY MOCK VALIDATION (KITE)")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Exchange: {exchange}")
    print(f"Date Range: {start_date} → {end_date}")
    print(f"Interval: {interval} (Kite: {map_interval_to_kite(interval)})")
    print(f"Param sets: {len(params_list)}")
    
    # Convert dates to Kite format (YYYY-MM-DD)
    start_date_kite = convert_date_format(start_date)
    end_date_kite = convert_date_format(end_date)
    
    # Store original backtest date range (for data leak prevention)
    original_start_date_kite = start_date_kite
    original_end_date_kite = end_date_kite
    
    # If validation is enabled, we need to fetch extra data from the past
    # to have enough data for validation after first validation_window_days
    # We need: days_to_validate days of past data for optimization
    # So actual start date should be: end_date - (backtest_days + days_to_validate)
    # IMPORTANT: Extra data is ONLY for validation, NOT for backtesting!
    if enable_parameter_validation:
        # Calculate how many days of backtest we have
        start_dt = datetime.strptime(start_date_kite, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date_kite, '%Y-%m-%d')
        backtest_days = (end_dt - start_dt).days
        
        # We need to fetch data starting from: end_date - backtest_days - days_to_validate
        # This ensures after first validation_window_days, we have days_to_validate days of past data
        actual_start_date = (end_dt - timedelta(days=backtest_days + days_to_validate)).strftime('%Y-%m-%d')
        print(f"📊 Validation enabled: Fetching extra {days_to_validate} days of past data for validation")
        print(f"   Fetching data: {actual_start_date} → {end_date_kite}")
        print(f"   ⚠️  Backtest will ONLY use: {original_start_date_kite} → {original_end_date_kite} (no data leak)")
        start_date_kite = actual_start_date
    
    # Convert interval to Kite format
    kite_interval = map_interval_to_kite(interval)
    
    # Auto-adjust max_loss_percent for 1x leverage to match optimization script
    adjusted_max_loss_percent = max_loss_percent
    if max_leverage == 1.0:
        adjusted_max_loss_percent = 100.0
        print(f"⚙️  Leverage is 1x: Auto-setting max_loss_percent to 100% (disables ATR-based position sizing)")
        print(f"   This ensures matching with optimization script behavior")
    
    print(f"📊 Max Leverage: {max_leverage}x")
    print(f"📊 Max Loss Percent: {adjusted_max_loss_percent}%")

    # Use Kite validator instead of regular validator
    validator = KiteComprehensiveStrategyValidator(
        initial_balance=initial_balance,
        max_leverage=max_leverage,
        max_loss_percent=adjusted_max_loss_percent,
        trading_fee=trading_fee,
        exchange=exchange
    )

    # Authenticate with Kite Connect
    validator.authenticate_kite()
    
    # Fetch data using Kite fetcher
    print("📥 Fetching data from Kite...")
    
    # We'll fetch all data and use it as test data (no train split for mock validation)
    data = validator.kite_fetcher.fetch_historical_data(
        symbol, start_date_kite, end_date_kite, interval=kite_interval
    )
    
    if data.empty:
        raise ValueError(f"No data fetched for {symbol} on {exchange}")
    
    # Filter for market hours if needed (Kite validator does this automatically in fetch_and_split_data)
    # For mock validation, we'll use all data but can filter if needed
    if exchange in ["NSE", "BSE"]:
        data = validator._filter_equity_market_hours(data)
    elif exchange == "MCX":
        data = validator._filter_mcx_market_hours(data)
    
    if data.empty:
        raise ValueError(f"No data remaining after filtering for {symbol}")
    
    # CRITICAL: Filter data to ONLY use original backtest date range (prevent data leak)
    # Extra past data is ONLY for parameter validation, NOT for backtesting
    if enable_parameter_validation:
        # Convert original dates to datetime for filtering
        original_start_dt = datetime.strptime(original_start_date_kite, '%Y-%m-%d')
        original_end_dt = datetime.strptime(original_end_date_kite, '%Y-%m-%d')
        
        # Filter data to only include backtest period
        def get_date_from_index(idx):
            ts = pd.Timestamp(idx)
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            return ts.date()
        
        backtest_data = data[
            data.index.map(lambda x: original_start_dt.date() <= get_date_from_index(x) <= original_end_dt.date())
        ].copy()
        
        # Store full data (including extra past) for validation purposes only
        validation_data = data.copy()
        
        print(f"📊 Data fetched: {len(data)} total points")
        print(f"   Backtest data: {len(backtest_data)} points ({original_start_date_kite} → {original_end_date_kite})")
        print(f"   Extra past data: {len(data) - len(backtest_data)} points (for validation only)")
        
        validator.train_data = backtest_data.iloc[:0].copy()
        validator.test_data = backtest_data.copy()  # Only backtest period for actual trading
        validator.symbol = symbol
        validator.interval = kite_interval
        
        # Store validation data separately (not used in backtest)
        validator._validation_data = validation_data
        
        print(f"✅ Backtest will use {len(backtest_data)} data points")
        print(f"📅 Backtest range: {backtest_data.index[0]} to {backtest_data.index[-1]}")
    else:
        # No validation, use all data as before
        validator.train_data = data.iloc[:0].copy()
        validator.test_data = data.copy()
        validator.symbol = symbol
        validator.interval = kite_interval
        
        print(f"✅ Successfully fetched {len(data)} data points")
        print(f"📅 Data range: {data.index[0]} to {data.index[-1]}")

    results: List[Dict[str, Any]] = []
    failed_results: List[Dict[str, Any]] = []  # Track failed parameter sets

    for idx, params in enumerate(params_list, start=1):
        print("-" * 80)
        print(f"🎯 Testing parameter set {idx}/{len(params_list)}: {params}")

        # Ensure trading_fee is present as required by the MA strategy
        params_with_fee = dict(params)
        if "trading_fee" not in params_with_fee:
            params_with_fee["trading_fee"] = validator.trading_fee
            print(f"🔄 Added trading_fee {validator.trading_fee} to parameters")

        ok = validator.strategy_manager.set_manual_parameters(ma_params=params_with_fee)
        if not ok:
            print(f"❌ Failed to set manual MA parameters for set {idx}")
            failed_results.append({
                "parameter_set": idx,
                "parameters": params,
                "error": "Failed to set manual MA parameters",
                "stage": "parameter_setup"
            })
            continue

        strategies = validator.strategy_manager.initialize_strategies(["ma"])
        if not strategies:
            print(f"❌ Failed to initialize MA strategy for set {idx}")
            failed_results.append({
                "parameter_set": idx,
                "parameters": params,
                "error": "Failed to initialize MA strategy",
                "stage": "strategy_initialization"
            })
            continue

        strategy = strategies[0]

        engine = TradingEngine(
            validator.initial_balance,
            validator.max_leverage,
            adjusted_max_loss_percent,  # Use adjusted value
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{validator.symbol}_{exchange}_ma_param_{idx}_mock_{timestamp}"
        engine.setup_logging(session_id, validator.symbol)

        print("🚦 Running simulation...")
        start_t = datetime.now()
        
        # Initialize parameter validator if enabled
        param_validator = None
        last_validation_date = None
        validation_results_history = []
        
        if enable_parameter_validation:
            param_validator = ParameterValidator(
                validation_frequency_days=validation_data_window_days,
                data_window_days=days_to_validate,  # Use days_to_validate for optimization window
                trading_fee=trading_fee,
                exchange=exchange
            )
            # Initialize with Kite fetcher from validator
            param_validator.kite_fetcher = validator.kite_fetcher
        
        for j in range(len(validator.test_data)):
            current_data = validator.test_data.iloc[: j + 1]
            current_time = validator.test_data.index[j]
            engine.process_strategy_signals(strategy, current_data, current_time)
            
            # Check if we need to run parameter validation
            if enable_parameter_validation and param_validator:
                # Extract date from timestamp (handle timezone-aware)
                current_ts = pd.Timestamp(current_time)
                if current_ts.tz is not None:
                    current_ts = current_ts.tz_localize(None)  # Remove timezone for date comparison
                current_date = current_ts.date()
                
                # Check if validation_window_days have passed since last validation
                should_validate = False
                if last_validation_date is None:
                    # First validation after validation_window_days
                    first_ts = pd.Timestamp(validator.test_data.index[0])
                    if first_ts.tz is not None:
                        first_ts = first_ts.tz_localize(None)
                    first_date = first_ts.date()
                    days_elapsed = (current_date - first_date).days
                    if days_elapsed >= validation_data_window_days:
                        should_validate = True
                else:
                    days_since_validation = (current_date - last_validation_date).days
                    if days_since_validation >= validation_data_window_days:
                        should_validate = True
                
                if should_validate:
                    print(f"\n{'='*80}")
                    print(f"🔍 PARAMETER VALIDATION CHECK (Day {current_date})")
                    print(f"{'='*80}")
                    
                    try:
                        # Get recent data up to current point for validation
                        # Use DAYS_TO_VALIDATE days of past data for optimization
                        validation_start_date = current_date - timedelta(days=days_to_validate)
                        validation_end_date = current_date
                        
                        # IMPORTANT: Use validation_data (includes extra past data), NOT test_data
                        # This ensures we can look back days_to_validate days even from early in backtest
                        validation_data_source = getattr(validator, '_validation_data', validator.test_data)
                        
                        # Filter validation_data to get recent window
                        # Handle timezone-aware indices
                        def get_date_from_index(idx):
                            ts = pd.Timestamp(idx)
                            if ts.tz is not None:
                                ts = ts.tz_localize(None)
                            return ts.date()
                        
                        recent_data = validation_data_source[
                            validation_data_source.index.map(lambda x: validation_start_date <= get_date_from_index(x) <= validation_end_date)
                        ].copy()
                        
                        if len(recent_data) < 50:  # Need minimum data points
                            print(f"⚠️  Insufficient data for validation ({len(recent_data)} points), skipping...")
                            print(f"   Need at least {days_to_validate} days of past data from {validation_start_date}")
                        else:
                            print(f"📊 Validating on {len(recent_data)} data points from {validation_start_date} to {validation_end_date}")
                            print(f"   Using {days_to_validate} days of past data for optimization")
                            
                            # Run optimization on recent data with specified ranges
                            new_optimal_params, new_optimal_metrics = param_validator.run_optimization(
                                recent_data,
                                short_window_range=validation_short_range,
                                long_window_range=validation_long_range,
                                risk_reward_range=validation_rr_range
                            )
                            
                            # Evaluate current parameters
                            current_metrics = param_validator.evaluate_parameters(recent_data, params_with_fee)
                            
                            # Calculate distance and gap
                            param_distance = param_validator.calculate_parameter_distance(params_with_fee, new_optimal_params)
                            performance_gap = new_optimal_metrics.get('total_pnl', 0) - current_metrics.get('total_pnl', 0)
                            
                            # Determine alert level
                            alert_level = 'none'
                            if param_distance >= param_validator.distance_threshold_critical or performance_gap >= param_validator.performance_gap_threshold * 2:
                                alert_level = 'critical'
                            elif param_distance >= param_validator.distance_threshold_warning or performance_gap >= param_validator.performance_gap_threshold:
                                alert_level = 'warning'
                            elif param_distance >= param_validator.distance_threshold_monitor:
                                alert_level = 'monitor'
                            
                            should_reopt = alert_level in ['warning', 'critical']
                            
                            # Report results
                            status_emoji = {
                                'none': '✅',
                                'monitor': '📊',
                                'warning': '⚠️',
                                'critical': '🚨'
                            }
                            emoji = status_emoji.get(alert_level, '❓')
                            
                            print(f"{emoji} Validation Results:")
                            print(f"   Parameter Distance: {param_distance:.2f}")
                            print(f"   Performance Gap: {performance_gap:.2%}")
                            print(f"   Alert Level: {alert_level.upper()}")
                            print(f"   Current Params: Short={params_with_fee['short_window']}, Long={params_with_fee['long_window']}, RR={params_with_fee['risk_reward_ratio']}")
                            print(f"   New Optimal: Short={new_optimal_params['short_window']}, Long={new_optimal_params['long_window']}, RR={new_optimal_params['risk_reward_ratio']}")
                            
                            if should_reopt:
                                print(f"   ⚠️  RECOMMENDATION: Parameters have drifted - consider re-optimization!")
                                if alert_level == 'critical':
                                    print(f"   🚨 CRITICAL: Strong recommendation to stop and re-optimize")
                            else:
                                print(f"   ✅ Parameters still optimal - continue trading")
                            
                            # Store validation result
                            validation_results_history.append({
                                'date': current_date.isoformat(),
                                'parameter_distance': param_distance,
                                'performance_gap': performance_gap,
                                'alert_level': alert_level,
                                'should_reoptimize': should_reopt,
                                'current_params': params_with_fee.copy(),
                                'new_optimal_params': new_optimal_params.copy(),
                                'current_performance': current_metrics,
                                'new_optimal_performance': new_optimal_metrics
                            })
                            
                            last_validation_date = current_date
                            print(f"{'='*80}\n")
                            
                    except Exception as e:
                        print(f"⚠️  Validation failed: {e}")
                        import traceback
                        traceback.print_exc()
            
            if mock_trading_delay > 0:
                import time
                time.sleep(mock_trading_delay)
        
        duration_s = (datetime.now() - start_t).total_seconds()

        final_status = engine.get_current_status()
        trade_history = engine.get_trade_history_df()
        performance_metrics = engine.calculate_performance_metrics()

        result = {
            "parameter_set": idx,
            "parameters": params,
            "final_status": final_status,
            "trade_history": trade_history,
            "performance_metrics": performance_metrics,
            "simulation_duration_sec": duration_s,
            "session_id": session_id,
        }
        
        # Store validation history in result (after result is created)
        if enable_parameter_validation and validation_results_history:
            result['parameter_validation_history'] = validation_results_history
        
        results.append(result)

        print(f"✅ Done. Final Balance: ${final_status['current_balance']:,.2f}")
        print(f"📋 Performance Metrics: {performance_metrics}")
        if "total_pnl" in performance_metrics:
            # Calculate rupee PnL
            total_pnl_rupees = final_status['current_balance'] - initial_balance
            print(f"📈 Total PnL: {performance_metrics['total_pnl']:.2%} (₹{total_pnl_rupees:+,.2f})")
        if "win_rate" in performance_metrics:
            print(f"🎯 Win Rate: {performance_metrics['win_rate']:.2%}")
        if "sharpe_ratio" in performance_metrics:
            print(f"📊 Sharpe: {performance_metrics['sharpe_ratio']:.3f}")
        if "max_drawdown" in performance_metrics:
            print(f"📉 Max DD: {performance_metrics['max_drawdown']:.2%}")

    # Save outputs
    if results:
        os.makedirs(output_dir, exist_ok=True)
        out_json = os.path.join(
            output_dir, f"ma_mock_results_{symbol}_{exchange}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Convert trade_history DataFrames to CSV paths and dict references
        serializable = []
        for r in results:
            csv_path = os.path.join(output_dir, f"{r['session_id']}_trades.csv")
            r["trade_history"].to_csv(csv_path, index=False)
            result_dict = {
                "parameter_set": r["parameter_set"],
                "parameters": r["parameters"],
                "final_status": r["final_status"],
                "performance_metrics": r["performance_metrics"],
                "simulation_duration_sec": r["simulation_duration_sec"],
                "session_id": r["session_id"],
                "trade_history_csv": csv_path,
            }
            # Include parameter validation history if available
            if "parameter_validation_history" in r:
                result_dict["parameter_validation_history"] = r["parameter_validation_history"]
            serializable.append(result_dict)

        with open(out_json, "w") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "start_date": start_date_kite,
                    "end_date": end_date_kite,
                    "interval": kite_interval,
                    "results": serializable,
                    "failed_results": failed_results,  # Include failed parameter sets
                },
                f,
                indent=2,
                default=str,
            )
        print(f"💾 Saved results to {out_json}")
        
        # Generate PnL charts
        print("\n📊 Generating Cumulative PnL charts...")
        chart_paths = []
        
        for r in results:
            if r.get('trade_history') is not None and not r['trade_history'].empty:
                chart_filename = f"{r['session_id']}_cumulative_pnl.html"
                chart_path = os.path.join(output_dir, chart_filename)
                
                try:
                    result_path = create_cumulative_pnl_chart(
                        trade_history=r['trade_history'],
                        params=r['parameters'],
                        symbol=symbol,
                        initial_balance=initial_balance,
                        output_path=chart_path
                    )
                    if result_path:
                        chart_paths.append(chart_path)
                except Exception as e:
                    print(f"⚠️  Failed to create chart for param set {r['parameter_set']}: {e}")
        
        # Create combined comparison chart if multiple param sets
        if len(results) > 1:
            combined_chart_path = os.path.join(
                output_dir, 
                f"ma_mock_combined_pnl_{symbol}_{exchange}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            try:
                create_combined_pnl_chart(
                    all_results=results,
                    symbol=symbol,
                    initial_balance=initial_balance,
                    output_path=combined_chart_path
                )
                chart_paths.append(combined_chart_path)
            except Exception as e:
                print(f"⚠️  Failed to create combined chart: {e}")
        
        # Open the first chart in browser
        if chart_paths:
            print(f"\n🌐 Opening PnL chart in browser...")
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(chart_paths[0])}")
            except Exception as e:
                print(f"⚠️  Could not open browser: {e}")
                print(f"   Please manually open: {os.path.abspath(chart_paths[0])}")

    # Report failed parameter sets
    if failed_results:
        print(f"\n⚠️  {len(failed_results)} parameter set(s) failed:")
        for failed in failed_results:
            print(f"   - Set {failed['parameter_set']}: {failed['error']} (stage: {failed['stage']})")

    # =========================================================================
    # FINAL PARAMETER VALIDATION SUMMARY
    # =========================================================================
    if enable_parameter_validation and results:
        print("\n" + "=" * 80)
        print("📋 PARAMETER VALIDATION SUMMARY")
        print("=" * 80)
        print("Summary of periodic validation checks during simulation...")
        print(f"Validation was performed every {validation_data_window_days} days during backtest")
        print(f"Each validation used {days_to_validate} days of past data for optimization")
        
        # Print validation history from during simulation
        for r in results:
            if r.get('parameter_validation_history'):
                params = r['parameters']
                print(f"\n📊 Param Set {r['parameter_set']} (S={params.get('short_window')}, L={params.get('long_window')}, RR={params.get('risk_reward_ratio')}):")
                print(f"   Validation checks performed: {len(r['parameter_validation_history'])}")
                
                for val_check in r['parameter_validation_history']:
                    alert_level = val_check.get('alert_level', 'none')
                    status_emoji = {
                        'none': '✅',
                        'monitor': '📊',
                        'warning': '⚠️',
                        'critical': '🚨'
                    }
                    emoji = status_emoji.get(alert_level, '❓')
                    
                    print(f"   {emoji} {val_check['date']}: Distance={val_check['parameter_distance']:.2f}, Gap={val_check['performance_gap']:.2%}, Level={alert_level.upper()}")
                    if val_check.get('should_reoptimize'):
                        print(f"      ⚠️  Re-optimization recommended at this point")
                
                # Overall recommendation
                critical_count = sum(1 for v in r['parameter_validation_history'] if v.get('alert_level') == 'critical')
                warning_count = sum(1 for v in r['parameter_validation_history'] if v.get('alert_level') == 'warning')
                
                if critical_count > 0:
                    print(f"   🚨 CRITICAL: {critical_count} critical drift(s) detected during simulation")
                elif warning_count > 0:
                    print(f"   ⚠️  WARNING: {warning_count} warning(s) detected during simulation")
                else:
                    print(f"   ✅ Parameters remained optimal throughout simulation")
            else:
                params = r['parameters']
                print(f"\n📊 Param Set {r['parameter_set']} (S={params.get('short_window')}, L={params.get('long_window')}, RR={params.get('risk_reward_ratio')}):")
                print(f"   ⚠️  No validation checks performed (insufficient data or validation disabled)")

    print("\n🎯 Completed MA mock validation.")
    return {
        "symbol": symbol,
        "exchange": exchange,
        "start_date": start_date_kite,
        "end_date": end_date_kite,
        "interval": kite_interval,
        "backtest_data": validator.test_data.copy() if validator.test_data is not None else pd.DataFrame(),
        "results": results,
        "failed_results": failed_results,  # Include failed parameter sets in return value
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MA strategy mock tests with Kite data and explicit params")
    p.add_argument("--symbol", default=None, help=f"Symbol, e.g., TCS, RELIANCE (default: {DEFAULT_SYMBOL})")
    p.add_argument("--exchange", default=None, help=f"Exchange: NSE, BSE, MCX (default: {DEFAULT_EXCHANGE})")
    p.add_argument("--start", default=None, help=f"Start date 'YYYY-MM-DD' (default: {DEFAULT_START_DATE})")
    p.add_argument("--end", default=None, help=f"End date 'YYYY-MM-DD' (default: {DEFAULT_END_DATE})")
    p.add_argument("--interval", default=None, help=f"Interval, e.g., 1m, 5m, 15m (default: {DEFAULT_INTERVAL})")
    p.add_argument("--params", help="JSON array string of parameter objects (overrides DEFAULT_PARAMS)")
    p.add_argument("--params-file", help="Path to JSON file with parameter array (overrides DEFAULT_PARAMS)")
    p.add_argument("--initial-balance", type=float, default=10000.0)
    p.add_argument("--max-leverage", type=float, default=10.0)
    p.add_argument("--max-loss-percent", type=float, default=2.0)
    p.add_argument("--trading-fee", type=float, default=0.0000)
    p.add_argument("--mock-delay", type=float, default=0.0, help="Seconds between ticks")
    p.add_argument("--out", default="ma_mock_results_kite", help="Output directory")
    p.add_argument("--no-parameter-validation", action="store_true", 
                   help="Disable parameter validation after backtest")
    p.add_argument("--validation-window-days", type=int, default=DEFAULT_VALIDATION_WINDOW_DAYS,
                   help=f"Days of recent data to use for parameter validation (default: {DEFAULT_VALIDATION_WINDOW_DAYS})")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Use global defaults if CLI args not provided
    symbol = args.symbol if args.symbol is not None else DEFAULT_SYMBOL
    exchange = args.exchange if args.exchange is not None else DEFAULT_EXCHANGE
    start_date = args.start if args.start is not None else DEFAULT_START_DATE
    end_date = args.end if args.end is not None else DEFAULT_END_DATE
    interval = args.interval if args.interval is not None else DEFAULT_INTERVAL
    
    # Load params - use global default if no CLI params provided
    params_list = _load_params_list(args.params, args.params_file, default_params=DEFAULT_PARAMS)
    
    run_ma_mock(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        params_list=params_list,
        exchange=exchange,
        initial_balance=args.initial_balance,
        max_leverage=args.max_leverage,
        max_loss_percent=args.max_loss_percent,
        trading_fee=args.trading_fee,
        mock_trading_delay=args.mock_delay,
        output_dir=args.out,
        enable_parameter_validation=not args.no_parameter_validation,
        validation_data_window_days=args.validation_window_days,
        days_to_validate=DAYS_TO_VALIDATE,
        validation_short_range=DEFAULT_VALIDATION_SHORT_RANGE,
        validation_long_range=DEFAULT_VALIDATION_LONG_RANGE,
        validation_rr_range=DEFAULT_VALIDATION_RR_RANGE,
    )


if __name__ == "__main__":
    main()
