#!/usr/bin/env python3
"""
Run mock tests for the Moving Average (MA) strategy over a specified date range
with user-specified parameter sets using cTrader data (Forex).
This script bypasses optimization and directly simulates provided parameter configurations.

QUICK START:
  Edit the global variables at the top of this script (DEFAULT_SYMBOL, 
  DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_INTERVAL, DEFAULT_PARAMS)
  and simply run: python run_ma_mock_validation_ctrader.py

Usage examples:

  - Using global defaults (edit DEFAULT_* variables in script):
    python run_ma_mock_validation_ctrader.py

  - Override symbol only:
    python run_ma_mock_validation_ctrader.py --symbol GBPUSD

  - Single param set via JSON string:
    python run_ma_mock_validation_ctrader.py \\
      --symbol EURUSD \\
      --start "2025-12-15" \\
      --end "2026-01-15" \\
      --interval 15m \\
      --params '[{"short_window":10,"long_window":40,"risk_reward_ratio":3.0}]'

  - Multiple param sets via JSON file:
    python run_ma_mock_validation_ctrader.py \\
      --symbol EURUSD \\
      --start "2025-12-01" \\
      --end "2026-01-01" \\
      --interval 15m \\
      --params-file params_ma.json

The params must be a list of objects with keys:
  short_window (int), long_window (int), risk_reward_ratio (float)
Optional: trading_fee will be taken from the validator config if omitted.
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# GLOBAL CONFIGURATION - Edit these values to set defaults
# ============================================================================
DEFAULT_SYMBOL = "USDJPY"  # Default forex symbol
DAYS_TO_VALIDATE = 30
DEFAULT_START_DATE = (datetime.now() - timedelta(days=DAYS_TO_VALIDATE)).strftime("%Y-%m-%d")
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")
# DEFAULT_START_DATE = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
# DEFAULT_END_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
DEFAULT_INTERVAL = "30m"  # cTrader intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d

DEFAULT_VALIDATION_WINDOW_DAYS = 32  # How often to run validation (every N days)

# Parameter ranges for validation optimization
VALIDATION_SHORT_VAL = 34
VALIDATION_LONG_VAL = 214
VALIDATION_RR_VAL = 9.0
VALIDATION_RANGE = 1
VALIDATION_SHORT_START = max(VALIDATION_SHORT_VAL - VALIDATION_RANGE, 4)
VALIDATION_SHORT_END = VALIDATION_SHORT_VAL + VALIDATION_RANGE
VALIDATION_LONG_START = max(VALIDATION_LONG_VAL - VALIDATION_RANGE, VALIDATION_SHORT_VAL + VALIDATION_RANGE + 1)
VALIDATION_LONG_END = VALIDATION_LONG_VAL + VALIDATION_RANGE
DEFAULT_VALIDATION_SHORT_RANGE = list(range(VALIDATION_SHORT_START, VALIDATION_SHORT_END + 1))
DEFAULT_VALIDATION_LONG_RANGE = list(range(VALIDATION_LONG_START, VALIDATION_LONG_END + 1))
DEFAULT_VALIDATION_RR_RANGE = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
# ============================================================================

DEFAULT_PARAMS = [
    {"short_window": VALIDATION_SHORT_VAL, "long_window": VALIDATION_LONG_VAL, "risk_reward_ratio": VALIDATION_RR_VAL}
]

# Import cTrader credentials
try:
    from config import (
        CTRADER_CLIENT_ID,
        CTRADER_CLIENT_SECRET,
        CTRADER_ACCESS_TOKEN,
        CTRADER_ACCOUNT_ID,
        CTRADER_DEMO
    )
    CTRADER_AVAILABLE = True
except ImportError:
    CTRADER_AVAILABLE = False
    print("⚠️  cTrader credentials not found in config.py")


def create_cumulative_pnl_chart(
    trade_history: pd.DataFrame,
    params: Dict,
    symbol: str,
    initial_balance: float,
    output_path: str
) -> str:
    """
    Create a cumulative PnL chart based on closed trades.
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
    
    if 'exit_time' not in closed_trades.columns:
        print("⚠️  No exit_time in trade history")
        return None
    
    closed_trades = closed_trades.sort_values('exit_time').reset_index(drop=True)
    
    # Calculate cumulative PnL
    cumulative_pnl = []
    cumulative_balance = []
    running_balance = initial_balance
    
    for idx, trade in closed_trades.iterrows():
        trade_pnl_pct = trade['pnl']
        leverage = trade.get('leverage', 1.0)
        position_size = trade.get('position_size', trade.get('quantity', 0) * trade.get('entry_price', 0))
        
        if position_size > 0 and leverage > 0:
            margin_used = position_size / leverage
            trade_profit = margin_used * trade_pnl_pct
        else:
            trade_profit = running_balance * trade_pnl_pct
        
        running_balance += trade_profit
        cumulative_balance.append(running_balance)
        cumulative_pnl.append((running_balance - initial_balance) / initial_balance * 100)
    
    # Create the chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35],
        subplot_titles=('📈 Cumulative PnL (%)', '📊 Per-Trade PnL (%)')
    )
    
    exit_times = closed_trades['exit_time'].tolist()
    trade_pnls = (closed_trades['pnl'] * 100).tolist()
    
    first_entry = closed_trades['entry_time'].iloc[0] if 'entry_time' in closed_trades.columns else exit_times[0]
    
    # Cumulative PnL line
    plot_times = [first_entry] + exit_times
    plot_pnls = [0] + cumulative_pnl
    
    colors_line = ['green' if p >= 0 else 'red' for p in plot_pnls]
    
    fig.add_trace(
        go.Scatter(
            x=plot_times,
            y=plot_pnls,
            mode='lines+markers',
            name='Cumulative PnL',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color=colors_line, line=dict(width=1, color='white')),
            hovertemplate='<b>Time:</b> %{x}<br><b>Cumulative PnL:</b> %{y:.2f}%<extra></extra>',
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.2)'
        ),
        row=1, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
    
    # Per-trade PnL bar chart
    bar_colors = ['green' if p >= 0 else 'red' for p in trade_pnls]
    
    fig.add_trace(
        go.Bar(
            x=exit_times,
            y=trade_pnls,
            name='Trade PnL',
            marker_color=bar_colors,
            hovertemplate='<b>Exit:</b> %{x}<br><b>PnL:</b> %{y:.2f}%<extra></extra>',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)
    
    # Summary stats
    final_pnl = cumulative_pnl[-1] if cumulative_pnl else 0
    total_trades = len(closed_trades)
    winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
    losing_trades = len(closed_trades[closed_trades['pnl'] < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # Max drawdown
    peak = 0
    max_dd = 0
    for pnl in cumulative_pnl:
        if pnl > peak:
            peak = pnl
        dd = peak - pnl
        if dd > max_dd:
            max_dd = dd
    
    params_str = f"Short={params.get('short_window')}, Long={params.get('long_window')}, RR={params.get('risk_reward_ratio')}"
    
    fig.update_layout(
        title=dict(
            text=f"📈 {symbol} MA Strategy - Cumulative PnL<br>"
                 f"<sup>{params_str} | Final PnL: {final_pnl:+.2f}% | "
                 f"Trades: {total_trades} (W:{winning_trades}/L:{losing_trades}) | "
                 f"Win Rate: {win_rate:.1f}% | Max DD: {max_dd:.2f}%</sup>",
            font=dict(size=16)
        ),
        xaxis2_title="Time",
        yaxis_title="Cumulative PnL (%)",
        yaxis2_title="Trade PnL (%)",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    fig.write_html(output_path)
    print(f"📊 Cumulative PnL chart saved: {output_path}")
    
    return output_path


def create_combined_pnl_chart(
    all_results: List[Dict],
    symbol: str,
    initial_balance: float,
    output_path: str
) -> str:
    """Create a combined cumulative PnL chart comparing all parameter sets."""
    if not all_results:
        print("⚠️  No results to plot")
        return None
    
    fig = go.Figure()
    colors = ['#2E86AB', '#E74C3C', '#27AE60', '#9B59B6', '#F39C12', '#1ABC9C', '#E91E63', '#3F51B5']
    
    for idx, result in enumerate(all_results):
        trade_history = result.get('trade_history')
        if trade_history is None or trade_history.empty:
            continue
        
        closed_statuses = ['closed', 'tp_hit', 'sl_hit', 'reversed']
        closed_trades = trade_history[trade_history['status'].isin(closed_statuses)].copy()
        
        if closed_trades.empty or 'exit_time' not in closed_trades.columns:
            continue
        
        closed_trades = closed_trades.sort_values('exit_time').reset_index(drop=True)
        
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
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=dict(text=f"📊 {symbol} - Parameter Comparison (Cumulative PnL %)", font=dict(size=16)),
        xaxis_title="Time",
        yaxis_title="Cumulative PnL (%)",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
    if default_params is not None:
        return default_params
    raise ValueError("You must provide parameter sets via --params, --params-file, or set DEFAULT_PARAMS in the script")


def fetch_ctrader_data(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Fetch data from cTrader"""
    if not CTRADER_AVAILABLE:
        raise ValueError("cTrader credentials not available in config.py")
    
    from data_fetcher import CTraderDataFetcher
    
    fetcher = CTraderDataFetcher(
        client_id=CTRADER_CLIENT_ID,
        client_secret=CTRADER_CLIENT_SECRET,
        access_token=CTRADER_ACCESS_TOKEN,
        account_id=CTRADER_ACCOUNT_ID,
        demo=CTRADER_DEMO
    )
    
    # Parse dates and format with time
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    except ValueError:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    
    try:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        # Set end time to end of day to get all data
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
    except ValueError:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    
    data = fetcher.fetch_historical_data(
        symbol=symbol,
        start_date=start_dt.strftime('%Y-%m-%d %H:%M:%S'),
        end_date=end_dt.strftime('%Y-%m-%d %H:%M:%S'),
        interval=interval
    )
    
    return data


def run_ma_mock(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    params_list: List[Dict[str, Any]],
    initial_balance: float = 10000.0,
    max_leverage: float = 10.0,
    max_loss_percent: float = 2.0,
    trading_fee: float = 0.0,
    mock_trading_delay: float = 0.0,
    output_dir: str = "ma_mock_results_ctrader",
    enable_parameter_validation: bool = True,
    validation_data_window_days: int = 7,
    days_to_validate: int = 30,
    validation_short_range: List[int] = None,
    validation_long_range: List[int] = None,
    validation_rr_range: List[float] = None
) -> Dict[str, Any]:
    # Use defaults if not provided
    if validation_short_range is None:
        validation_short_range = DEFAULT_VALIDATION_SHORT_RANGE
    if validation_long_range is None:
        validation_long_range = DEFAULT_VALIDATION_LONG_RANGE
    if validation_rr_range is None:
        validation_rr_range = DEFAULT_VALIDATION_RR_RANGE
    
    print("🚀 MA STRATEGY MOCK VALIDATION (CTRADER)")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Date Range: {start_date} → {end_date}")
    print(f"Interval: {interval}")
    print(f"Param sets: {len(params_list)}")
    print(f"Mode: {'Demo' if CTRADER_DEMO else 'Live'}")
    
    # Store original dates for data leak prevention
    original_start_date = start_date
    original_end_date = end_date
    
    # If validation is enabled, fetch extra past data
    if enable_parameter_validation:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') if len(start_date) == 10 else datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if len(end_date) == 10 else datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        backtest_days = (end_dt - start_dt).days
        
        actual_start_date = (end_dt - timedelta(days=backtest_days + days_to_validate)).strftime('%Y-%m-%d')
        print(f"📊 Validation enabled: Fetching extra {days_to_validate} days of past data")
        print(f"   Fetching data: {actual_start_date} → {end_date}")
        print(f"   ⚠️  Backtest will ONLY use: {original_start_date} → {original_end_date} (no data leak)")
        start_date = actual_start_date
    
    # Auto-adjust max_loss_percent for 1x leverage
    adjusted_max_loss_percent = max_loss_percent
    if max_leverage == 1.0:
        adjusted_max_loss_percent = 100.0
        print(f"⚙️  Leverage is 1x: Auto-setting max_loss_percent to 100%")
    
    print(f"📊 Max Leverage: {max_leverage}x")
    print(f"📊 Max Loss Percent: {adjusted_max_loss_percent}%")
    
    # Fetch data from cTrader
    print("\n📥 Fetching data from cTrader...")
    data = fetch_ctrader_data(symbol, start_date, end_date, interval)
    
    if data.empty:
        raise ValueError(f"No data fetched for {symbol}")
    
    print(f"✅ Successfully fetched {len(data)} data points")
    print(f"📅 Data range: {data.index[0]} to {data.index[-1]}")
    
    # Filter data for backtest only (prevent data leak)
    if enable_parameter_validation:
        original_start_dt = datetime.strptime(original_start_date, '%Y-%m-%d') if len(original_start_date) == 10 else datetime.strptime(original_start_date, '%Y-%m-%d %H:%M:%S')
        original_end_dt = datetime.strptime(original_end_date, '%Y-%m-%d') if len(original_end_date) == 10 else datetime.strptime(original_end_date, '%Y-%m-%d %H:%M:%S')
        
        def get_date_from_index(idx):
            ts = pd.Timestamp(idx)
            if ts.tz is not None:
                ts = ts.tz_localize(None)
            return ts.date()
        
        backtest_data = data[
            data.index.map(lambda x: original_start_dt.date() <= get_date_from_index(x) <= original_end_dt.date())
        ].copy()
        
        validation_data = data.copy()
        
        print(f"📊 Data split: {len(backtest_data)} backtest points, {len(data) - len(backtest_data)} extra for validation")
    else:
        backtest_data = data.copy()
        validation_data = data.copy()
    
    # Import strategy components
    from strategy_manager import StrategyManager
    from trading_engine import TradingEngine
    
    results: List[Dict[str, Any]] = []
    failed_results: List[Dict[str, Any]] = []
    
    for idx, params in enumerate(params_list, start=1):
        print("-" * 80)
        print(f"🎯 Testing parameter set {idx}/{len(params_list)}: {params}")
        
        # Ensure trading_fee is present
        params_with_fee = dict(params)
        if "trading_fee" not in params_with_fee:
            params_with_fee["trading_fee"] = trading_fee
        
        # Initialize strategy manager
        strategy_manager = StrategyManager()
        
        ok = strategy_manager.set_manual_parameters(ma_params=params_with_fee)
        if not ok:
            print(f"❌ Failed to set manual MA parameters for set {idx}")
            failed_results.append({
                "parameter_set": idx,
                "parameters": params,
                "error": "Failed to set manual MA parameters",
                "stage": "parameter_setup"
            })
            continue
        
        strategies = strategy_manager.initialize_strategies(["ma"])
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
            initial_balance,
            max_leverage,
            adjusted_max_loss_percent,
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{symbol}_ctrader_ma_param_{idx}_mock_{timestamp}"
        engine.setup_logging(session_id, symbol)
        
        print("🚦 Running simulation...")
        start_t = datetime.now()
        
        # Parameter validation setup
        last_validation_date = None
        validation_results_history = []
        
        for j in range(len(backtest_data)):
            current_data = backtest_data.iloc[: j + 1]
            current_time = backtest_data.index[j]
            engine.process_strategy_signals(strategy, current_data, current_time)
            
            # Parameter validation check
            if enable_parameter_validation:
                current_ts = pd.Timestamp(current_time)
                if current_ts.tz is not None:
                    current_ts = current_ts.tz_localize(None)
                current_date = current_ts.date()
                
                should_validate = False
                if last_validation_date is None:
                    first_ts = pd.Timestamp(backtest_data.index[0])
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
                    print(f"\n{'='*60}")
                    print(f"🔍 PARAMETER VALIDATION CHECK (Day {current_date})")
                    print(f"{'='*60}")
                    
                    try:
                        validation_start_date = current_date - timedelta(days=days_to_validate)
                        validation_end_date = current_date
                        
                        def get_date_from_idx(idx):
                            ts = pd.Timestamp(idx)
                            if ts.tz is not None:
                                ts = ts.tz_localize(None)
                            return ts.date()
                        
                        recent_data = validation_data[
                            validation_data.index.map(lambda x: validation_start_date <= get_date_from_idx(x) <= validation_end_date)
                        ].copy()
                        
                        if len(recent_data) < 50:
                            print(f"⚠️  Insufficient data for validation ({len(recent_data)} points), skipping...")
                        else:
                            print(f"📊 Validating on {len(recent_data)} data points")
                            
                            # Run simple optimization on recent data
                            from ma_3d_optimization_visualizer import MAOptimization3DVisualizer
                            import tempfile
                            
                            # Use temp directory for validation (we don't need to save plots)
                            with tempfile.TemporaryDirectory() as temp_dir:
                                visualizer = MAOptimization3DVisualizer(
                                    recent_data,
                                    trading_fee=trading_fee,
                                    auto_open=False,
                                    output_dir=temp_dir
                                )
                                
                                opt_results = visualizer.run_optimization_grid(
                                    validation_short_range,
                                    validation_long_range,
                                    validation_rr_range
                                )
                            
                            # Find best parameters
                            if opt_results:
                                best_result = max(opt_results, key=lambda x: x.get('composite_score', 0))
                                new_optimal_params = {
                                    'short_window': best_result['short_window'],
                                    'long_window': best_result['long_window'],
                                    'risk_reward_ratio': best_result['risk_reward_ratio']
                                }
                                new_optimal_metrics = best_result
                                
                                # Calculate distance
                                param_distance = abs(params_with_fee['short_window'] - new_optimal_params['short_window']) + \
                                               abs(params_with_fee['long_window'] - new_optimal_params['long_window']) + \
                                               abs(params_with_fee['risk_reward_ratio'] - new_optimal_params['risk_reward_ratio']) * 2
                                
                                # Determine alert level
                                alert_level = 'none'
                                if param_distance >= 15:
                                    alert_level = 'critical'
                                elif param_distance >= 10:
                                    alert_level = 'warning'
                                elif param_distance >= 5:
                                    alert_level = 'monitor'
                                
                                status_emoji = {'none': '✅', 'monitor': '📊', 'warning': '⚠️', 'critical': '🚨'}
                                emoji = status_emoji.get(alert_level, '❓')
                                
                                print(f"{emoji} Validation Results:")
                                print(f"   Parameter Distance: {param_distance:.2f}")
                                print(f"   Alert Level: {alert_level.upper()}")
                                print(f"   Current: S={params_with_fee['short_window']}, L={params_with_fee['long_window']}, RR={params_with_fee['risk_reward_ratio']}")
                                print(f"   Optimal: S={new_optimal_params['short_window']}, L={new_optimal_params['long_window']}, RR={new_optimal_params['risk_reward_ratio']}")
                                
                                validation_results_history.append({
                                    'date': current_date.isoformat(),
                                    'parameter_distance': param_distance,
                                    'alert_level': alert_level,
                                    'current_params': params_with_fee.copy(),
                                    'new_optimal_params': new_optimal_params.copy()
                                })
                        
                        print(f"{'='*60}\n")
                        
                    except Exception as e:
                        print(f"⚠️  Validation failed: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        # Always update last_validation_date to prevent repeated attempts
                        last_validation_date = current_date
            
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
        
        if enable_parameter_validation and validation_results_history:
            result['parameter_validation_history'] = validation_results_history
        
        results.append(result)
        
        print(f"✅ Done. Final Balance: ${final_status['current_balance']:,.2f}")
        if "total_pnl" in performance_metrics:
            print(f"📈 Total PnL: {performance_metrics['total_pnl']:.2%}")
        if "win_rate" in performance_metrics:
            print(f"🎯 Win Rate: {performance_metrics['win_rate']:.2%}")
        if "max_drawdown" in performance_metrics:
            print(f"📉 Max DD: {performance_metrics['max_drawdown']:.2%}")
    
    # Save outputs
    if results:
        os.makedirs(output_dir, exist_ok=True)
        out_json = os.path.join(
            output_dir, f"ma_mock_results_{symbol}_ctrader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
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
            if "parameter_validation_history" in r:
                result_dict["parameter_validation_history"] = r["parameter_validation_history"]
            serializable.append(result_dict)
        
        with open(out_json, "w") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "interval": interval,
                    "results": serializable,
                    "failed_results": failed_results,
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
        
        # Create combined chart if multiple param sets
        if len(results) > 1:
            combined_chart_path = os.path.join(
                output_dir,
                f"ma_mock_combined_pnl_{symbol}_ctrader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
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
        
        # Open chart in browser
        if chart_paths:
            print(f"\n🌐 Opening PnL chart in browser...")
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(chart_paths[0])}")
            except Exception as e:
                print(f"⚠️  Could not open browser: {e}")
    
    # Report failed parameter sets
    if failed_results:
        print(f"\n⚠️  {len(failed_results)} parameter set(s) failed:")
        for failed in failed_results:
            print(f"   - Set {failed['parameter_set']}: {failed['error']}")
    
    # Parameter validation summary
    if enable_parameter_validation and results:
        print("\n" + "=" * 80)
        print("📋 PARAMETER VALIDATION SUMMARY")
        print("=" * 80)
        
        for r in results:
            if r.get('parameter_validation_history'):
                params = r['parameters']
                print(f"\n📊 Param Set {r['parameter_set']} (S={params.get('short_window')}, L={params.get('long_window')}, RR={params.get('risk_reward_ratio')}):")
                
                for val_check in r['parameter_validation_history']:
                    alert_level = val_check.get('alert_level', 'none')
                    status_emoji = {'none': '✅', 'monitor': '📊', 'warning': '⚠️', 'critical': '🚨'}
                    emoji = status_emoji.get(alert_level, '❓')
                    print(f"   {emoji} {val_check['date']}: Distance={val_check['parameter_distance']:.2f}, Level={alert_level.upper()}")
    
    print("\n🎯 Completed MA mock validation for cTrader.")
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "results": results,
        "failed_results": failed_results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MA strategy mock tests with cTrader data")
    p.add_argument("--symbol", default=None, help=f"Symbol, e.g., EURUSD, GBPUSD (default: {DEFAULT_SYMBOL})")
    p.add_argument("--start", default=None, help=f"Start date 'YYYY-MM-DD' (default: {DEFAULT_START_DATE})")
    p.add_argument("--end", default=None, help=f"End date 'YYYY-MM-DD' (default: {DEFAULT_END_DATE})")
    p.add_argument("--interval", default=None, help=f"Interval: 1m, 5m, 15m, 30m, 1h, 4h, 1d (default: {DEFAULT_INTERVAL})")
    p.add_argument("--params", help="JSON array string of parameter objects")
    p.add_argument("--params-file", help="Path to JSON file with parameter array")
    p.add_argument("--initial-balance", type=float, default=10000.0)
    p.add_argument("--max-leverage", type=float, default=10.0)
    p.add_argument("--max-loss-percent", type=float, default=0.2)
    p.add_argument("--trading-fee", type=float, default=0.0)
    p.add_argument("--mock-delay", type=float, default=0.0, help="Seconds between ticks")
    p.add_argument("--out", default="ma_mock_results_ctrader", help="Output directory")
    p.add_argument("--no-parameter-validation", action="store_true", help="Disable parameter validation")
    p.add_argument("--validation-window-days", type=int, default=DEFAULT_VALIDATION_WINDOW_DAYS)
    return p.parse_args()


def main():
    args = parse_args()
    
    symbol = args.symbol if args.symbol is not None else DEFAULT_SYMBOL
    start_date = args.start if args.start is not None else DEFAULT_START_DATE
    end_date = args.end if args.end is not None else DEFAULT_END_DATE
    interval = args.interval if args.interval is not None else DEFAULT_INTERVAL
    
    params_list = _load_params_list(args.params, args.params_file, default_params=DEFAULT_PARAMS)
    
    run_ma_mock(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        params_list=params_list,
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

