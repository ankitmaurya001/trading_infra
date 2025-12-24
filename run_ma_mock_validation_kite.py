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
import config as cfg


# ============================================================================
# GLOBAL CONFIGURATION - Edit these values to set defaults
# ============================================================================
# You can override these via command-line arguments if needed
DEFAULT_SYMBOL = "SILVERMIC26FEBFUT"  # Default Indian stock symbol
DEFAULT_EXCHANGE = "MCX"  # Default exchange (NSE, BSE, MCX)
# use 30 days ago (Kite uses YYYY-MM-DD format)
DEFAULT_START_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
# use today's date
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_INTERVAL = "15m"  # Will be converted to "15minute" for Kite
DEFAULT_PARAMS = [
    {"short_window": 4, "long_window": 58, "risk_reward_ratio": 6.0}
]
# ============================================================================


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
    cumulative_balance = []
    running_balance = initial_balance
    
    for idx, trade in closed_trades.iterrows():
        # Get the PnL percentage for this trade
        trade_pnl_pct = trade['pnl']  # This is already a decimal (e.g., 0.05 for 5%)
        
        # Calculate dollar profit/loss based on position size at entry
        # The margin used determines how much of balance was at risk
        leverage = trade.get('leverage', 1.0)
        position_size = trade.get('position_size', trade.get('quantity', 0) * trade.get('entry_price', 0))
        
        if position_size > 0 and leverage > 0:
            margin_used = position_size / leverage
            trade_profit = margin_used * trade_pnl_pct
        else:
            # Fallback: use running balance with pnl percentage
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
    trade_pnls = (closed_trades['pnl'] * 100).tolist()  # Convert to percentage
    
    # Add starting point (0% at first trade entry or start)
    first_entry = closed_trades['entry_time'].iloc[0] if 'entry_time' in closed_trades.columns else exit_times[0]
    
    # 1. Cumulative PnL line
    plot_times = [first_entry] + exit_times
    plot_pnls = [0] + cumulative_pnl
    
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
            hovertemplate='<b>Time:</b> %{x}<br><b>Cumulative PnL:</b> %{y:.2f}%<extra></extra>',
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
            hovertemplate='<b>Exit:</b> %{x}<br><b>PnL:</b> %{y:.2f}%<extra></extra>',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add zero line for bar chart
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)
    
    # Calculate summary stats
    final_pnl = cumulative_pnl[-1] if cumulative_pnl else 0
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
    output_dir: str = "ma_mock_results_kite"
) -> Dict[str, Any]:
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
        
        for j in range(len(validator.test_data)):
            current_data = validator.test_data.iloc[: j + 1]
            current_time = validator.test_data.index[j]
            engine.process_strategy_signals(strategy, current_data, current_time)
            
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
        results.append(result)

        print(f"✅ Done. Final Balance: ${final_status['current_balance']:,.2f}")
        print(f"📋 Performance Metrics: {performance_metrics}")
        if "total_pnl" in performance_metrics:
            print(f"📈 Total PnL: {performance_metrics['total_pnl']:.2%}")
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
            serializable.append(
                {
                    "parameter_set": r["parameter_set"],
                    "parameters": r["parameters"],
                    "final_status": r["final_status"],
                    "performance_metrics": r["performance_metrics"],
                    "simulation_duration_sec": r["simulation_duration_sec"],
                    "session_id": r["session_id"],
                    "trade_history_csv": csv_path,
                }
            )

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

    print("\n🎯 Completed MA mock validation.")
    return {
        "symbol": symbol,
        "exchange": exchange,
        "start_date": start_date_kite,
        "end_date": end_date_kite,
        "interval": kite_interval,
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
    p.add_argument("--trading-fee", type=float, default=0.0003)
    p.add_argument("--mock-delay", type=float, default=0.0, help="Seconds between ticks")
    p.add_argument("--out", default="ma_mock_results_kite", help="Output directory")
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
    )


if __name__ == "__main__":
    main()

