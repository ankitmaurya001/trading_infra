#!/usr/bin/env python3
"""
Streamlit Components for Trading Applications
Modular components for displaying trading data, charts, and visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
from typing import Dict, List, Optional
import glob

def get_trade_log_path(session_id: str) -> Optional[str]:
    """
    Get the trade log file path using the session ID.
    
    Args:
        session_id: Session ID (e.g., 'BTC-USD_20250831_163405_mock')
        
    Returns:
        Path to the trade log file, or None if not found
    """
    if not session_id:
        return None
    
    trade_log_path = f"logs/live_trades_{session_id}.csv"
    
    if os.path.exists(trade_log_path):
        print(f"📁 Using trade log: {trade_log_path}")
        return trade_log_path
    else:
        print(f"⚠️ Trade log not found: {trade_log_path}")
        return None

def parse_trades_from_csv(log_file_path: str) -> pd.DataFrame:
    """
    Parse all trades (completed and open) from CSV file.
    
    Args:
        log_file_path: Path to the trade log CSV file
        
    Returns:
        DataFrame with all trades (completed and open)
    """
    if not os.path.exists(log_file_path):
        print(f"⚠️ Trade log file not found: {log_file_path}")
        return pd.DataFrame()
    
    try:
        # Read the CSV file
        df = pd.read_csv(log_file_path)
        print(f"📋 Loaded trade log: {len(df)} records from {log_file_path}")
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by trade_id to process all trades
        all_trades = []
        
        for trade_id in df['trade_id'].unique():
            trade_records = df[df['trade_id'] == trade_id]
            
            # Find entry record (BUY/SELL with open status)
            entry_record = trade_records[
                (trade_records['action'].isin(['BUY', 'SELL'])) & 
                (trade_records['status'] == 'open')
            ]
            
            # Find exit record (EXIT with closed status)
            exit_record = trade_records[
                (trade_records['action'] == 'EXIT') & 
                (trade_records['status'].isin(['tp_hit', 'sl_hit', 'closed']))
            ]
            
            if not entry_record.empty:
                entry = entry_record.iloc[0]
                
                if not exit_record.empty:
                    # Completed trade
                    exit = exit_record.iloc[0]
                    trade = {
                        'trade_id': int(trade_id),
                        'entry_time': entry['timestamp'],
                        'entry_price': float(entry['price']),
                        'exit_time': exit['timestamp'],
                        'exit_price': float(exit['price']),
                        'pnl': float(exit['pnl']),
                        'strategy': entry['strategy'],
                        'action': entry['action'],
                        'exit_reason': exit['status'],
                        'status': 'completed'
                    }
                    all_trades.append(trade)
                    print(f"📊 Found completed trade {trade_id}: {entry['action']} @ ${entry['price']:.2f} → EXIT @ ${exit['price']:.2f} (PnL: {exit['pnl']:.2%})")
                else:
                    # Open trade (no exit record yet)
                    trade = {
                        'trade_id': int(trade_id),
                        'entry_time': entry['timestamp'],
                        'entry_price': float(entry['price']),
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': None,
                        'strategy': entry['strategy'],
                        'action': entry['action'],
                        'exit_reason': None,
                        'status': 'open'
                    }
                    all_trades.append(trade)
                    print(f"📊 Found open trade {trade_id}: {entry['action']} @ ${entry['price']:.2f} (Status: Open)")
        
        if all_trades:
            result_df = pd.DataFrame(all_trades)
            completed_count = len(result_df[result_df['status'] == 'completed'])
            open_count = len(result_df[result_df['status'] == 'open'])
            print(f"✅ Found {completed_count} completed trades and {open_count} open trades")
            return result_df
        else:
            print("⚠️ No trades found in log file")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error parsing trade log: {str(e)}")
        return pd.DataFrame()

def display_trading_status(status: Dict, simulator=None):
    """
    Display current trading status with metrics.
    
    Args:
        status: Current trading status dictionary
        simulator: Simulator instance for additional info
    """
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        status_text = "🟢 Running" if status['is_running'] else "🔴 Stopped"
        if status['is_running'] and simulator and simulator.mock_mode:
            status_text += " (Mock)"
        st.metric("Status", status_text)
    
    with status_col2:
        st.metric(
            "Balance",
            f"${status['current_balance']:,.2f}",
            f"{status['total_pnl']:+.2f}"
        )
        st.caption(f"Total Value: ${status['total_value']:,.2f}")
        if status.get('unrealized_pnl', 0) != 0:
            st.caption(f"Unrealized: {status['unrealized_pnl']:+.2f}")
        st.caption("💡 Full balance used per trade for maximum compounding")
    
    with status_col3:
        st.metric("Active Trades", status['active_trades'])
        if status.get('active_trade_info'):
            trade_info = status['active_trade_info']
            st.caption(f"📈 {trade_info['strategy']} - {trade_info['action']}")
            st.caption(f"Entry: ${trade_info['entry_price']:.2f}")
            if 'quantity' in trade_info:
                st.caption(f"Qty: {trade_info['quantity']:.4f}")
            if 'entry_time' in trade_info:
                st.caption(f"Time: {trade_info['entry_time'].strftime('%H:%M:%S')}")
        elif status['active_trades'] == 0 and status.get('can_trade', True):
            st.caption("✅ Ready to trade")
        elif status['active_trades'] == 0 and not status.get('can_trade', True):
            st.caption("⚠️ No balance available")
    
    with status_col4:
        st.metric("Total Trades", status['total_trades'])

def display_strategy_parameters(params: Dict):
    """
    Display current strategy parameters.
    
    Args:
        params: Dictionary containing strategy parameters
    """
    if not params:
        st.info("No strategy parameters set. Please setup strategies first.")
        return
    
    st.subheader("📋 Current Strategy Parameters")
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        if 'ma' in params:
            st.markdown("**Moving Average Crossover**")
            p = params['ma']
            st.write(f"Short MA: {p['short_window']}")
            st.write(f"Long MA: {p['long_window']}")
            st.write(f"Risk/Reward: {p['risk_reward_ratio']}")
            st.write(f"Trading Fee: {p['trading_fee']:.3f}")
    
    with param_col2:
        if 'rsi' in params:
            st.markdown("**RSI Strategy**")
            p = params['rsi']
            st.write(f"Period: {p['period']}")
            st.write(f"Overbought: {p['overbought']}")
            st.write(f"Oversold: {p['oversold']}")
            st.write(f"Risk/Reward: {p['risk_reward_ratio']}")
            st.write(f"Trading Fee: {p['trading_fee']:.3f}")
    
    with param_col3:
        if 'donchian' in params:
            st.markdown("**Donchian Channel**")
            p = params['donchian']
            st.write(f"Channel Period: {p['channel_period']}")
            st.write(f"Risk/Reward: {p['risk_reward_ratio']}")
            st.write(f"Trading Fee: {p['trading_fee']:.3f}")

def display_trade_history(trade_history_df: pd.DataFrame, symbol: str):
    """
    Display trade history with formatting and download options.
    
    Args:
        trade_history_df: DataFrame containing trade history
        symbol: Trading symbol for file naming
    """
    st.subheader("📋 Trade History")
    
    if not trade_history_df.empty:
        # Format the display
        display_df = trade_history_df.copy()
        display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        if 'exit_time' in display_df.columns:
            display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['pnl_pct'] = (display_df['pnl'] * 100).round(2)
        
        # Color code PnL
        def color_pnl(val):
            if val > 0:
                return 'color: green; font-weight: bold'
            elif val < 0:
                return 'color: red; font-weight: bold'
            return 'color: black'
        
        styled_df = display_df.style.map(color_pnl, subset=['pnl_pct'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download trade history
        csv = trade_history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History",
            data=csv,
            file_name=f"trade_history_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trades executed yet. Start live trading to see trade history.")

def display_performance_metrics(metrics: Dict):
    """
    Display comprehensive performance metrics.
    
    Args:
        metrics: Dictionary containing performance metrics
    """
    if not metrics or metrics.get('total_trades', 0) == 0:
        st.info("No performance metrics available yet. Metrics will appear here once trades are executed.")
        return
    
    st.subheader("📊 Performance Metrics")
    
    # Main performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", metrics["total_trades"])
        st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
    with col2:
        st.metric("Total PnL", f"{metrics['total_pnl']:.2%}")
        st.metric("Avg Return per Trade", f"{metrics['avg_return']:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
        st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.3f}")
    with col4:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    
    # Risk-adjusted metrics
    st.subheader("Risk-Adjusted Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
        st.metric("Geometric Mean Return", f"{metrics['geometric_mean_return']:.2%}")
    with col2:
        st.metric("Avg Win", f"{metrics['avg_win']:.2%}")
        st.metric("Avg Loss", f"{metrics['avg_loss']:.2%}")
    with col3:
        st.metric("Win/Loss Ratio", f"{metrics['risk_reward_ratio']:.2f}")
        st.metric("Largest Win", f"{metrics['largest_win']:.2%}")
    with col4:
        st.metric("Largest Loss", f"{metrics['largest_loss']:.2%}")
        st.metric("Consecutive Wins", metrics['consecutive_wins'])
    
    # Performance summary with color coding
    st.subheader("Performance Summary")
    
    # Color-code performance based on metrics
    sharpe_color = "green" if metrics['sharpe_ratio'] > 1.0 else "orange" if metrics['sharpe_ratio'] > 0.5 else "red"
    calmar_color = "green" if metrics['calmar_ratio'] > 1.0 else "orange" if metrics['calmar_ratio'] > 0.5 else "red"
    profit_factor_color = "green" if metrics['profit_factor'] > 1.5 else "orange" if metrics['profit_factor'] > 1.0 else "red"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; border-radius: 5px; background-color: {sharpe_color}20;'>
            <h4>Risk-Adjusted Returns</h4>
            <p style='font-size: 24px; color: {sharpe_color};'>{metrics['sharpe_ratio']:.2f}</p>
            <p>Sharpe Ratio</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; border-radius: 5px; background-color: {calmar_color}20;'>
            <h4>Return vs Drawdown</h4>
            <p style='font-size: 24px; color: {calmar_color};'>{metrics['calmar_ratio']:.2f}</p>
            <p>Calmar Ratio</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; border-radius: 5px; background-color: {profit_factor_color}20;'>
            <h4>Profit Efficiency</h4>
            <p style='font-size: 24px; color: {profit_factor_color};'>{metrics['profit_factor']:.2f}</p>
            <p>Profit Factor</p>
        </div>
        """, unsafe_allow_html=True)

def create_performance_chart(trade_history_df: pd.DataFrame, initial_balance: float):
    """
    Create cumulative performance chart.
    
    Args:
        trade_history_df: DataFrame containing trade history
        initial_balance: Initial trading balance
        
    Returns:
        Plotly figure object
    """
    if trade_history_df.empty:
        return None
    
    # Filter for closed trades and check if exit_time column exists
    closed_trades = trade_history_df[trade_history_df['status'].isin(['closed', 'tp_hit', 'sl_hit', 'reversed'])]
    
    if closed_trades.empty:
        return None
    
    # Check if exit_time column exists
    if 'exit_time' not in closed_trades.columns:
        st.warning("Trade history missing exit_time column. Cannot create performance chart.")
        return None
    
    # Sort trades by exit time to ensure chronological order
    sorted_trades = closed_trades.sort_values('exit_time')
    
    if sorted_trades.empty:
        return None
    
    # Calculate cumulative PnL with proper compounding
    cumulative_pnl = []
    balance_history = []
    dates = []
    
    current_balance = initial_balance
    for _, trade in sorted_trades.iterrows():
        # Calculate the actual dollar profit/loss for this trade
        position_value = trade['quantity'] * trade['entry_price']
        trade_profit_loss = position_value * trade['pnl']  # Convert percentage to dollars
        
        # Update balance with this trade's profit/loss
        current_balance += trade_profit_loss
        
        # Calculate cumulative return from initial balance
        cumulative_return = (current_balance - initial_balance) / initial_balance
        
        cumulative_pnl.append(cumulative_return)
        balance_history.append(current_balance)
        dates.append(trade['exit_time'])
    
    if not dates:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_pnl,
        mode='lines+markers',
        name='Cumulative PnL',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>PnL:</b> %{y:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Cumulative Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL",
        yaxis_tickformat='.1%',
        height=400
    )
    
    return fig

def display_decision_log_viewer(decision_log_df: pd.DataFrame):
    """
    Display decision log with filtering and statistics.
    
    Args:
        decision_log_df: DataFrame containing decision log
    """
    if decision_log_df.empty:
        st.info("No decision log data available yet. Decisions will appear here as they are made.")
        return
    
    # Decision log controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Strategy filter
        strategies = decision_log_df['strategy'].unique()
        selected_strategy = st.selectbox(
            "Filter by Strategy",
            options=["All"] + list(strategies),
            index=0
        )
    
    with col2:
        # Signal filter
        signals = decision_log_df['signal_name'].unique()
        selected_signal = st.selectbox(
            "Filter by Signal",
            options=["All"] + list(signals),
            index=0
        )
    
    with col3:
        # Date range filter
        if len(decision_log_df) > 1:
            min_date = decision_log_df['timestamp'].min()
            max_date = decision_log_df['timestamp'].max()
            date_range = st.date_input(
                "Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
        else:
            date_range = None
    
    # Apply filters
    filtered_df = decision_log_df.copy()
    
    if selected_strategy != "All":
        filtered_df = filtered_df[filtered_df['strategy'] == selected_strategy]
    
    if selected_signal != "All":
        filtered_df = filtered_df[filtered_df['signal_name'] == selected_signal]
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) &
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    
    # Display decision log
    if not filtered_df.empty:
        # Format the display
        display_df = filtered_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['current_price'] = display_df['current_price'].round(2)
        display_df['current_balance'] = display_df['current_balance'].round(2)
        
        # Color code signals
        def color_signal(val):
            if val == 'LONG_ENTRY':
                return 'background-color: lightgreen'
            elif val == 'SHORT_ENTRY':
                return 'background-color: lightcoral'
            elif val == 'LONG_EXIT':
                return 'background-color: lightblue'
            elif val == 'SHORT_EXIT':
                return 'background-color: lightyellow'
            elif val == 'HOLD':
                return 'background-color: lightgray'
            else:
                return ''
        
        styled_df = display_df.style.applymap(color_signal, subset=['signal_name'])
        
        # Show decision log
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Decision statistics
        st.subheader("📊 Decision Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_decisions = len(filtered_df)
            st.metric("Total Decisions", total_decisions)
        
        with col2:
            if not filtered_df.empty:
                signal_counts = filtered_df['signal_name'].value_counts()
                most_common = signal_counts.index[0] if len(signal_counts) > 0 else "None"
                st.metric("Most Common Signal", most_common)
        
        with col3:
            if not filtered_df.empty:
                avg_price = filtered_df['current_price'].mean()
                st.metric("Avg Price", f"${avg_price:.2f}")
        
        with col4:
            if not filtered_df.empty:
                price_change = filtered_df['current_price'].iloc[-1] - filtered_df['current_price'].iloc[0]
                st.metric("Price Change", f"${price_change:.2f}")
        
        # Signal distribution chart
        if not filtered_df.empty:
            st.subheader("📈 Signal Distribution")
            
            signal_counts = filtered_df['signal_name'].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=signal_counts.index,
                    y=signal_counts.values,
                    marker_color=['lightgreen', 'lightcoral', 'lightblue', 'lightyellow', 'lightgray'][:len(signal_counts)]
                )
            ])
            
            fig.update_layout(
                title="Signal Distribution",
                xaxis_title="Signal Type",
                yaxis_title="Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key="signal_distribution_chart")
        
        # Strategy performance comparison
        if len(strategies) > 1:
            st.subheader("🎯 Strategy Comparison")
            
            strategy_stats = []
            for strategy in strategies:
                strategy_df = filtered_df[filtered_df['strategy'] == strategy]
                if not strategy_df.empty:
                    stats = {
                        'Strategy': strategy,
                        'Total Decisions': len(strategy_df),
                        'Avg Price': strategy_df['current_price'].mean(),
                        'Price Change': strategy_df['current_price'].iloc[-1] - strategy_df['current_price'].iloc[0],
                        'Most Common Signal': strategy_df['signal_name'].mode().iloc[0] if len(strategy_df['signal_name'].mode()) > 0 else "None"
                    }
                    strategy_stats.append(stats)
            
            if strategy_stats:
                strategy_df = pd.DataFrame(strategy_stats)
                st.dataframe(strategy_df, use_container_width=True, hide_index=True)
    else:
        st.info("No decisions match the selected filters.")

def display_decision_log_summary(decision_log_df: pd.DataFrame):
    """
    Display decision log summary with recent decisions.
    
    Args:
        decision_log_df: DataFrame containing decision log
    """
    if decision_log_df.empty:
        return
    
    st.subheader("📋 Decision Log Summary")
    
    # Show recent decisions
    recent_decisions = decision_log_df.tail(10)
    if not recent_decisions.empty:
        st.caption("Recent Decisions:")
        for _, row in recent_decisions.iterrows():
            signal_color = {
                'LONG_ENTRY': '🟢',
                'SHORT_ENTRY': '🔴', 
                'LONG_EXIT': '🔵',
                'SHORT_EXIT': '🟡',
                'HOLD': '⚪'
            }.get(row['signal_name'], '❓')
            
            st.write(f"{signal_color} **{row['strategy']}** - {row['signal_name']} @ ${row['current_price']:.2f} ({row['timestamp'].strftime('%H:%M:%S')})")

def display_log_files_info(log_info: Dict):
    """
    Display log files information.
    
    Args:
        log_info: Dictionary containing log file paths
    """
    st.subheader("📁 Log Files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("Trade Log")
        st.code(log_info['trade_log'], language=None)
        
    with col2:
        st.caption("Data Log")
        st.code(log_info['data_log'], language=None)
        
    with col3:
        st.caption("Decision Log")
        st.code(log_info['decision_log'], language=None)

def create_live_trading_chart(current_data: pd.DataFrame, trade_history_df: pd.DataFrame, strategies: List, symbol: str = "BTC-USD", active_trade_info: Dict = None, session_id: str = None):
    """
    Create a comprehensive live trading chart with OHLC candles, indicators, and trade markers.
    
    Args:
        current_data: Current market data DataFrame
        trade_history_df: DataFrame containing trade history
        strategies: List of strategy objects
        symbol: Trading symbol
        active_trade_info: Information about currently active trade
        
    Returns:
        Plotly figure object
    """
    if current_data.empty:
        return None
    
    # Create a continuous index to remove gaps between trading sessions
    # This makes the chart look like TradingView with no gaps
    data_with_continuous_index = current_data.copy()
    data_with_continuous_index['continuous_index'] = range(len(data_with_continuous_index))
    continuous_index = data_with_continuous_index['continuous_index']
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,  # Increase spacing between subplots
        row_heights=[0.65, 0.2, 0.15],  # Give more space to main chart
        subplot_titles=(f'{symbol} - Live Trading Chart', 'Volume', 'RSI'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Add OHLC candlestick chart with continuous index
    fig.add_trace(
        go.Candlestick(
            x=continuous_index,
            open=current_data['Open'],
            high=current_data['High'],
            low=current_data['Low'],
            close=current_data['Close'],
            name='OHLC',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350',
            increasing_fillcolor='#26A69A',
            decreasing_fillcolor='#EF5350',
            line=dict(width=1),
            hovertext=[
                f"<b>{date}</b><br>"
                f"Open: ${open:.2f}<br>"
                f"High: ${high:.2f}<br>"
                f"Low: ${low:.2f}<br>"
                f"Close: ${close:.2f}"
                for date, open, high, low, close in zip(
                    current_data.index,
                    current_data['Open'],
                    current_data['High'],
                    current_data['Low'],
                    current_data['Close']
                )
            ]
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['#26A69A' if close >= open else '#EF5350' 
              for close, open in zip(current_data['Close'], current_data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=continuous_index,
            y=current_data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7,
            hovertemplate=(
                "<b>Volume</b><br>" +
                "Date: %{customdata}<br>" +
                "Volume: %{y:,.0f}<br>" +
                "<extra></extra>"
            ),
            customdata=current_data.index
        ),
        row=2, col=1
    )
    
    # Add strategy indicators and signals
    strategy_colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50', '#F44336']
    
    for i, strategy in enumerate(strategies):
        if strategy is None:
            continue
            
        # Generate indicators for the strategy (safe to call multiple times)
        try:
            strategy_data = strategy.generate_indicators(current_data)
        except Exception as e:
            # If there's an error, create a basic dataframe with just the indicators
            strategy_data = current_data.copy()
            strategy_data['Signal'] = 0
            
            # Add basic indicators based on strategy type
            if hasattr(strategy, 'short_window') and hasattr(strategy, 'long_window'):
                strategy_data['SMA_short'] = strategy_data['Close'].rolling(window=strategy.short_window).mean()
                strategy_data['SMA_long'] = strategy_data['Close'].rolling(window=strategy.long_window).mean()
            elif hasattr(strategy, 'period'):
                # Calculate RSI
                delta = strategy_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=strategy.period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=strategy.period).mean()
                rs = gain / loss
                strategy_data['RSI'] = 100 - (100 / (1 + rs))
        
        # Add strategy-specific indicators
        if hasattr(strategy, 'short_window') and hasattr(strategy, 'long_window'):
            # Moving Average Crossover
            if 'SMA_short' in strategy_data.columns and 'SMA_long' in strategy_data.columns:
                # Short MA
                fig.add_trace(
                    go.Scatter(
                        x=continuous_index,
                        y=strategy_data['SMA_short'],
                        name=f'{strategy.name} - Short MA',
                        line=dict(color=strategy_colors[i % len(strategy_colors)], width=2),
                        hovertemplate=(
                            "<b>Short MA</b><br>" +
                            "Date: %{customdata}<br>" +
                            "Value: $%{y:.2f}<br>" +
                            "<extra></extra>"
                        ),
                        customdata=current_data.index
                    ),
                    row=1, col=1
                )
                
                # Long MA
                fig.add_trace(
                    go.Scatter(
                        x=continuous_index,
                        y=strategy_data['SMA_long'],
                        name=f'{strategy.name} - Long MA',
                        line=dict(color=strategy_colors[(i + 1) % len(strategy_colors)], width=2),
                        hovertemplate=(
                            "<b>Long MA</b><br>" +
                            "Date: %{customdata}<br>" +
                            "Value: $%{y:.2f}<br>" +
                            "<extra></extra>"
                        ),
                        customdata=current_data.index
                    ),
                    row=1, col=1
                )
        
        elif hasattr(strategy, 'period'):
            # RSI Strategy
            if 'RSI' in strategy_data.columns:
                # RSI line
                fig.add_trace(
                    go.Scatter(
                        x=continuous_index,
                        y=strategy_data['RSI'],
                        name=f'{strategy.name} - RSI',
                        line=dict(color=strategy_colors[i % len(strategy_colors)], width=2),
                        hovertemplate=(
                            "<b>RSI</b><br>" +
                            "Date: %{customdata}<br>" +
                            "Value: %{y:.2f}<br>" +
                            "<extra></extra>"
                        ),
                        customdata=current_data.index
                    ),
                    row=3, col=1
                )
                
                # Overbought line
                fig.add_trace(
                    go.Scatter(
                        x=continuous_index,
                        y=[strategy.overbought] * len(current_data),
                        name=f'{strategy.name} - Overbought',
                        line=dict(color='#F44336', dash='dash', width=1),
                        showlegend=False,
                        hovertemplate=(
                            "<b>Overbought Level</b><br>" +
                            "Value: %{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    ),
                    row=3, col=1
                )
                
                # Oversold line
                fig.add_trace(
                    go.Scatter(
                        x=continuous_index,
                        y=[strategy.oversold] * len(current_data),
                        name=f'{strategy.name} - Oversold',
                        line=dict(color='#4CAF50', dash='dash', width=1),
                        showlegend=False,
                        hovertemplate=(
                            "<b>Oversold Level</b><br>" +
                            "Value: %{y:.2f}<br>" +
                            "<extra></extra>"
                        )
                    ),
                    row=3, col=1
                )
        
        # Add buy signals
        buy_signals = strategy_data[strategy_data['Signal'] == 1]
        if not buy_signals.empty:
            # Get continuous indices for buy signals
            buy_continuous_indices = [data_with_continuous_index.loc[idx, 'continuous_index'] for idx in buy_signals.index]
            fig.add_trace(
                go.Scatter(
                    x=buy_continuous_indices,
                    y=buy_signals['Close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color=strategy_colors[i % len(strategy_colors)],
                        line=dict(color='white', width=1)
                    ),
                    name=f'{strategy.name} - Buy Signal',
                    hovertemplate=(
                        f"<b>{strategy.name} Buy Signal</b><br>" +
                        "Date: %{customdata}<br>" +
                        "Price: $%{y:.2f}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=buy_signals.index
                ),
                row=1, col=1
            )
        
        # Add sell signals
        sell_signals = strategy_data[strategy_data['Signal'] == -1]
        if not sell_signals.empty:
            # Get continuous indices for sell signals
            sell_continuous_indices = [data_with_continuous_index.loc[idx, 'continuous_index'] for idx in sell_signals.index]
            fig.add_trace(
                go.Scatter(
                    x=sell_continuous_indices,
                    y=sell_signals['Close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color=strategy_colors[i % len(strategy_colors)],
                        line=dict(color='white', width=1)
                    ),
                    name=f'{strategy.name} - Sell Signal',
                    hovertemplate=(
                        f"<b>{strategy.name} Sell Signal</b><br>" +
                        "Date: %{customdata}<br>" +
                        "Price: $%{y:.2f}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=sell_signals.index
                ),
                row=1, col=1
            )
    
        # Get all trades from CSV logs using session_id
    trade_log_path = get_trade_log_path(session_id)
    if trade_log_path:
        all_trades = parse_trades_from_csv(trade_log_path)
    else:
        print(f"⚠️ No trade log found for session: {session_id}")
        all_trades = pd.DataFrame()
    
    # Add trade markers if we have trades
    if not all_trades.empty:
        # Helper function to convert datetime to continuous index
        def get_continuous_index_for_datetime(dt):
            try:
                # Find the closest datetime in our data
                closest_idx = current_data.index.get_indexer([dt], method='nearest')[0]
                if closest_idx >= 0:
                    return data_with_continuous_index.iloc[closest_idx]['continuous_index']
                return 0
            except:
                return 0
        
        # Convert trade times to continuous indices
        entry_continuous_indices = [get_continuous_index_for_datetime(dt) for dt in all_trades['entry_time']]
        
        # Entry points
        fig.add_trace(
            go.Scatter(
                x=entry_continuous_indices,
                y=all_trades['entry_price'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='#4CAF50',
                        line=dict(color='white', width=2)
                    ),
                                    name='Trade Entry',
                hovertemplate=(
                    "<b>Trade Entry</b><br>" +
                    "Trade ID: %{customdata[0]}<br>" +
                    "Type: %{customdata[1]}<br>" +
                    "Status: %{customdata[2]}<br>" +
                    "Date: %{customdata[3]}<br>" +
                    "Price: $%{y:.2f}<br>" +
                    "<extra></extra>"
                ),
                customdata=list(zip(
                    all_trades['trade_id'],
                    all_trades['action'],
                    all_trades['status'],
                    all_trades['entry_time']
                ))
            ),
            row=1, col=1
        )
        
        # Open trades (different marker style)
        open_trades = all_trades[all_trades['status'] == 'open']
        if not open_trades.empty:
            open_entry_continuous_indices = [get_continuous_index_for_datetime(dt) for dt in open_trades['entry_time']]
            fig.add_trace(
                go.Scatter(
                    x=open_entry_continuous_indices,
                    y=open_trades['entry_price'],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=14,
                        color='#FFD700',  # Gold color for open trades
                        line=dict(color='black', width=2)
                    ),
                    name='Open Trade Entry',
                    hovertemplate=(
                        "<b>Open Trade Entry</b><br>" +
                        "Trade ID: %{customdata[0]}<br>" +
                        "Type: %{customdata[1]}<br>" +
                        "Status: %{customdata[2]}<br>" +
                        "Date: %{customdata[3]}<br>" +
                        "Price: $%{y:.2f}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=list(zip(
                        open_trades['trade_id'],
                        open_trades['action'],
                        open_trades['status'],
                        open_trades['entry_time']
                    ))
                ),
                row=1, col=1
            )
            
        # Exit points (only for completed trades)
        completed_trades = all_trades[all_trades['status'] == 'completed']
        if not completed_trades.empty:
            exit_continuous_indices = [get_continuous_index_for_datetime(dt) for dt in completed_trades['exit_time']]
            fig.add_trace(
                go.Scatter(
                    x=exit_continuous_indices,
                    y=completed_trades['exit_price'],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='#F44336',
                    line=dict(color='white', width=2)
                ),
                name='Trade Exit',
                hovertemplate=(
                    "<b>Trade Exit</b><br>" +
                    "Trade ID: %{customdata[0]}<br>" +
                    "Type: %{customdata[1]}<br>" +
                    "Exit Reason: %{customdata[2]}<br>" +
                    "Date: %{customdata[3]}<br>" +
                    "Price: $%{y:.2f}<br>" +
                    "<extra></extra>"
                ),
                customdata=list(zip(
                    completed_trades['trade_id'],
                    completed_trades['action'],
                    completed_trades['exit_reason'],
                    completed_trades['exit_time']
                ))
            ),
            row=1, col=1
        )
            
        
        # Add trade lines connecting entry to exit (only for completed trades)
        for _, trade in completed_trades.iterrows():
            entry_continuous_idx = get_continuous_index_for_datetime(trade['entry_time'])
            exit_continuous_idx = get_continuous_index_for_datetime(trade['exit_time'])
            fig.add_trace(
                go.Scatter(
                    x=[entry_continuous_idx, exit_continuous_idx],
                    y=[trade['entry_price'], trade['exit_price']],
                    mode='lines',
                    line=dict(
                        color='#FF9800' if trade['pnl'] > 0 else '#F44336',
                        width=2,
                        dash='dot'
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Trade {trade['trade_id']} PnL: {trade['pnl']:.2%}</b><br>" +
                        f"Type: {trade['action']}<br>" +
                        f"Exit Reason: {trade['exit_reason']}<br>" +
                        f"Entry: ${trade['entry_price']:.2f}<br>" +
                        f"Exit: ${trade['exit_price']:.2f}<br>" +
                        "<extra></extra>"
                    )
                ),
                row=1, col=1
            )
    
    # Add active trade marker if there's an active trade
    if active_trade_info and current_data is not None and not current_data.empty:
        current_price = current_data['Close'].iloc[-1]
        
        # Get continuous indices for active trade
        active_entry_continuous_idx = get_continuous_index_for_datetime(active_trade_info['entry_time'])
        current_continuous_idx = len(continuous_index) - 1  # Last index
        
        # Active trade entry point
        fig.add_trace(
            go.Scatter(
                x=[active_entry_continuous_idx],
                y=[active_trade_info['entry_price']],
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=15,
                    color='#FFD700',  # Gold color for active trade
                    line=dict(color='black', width=2)
                ),
                name='Active Trade Entry',
                hovertemplate=(
                    "<b>Active Trade Entry</b><br>" +
                    f"Strategy: {active_trade_info['strategy']}<br>" +
                    f"Action: {active_trade_info['action']}<br>" +
                    "Entry Price: $%{y:.2f}<br>" +
                    f"Quantity: {active_trade_info['quantity']:.4f}<br>" +
                    "<extra></extra>"
                )
            ),
            row=1, col=1
        )
        
        # Current price line for active trade
        fig.add_trace(
            go.Scatter(
                x=[active_entry_continuous_idx, current_continuous_idx],
                y=[active_trade_info['entry_price'], current_price],
                mode='lines',
                line=dict(
                    color='#FFD700',
                    width=3,
                    dash='solid'
                ),
                name='Active Trade PnL',
                hovertemplate=(
                    "<b>Active Trade</b><br>" +
                    "Current Price: $%{y:.2f}<br>" +
                    "<extra></extra>"
                )
            ),
            row=1, col=1
        )
        
        # Current price marker
        fig.add_trace(
            go.Scatter(
                x=[current_continuous_idx],
                y=[current_price],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=12,
                    color='#FFD700',
                    line=dict(color='black', width=1)
                ),
                name='Current Price',
                hovertemplate=(
                    "<b>Current Price</b><br>" +
                    "Price: $%{y:.2f}<br>" +
                    "<extra></extra>"
                )
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} - Live Trading Chart with Indicators',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='white'),
            y=0.98
        ),
        xaxis_rangeslider_visible=False,
        height=1000,  # Increased height for better visibility
        width=None,  # Let it auto-size to container width
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,  # Position legend at the very top
            xanchor="center",
            x=0.5,   # Center the legend
            bgcolor='rgba(0,0,0,0.9)',  # Very dark background for visibility
            bordercolor='rgba(255,255,255,0.5)',
            borderwidth=2,
            font=dict(size=14, color='white'),  # Larger white text for visibility
            itemsizing='constant',
            itemwidth=30
        ),
        margin=dict(t=160, b=50, l=50, r=50),  # Increase top margin for legend
        plot_bgcolor='rgba(0,0,0,0.05)',  # Very dark background
        paper_bgcolor='rgba(0,0,0,0.05)',  # Very dark background
        hovermode='x unified',  # Better hover behavior
        autosize=True,  # Enable autosize for full width
        # Enhanced responsive settings
        uirevision=True,  # Maintain zoom/pan state
        dragmode='zoom'  # Enable zoom by default
    )
    
    # Update y-axis ranges with better formatting for dark theme
    fig.update_yaxes(
        title_text="Price ($)", 
        row=1, col=1,
        tickformat=",.0f",
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False,
        tickfont=dict(color='white'),
        title_font=dict(color='white')
    )
    fig.update_yaxes(
        title_text="Volume", 
        row=2, col=1,
        tickformat=",.0f",
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False,
        tickfont=dict(color='white'),
        title_font=dict(color='white')
    )
    fig.update_yaxes(
        title_text="RSI", 
        range=[0, 100], 
        row=3, col=1,
        tickformat=".0f",
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False,
        tickfont=dict(color='white'),
        title_font=dict(color='white')
    )
    
    # Update x-axis with proper time handling to remove gaps
    # Create custom tick labels that show actual datetime but use continuous index
    tick_interval = max(1, len(continuous_index) // 10)  # Show about 10 ticks
    tick_positions = list(range(0, len(continuous_index), tick_interval))
    tick_labels = [current_data.index[i].strftime('%m/%d %H:%M') for i in tick_positions]
    
    fig.update_xaxes(
        title_text="Date", 
        row=3, col=1,
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False,
        tickfont=dict(color='white'),
        title_font=dict(color='white'),
        # Remove gaps between trading sessions
        rangeslider=dict(visible=False),
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels
    )
    
    # Update all x-axes to remove gaps
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False,
        tickfont=dict(color='white'),
        title_font=dict(color='white'),
        tickmode='array',
        tickvals=tick_positions,
        ticktext=tick_labels
    )
    
    return fig

def display_market_data(current_data: pd.DataFrame):
    """
    Display current market data.
    
    Args:
        current_data: DataFrame containing market data
    """
    if current_data.empty:
        return
    
    st.subheader("📈 Current Market Data")
    
    # Show latest data
    latest_data = current_data.tail(10)
    st.dataframe(latest_data[['Open', 'High', 'Low', 'Close', 'Volume']], use_container_width=True)

def create_strategy_parameter_inputs(enabled_strategies: List[str]):
    """
    Create strategy parameter input widgets.
    
    Args:
        enabled_strategies: List of enabled strategies
        
    Returns:
        Dictionary containing parameter values
    """
    st.sidebar.subheader("⚙️ Strategy Parameters")
    
    ma_params = None
    rsi_params = None
    donchian_params = None
    
    if 'ma' in enabled_strategies:
        st.sidebar.markdown("**Moving Average Parameters**")
        ma_params = {
            'short_window': st.sidebar.slider("Short MA Window", 5, 50, 10, key="ma_short"),
            'long_window': st.sidebar.slider("Long MA Window", 20, 200, 50, key="ma_long"),
            'risk_reward_ratio': st.sidebar.slider("Risk/Reward Ratio", 1.0, 5.0, 2.0, 0.1, key="ma_rr"),
            'trading_fee': st.sidebar.slider("Trading Fee (%)", 0.0, 1.0, 0.1, 0.01, key="ma_fee") / 100
        }
    
    if 'rsi' in enabled_strategies:
        st.sidebar.markdown("**RSI Parameters**")
        rsi_params = {
            'period': st.sidebar.slider("RSI Period", 5, 30, 14, key="rsi_period"),
            'overbought': st.sidebar.slider("Overbought Level", 60, 90, 70, key="rsi_overbought"),
            'oversold': st.sidebar.slider("Oversold Level", 10, 40, 30, key="rsi_oversold"),
            'risk_reward_ratio': st.sidebar.slider("Risk/Reward Ratio", 1.0, 5.0, 2.0, 0.1, key="rsi_rr"),
            'trading_fee': st.sidebar.slider("Trading Fee (%)", 0.0, 1.0, 0.1, 0.01, key="rsi_fee") / 100
        }
    
    if 'donchian' in enabled_strategies:
        st.sidebar.markdown("**Donchian Channel Parameters**")
        donchian_params = {
            'channel_period': st.sidebar.slider("Channel Period", 10, 100, 20, key="donchian_period"),
            'risk_reward_ratio': st.sidebar.slider("Risk/Reward Ratio", 1.0, 5.0, 2.0, 0.1, key="donchian_rr"),
            'trading_fee': st.sidebar.slider("Trading Fee (%)", 0.0, 1.0, 0.1, 0.01, key="donchian_fee") / 100
        }
    
    return {
        'ma_params': ma_params,
        'rsi_params': rsi_params,
        'donchian_params': donchian_params
    }
