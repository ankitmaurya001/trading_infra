import streamlit as st
import pandas as pd
from data_fetcher import DataFetcher
from strategies import MovingAverageCrossover, RSIStrategy, DonchianChannelBreakout
from visualizer import Visualizer
import plotly.graph_objects as go

def main():
    st.set_page_config(page_title="Trading Strategy Visualizer", layout="wide")
    st.title("Trading Strategy Visualizer")
    
    # Sidebar for controls
    st.sidebar.header("Data Settings")
    
    # Stock selection
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.Timestamp('2023-01-01'))
    with col2:
        end_date = st.date_input("End Date", pd.Timestamp.now().date())

    # Sidebar for interval selection
    st.sidebar.subheader("Data Interval")
    interval = st.sidebar.selectbox(
        "Select data interval",
        options=["1d", "5m", "15m", "30m", "1h", "4h"],
        index=0,
        help="Intraday intervals (e.g., 5m, 15m, 30m, 1h, 4h) are limited to recent ~60 days. 1d is daily bars."
    )
    
    # Key point settings
    st.sidebar.subheader("Key Point Settings")
    key_point_multiplier = st.sidebar.slider(
        "Key Point Multiplier",
        min_value=1,
        max_value=500,
        value=400,
        step=1,
        help="Multiplier for average daily return to identify significant price movements"
    )
    
    # Strategy selection
    st.sidebar.header("Strategy Settings")
    
    # Risk-Reward settings
    st.sidebar.subheader("Risk-Reward Settings")
    risk_reward_ratio = st.sidebar.slider(
        "Risk-Reward Ratio",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Ratio of potential reward to risk (e.g., 2.0 means potential reward is twice the risk)"
    )
    
    # Trading fee settings
    trading_fee = st.sidebar.slider(
        "Trading Fee (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Trading fee as a percentage of trade value"
    )
    
    # Moving Average Crossover settings
    st.sidebar.subheader("Moving Average Crossover")
    use_ma = st.sidebar.checkbox("Enable MA Crossover", value=True)
    if use_ma:
        ma_short = st.sidebar.slider("Short MA Period", 5, 50, 20)
        ma_long = st.sidebar.slider("Long MA Period", 14, 200, 50)
    
    # RSI settings
    st.sidebar.subheader("RSI Strategy")
    use_rsi = st.sidebar.checkbox("Enable RSI", value=True)
    if use_rsi:
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
        rsi_overbought = st.sidebar.slider("Overbought Level", 50, 90, 70)
        rsi_oversold = st.sidebar.slider("Oversold Level", 10, 50, 30)
    
    # Donchian Channel Breakout settings
    st.sidebar.subheader("Donchian Channel Breakout")
    use_donchian = st.sidebar.checkbox("Enable Donchian Channel Breakout", value=False)
    if use_donchian:
        donchian_period = st.sidebar.slider("Channel Period", 5, 50, 20)
    
    # User-configurable recent days for metrics
    st.sidebar.subheader("Performance Metrics Window")
    recent_days = st.sidebar.number_input(
        "Show metrics for last N ticks",
        min_value=1,
        max_value=730,
        value=30,
        step=1,
        help="Number of most recent days to show strategy metrics for"
    )
    
    # Fetch data
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_data(
        symbol, 
        str(start_date), 
        str(end_date),
        key_point_multiplier=key_point_multiplier,
        interval=interval
    )
    
    if data.empty:
        st.error(f"Error fetching data for {symbol}")
        return
    
    # Create strategies based on user selection
    strategies = []
    if use_ma:
        ma_strategy = MovingAverageCrossover(
            short_window=ma_short, 
            long_window=ma_long,
            risk_reward_ratio=risk_reward_ratio,
            trading_fee=trading_fee/100  # Convert percentage to decimal
        )
        strategies.append(ma_strategy)
        st.sidebar.info(f"MA Strategy: Short={ma_short}, Long={ma_long}, R/R={risk_reward_ratio}, Fee={trading_fee/100:.4f}")
    
    if use_rsi:
        rsi_strategy = RSIStrategy(
            period=rsi_period, 
            overbought=rsi_overbought, 
            oversold=rsi_oversold,
            risk_reward_ratio=risk_reward_ratio,
            trading_fee=trading_fee/100  # Convert percentage to decimal
        )
        strategies.append(rsi_strategy)
        st.sidebar.info(f"RSI Strategy: Period={rsi_period}, Overbought={rsi_overbought}, Oversold={rsi_oversold}, R/R={risk_reward_ratio}, Fee={trading_fee/100:.4f}")
    
    if use_donchian:
        donchian_strategy = DonchianChannelBreakout(
            channel_period=donchian_period,
            risk_reward_ratio=risk_reward_ratio,
            trading_fee=trading_fee/100  # Convert percentage to decimal
        )
        strategies.append(donchian_strategy)
        st.sidebar.info(f"Donchian Strategy: Period={donchian_period}, R/R={risk_reward_ratio}, Fee={trading_fee/100:.4f}")
    
    if not strategies:
        st.warning("Please enable at least one strategy")
        return
    
    # Create and display the chart
    visualizer = Visualizer()
    
    # Generate signals for all strategies first (to avoid double execution)
    signals_data = {}
    for strategy in strategies:
        signals_data[strategy.name] = strategy.generate_signals(data)
    
    fig = visualizer.create_chart(data, strategies, signals_data)
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display strategy performance metrics
    st.header("Strategy Performance Metrics")
    
    for strategy in strategies:
        # Get metrics (signals already generated above)
        metrics = strategy.get_strategy_metrics()
        
        # Recent metrics: use last N bars/ticks
        recent_bars = set(data.index[-recent_days:])
        recent_metrics = strategy.get_strategy_metrics(recent_bars=recent_bars)
        
        # Get cumulative PnL data for plotting
        try:
            pnl_data = strategy.get_cumulative_pnl_data(data)
        except Exception as e:
            st.error(f"Error generating PnL data for {strategy.name}: {str(e)}")
            pnl_data = pd.DataFrame({
                'Date': data.index,
                'Cumulative_PnL': [1.0] * len(data),
                'Drawdown': [0.0] * len(data),
                'Peak': [1.0] * len(data)
            })
        
        st.subheader(f"{strategy.name} Performance")
        
        # Show strategy parameters for verification
        st.caption("Strategy Parameters:")
        if hasattr(strategy, 'short_window'):
            st.caption(f"Short MA: {strategy.short_window}, Long MA: {strategy.long_window}, Risk/Reward: {strategy.risk_reward_ratio}, Trading Fee: {strategy.trading_fee:.4f}")
        elif hasattr(strategy, 'period'):
            st.caption(f"RSI Period: {strategy.period}, Overbought: {strategy.overbought}, Oversold: {strategy.oversold}, Risk/Reward: {strategy.risk_reward_ratio}, Trading Fee: {strategy.trading_fee:.4f}")
        elif hasattr(strategy, 'channel_period'):
            st.caption(f"Channel Period: {strategy.channel_period}, Risk/Reward: {strategy.risk_reward_ratio}, Trading Fee: {strategy.trading_fee:.4f}")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Metrics", "📈 Performance Chart", "📋 Trade History"])
        
        with tab1:
            # Main performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", metrics["total_trades"])
                st.metric(f"Recent Trades (last {recent_days} bars)", recent_metrics["total_trades"])
            with col2:
                st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
                st.metric(f"Recent Win Rate", f"{recent_metrics['win_rate']:.1%}")
            with col3:
                st.metric("Total PnL", f"{metrics['total_pnl']:.2%}")
                st.metric(f"Recent PnL", f"{recent_metrics['total_pnl']:.2%}")
            with col4:
                st.metric("Avg Return per Trade", f"{metrics['avg_return']:.2%}")
                st.metric(f"Recent Avg Return", f"{recent_metrics['avg_return']:.2%}")
            
            # Risk-adjusted metrics
            st.subheader("Risk-Adjusted Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
                st.metric(f"Recent Sharpe", f"{recent_metrics['sharpe_ratio']:.3f}")
            with col2:
                st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.3f}")
                st.metric(f"Recent Calmar", f"{recent_metrics['calmar_ratio']:.3f}")
            with col3:
                st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
                st.metric(f"Recent Sortino", f"{recent_metrics['sortino_ratio']:.3f}")
            with col4:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                st.metric(f"Recent Max DD", f"{recent_metrics['max_drawdown']:.2%}")
            
            # Trade quality metrics
            st.subheader("Trade Quality Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                st.metric(f"Recent PF", f"{recent_metrics['profit_factor']:.2f}")
            with col2:
                st.metric("Avg Win", f"{metrics['avg_win']:.2%}")
                st.metric(f"Recent Avg Win", f"{recent_metrics['avg_win']:.2%}")
            with col3:
                st.metric("Avg Loss", f"{metrics['avg_loss']:.2%}")
                st.metric(f"Recent Avg Loss", f"{recent_metrics['avg_loss']:.2%}")
            with col4:
                st.metric("Win/Loss Ratio", f"{metrics['risk_reward_ratio']:.2f}")
                st.metric(f"Recent W/L Ratio", f"{recent_metrics['risk_reward_ratio']:.2f}")
            
            # Additional metrics
            st.subheader("Additional Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Largest Win", f"{metrics['largest_win']:.2%}")
                st.metric(f"Largest Loss", f"{metrics['largest_loss']:.2%}")
            with col2:
                st.metric("Consecutive Wins", metrics['consecutive_wins'])
                st.metric("Consecutive Losses", metrics['consecutive_losses'])
            with col3:
                st.metric("Geometric Mean Return", f"{metrics['geometric_mean_return']:.2%}")
                st.metric(f"Recent Geo Mean", f"{recent_metrics['geometric_mean_return']:.2%}")
            with col4:
                # Display strategy-specific parameters
                if hasattr(strategy, 'short_window'):
                    st.metric("Short MA", strategy.short_window)
                    st.metric("Long MA", strategy.long_window)
                elif hasattr(strategy, 'period'):
                    st.metric("RSI Period", strategy.period)
                    st.metric("Overbought/Oversold", f"{strategy.overbought}/{strategy.oversold}")
                elif hasattr(strategy, 'channel_period'):
                    st.metric("Channel Period", strategy.channel_period)
                    st.metric("Risk/Reward", f"{strategy.risk_reward_ratio:.1f}")
            
            # Performance summary
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
        
        with tab2:
            # Performance charts
            st.subheader("Cumulative Performance Chart")
            
            try:
                if not pnl_data.empty and len(pnl_data) > 1:
                    # Debug: Check the structure of pnl_data
                    st.write(f"PnL data shape: {pnl_data.shape}")
                    st.write(f"PnL data columns: {list(pnl_data.columns)}")
                    st.write(f"First few rows of PnL data:")
                    st.write(pnl_data.head())
                    
                    # Ensure we have the required columns
                    required_columns = ['Date', 'Cumulative_PnL', 'Drawdown', 'Peak']
                    missing_columns = [col for col in required_columns if col not in pnl_data.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns in PnL data: {missing_columns}")
                        st.write("Available columns:", list(pnl_data.columns))
                        return
                    
                    # Create cumulative PnL chart
                    fig_pnl = go.Figure()
                    
                    # Add cumulative PnL line
                    fig_pnl.add_trace(go.Scatter(
                        x=pnl_data['Date'],
                        y=pnl_data['Cumulative_PnL'],
                        mode='lines',
                        name='Cumulative PnL',
                        line=dict(color='#1f77b4', width=2),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative PnL:</b> %{y:.2%}<extra></extra>'
                    ))
                    
                    # Add peak line
                    fig_pnl.add_trace(go.Scatter(
                        x=pnl_data['Date'],
                        y=pnl_data['Peak'],
                        mode='lines',
                        name='Peak',
                        line=dict(color='green', width=1, dash='dash'),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Peak:</b> %{y:.2%}<extra></extra>'
                    ))
                    
                    # Add drawdown area
                    fig_pnl.add_trace(go.Scatter(
                        x=pnl_data['Date'],
                        y=pnl_data['Cumulative_PnL'],
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.1)',
                        line=dict(color='rgba(255, 0, 0, 0.1)', width=0),
                        showlegend=False,
                        hovertemplate='<b>Date:</b> %{x}<br><b>Drawdown:</b> %{text:.2%}<extra></extra>',
                        text=[f"{dd:.2%}" for dd in pnl_data['Drawdown']]
                    ))
                    
                    # Update layout
                    fig_pnl.update_layout(
                        title=f"{strategy.name} - Cumulative Performance",
                        xaxis_title="Date",
                        yaxis_title="Cumulative PnL",
                        yaxis_tickformat='.1%',
                        hovermode='x unified',
                        showlegend=True,
                        height=500
                    )
                    
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    
                    # Add drawdown chart
                    st.subheader("Drawdown Analysis")
                    fig_dd = go.Figure()
                    
                    fig_dd.add_trace(go.Scatter(
                        x=pnl_data['Date'],
                        y=pnl_data['Drawdown'],
                        mode='lines',
                        name='Drawdown',
                        line=dict(color='red', width=2),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        hovertemplate='<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2%}<extra></extra>'
                    ))
                    
                    fig_dd.update_layout(
                        title=f"{strategy.name} - Drawdown Over Time",
                        xaxis_title="Date",
                        yaxis_title="Drawdown",
                        yaxis_tickformat='.1%',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig_dd, use_container_width=True)
                    
                    # Performance statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        final_pnl = pnl_data['Cumulative_PnL'].iloc[-1] - 1
                        st.metric("Final PnL", f"{final_pnl:.2%}")
                    
                    with col2:
                        max_dd = pnl_data['Drawdown'].max()
                        st.metric("Max Drawdown", f"{max_dd:.2%}")
                    
                    with col3:
                        peak_value = pnl_data['Peak'].max()
                        st.metric("Peak Value", f"{peak_value:.2%}")
                    
                    with col4:
                        recovery_factor = final_pnl / max_dd if max_dd > 0 else float('inf')
                        st.metric("Recovery Factor", f"{recovery_factor:.2f}")
                    
                else:
                    st.info("No trades completed yet. Performance chart will appear after trades are executed.")
                    
            except Exception as e:
                st.error(f"Error creating performance charts for {strategy.name}: {str(e)}")
                st.error(f"Error details: {type(e).__name__}")
                st.info("Performance charts are temporarily unavailable.")
        
        with tab3:
            # Trade history
            st.subheader("Trade History")
            
            trade_history = strategy.get_trade_history()
            
            if not trade_history.empty:
                # Format the trade history for display
                display_history = trade_history.copy()
                display_history['Entry_Date'] = display_history['Entry_Date'].dt.strftime('%Y-%m-%d %H:%M')
                display_history['Exit_Date'] = display_history['Exit_Date'].dt.strftime('%Y-%m-%d %H:%M')
                display_history['PnL_Pct'] = display_history['PnL_Pct'].round(2)
                
                # Color code PnL
                def color_pnl(val):
                    if val > 0:
                        return 'color: green'
                    elif val < 0:
                        return 'color: red'
                    return 'color: black'
                
                styled_history = display_history.style.map(color_pnl, subset=['PnL_Pct'])
                
                st.dataframe(
                    styled_history,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Trade statistics
                st.subheader("Trade Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    winning_trades = len(trade_history[trade_history['PnL'] > 0])
                    total_trades = len(trade_history)
                    st.metric("Winning Trades", f"{winning_trades}/{total_trades}")
                
                with col2:
                    avg_win = trade_history[trade_history['PnL'] > 0]['PnL_Pct'].mean()
                    st.metric("Avg Win", f"{avg_win:.2f}%" if not pd.isna(avg_win) else "N/A")
                
                with col3:
                    avg_loss = trade_history[trade_history['PnL'] < 0]['PnL_Pct'].mean()
                    st.metric("Avg Loss", f"{avg_loss:.2f}%" if not pd.isna(avg_loss) else "N/A")
                
                with col4:
                    best_trade = trade_history['PnL_Pct'].max()
                    worst_trade = trade_history['PnL_Pct'].min()
                    st.metric("Best/Worst Trade", f"{best_trade:.2f}% / {worst_trade:.2f}%")
                
            else:
                st.info("No completed trades to display.")
        
        st.divider()
    
    # Display key statistics
    st.header("Market Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Days", len(data))
        st.metric("Average Daily Return", f"{data['Returns'].mean():.2%}")
    
    with col2:
        st.metric("Volatility", f"{data['Volatility'].mean():.2%}")
        st.metric("Key Points", len(data[data['Key_Point'] == True]))
    
    with col3:
        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
        st.metric("52-Week High", f"${data['High'].max():.2f}")
    
    # Display raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(data)

if __name__ == "__main__":
    main() 