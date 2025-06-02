import streamlit as st
import pandas as pd
from data_fetcher import DataFetcher
from strategies import MovingAverageCrossover, RSIStrategy
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
    
    # Key point settings
    st.sidebar.subheader("Key Point Settings")
    key_point_multiplier = st.sidebar.slider(
        "Key Point Multiplier",
        min_value=1,
        max_value=500,
        value=50,
        step=1,
        help="Multiplier for average daily return to identify significant price movements"
    )
    
    # Strategy selection
    st.sidebar.header("Strategy Settings")
    
    # Moving Average Crossover settings
    st.sidebar.subheader("Moving Average Crossover")
    use_ma = st.sidebar.checkbox("Enable MA Crossover", value=True)
    if use_ma:
        ma_short = st.sidebar.slider("Short MA Period", 5, 50, 20)
        ma_long = st.sidebar.slider("Long MA Period", 20, 200, 50)
    
    # RSI settings
    st.sidebar.subheader("RSI Strategy")
    use_rsi = st.sidebar.checkbox("Enable RSI", value=True)
    if use_rsi:
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
        rsi_overbought = st.sidebar.slider("Overbought Level", 50, 90, 70)
        rsi_oversold = st.sidebar.slider("Oversold Level", 10, 50, 30)
    
    # Fetch data
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_data(
        symbol, 
        str(start_date), 
        str(end_date),
        key_point_multiplier=key_point_multiplier
    )
    
    if data.empty:
        st.error(f"Error fetching data for {symbol}")
        return
    
    # Create strategies based on user selection
    strategies = []
    if use_ma:
        strategies.append(MovingAverageCrossover(short_window=ma_short, long_window=ma_long))
    if use_rsi:
        strategies.append(RSIStrategy(period=rsi_period, overbought=rsi_overbought, oversold=rsi_oversold))
    
    if not strategies:
        st.warning("Please enable at least one strategy")
        return
    
    # Create and display the chart
    visualizer = Visualizer()
    fig = visualizer.create_chart(data, strategies)
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display key statistics
    st.header("Key Statistics")
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