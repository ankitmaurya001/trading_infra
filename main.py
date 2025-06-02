from data_fetcher import DataFetcher
from strategies import MovingAverageCrossover, RSIStrategy
from visualizer import Visualizer
import pandas as pd

def main():
    # Initialize components
    data_fetcher = DataFetcher()
    visualizer = Visualizer()
    
    # Define strategies
    strategies = [
        MovingAverageCrossover(short_window=20, long_window=50),
        RSIStrategy(period=14, overbought=70, oversold=30)
    ]
    
    # Fetch data
    symbol = 'AAPL'  # Example: Apple stock
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    print(f"Fetching data for {symbol}...")
    data = data_fetcher.fetch_data(symbol, start_date, end_date)
    
    if data.empty:
        print("Error: No data fetched")
        return
    
    # Generate signals for each strategy
    for strategy in strategies:
        print(f"Generating signals for {strategy.name}...")
        data = strategy.generate_signals(data)
    
    # Create and show the chart
    print("Creating interactive chart...")
    fig = visualizer.create_chart(data, strategies)
    visualizer.show()

if __name__ == "__main__":
    main() 