#!/usr/bin/env python3
"""
Example usage of BinanceDataFetcher
Demonstrates how to fetch cryptocurrency data from Binance
"""

from data_fetcher import BinanceDataFetcher
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import config as cfg

def main():
    # Initialize Binance data fetcher (no API key needed for historical data)
    binance_fetcher = BinanceDataFetcher(api_key=cfg.BINANCE_API_KEY, api_secret=cfg.BINANCE_SECRET_KEY)
    
    # Example 1: Fetch BTC/USDT data for the last 30 days with current time
    print("Fetching BTC/USDT data...")
    end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    
    btc_data = binance_fetcher.fetch_btc_data(
        start_date=start_date,
        end_date=end_date,
        interval='15m'  # 15-minute intervals
    )
    
    if not btc_data.empty:
        print(f"BTC Data shape: {btc_data.shape}")
        print(f"Date range: {btc_data.index.min()} to {btc_data.index.max()}")
        print("\nFirst few rows:")
        print(btc_data.head())
        print("\nLast few rows:")
        print(btc_data.tail())
        
        # Get key points (significant price movements)
        key_points = binance_fetcher.get_key_points(btc_data)
        print(f"\nKey points found: {len(key_points)}")
        if not key_points.empty:
            print("Sample key points:")
            print(key_points[['Close', 'Returns', 'Volatility']].head())
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Fetch ETH/USDT data with different interval
    print("Fetching ETH/USDT data...")
    eth_data = binance_fetcher.fetch_eth_data(
        start_date=start_date,
        end_date=end_date,
        interval='15m'  # 15-minute intervals
    )
    
    if not eth_data.empty:
        print(f"ETH Data shape: {eth_data.shape}")
        print(f"Date range: {eth_data.index.min()} to {eth_data.index.max()}")
        print("\nData columns:", eth_data.columns.tolist())
        print("\nBasic statistics:")
        print(eth_data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Fetch any cryptocurrency pair
    print("Fetching ADA/USDT data...")
    ada_data = binance_fetcher.fetch_historical_data(
        symbol='ADAUSDT',
        start_date=start_date,
        end_date=end_date,
        interval='1d'  # Daily intervals
    )
    
    if not ada_data.empty:
        print(f"ADA Data shape: {ada_data.shape}")
        print(f"Date range: {ada_data.index.min()} to {ada_data.index.max()}")
        print("\nPrice range:")
        print(f"Highest: ${ada_data['High'].max():.4f}")
        print(f"Lowest: ${ada_data['Low'].min():.4f}")
        print(f"Average: ${ada_data['Close'].mean():.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Get symbol information
    print("Getting BTC/USDT symbol information...")
    symbol_info = binance_fetcher.get_symbol_info('BTCUSDT')
    if symbol_info:
        print(f"Symbol: {symbol_info.get('symbol', 'N/A')}")
        print(f"Base Asset: {symbol_info.get('baseAsset', 'N/A')}")
        print(f"Quote Asset: {symbol_info.get('quoteAsset', 'N/A')}")
        print(f"Status: {symbol_info.get('status', 'N/A')}")
    
    # Example 5: Get 24hr ticker (if you have API credentials)
    print("\nGetting 24hr ticker for BTC/USDT...")
    ticker = binance_fetcher.get_24hr_ticker('BTCUSDT')
    if ticker:
        print(f"Current Price: ${float(ticker.get('lastPrice', 0)):.2f}")
        print(f"24h Change: {float(ticker.get('priceChangePercent', 0)):.2f}%")
        print(f"24h Volume: {float(ticker.get('volume', 0)):.2f} BTC")

if __name__ == "__main__":
    main()
