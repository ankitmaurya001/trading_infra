#!/usr/bin/env python3
"""
Test script to verify MA crossover logic and debug the trade execution issue
"""

import pandas as pd
import numpy as np
from datetime import datetime
from strategies import MovingAverageCrossover

def test_ma_crossover_logic():
    """Test the MA crossover logic with sample data."""
    
    # Create sample data that should trigger a LONG signal
    dates = pd.date_range('2025-08-22 07:00:00', periods=10, freq='15min')
    prices = [113000, 113100, 113200, 113300, 113400, 113500, 113600, 113700, 113800, 113900]
    
    data = pd.DataFrame({
        'Open': prices,
        'High': [p + 50 for p in prices],
        'Low': [p - 50 for p in prices],
        'Close': prices,
        'Volume': [1000000] * len(prices)
    }, index=dates)
    
    print("🧪 Testing MA Crossover Logic")
    print("=" * 50)
    print(f"Sample data range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}")
    
    # Create strategy with short=3, long=5 for easy testing
    strategy = MovingAverageCrossover(short_window=3, long_window=5)
    
    # Generate signals
    signals_data = strategy.generate_signals(data)
    
    print("\n📊 MA Values:")
    for i, (date, row) in enumerate(signals_data.iterrows()):
        short_ma = row.get('SMA_short', np.nan)
        long_ma = row.get('SMA_long', np.nan)
        signal = row.get('Signal', 0)
        price = row['Close']
        
        if not np.isnan(short_ma) and not np.isnan(long_ma):
            signal_name = {0: "HOLD", 1: "LONG", -1: "SHORT"}.get(signal, f"UNKNOWN({signal})")
            print(f"{date}: Price=${price:.2f}, Short_MA=${short_ma:.2f}, Long_MA=${long_ma:.2f}, Signal={signal_name}")
    
    # Check for signals
    long_signals = signals_data[signals_data['Signal'] == 1]
    short_signals = signals_data[signals_data['Signal'] == -1]
    
    print(f"\n🎯 Signals Found:")
    print(f"LONG signals: {len(long_signals)}")
    print(f"SHORT signals: {len(short_signals)}")
    
    if not long_signals.empty:
        print("\nLONG Signal Details:")
        for date, row in long_signals.iterrows():
            print(f"  {date}: Price=${row['Close']:.2f}, Short_MA=${row['SMA_short']:.2f}, Long_MA=${row['SMA_long']:.2f}")
    
    if not short_signals.empty:
        print("\nSHORT Signal Details:")
        for date, row in short_signals.iterrows():
            print(f"  {date}: Price=${row['Close']:.2f}, Short_MA=${row['SMA_short']:.2f}, Long_MA=${row['SMA_long']:.2f}")

def analyze_trade_execution():
    """Analyze the actual trade execution from the CSV data."""
    
    print("\n🔍 Analyzing Trade Execution from CSV")
    print("=" * 50)
    
    # Read the trade CSV
    try:
        trade_df = pd.read_csv("logs/live_trades_BTC-USD_20250831_162920_mock.csv")
        trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])
        
        print(f"Total trade records: {len(trade_df)}")
        
        # Find BUY trades (LONG entries)
        buy_trades = trade_df[trade_df['action'] == 'BUY']
        print(f"BUY trades: {len(buy_trades)}")
        
        for _, trade in buy_trades.iterrows():
            print(f"\n🟢 BUY Trade at {trade['timestamp']}:")
            print(f"  Price: ${trade['price']:.2f}")
            print(f"  Strategy: {trade['strategy']}")
            print(f"  Trade ID: {trade['trade_id']}")
            
            # Find corresponding EXIT
            exit_trade = trade_df[(trade_df['trade_id'] == trade['trade_id']) & (trade_df['action'] == 'EXIT')]
            if not exit_trade.empty:
                exit_row = exit_trade.iloc[0]
                print(f"  Exit at {exit_row['timestamp']}: ${exit_row['price']:.2f}")
                print(f"  PnL: {exit_row['pnl']:.4f}")
        
        # Find SELL trades (SHORT entries)
        sell_trades = trade_df[trade_df['action'] == 'SELL']
        print(f"\nSELL trades: {len(sell_trades)}")
        
        for _, trade in sell_trades.iterrows():
            print(f"\n🔴 SELL Trade at {trade['timestamp']}:")
            print(f"  Price: ${trade['price']:.2f}")
            print(f"  Strategy: {trade['strategy']}")
            print(f"  Trade ID: {trade['trade_id']}")
            
            # Find corresponding EXIT
            exit_trade = trade_df[(trade_df['trade_id'] == trade['trade_id']) & (trade_df['action'] == 'EXIT')]
            if not exit_trade.empty:
                exit_row = exit_trade.iloc[0]
                print(f"  Exit at {exit_row['timestamp']}: ${exit_row['price']:.2f}")
                print(f"  PnL: {exit_row['pnl']:.4f}")
                
    except Exception as e:
        print(f"Error reading trade CSV: {e}")

if __name__ == "__main__":
    test_ma_crossover_logic()
    analyze_trade_execution()
