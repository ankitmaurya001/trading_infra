#!/usr/bin/env python3
"""
Analyze leverage usage in mock validation to understand PnL discrepancy.
Shows actual leverage used per trade and average leverage.
"""

import pandas as pd
from datetime import datetime, timedelta
from comprehensive_strategy_validation import ComprehensiveStrategyValidator

# Same config as mock validation
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_START_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
DEFAULT_INTERVAL = "15m"
DEFAULT_PARAMS = [
    {"short_window": 10, "long_window": 40, "risk_reward_ratio": 4.0}
]

print("🔍 ANALYZING LEVERAGE USAGE")
print("=" * 80)

validator = ComprehensiveStrategyValidator(
    initial_balance=10000.0,
    max_leverage=10.0,
    max_loss_percent=2.0,
    trading_fee=0.0,
)

# Fetch data
print("📥 Fetching data...")
data = validator.data_fetcher.fetch_historical_data(
    DEFAULT_SYMBOL, DEFAULT_START_DATE, DEFAULT_END_DATE, interval=DEFAULT_INTERVAL
)
if data.empty:
    raise ValueError(f"No data fetched for {DEFAULT_SYMBOL}")

validator.train_data = data.iloc[:0].copy()
validator.test_data = data.copy()
validator.symbol = DEFAULT_SYMBOL
validator.interval = DEFAULT_INTERVAL

print(f"✅ Fetched {len(data)} data points")

# Run mock validation
params = DEFAULT_PARAMS[0].copy()
params["trading_fee"] = 0.0

validator.strategy_manager.set_manual_parameters(ma_params=params)
strategies = validator.strategy_manager.initialize_strategies(["ma"])
strategy = strategies[0]

from trading_engine import TradingEngine
engine = TradingEngine(10000.0, 10.0, 2.0)

print("\n🚦 Running simulation...")
for j in range(len(validator.test_data)):
    current_data = validator.test_data.iloc[: j + 1]
    current_time = validator.test_data.index[j]
    engine.process_strategy_signals(strategy, current_data, current_time)

# Analyze leverage usage
trade_history = engine.get_trade_history_df()
closed_trades = trade_history[trade_history['status'].isin(['closed', 'tp_hit', 'sl_hit', 'reversed'])].copy()

if not closed_trades.empty:
    # Calculate statistics
    leverages = closed_trades['leverage'].dropna()
    avg_leverage = leverages.mean()
    min_leverage = leverages.min()
    max_leverage = leverages.max()
    
    print("\n📊 LEVERAGE ANALYSIS")
    print("=" * 80)
    print(f"Total Trades: {len(closed_trades)}")
    print(f"Average Leverage: {avg_leverage:.2f}x")
    print(f"Min Leverage: {min_leverage:.2f}x")
    print(f"Max Leverage: {max_leverage:.2f}x")
    print(f"\nLeverage Distribution:")
    print(leverages.describe())
    
    # Show leverage by ATR
    if 'atr' in closed_trades.columns:
        closed_trades['atr_pct'] = (closed_trades['atr'] / closed_trades['entry_price']) * 100
        print(f"\n📈 Leverage vs ATR Percentage:")
        for idx, row in closed_trades.head(20).iterrows():
            print(f"  Trade {row['id']}: ATR={row['atr_pct']:.3f}%, Leverage={row['leverage']:.2f}x")
    
    # Calculate what PnL would be with fixed 10x leverage
    print(f"\n💡 EXPLANATION:")
    print(f"  - Max leverage allowed: 10x")
    print(f"  - Actual average leverage: {avg_leverage:.2f}x")
    print(f"  - Reason: Dynamic leverage adjusts based on ATR to keep stop-loss ≤ 2%")
    print(f"  - This is why PnL is ~{avg_leverage:.1f}x instead of 10x")
    
    # Calculate theoretical 10x PnL
    unleveraged_pnl = 0.1882  # From optimization (18.82%)
    actual_pnl = 1.2188  # From mock validation (121.88%)
    expected_10x_pnl = unleveraged_pnl * 10
    
    print(f"\n🔢 THEORETICAL COMPARISON:")
    print(f"  - Unleveraged PnL (optimization): {unleveraged_pnl:.2%}")
    print(f"  - Expected with 10x leverage: {expected_10x_pnl:.2%} ({expected_10x_pnl:.1f}x)")
    print(f"  - Actual PnL (mock validation): {actual_pnl:.2%} ({actual_pnl/unleveraged_pnl:.1f}x)")
    print(f"  - Average leverage used: {avg_leverage:.2f}x")
    print(f"  - Match: {'✅ Yes' if abs(avg_leverage - (actual_pnl/unleveraged_pnl)) < 1.0 else '⚠️  Close'}")

else:
    print("❌ No closed trades found")

