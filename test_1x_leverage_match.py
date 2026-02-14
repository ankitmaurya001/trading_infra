#!/usr/bin/env python3
"""
Test if run_ma_optimization.py and run_ma_mock_validation.py match at 1x leverage.
"""

from datetime import datetime, timedelta
from comprehensive_strategy_validation import ComprehensiveStrategyValidator
from trading_engine import TradingEngine
from ma_3d_optimization_visualizer import MAOptimization3DVisualizer
import pandas as pd

print("🧪 TESTING 1X LEVERAGE MATCH")
print("=" * 80)

# Same config
symbol = "BTCUSDT"
start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
interval = "15m"
params = {"short_window": 10, "long_window": 40, "risk_reward_ratio": 4.0, "trading_fee": 0.0}

# Fetch data
validator = ComprehensiveStrategyValidator(
    initial_balance=10000.0,
    max_leverage=1.0,  # 1x leverage
    max_loss_percent=2.0,
    trading_fee=0.0,
)

print("📥 Fetching data...")
data = validator.data_fetcher.fetch_historical_data(symbol, start_date, end_date, interval=interval)
if data.empty:
    raise ValueError(f"No data fetched for {symbol}")

print(f"✅ Fetched {len(data)} data points\n")

# Method 1: run_ma_optimization.py approach (strategy.get_strategy_metrics)
print("🔍 Method 1: Strategy.get_strategy_metrics() (optimization approach)")
print("-" * 80)
from strategies import MovingAverageCrossover

strategy1 = MovingAverageCrossover(
    short_window=params["short_window"],
    long_window=params["long_window"],
    risk_reward_ratio=params["risk_reward_ratio"],
    trading_fee=params["trading_fee"]
)

signals = strategy1.generate_signals(data)
metrics1 = strategy1.get_strategy_metrics()

print(f"Total PnL: {metrics1.get('total_pnl', 0):.4%}")
print(f"Total Trades: {metrics1.get('total_trades', 0)}")
print(f"Win Rate: {metrics1.get('win_rate', 0):.2%}")
print(f"Sharpe Ratio: {metrics1.get('sharpe_ratio', 0):.3f}")

# Method 2: run_ma_mock_validation.py approach (TradingEngine with 1x leverage)
print("\n🔍 Method 2: TradingEngine with 1x leverage (mock validation approach)")
print("-" * 80)

validator2 = ComprehensiveStrategyValidator(
    initial_balance=10000.0,
    max_leverage=1.0,  # 1x leverage
    max_loss_percent=2.0,
    trading_fee=0.0,
)
validator2.train_data = data.iloc[:0].copy()
validator2.test_data = data.copy()

validator2.strategy_manager.set_manual_parameters(ma_params=params)
strategies2 = validator2.strategy_manager.initialize_strategies(["ma"])
strategy2 = strategies2[0]

engine = TradingEngine(10000.0, 1.0, 2.0)  # 1x leverage

for j in range(len(data)):
    current_data = data.iloc[: j + 1]
    current_time = data.index[j]
    engine.process_strategy_signals(strategy2, current_data, current_time)

metrics2 = engine.calculate_performance_metrics()
final_status = engine.get_current_status()

print(f"Total PnL: {metrics2.get('total_pnl', 0):.4%}")
print(f"Total Trades: {metrics2.get('total_trades', 0)}")
print(f"Win Rate: {metrics2.get('win_rate', 0):.2%}")
print(f"Sharpe Ratio: {metrics2.get('sharpe_ratio', 0):.3f}")
print(f"Final Balance: ${final_status['current_balance']:,.2f}")

# Comparison
print("\n📊 COMPARISON")
print("=" * 80)
pnl1 = metrics1.get('total_pnl', 0)
pnl2 = metrics2.get('total_pnl', 0)

print(f"Method 1 (strategy.get_strategy_metrics): {pnl1:.4%}")
print(f"Method 2 (TradingEngine 1x):             {pnl2:.4%}")
print(f"\nDifference: {abs(pnl1 - pnl2):.4%} ({abs(pnl1 - pnl2) / max(abs(pnl1), 0.0001) * 100:.2f}% relative difference)")

if abs(pnl1 - pnl2) < 0.01:  # Within 1%
    print("✅ Methods 1 and 2 MATCH (within 1%)")
else:
    print("⚠️  Methods 1 and 2 DO NOT MATCH")
    print("\n💡 Potential reasons:")
    print("  - TradingEngine applies ATR-based position sizing even at 1x leverage")
    print("  - TradingEngine manages balance differently (subtracts margin)")
    print("  - Strategy.get_strategy_metrics() uses full position size assumption")

# Analyze trade-by-trade differences
print("\n🔬 TRADE ANALYSIS")
print("-" * 80)
trade_history = engine.get_trade_history_df()
closed_trades = trade_history[trade_history['status'].isin(['closed', 'tp_hit', 'sl_hit', 'reversed'])].copy()

if not closed_trades.empty and len(strategy1.closed_trades) == len(closed_trades):
    print(f"Comparing {len(closed_trades)} trades...")
    differences = []
    for i, (trade1, trade2) in enumerate(zip(strategy1.closed_trades, closed_trades.itertuples())):
        pnl_diff = abs(trade1.pnl - trade2.pnl)
        differences.append(pnl_diff)
        if i < 5:  # Show first 5
            print(f"  Trade {i+1}: Strategy={trade1.pnl:.4%}, Engine={trade2.pnl:.4%}, Diff={pnl_diff:.4%}")
    
    avg_diff = sum(differences) / len(differences) if differences else 0
    print(f"\nAverage PnL difference per trade: {avg_diff:.4%}")
else:
    print(f"⚠️  Trade count mismatch: Strategy={len(strategy1.closed_trades)}, Engine={len(closed_trades)}")

