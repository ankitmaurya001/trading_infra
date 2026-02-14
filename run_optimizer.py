from data_fetcher import DataFetcher
from strategy_optimizer import (
    optimize_moving_average_crossover, print_optimization_report,
    optimize_rsi_strategy, SHORT_WINDOW_RANGE, LONG_WINDOW_RANGE, RISK_REWARD_RANGE,
    RSI_PERIOD_RANGE, overbought_range, oversold_range
)
import itertools

TRADING_FEE = 0.0003 # 0.03% kite charges or flat 20rs

# Parameters for data fetching
symbol = "TATAMOTORS.NS"
start_date = "2025-06-01"
end_date = "2025-07-20"
interval = "5m"

print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
fetcher = DataFetcher()
data = fetcher.fetch_data(symbol, start_date, end_date, interval=interval)

data.to_csv(f"{symbol}_{start_date}_{end_date}_{interval}.csv")
if data.empty:
    print(f"No data fetched for {symbol}")
else:
    print(f"Fetched {len(data)} rows of data for {symbol}.")

    print("\n--- Moving Average Crossover Optimization ---")
    n_ma = sum(1 for _ in itertools.product(SHORT_WINDOW_RANGE, LONG_WINDOW_RANGE, RISK_REWARD_RANGE) if _[0] < _[1])
    print(f"Testing {n_ma} parameter combinations for Moving Average Crossover...")
    best_params_ma, results_df_ma = optimize_moving_average_crossover(data, trading_fee=TRADING_FEE)
    print("Moving Average Crossover optimization complete.")
    print_optimization_report(best_params_ma, results_df_ma)

    print("\n--- RSI Strategy Optimization ---")
    n_rsi = sum(1 for _ in itertools.product(RSI_PERIOD_RANGE, overbought_range, oversold_range, RISK_REWARD_RANGE) if _[2] < _[1])
    print(f"Testing {n_rsi} parameter combinations for RSI Strategy...")
    best_params_rsi, results_df_rsi = optimize_rsi_strategy(data, trading_fee=TRADING_FEE)
    print("RSI Strategy optimization complete.")
    print_optimization_report(best_params_rsi, results_df_rsi) 