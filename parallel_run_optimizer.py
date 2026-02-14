import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_fetcher import DataFetcher
from strategy_optimizer import (
    SHORT_WINDOW_RANGE, LONG_WINDOW_RANGE, RISK_REWARD_RANGE,
    RSI_PERIOD_RANGE, overbought_range, oversold_range,
    DONCHIAN_PERIOD_RANGE, print_optimization_report
)
from strategies import MovingAverageCrossover, RSIStrategy, DonchianChannelBreakout

#TRADING_FEE = 0.0003 # 0.03% kite charges or flat 20rs
TRADING_FEE = 0

def evaluate_ma_combo(args):
    data, short_w, long_w, rr = args
    strategy = MovingAverageCrossover(short_window=short_w, long_window=long_w, risk_reward_ratio=rr, trading_fee=TRADING_FEE)
    strategy.generate_signals(data)
    metrics = strategy.get_strategy_metrics()
    return {
        "short_window": short_w,
        "long_window": long_w,
        "risk_reward_ratio": rr,
        **metrics
    }

def parallel_optimize_moving_average_crossover(data, metric="total_pnl"):
    combos = [(data, short_w, long_w, rr)
              for short_w, long_w, rr in itertools.product(SHORT_WINDOW_RANGE, LONG_WINDOW_RANGE, RISK_REWARD_RANGE)
              if short_w < long_w]
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_ma_combo, combo) for combo in combos]
        for future in as_completed(futures):
            results.append(future.result())
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return {}, results_df
    best_idx = results_df[metric].idxmax()
    best_params = results_df.loc[best_idx].to_dict()
    return best_params, results_df

def evaluate_rsi_combo(args):
    data, period, overbought, oversold, rr = args
    strategy = RSIStrategy(period=period, overbought=overbought, oversold=oversold, risk_reward_ratio=rr, trading_fee=TRADING_FEE)
    strategy.generate_signals(data)
    metrics = strategy.get_strategy_metrics()
    return {
        "period": period,
        "overbought": overbought,
        "oversold": oversold,
        "risk_reward_ratio": rr,
        **metrics
    }

def parallel_optimize_rsi_strategy(data, metric="total_pnl"):
    combos = [(data, period, overbought, oversold, rr)
              for period, overbought, oversold, rr in itertools.product(RSI_PERIOD_RANGE, overbought_range, oversold_range, RISK_REWARD_RANGE)
              if oversold < overbought]
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_rsi_combo, combo) for combo in combos]
        for future in as_completed(futures):
            results.append(future.result())
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return {}, results_df
    best_idx = results_df[metric].idxmax()
    best_params = results_df.loc[best_idx].to_dict()
    return best_params, results_df

def evaluate_donchian_combo(args):
    data, period, rr = args
    strategy = DonchianChannelBreakout(channel_period=period, risk_reward_ratio=rr, trading_fee=TRADING_FEE)
    strategy.generate_signals(data)
    metrics = strategy.get_strategy_metrics()
    return {
        "channel_period": period,
        "risk_reward_ratio": rr,
        **metrics
    }

def parallel_optimize_donchian_channel_breakout(data, metric="total_pnl"):
    combos = [(data, period, rr)
              for period, rr in itertools.product(DONCHIAN_PERIOD_RANGE, RISK_REWARD_RANGE)]
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_donchian_combo, combo) for combo in combos]
        for future in as_completed(futures):
            results.append(future.result())
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return {}, results_df
    best_idx = results_df[metric].idxmax()
    best_params = results_df.loc[best_idx].to_dict()
    return best_params, results_df

if __name__ == "__main__":
    # Parameters for data fetching
    symbol = "BTC-USD"
    start_date = "2025-06-15"
    #end_date = "2025-07-20"
    end_date = "2025-07-15"
    interval = "15m"

    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    fetcher = DataFetcher()
    data = fetcher.fetch_data(symbol, start_date, end_date, interval=interval)

    data.to_csv(f"data/{symbol}_{start_date}_{end_date}_{interval}.csv")
    if data.empty:
        print(f"No data fetched for {symbol}")
    else:
        print(f"Fetched {len(data)} rows of data for {symbol}.")

        print("\n--- Moving Average Crossover Parallel Optimization ---")
        n_ma = sum(1 for _ in itertools.product(SHORT_WINDOW_RANGE, LONG_WINDOW_RANGE, RISK_REWARD_RANGE) if _[0] < _[1])
        print(f"Testing {n_ma} parameter combinations for Moving Average Crossover (parallel)...")
        best_params_ma, results_df_ma = parallel_optimize_moving_average_crossover(data)
        print("Moving Average Crossover parallel optimization complete.")
        print_optimization_report(best_params_ma, results_df_ma)

        # print("\n--- RSI Strategy Parallel Optimization ---")
        # n_rsi = sum(1 for _ in itertools.product(RSI_PERIOD_RANGE, overbought_range, oversold_range, RISK_REWARD_RANGE) if _[2] < _[1])
        # print(f"Testing {n_rsi} parameter combinations for RSI Strategy (parallel)...")
        # best_params_rsi, results_df_rsi = parallel_optimize_rsi_strategy(data)
        # print("RSI Strategy parallel optimization complete.")
        # print_optimization_report(best_params_rsi, results_df_rsi)

        print("\n--- Donchian Channel Breakout Parallel Optimization ---")
        n_donchian = sum(1 for _ in itertools.product(DONCHIAN_PERIOD_RANGE, RISK_REWARD_RANGE))
        print(f"Testing {n_donchian} parameter combinations for Donchian Channel Breakout (parallel)...")
        best_params_donchian, results_df_donchian = parallel_optimize_donchian_channel_breakout(data)
        print("Donchian Channel Breakout parallel optimization complete.")
        print_optimization_report(best_params_donchian, results_df_donchian) 