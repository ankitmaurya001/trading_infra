#!/usr/bin/env python3
"""
Run mock tests for the Moving Average (MA) strategy over a specified date range
with user-specified parameter sets. This script bypasses optimization and
directly simulates provided parameter configurations.

QUICK START:
  Edit the global variables at the top of this script (DEFAULT_SYMBOL, 
  DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_INTERVAL, DEFAULT_PARAMS)
  and simply run: python run_ma_mock_validation.py

Usage examples:

  - Using global defaults (edit DEFAULT_* variables in script):
    python run_ma_mock_validation.py

  - Override symbol only:
    python run_ma_mock_validation.py --symbol ETHUSDT

  - Single param set via JSON string:
    python run_ma_mock_validation.py \
      --symbol SOLUSDT \
      --start "2025-09-15 00:00:00" \
      --end "2025-10-15 00:00:00" \
      --interval 15m \
      --params '[{"short_window":10,"long_window":40,"risk_reward_ratio":3.0}]'

  - Multiple param sets via JSON file:
    python run_ma_mock_validation.py \
      --symbol ETHUSDT \
      --start "2025-10-01 00:00:00" \
      --end "2025-10-07 00:00:00" \
      --interval 15m \
      --params-file params_ma.json

The params must be a list of objects with keys:
  short_window (int), long_window (int), risk_reward_ratio (float)
Optional: trading_fee will be taken from the validator config if omitted.
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from comprehensive_strategy_validation import ComprehensiveStrategyValidator
from trading_engine import TradingEngine
from datetime import timedelta


# ============================================================================
# GLOBAL CONFIGURATION - Edit these values to set defaults
# ============================================================================
# You can override these via command-line arguments if needed
DEFAULT_SYMBOL = "BNBUSDT"
# use 30 days ago
DEFAULT_START_DATE = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S")
# use today's date
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
DEFAULT_INTERVAL = "15m"
DEFAULT_PARAMS = [
    {"short_window": 22, "long_window": 30, "risk_reward_ratio": 3.5}
]
# ============================================================================


def _load_params_list(params_str: str, params_file: str, default_params: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if params_str and params_file:
        raise ValueError("Provide either --params or --params-file, not both.")
    if params_str:
        parsed = json.loads(params_str)
        if not isinstance(parsed, list):
            raise ValueError("--params must be a JSON array of parameter objects")
        return parsed
    if params_file:
        with open(params_file, "r") as f:
            parsed = json.load(f)
        if not isinstance(parsed, list):
            raise ValueError("--params-file must contain a JSON array of parameter objects")
        return parsed
    # Use default params if provided
    if default_params is not None:
        return default_params
    raise ValueError("You must provide parameter sets via --params, --params-file, or set DEFAULT_PARAMS in the script")


def run_ma_mock(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    params_list: List[Dict[str, Any]],
    initial_balance: float = 10000.0,
    max_leverage: float = 10.0,
    max_loss_percent: float = 2.0,
    trading_fee: float = 0.001,
    mock_trading_delay: float = 0.0,
    output_dir: str = "ma_mock_results"
) -> Dict[str, Any]:
    print("🚀 MA STRATEGY MOCK VALIDATION")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Date Range: {start_date} → {end_date}")
    print(f"Interval: {interval}")
    print(f"Param sets: {len(params_list)}")
    
    # Auto-adjust max_loss_percent for 1x leverage to match optimization script
    adjusted_max_loss_percent = max_loss_percent
    if max_leverage == 1.0:
        adjusted_max_loss_percent = 100.0
        print(f"⚙️  Leverage is 1x: Auto-setting max_loss_percent to 100% (disables ATR-based position sizing)")
        print(f"   This ensures matching with optimization script behavior")
    
    print(f"📊 Max Leverage: {max_leverage}x")
    print(f"📊 Max Loss Percent: {adjusted_max_loss_percent}%")

    validator = ComprehensiveStrategyValidator(
        initial_balance=initial_balance,
        max_leverage=max_leverage,
        max_loss_percent=adjusted_max_loss_percent,
        trading_fee=trading_fee,
    )

    # Fetch data and use full range as test data (no train split)
    print("📥 Fetching data...")
    data = validator.data_fetcher.fetch_historical_data(
        symbol, start_date, end_date, interval=interval
    )
    if data.empty:
        raise ValueError(f"No data fetched for {symbol}")
    validator.train_data = data.iloc[:0].copy()
    validator.test_data = data.copy()
    validator.symbol = symbol
    validator.interval = interval
    print(f"✅ Successfully fetched {len(data)} data points")
    print(f"📅 Data range: {data.index[0]} to {data.index[-1]}")

    results: List[Dict[str, Any]] = []
    failed_results: List[Dict[str, Any]] = []  # Track failed parameter sets

    for idx, params in enumerate(params_list, start=1):
        print("-" * 80)
        print(f"🎯 Testing parameter set {idx}/{len(params_list)}: {params}")

        # Ensure trading_fee is present as required by the MA strategy
        params_with_fee = dict(params)
        if "trading_fee" not in params_with_fee:
            params_with_fee["trading_fee"] = validator.trading_fee
            print(f"🔄 Added trading_fee {validator.trading_fee} to parameters")

        ok = validator.strategy_manager.set_manual_parameters(ma_params=params_with_fee)
        if not ok:
            print(f"❌ Failed to set manual MA parameters for set {idx}")
            failed_results.append({
                "parameter_set": idx,
                "parameters": params,
                "error": "Failed to set manual MA parameters",
                "stage": "parameter_setup"
            })
            continue

        strategies = validator.strategy_manager.initialize_strategies(["ma"])
        if not strategies:
            print(f"❌ Failed to initialize MA strategy for set {idx}")
            failed_results.append({
                "parameter_set": idx,
                "parameters": params,
                "error": "Failed to initialize MA strategy",
                "stage": "strategy_initialization"
            })
            continue

        strategy = strategies[0]

        engine = TradingEngine(
            validator.initial_balance,
            validator.max_leverage,
            adjusted_max_loss_percent,  # Use adjusted value
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{validator.symbol}_ma_param_{idx}_mock_{timestamp}"
        engine.setup_logging(session_id, validator.symbol)

        print("🚦 Running simulation...")
        start_t = datetime.now()
        for j in range(len(validator.test_data)):
            current_data = validator.test_data.iloc[: j + 1]
            current_time = validator.test_data.index[j]
            engine.process_strategy_signals(strategy, current_data, current_time)
            if mock_trading_delay > 0:
                import time
                time.sleep(mock_trading_delay)
        duration_s = (datetime.now() - start_t).total_seconds()

        final_status = engine.get_current_status()
        trade_history = engine.get_trade_history_df()
        performance_metrics = engine.calculate_performance_metrics()

        result = {
            "parameter_set": idx,
            "parameters": params,
            "final_status": final_status,
            "trade_history": trade_history,
            "performance_metrics": performance_metrics,
            "simulation_duration_sec": duration_s,
            "session_id": session_id,
        }
        results.append(result)

        print(f"✅ Done. Final Balance: ${final_status['current_balance']:,.2f}")
        print(f"📋 Performance Metrics: {performance_metrics}")
        if "total_pnl" in performance_metrics:
            print(f"📈 Total PnL: {performance_metrics['total_pnl']:.2%}")
        if "win_rate" in performance_metrics:
            print(f"🎯 Win Rate: {performance_metrics['win_rate']:.2%}")
        if "sharpe_ratio" in performance_metrics:
            print(f"📊 Sharpe: {performance_metrics['sharpe_ratio']:.3f}")
        if "max_drawdown" in performance_metrics:
            print(f"📉 Max DD: {performance_metrics['max_drawdown']:.2%}")

    # Save outputs
    if results:
        os.makedirs(output_dir, exist_ok=True)
        out_json = os.path.join(
            output_dir, f"ma_mock_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Convert trade_history DataFrames to CSV paths and dict references
        serializable = []
        for r in results:
            csv_path = os.path.join(output_dir, f"{r['session_id']}_trades.csv")
            r["trade_history"].to_csv(csv_path, index=False)
            serializable.append(
                {
                    "parameter_set": r["parameter_set"],
                    "parameters": r["parameters"],
                    "final_status": r["final_status"],
                    "performance_metrics": r["performance_metrics"],
                    "simulation_duration_sec": r["simulation_duration_sec"],
                    "session_id": r["session_id"],
                    "trade_history_csv": csv_path,
                }
            )

        with open(out_json, "w") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "interval": interval,
                    "results": serializable,
                    "failed_results": failed_results,  # Include failed parameter sets
                },
                f,
                indent=2,
                default=str,
            )
        print(f"💾 Saved results to {out_json}")

    # Report failed parameter sets
    if failed_results:
        print(f"\n⚠️  {len(failed_results)} parameter set(s) failed:")
        for failed in failed_results:
            print(f"   - Set {failed['parameter_set']}: {failed['error']} (stage: {failed['stage']})")

    print("\n🎯 Completed MA mock validation.")
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "results": results,
        "failed_results": failed_results,  # Include failed parameter sets in return value
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MA strategy mock tests with explicit params")
    p.add_argument("--symbol", default=None, help=f"Symbol, e.g., ETHUSDT (default: {DEFAULT_SYMBOL})")
    p.add_argument("--start", default=None, help=f"Start datetime 'YYYY-MM-DD HH:MM:SS' (default: {DEFAULT_START_DATE})")
    p.add_argument("--end", default=None, help=f"End datetime 'YYYY-MM-DD HH:MM:SS' (default: {DEFAULT_END_DATE})")
    p.add_argument("--interval", default=None, help=f"Interval, e.g., 1m, 5m, 15m (default: {DEFAULT_INTERVAL})")
    p.add_argument("--params", help="JSON array string of parameter objects (overrides DEFAULT_PARAMS)")
    p.add_argument("--params-file", help="Path to JSON file with parameter array (overrides DEFAULT_PARAMS)")
    p.add_argument("--initial-balance", type=float, default=10000.0)
    p.add_argument("--max-leverage", type=float, default=10.0)
    p.add_argument("--max-loss-percent", type=float, default=2.0)
    p.add_argument("--trading-fee", type=float, default=0.0)
    p.add_argument("--mock-delay", type=float, default=0.0, help="Seconds between ticks")
    p.add_argument("--out", default="ma_mock_results", help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Use global defaults if CLI args not provided
    symbol = args.symbol if args.symbol is not None else DEFAULT_SYMBOL
    start_date = args.start if args.start is not None else DEFAULT_START_DATE
    end_date = args.end if args.end is not None else DEFAULT_END_DATE
    interval = args.interval if args.interval is not None else DEFAULT_INTERVAL
    
    # Load params - use global default if no CLI params provided
    params_list = _load_params_list(args.params, args.params_file, default_params=DEFAULT_PARAMS)
    
    run_ma_mock(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        params_list=params_list,
        initial_balance=args.initial_balance,
        max_leverage=args.max_leverage,
        max_loss_percent=args.max_loss_percent,
        trading_fee=args.trading_fee,
        mock_trading_delay=args.mock_delay,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()


