#!/usr/bin/env python3
"""
Parallel MA optimization script focused on selecting top-3 robust parameter sets.

Selection logic:
1) Run the same parallel optimization grid as run_ma_optimization_kite_parallel.py
2) Take top-3 entries from "most robust parameters"
3) For each (short, long), choose RR from "best parameters by RR" where the pair matches
   and has the highest score for that pair
4) Print final top-3 parameter sets with score, PnL, Sharpe, and trades

Plotting:
- Generates ONLY the MA Optimal Regions Composite Score chart.
"""

import os
import numpy as np

from ma_3d_optimization_visualizer import MAOptimization3DVisualizer
from run_ma_optimization_kite_parallel import (
    SYMBOL,
    EXCHANGE,
    DAYS_TO_FETCH,
    INTERVAL,
    MAX_WORKERS,
    fetch_real_data,
    generate_sample_data,
    run_parallel_optimization_grid,
)


def build_best_by_rr(results: dict, metric: str = 'composite_score') -> dict:
    """Return best result per RR ratio."""
    best_by_rr = {}
    for rr, rr_data in results.items():
        if not rr_data.get(metric):
            continue
        best_idx = int(np.argmax(rr_data[metric]))
        best_by_rr[rr] = {
            'risk_reward_ratio': rr,
            'short_window': int(rr_data['short_windows'][best_idx]),
            'long_window': int(rr_data['long_windows'][best_idx]),
            'score': float(rr_data[metric][best_idx]),
            'total_pnl': float(rr_data['total_pnl'][best_idx]),
            'sharpe_ratio': float(rr_data['sharpe_ratio'][best_idx]),
            'total_trades': int(rr_data['total_trades'][best_idx]),
        }
    return best_by_rr


def find_best_rr_for_pair_from_best_by_rr(short_w: int, long_w: int, best_by_rr: dict):
    """Pick RR for a short/long pair using only best-by-RR rows."""
    matches = [
        row for row in best_by_rr.values()
        if row['short_window'] == short_w and row['long_window'] == long_w
    ]
    if not matches:
        return None
    return max(matches, key=lambda row: row['score'])


def fallback_best_rr_for_pair_from_full_results(short_w: int, long_w: int, results: dict,
                                                metric: str = 'composite_score'):
    """Fallback: if pair not present in best-by-RR list, use best RR from full grid."""
    candidates = []
    for rr, rr_data in results.items():
        for i in range(len(rr_data['short_windows'])):
            if int(rr_data['short_windows'][i]) == short_w and int(rr_data['long_windows'][i]) == long_w:
                candidates.append({
                    'risk_reward_ratio': rr,
                    'short_window': short_w,
                    'long_window': long_w,
                    'score': float(rr_data[metric][i]),
                    'total_pnl': float(rr_data['total_pnl'][i]),
                    'sharpe_ratio': float(rr_data['sharpe_ratio'][i]),
                    'total_trades': int(rr_data['total_trades'][i]),
                })
    if not candidates:
        return None
    return max(candidates, key=lambda row: row['score'])


def main():
    print("🚀 PARALLEL MA OPTIMIZATION (TOP-3 ROBUST PARAMS)")
    print("=" * 60)

    print("\n📊 Data Source Options:")
    print(f"1. Real Kite Data ({SYMBOL}, {EXCHANGE}, {DAYS_TO_FETCH} days, {INTERVAL})")
    print("2. Demo Data (Sample)")
    print("3. Real Kite Data (Custom)")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        data = fetch_real_data(symbol=SYMBOL, days=DAYS_TO_FETCH, interval=INTERVAL, exchange=EXCHANGE)
    elif choice == "2":
        data = generate_sample_data()
    elif choice == "3":
        symbol = input("Enter symbol (e.g., TATAMOTORS, RELIANCE, TCS): ").strip().upper()
        exchange = input(f"Enter exchange (NSE, BSE, MCX, default {EXCHANGE}): ").strip().upper() or EXCHANGE
        days = input(f"Enter number of days (default {DAYS_TO_FETCH}): ").strip()
        days = int(days) if days.isdigit() else DAYS_TO_FETCH
        interval = input(f"Enter interval (default {INTERVAL}): ").strip() or INTERVAL
        data = fetch_real_data(symbol=symbol, days=days, interval=interval, exchange=exchange)
    else:
        print("❌ Invalid choice. Using demo data...")
        data = generate_sample_data()

    if data is None or data.empty:
        print("❌ No data available. Exiting...")
        return

    print("\n⚙️  Parallel Processing Configuration:")
    print(f"   Available CPUs: {os.cpu_count()}")
    workers_input = input("   Number of workers (default: all CPUs, or enter number): ").strip()
    max_workers = int(workers_input) if workers_input.isdigit() else MAX_WORKERS

    visualizer = MAOptimization3DVisualizer(
        data,
        trading_fee=0.0,
        auto_open=False,
        output_dir="ma_optimization_plots_kite_top3"
    )

    SHORT_VAL = 5
    LONG_VAL = 180
    RANGE = 9

    short_start = max(SHORT_VAL - RANGE, 4)
    short_end = SHORT_VAL + RANGE
    long_start = max(LONG_VAL - RANGE, SHORT_VAL + RANGE + 1)
    long_end = LONG_VAL + RANGE
    short_window_range = np.arange(short_start, short_end, 1)
    long_window_range = np.arange(long_start, long_end, 1)

    risk_reward_ratios = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]

    print("\n🔍 Running parallel optimization...")
    results = run_parallel_optimization_grid(
        data=data,
        short_window_range=short_window_range,
        long_window_range=long_window_range,
        risk_reward_ratios=risk_reward_ratios,
        trading_fee=0.0,
        max_workers=max_workers,
    )

    visualizer.results = results

    print("\n📊 Optimization summary:")
    recommendations = visualizer.create_parameter_recommendations(metric='composite_score')

    print("\n🎨 Generating MA Optimal Regions Composite Score chart only...")
    visualizer.create_optimal_regions_plot(metric='composite_score', percentile_threshold=80.0)

    robust_parameters = recommendations.get('robustness_analysis', {}).get('robust_parameters', [])
    top_robust_3 = robust_parameters[:3]
    best_by_rr = build_best_by_rr(results, metric='composite_score')

    print("\n🏁 TOP 3 PARAMETER SETS (Robust + RR-selected)")
    print("-" * 70)

    final_rows = []
    for _, robust_data in top_robust_3:
        short_w = int(robust_data['short_window'])
        long_w = int(robust_data['long_window'])

        selected = find_best_rr_for_pair_from_best_by_rr(short_w, long_w, best_by_rr)
        if selected is None:
            selected = fallback_best_rr_for_pair_from_full_results(
                short_w, long_w, results, metric='composite_score'
            )

        if selected:
            final_rows.append(selected)

    if not final_rows:
        print("No top-3 rows could be derived from the optimization output.")
        return

    for i, row in enumerate(final_rows, 1):
        print(
            f"{i}. Short{ i }={row['short_window']}, Long{ i }={row['long_window']}, "
            f"RR{ i }={row['risk_reward_ratio']}, Score{ i }={row['score']:.2f}, "
            f"PNL{ i }={row['total_pnl']:.2%}, Sharpe{ i }={row['sharpe_ratio']:.4f}, "
            f"Trades{ i }={row['total_trades']}"
        )


if __name__ == "__main__":
    main()
