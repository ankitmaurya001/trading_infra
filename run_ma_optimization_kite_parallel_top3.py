#!/usr/bin/env python3
"""
Parallel MA optimization runner for Kite focused on configurable top-N selection.

Compared to run_ma_optimization_kite_parallel.py, this script:
1. Generates only the MA Optimal Regions Composite Score chart.
2. Selects configurable top-N parameter sets from BEST PARAMETERS BY RR.
3. Ranks those selections by composite score (descending).
"""

import os
import numpy as np
from datetime import datetime, timedelta

from ma_3d_optimization_visualizer import MAOptimization3DVisualizer
from run_ma_optimization_kite_parallel import (
    fetch_real_data,
    generate_sample_data,
    run_parallel_optimization_grid,
)

# ============================================================================
# GLOBAL CONFIGURATION - Edit these values to set defaults for this script
# ============================================================================
SYMBOL = "NATGASMINI26FEBFUT"
EXCHANGE = "MCX"
DAYS_TO_FETCH = 30
INTERVAL = "15m"
MAX_WORKERS = None  # None = use all available CPUs
TOP_N = 3


def _build_best_by_rr_rows(results: dict, metric: str = "composite_score") -> list:
    """Build rows for the "best parameters by RR" table from optimization results."""
    rows = []
    for rr, data in results.items():
        if not data.get(metric):
            continue
        best_idx = int(np.argmax(data[metric]))
        rows.append(
            {
                "risk_reward_ratio": rr,
                "short_window": int(data["short_windows"][best_idx]),
                "long_window": int(data["long_windows"][best_idx]),
                "score": float(data[metric][best_idx]),
                "total_pnl": float(data["total_pnl"][best_idx]),
                "sharpe_ratio": float(data["sharpe_ratio"][best_idx]),
                "total_trades": int(data["total_trades"][best_idx]),
            }
        )
    return sorted(rows, key=lambda r: r["risk_reward_ratio"])


def select_top_n_from_best_by_rr(results: dict, top_n: int, metric: str = "composite_score") -> list:
    """Select configurable top-N rows from BEST PARAMETERS BY RR sorted by score."""
    best_by_rr_rows = _build_best_by_rr_rows(results, metric=metric)
    sorted_rows = sorted(best_by_rr_rows, key=lambda row: row["score"], reverse=True)
    return sorted_rows[: max(1, top_n)]


def main() -> None:
    print("🚀 PARALLEL MA OPTIMIZATION (KITE) - CONFIGURABLE TOP N")
    print("=" * 58)

    print("\n📊 Data Source Options:")
    print(f"1. Real Kite Data ({SYMBOL}, {EXCHANGE}, {DAYS_TO_FETCH} days, {INTERVAL})")
    print("2. Demo Data (Sample)")
    print("3. Real Kite Data (Custom)")

    choice = input("\nEnter your choice (1-3): ").strip()
    end_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    if choice == "1":
        data = fetch_real_data(
            symbol=SYMBOL,
            days=DAYS_TO_FETCH,
            interval=INTERVAL,
            exchange=EXCHANGE,
            start_date=start_date,
            end_date=end_date,
        )
    elif choice == "2":
        data = generate_sample_data()
    elif choice == "3":
        symbol = (
            input("Enter symbol (e.g., TATAMOTORS, RELIANCE, TCS): ").strip().upper()
        )
        exchange = (
            input(f"Enter exchange (NSE, BSE, MCX, default {EXCHANGE}): ")
            .strip()
            .upper()
            or EXCHANGE
        )
        interval = input(f"Enter interval (default {INTERVAL}): ").strip() or INTERVAL
        use_date_range = input("Use explicit date range? (y/N): ").strip().lower() in (
            "y",
            "yes",
        )
        if use_date_range:
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter end date (YYYY-MM-DD, blank = tomorrow): ").strip()
            days = input(
                f"Fallback days if one date is missing (default {DAYS_TO_FETCH}): "
            ).strip()
            days = int(days) if days.isdigit() else DAYS_TO_FETCH
            data = fetch_real_data(
                symbol=symbol,
                days=days,
                interval=interval,
                exchange=exchange,
                start_date=start_date or None,
                end_date=end_date or None,
            )
        else:
            days = input(f"Enter number of days (default {DAYS_TO_FETCH}): ").strip()
            days = int(days) if days.isdigit() else DAYS_TO_FETCH
            data = fetch_real_data(
                symbol=symbol, days=days, interval=interval, exchange=exchange
            )
    else:
        print("❌ Invalid choice. Using demo data...")
        data = generate_sample_data()

    if data is None or data.empty:
        print("❌ No data available. Exiting...")
        return

    print("\n⚙️  Parallel Processing Configuration:")
    print(f"   Available CPUs: {os.cpu_count()}")
    workers_input = input(
        "   Number of workers (default: all CPUs, or enter number): "
    ).strip()
    max_workers = int(workers_input) if workers_input.isdigit() else MAX_WORKERS

    top_n_input = input(f"   Number of top parameter sets to show (default {TOP_N}): ").strip()
    top_n = int(top_n_input) if top_n_input.isdigit() else TOP_N

    visualizer = MAOptimization3DVisualizer(
        data,
        trading_fee=0.0,
        auto_open=True,
        output_dir="ma_optimization_plots_kite",
    )

    short_window_range = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
    long_window_range = [
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
    ]
    risk_reward_ratios = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]

    print("\n🔍 Running parallel optimization...")
    print(f"   Short windows: {list(short_window_range)}")
    print(f"   Long windows: {list(long_window_range)}")
    print(f"   Risk-reward ratios: {risk_reward_ratios}")

    results = run_parallel_optimization_grid(
        data=data,
        short_window_range=short_window_range,
        long_window_range=long_window_range,
        risk_reward_ratios=risk_reward_ratios,
        trading_fee=0.0,
        max_workers=max_workers,
    )

    visualizer.results = results

    print("\n📊 Creating only MA Optimal Regions Composite Score chart...")
    visualizer.create_optimal_regions_plot(
        metric="composite_score", percentile_threshold=80.0
    )

    top_params = select_top_n_from_best_by_rr(
        visualizer.results,
        top_n=top_n,
        metric="composite_score",
    )

    print("\n🏁 TOP PARAMETER SETS (From BEST PARAMETERS BY RR sorted by Score):")
    print("-" * 108)
    if not top_params:
        print("No top parameters found.")
        return

    for idx, row in enumerate(top_params, start=1):
        print(
            f"{idx}. Short={row['short_window']:>2}, Long={row['long_window']:>3} | "
            f"RR={row['risk_reward_ratio']:>3.1f} | Score={row['score']:>8.2f} | "
            f"PnL={row['total_pnl']:>8.2%} | Sharpe={row['sharpe_ratio']:>7.4f} | "
            f"Trades={row['total_trades']}"
        )


if __name__ == "__main__":
    main()
