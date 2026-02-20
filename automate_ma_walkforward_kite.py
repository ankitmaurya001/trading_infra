#!/usr/bin/env python3
"""Automate walk-forward MA optimization + majority-vote validation on Kite data."""

import argparse
import contextlib
import io
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ma_3d_optimization_visualizer import MAOptimization3DVisualizer
from run_ma_mock_validation_majority_kite import (
    calculate_performance_metrics,
    create_cumulative_pnl_chart,
    create_ohlc_trade_chart,
    extract_trade_history,
    fetch_kite_data,
    run_majority_vote_validation,
)
from run_ma_optimization_kite_parallel import run_parallel_optimization_grid
from run_ma_optimization_kite_parallel_top3 import select_top_n_from_best_by_rr


SHORT_WINDOW_RANGE = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
LONG_WINDOW_RANGE = [
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
RISK_REWARD_RATIOS = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]

# Global defaults for CLI-required runtime fields
DEFAULT_SYMBOL = "NATGASMINI26FEBFUT"
DEFAULT_EXCHANGE = "MCX"
DEFAULT_START_DATE = (date.today() - timedelta(days=150)).strftime("%Y-%m-%d")
DEFAULT_OPTIMIZATION_DAYS = 30
DEFAULT_VALIDATION_DAYS = 30
DEFAULT_MAX_CONSECUTIVE_LOSSES = 3
TOP_N = 1


@dataclass
class IterationSummary:
    iteration: int
    optimization_start: str
    optimization_end: str
    validation_start: str
    validation_end: str
    validation_requested_end: str
    validation_stop_reason: str
    optimization_points: int
    validation_points: int
    selected_params: List[Dict]
    validation_metrics: Dict
    validation_trades: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward automation: optimize on window N, validate on next M days."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--exchange", default=DEFAULT_EXCHANGE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="YYYY-MM-DD")
    parser.add_argument(
        "--optimization-days", type=int, default=DEFAULT_OPTIMIZATION_DAYS
    )
    parser.add_argument("--validation-days", type=int, default=DEFAULT_VALIDATION_DAYS)
    parser.add_argument(
        "--validation-stop-mode",
        choices=["time", "result"],
        default="time",
        help="How to stop validation: fixed days (time) or trade outcome (result).",
    )
    parser.add_argument(
        "--validation-stop-days",
        type=int,
        default=None,
        help="When stop-mode=time, stop validation after this many days (defaults to --validation-days).",
    )
    parser.add_argument(
        "--max-consecutive-losses",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_LOSSES,
        help="When stop-mode=result, stop after this many consecutive losing closed trades.",
    )
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--out", default="ma_walkforward_results_kite")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def to_iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def run_iteration(
    iteration: int,
    symbol: str,
    exchange: str,
    interval: str,
    optimization_start: date,
    optimization_end: date,
    validation_start: date,
    requested_validation_end: date,
    validation_stop_mode: str,
    validation_stop_days: int,
    max_consecutive_losses: int,
    out_dir: Path,
    initial_balance: float,
    quiet: bool,
    max_workers: int,
) -> Tuple[IterationSummary, date]:
    iter_dir = out_dir / f"iteration_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔁 Iteration {iteration}")
    print(f"   Optimization: {optimization_start} -> {optimization_end}")
    print(
        f"   Validation:   {validation_start} -> {requested_validation_end} (requested)"
    )

    optimization_data = fetch_kite_data(
        symbol=symbol,
        exchange=exchange,
        start_date=to_iso(optimization_start),
        end_date=to_iso(optimization_end),
        interval=interval,
    )

    visualizer = MAOptimization3DVisualizer(
        optimization_data,
        trading_fee=0.0,
        auto_open=False,
        output_dir=str(iter_dir),
    )
    optimization_results = run_parallel_optimization_grid(
        data=optimization_data,
        short_window_range=SHORT_WINDOW_RANGE,
        long_window_range=LONG_WINDOW_RANGE,
        risk_reward_ratios=RISK_REWARD_RATIOS,
        trading_fee=0.0,
        max_workers=max_workers,
    )
    visualizer.results = optimization_results

    with contextlib.redirect_stdout(io.StringIO()):
        visualizer.create_optimal_regions_plot(
            metric="composite_score", percentile_threshold=80.0
        )
    top_n = select_top_n_from_best_by_rr(
        optimization_results,
        top_n=TOP_N,
        metric="composite_score",
    )
    if not top_n:
        raise RuntimeError(
            f"No top-n parameter sets were selected for iteration {iteration}"
        )

    top_n_path = iter_dir / "top_n_params.json"
    with open(top_n_path, "w", encoding="utf-8") as f:
        json.dump(top_n, f, indent=2)

    validation_data = fetch_kite_data(
        symbol=symbol,
        exchange=exchange,
        start_date=to_iso(validation_start),
        end_date=to_iso(requested_validation_end),
        interval=interval,
    )

    stop_reason = "max_window_reached"
    validation_end = requested_validation_end

    if validation_stop_mode == "result":
        filtered_result, stop_timestamp, stop_reason_from_engine = (
            run_majority_vote_validation(
                data=validation_data,
                param_sets=top_n,
                symbol=symbol,
                initial_balance=initial_balance,
                verbose=not quiet,
                mock_delay=0.0,
                stop_on_result=True,
                max_consecutive_losses=max_consecutive_losses,
                return_stop_metadata=True,
            )
        )
        if stop_reason_from_engine is not None:
            stop_reason = stop_reason_from_engine
        if stop_timestamp is not None:
            validation_end = stop_timestamp.date()
    else:
        result = run_majority_vote_validation(
            data=validation_data,
            param_sets=top_n,
            symbol=symbol,
            initial_balance=initial_balance,
            verbose=not quiet,
            mock_delay=0.0,
        )
        time_stop_date = min(
            validation_start + timedelta(days=validation_stop_days),
            requested_validation_end,
        )
        filtered_result = result[result.index.date <= time_stop_date]
        validation_end = time_stop_date
        stop_reason = f"time_stop_after_{validation_stop_days}_days"

    if filtered_result.empty:
        raise RuntimeError(
            f"Validation produced no rows after applying stop rule for iteration {iteration}."
        )

    result_csv_path = iter_dir / "validation_detail.csv"
    filtered_result.to_csv(result_csv_path)

    trades = extract_trade_history(filtered_result, initial_balance=initial_balance)
    trades_path = iter_dir / "validation_trades.csv"
    trades.to_csv(trades_path, index=False)

    metrics = calculate_performance_metrics(trades, initial_balance=initial_balance)
    metrics_path = iter_dir / "validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    total_trades = int(metrics.get("total_trades", 0))
    pnl_pct = float(metrics.get("total_pnl", 0.0)) * 100.0
    final_balance = float(metrics.get("final_balance", initial_balance))
    pnl_rupees = final_balance - initial_balance
    win_rate_pct = float(metrics.get("win_rate", 0.0)) * 100.0
    chart_summary = (
        f"Range: {validation_start} -> {validation_end} | "
        f"Trades: {total_trades} | "
        f"PnL: {pnl_pct:.2f}% (₹{pnl_rupees:+.2f}) | "
        f"Win Rate: {win_rate_pct:.2f}%"
    )

    create_cumulative_pnl_chart(
        trades=trades,
        initial_balance=initial_balance,
        output_path=str(iter_dir / "validation_pnl.html"),
        title_suffix=chart_summary,
        auto_open=True,
    )
    validation_ohlc = validation_data[validation_data.index.date <= validation_end]
    create_ohlc_trade_chart(
        ohlc=validation_ohlc,
        trades=trades,
        param_sets=top_n,
        symbol=symbol,
        exchange=exchange,
        output_path=str(iter_dir / "validation_ohlc_trades.html"),
        title_suffix=chart_summary,
        auto_open=True,
    )

    return (
        IterationSummary(
            iteration=iteration,
            optimization_start=to_iso(optimization_start),
            optimization_end=to_iso(optimization_end),
            validation_start=to_iso(validation_start),
            validation_end=to_iso(validation_end),
            validation_requested_end=to_iso(requested_validation_end),
            validation_stop_reason=stop_reason,
            optimization_points=len(optimization_data),
            validation_points=len(filtered_result),
            selected_params=top_n,
            validation_metrics=metrics,
            validation_trades=int(metrics.get("total_trades", 0)),
        ),
        validation_end,
    )


def build_combined_summary(
    iterations: List[IterationSummary], initial_balance: float
) -> Dict:
    total_trades = sum(i.validation_trades for i in iterations)
    total_return = sum(i.validation_metrics.get("total_pnl", 0.0) for i in iterations)
    avg_win_rate = (
        sum(i.validation_metrics.get("win_rate", 0.0) for i in iterations)
        / len(iterations)
        if iterations
        else 0.0
    )

    compounded_balance = initial_balance
    for row in iterations:
        compounded_balance *= 1 + row.validation_metrics.get("total_pnl", 0.0)

    return {
        "iterations": len(iterations),
        "total_trades": int(total_trades),
        "sum_of_iteration_returns": float(total_return),
        "average_win_rate": float(avg_win_rate),
        "initial_balance": float(initial_balance),
        "compounded_final_balance": float(compounded_balance),
        "compounded_total_return": float(
            (compounded_balance - initial_balance) / initial_balance
        ),
    }


def print_iteration_validation_summary(
    summary: IterationSummary, initial_balance: float
) -> None:
    total_trades = int(summary.validation_metrics.get("total_trades", 0))
    pnl_pct = float(summary.validation_metrics.get("total_pnl", 0.0)) * 100.0
    final_balance = float(
        summary.validation_metrics.get("final_balance", initial_balance)
    )
    pnl_rupees = final_balance - initial_balance
    win_rate_pct = float(summary.validation_metrics.get("win_rate", 0.0)) * 100.0

    print(
        f"📌 Validation Summary (Iteration {summary.iteration}) | "
        f"{summary.validation_start} -> {summary.validation_end} | "
        f"Trades={total_trades} | "
        f"PnL={pnl_pct:.2f}% (₹{pnl_rupees:+.2f}) | "
        f"Win Rate={win_rate_pct:.2f}%"
    )


def print_final_validation_summary(
    iterations: List[IterationSummary], initial_balance: float
) -> None:
    print("\n📋 FINAL VALIDATION SUMMARY")
    print("Iter | Date Range               | Trades | PnL%    | PnL ₹       | Win Rate")
    print("-----+--------------------------+--------+---------+-------------+---------")

    total_trades = 0
    total_pnl_rupees = 0.0
    weighted_win_numerator = 0.0

    for item in iterations:
        trades = int(item.validation_metrics.get("total_trades", 0))
        pnl_pct = float(item.validation_metrics.get("total_pnl", 0.0)) * 100.0
        final_balance = float(
            item.validation_metrics.get("final_balance", initial_balance)
        )
        pnl_rupees = final_balance - initial_balance
        win_rate_pct = float(item.validation_metrics.get("win_rate", 0.0)) * 100.0

        total_trades += trades
        total_pnl_rupees += pnl_rupees
        weighted_win_numerator += win_rate_pct * trades

        date_range = f"{item.validation_start} -> {item.validation_end}"
        print(
            f"{item.iteration:>4} | {date_range:<24} | "
            f"{trades:>6} | {pnl_pct:>6.2f}% | "
            f"₹{pnl_rupees:>+10.2f} | {win_rate_pct:>6.2f}%"
        )

    avg_win_rate_weighted = (
        weighted_win_numerator / total_trades if total_trades > 0 else 0.0
    )
    print("-----+--------------------------+--------+---------+-------------+---------")
    print(
        f"TOTAL| {'ALL ITERATIONS':<24} | "
        f"{total_trades:>6} | {'-':>6}   | "
        f"₹{total_pnl_rupees:>+10.2f} | {avg_win_rate_weighted:>6.2f}%"
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    today = date.today()

    if args.optimization_days <= 0 or args.validation_days <= 0:
        raise ValueError("optimization-days and validation-days must both be > 0")

    validation_stop_days = args.validation_stop_days or args.validation_days
    if validation_stop_days <= 0:
        raise ValueError("validation-stop-days must be > 0")
    if args.max_consecutive_losses <= 0:
        raise ValueError("max-consecutive-losses must be > 0")

    iterations: List[IterationSummary] = []
    cursor = start_date
    iteration = 1

    while True:
        optimization_start = cursor
        optimization_end = cursor + timedelta(days=args.optimization_days)
        validation_start = optimization_end
        requested_validation_end = validation_start + timedelta(
            days=args.validation_days
        )

        if validation_start >= today:
            break

        requested_validation_end = min(requested_validation_end, today)

        iteration_summary, actual_validation_end = run_iteration(
            iteration=iteration,
            symbol=args.symbol,
            exchange=args.exchange,
            interval=args.interval,
            optimization_start=optimization_start,
            optimization_end=optimization_end,
            validation_start=validation_start,
            requested_validation_end=requested_validation_end,
            validation_stop_mode=args.validation_stop_mode,
            validation_stop_days=validation_stop_days,
            max_consecutive_losses=args.max_consecutive_losses,
            out_dir=out_dir,
            initial_balance=args.initial_balance,
            quiet=args.quiet,
            max_workers=args.max_workers,
        )
        iterations.append(iteration_summary)
        print_iteration_validation_summary(
            iteration_summary, initial_balance=args.initial_balance
        )

        cursor = actual_validation_end
        iteration += 1

    if not iterations:
        raise RuntimeError(
            "No iterations were executed. Check start-date and day windows."
        )

    summary_rows = []
    for item in iterations:
        summary_rows.append(
            {
                "iteration": item.iteration,
                "optimization_start": item.optimization_start,
                "optimization_end": item.optimization_end,
                "validation_start": item.validation_start,
                "validation_end": item.validation_end,
                "validation_requested_end": item.validation_requested_end,
                "validation_stop_reason": item.validation_stop_reason,
                "validation_total_trades": item.validation_metrics.get(
                    "total_trades", 0
                ),
                "validation_win_rate": item.validation_metrics.get("win_rate", 0.0),
                "validation_total_pnl": item.validation_metrics.get("total_pnl", 0.0),
                "validation_final_balance": item.validation_metrics.get(
                    "final_balance", args.initial_balance
                ),
                "validation_max_drawdown": item.validation_metrics.get(
                    "max_drawdown", 0.0
                ),
                "selected_params": json.dumps(item.selected_params),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    iteration_summary_csv = out_dir / "walkforward_iteration_summary.csv"
    summary_df.to_csv(iteration_summary_csv, index=False)

    combined_summary = {
        "config": {
            "symbol": args.symbol,
            "exchange": args.exchange,
            "interval": args.interval,
            "start_date": args.start_date,
            "optimization_days": args.optimization_days,
            "validation_days": args.validation_days,
            "validation_stop_mode": args.validation_stop_mode,
            "validation_stop_days": validation_stop_days,
            "max_consecutive_losses": args.max_consecutive_losses,
            "initial_balance": args.initial_balance,
        },
        "combined_metrics": build_combined_summary(iterations, args.initial_balance),
        "iterations": [asdict(i) for i in iterations],
    }

    combined_summary_path = out_dir / "walkforward_summary.json"
    with open(combined_summary_path, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)

    print_final_validation_summary(iterations, initial_balance=args.initial_balance)
    print("\n✅ Walk-forward automation completed")
    print(f"📁 Output directory: {out_dir.resolve()}")
    print(f"📄 Iteration summary CSV: {iteration_summary_csv}")
    print(f"📄 Combined summary JSON: {combined_summary_path}")


if __name__ == "__main__":
    main()
