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
from typing import Dict, List, Optional, Tuple

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
from monte_carlo_significance import run_monte_carlo_permutation_test


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
RISK_REWARD_RATIOS = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]

# SHORT_WINDOW_RANGE = [10, 20]
# LONG_WINDOW_RANGE = [
#     90,
#     100,
# ]
# RISK_REWARD_RATIOS = [5.0, 6.0]

# Global defaults for CLI-required runtime fields
DEFAULT_SYMBOL = "NATGASMINI26MARFUT"
# DEFAULT_SYMBOL = "CRUDEOILM26MARFUT"
DEFAULT_EXCHANGE = "MCX"
DEFAULT_START_DATE = (date.today() - timedelta(days=130)).strftime("%Y-%m-%d")
DEFAULT_OPTIMIZATION_DAYS = 60
DEFAULT_VALIDATION_DAYS = 30
DEFAULT_MAX_CONSECUTIVE_LOSSES = 5
TOP_N = 1
MINIMUM_TRADES_REQUIRED = 5
ENABLE_MONTE_CARLO_SIGNIFICANCE = False
MONTE_CARLO_PERMUTATIONS = 100
MONTE_CARLO_P_VALUE_THRESHOLD = 0.05
MONTE_CARLO_RANDOM_SEED = 42
MONTE_CARLO_EARLY_STOPPING = True
MONTE_CARLO_SHORT_STEP = 5
MONTE_CARLO_LONG_STEP = 5
MONTE_CARLO_RR_STEP = 3


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


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _build_optimization_chart_snippet(top_n_params: List[Dict]) -> str:
    """Create an optimization summary suitable for chart subtitles."""
    snippets = []
    for idx, row in enumerate(top_n_params, start=1):
        snippets.append(
            "#{} SW{}-LW{} RR{} Score={:.3f} Trades={} WR={} PnL={} DD={}".format(
                idx,
                row.get("short_window", "-"),
                row.get("long_window", "-"),
                row.get("risk_reward_ratio", "-"),
                float(row.get("score", 0.0)),
                int(row.get("total_trades", 0)),
                _format_pct(float(row.get("win_rate", 0.0))),
                _format_pct(float(row.get("total_pnl", 0.0))),
                _format_pct(float(row.get("max_drawdown", 0.0))),
            )
        )
    return "<br>".join(snippets)


def print_optimization_summary(
    iteration: int, top_n_params: List[Dict], min_trades_required: int
) -> None:
    print(f"\n🧠 Optimization Summary (Iteration {iteration})")
    print(
        "   Note: Unique (short,long) MA pairs only; highest RR kept per pair; "
        f"min trades filter >= {min_trades_required}."
    )
    print("Rank | Short | Long | RR  | Trades | Score   | Win Rate | PnL      | Max DD")
    print(
        "-----+-------+------+-----+--------+---------+----------+----------+--------"
    )

    pnl_values: List[float] = []
    for idx, row in enumerate(top_n_params, start=1):
        pnl = float(row.get("total_pnl", 0.0))
        pnl_values.append(pnl)
        print(
            f"{idx:>4} | "
            f"{int(row.get('short_window', 0)):>5} | "
            f"{int(row.get('long_window', 0)):>4} | "
            f"{float(row.get('risk_reward_ratio', 0.0)):>3.1f} | "
            f"{int(row.get('total_trades', 0)):>6} | "
            f"{float(row.get('score', 0.0)):>7.3f} | "
            f"{_format_pct(float(row.get('win_rate', 0.0))):>8} | "
            f"{_format_pct(pnl):>8} | "
            f"{_format_pct(float(row.get('max_drawdown', 0.0))):>6}"
        )

    print(
        "-----+-------+------+-----+--------+---------+----------+----------+--------"
    )
    print(
        "Top-N PnL Range: {} to {}".format(
            _format_pct(min(pnl_values)) if pnl_values else "0.00%",
            _format_pct(max(pnl_values)) if pnl_values else "0.00%",
        )
    )


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
    parser.add_argument(
        "--enable-monte-carlo",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_MONTE_CARLO_SIGNIFICANCE,
        help="Enable Monte Carlo permutation significance gate before validation.",
    )
    parser.add_argument(
        "--monte-carlo-permutations",
        type=int,
        default=MONTE_CARLO_PERMUTATIONS,
        help="Number of shuffled permutations used for significance test.",
    )
    parser.add_argument(
        "--monte-carlo-p-threshold",
        type=float,
        default=MONTE_CARLO_P_VALUE_THRESHOLD,
        help="Maximum p-value to accept optimization as significant.",
    )
    parser.add_argument(
        "--monte-carlo-random-seed",
        type=int,
        default=MONTE_CARLO_RANDOM_SEED,
        help="Seed for permutation generation.",
    )
    parser.add_argument(
        "--monte-carlo-early-stop",
        action=argparse.BooleanOptionalAction,
        default=MONTE_CARLO_EARLY_STOPPING,
        help="Enable/disable early stopping using p-value bounds during Monte Carlo.",
    )
    parser.add_argument(
        "--monte-carlo-short-step",
        type=int,
        default=MONTE_CARLO_SHORT_STEP,
        help="Downsample short MA candidates for Monte Carlo (1 keeps full range).",
    )
    parser.add_argument(
        "--monte-carlo-long-step",
        type=int,
        default=MONTE_CARLO_LONG_STEP,
        help="Downsample long MA candidates for Monte Carlo (1 keeps full range).",
    )
    parser.add_argument(
        "--monte-carlo-rr-step",
        type=int,
        default=MONTE_CARLO_RR_STEP,
        help="Downsample RR candidates for Monte Carlo (1 keeps full range).",
    )
    parser.add_argument("--out", default="ma_walkforward_results_kite")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def to_iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _downsample_values(values: List, step: int) -> List:
    values = list(values)
    if step <= 1 or len(values) <= 1:
        return values
    sampled = values[::step]
    if sampled[-1] != values[-1]:
        sampled.append(values[-1])
    return sampled


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
    enable_monte_carlo: bool,
    monte_carlo_permutations: int,
    monte_carlo_p_threshold: float,
    monte_carlo_random_seed: int,
    monte_carlo_early_stop: bool,
    monte_carlo_short_step: int,
    monte_carlo_long_step: int,
    monte_carlo_rr_step: int,
) -> Tuple[Optional[IterationSummary], date]:
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
        min_total_trades=MINIMUM_TRADES_REQUIRED,
    )
    if not top_n:
        raise RuntimeError(
            "No top-n parameter sets were selected for iteration "
            f"{iteration} with min trades >= {MINIMUM_TRADES_REQUIRED}"
        )
    print_optimization_summary(
        iteration=iteration,
        top_n_params=top_n,
        min_trades_required=MINIMUM_TRADES_REQUIRED,
    )

    if enable_monte_carlo:
        mc_short_range = _downsample_values(SHORT_WINDOW_RANGE, monte_carlo_short_step)
        mc_long_range = _downsample_values(LONG_WINDOW_RANGE, monte_carlo_long_step)
        mc_rr_range = _downsample_values(RISK_REWARD_RATIOS, monte_carlo_rr_step)
        print(
            f"🎲 Running Monte Carlo permutation test ({monte_carlo_permutations} permutations)..."
        )
        print(
            "   Monte Carlo grid | "
            f"short {len(SHORT_WINDOW_RANGE)}->{len(mc_short_range)} "
            f"(step={monte_carlo_short_step}) | "
            f"long {len(LONG_WINDOW_RANGE)}->{len(mc_long_range)} "
            f"(step={monte_carlo_long_step}) | "
            f"rr {len(RISK_REWARD_RATIOS)}->{len(mc_rr_range)} "
            f"(step={monte_carlo_rr_step})"
        )
        monte_carlo_summary = run_monte_carlo_permutation_test(
            data=optimization_data,
            short_window_range=mc_short_range,
            long_window_range=mc_long_range,
            risk_reward_ratios=mc_rr_range,
            actual_optimization_results=optimization_results,
            trading_fee=0.0,
            n_permutations=monte_carlo_permutations,
            p_value_threshold=monte_carlo_p_threshold,
            random_seed=monte_carlo_random_seed,
            max_workers=max_workers,
            min_total_trades=MINIMUM_TRADES_REQUIRED,
            enable_early_stopping=monte_carlo_early_stop,
        )
        mc_path = iter_dir / "monte_carlo_significance.json"
        with open(mc_path, "w", encoding="utf-8") as f:
            json.dump(monte_carlo_summary, f, indent=2)

        print(
            "   Monte Carlo result | "
            f"actual={monte_carlo_summary['actual_score']:.6f} | "
            f"p-value={monte_carlo_summary['p_value']:.4f} "
            f"(threshold={monte_carlo_p_threshold:.4f}) | "
            f"completed={monte_carlo_summary['n_permutations_completed']}/"
            f"{monte_carlo_summary['n_permutations']}"
        )
        print(
            "   Monte Carlo p-value bounds | "
            f"[{monte_carlo_summary['p_value_lower_bound']:.4f}, "
            f"{monte_carlo_summary['p_value_upper_bound']:.4f}]"
        )
        if monte_carlo_summary.get("early_stopped"):
            print(
                "   Monte Carlo early stop triggered | "
                f"reason={monte_carlo_summary.get('early_stop_reason', 'unknown')}"
            )
        best_params = monte_carlo_summary.get("actual_best_params")
        if best_params:
            print(
                "   Monte Carlo baseline best mean-return/trade | "
                f"Short={best_params['short_window']} | "
                f"Long={best_params['long_window']} | "
                f"RR={best_params['risk_reward_ratio']:.2f} | "
                f"Trades={best_params['total_trades']} | "
                f"PnL={best_params['total_pnl']*100:.2f}% | "
                f"Mean/Trade={best_params['mean_return_per_trade']:.6f}"
            )
        if not monte_carlo_summary["is_significant"]:
            print(
                "   ⏭️  Skipping validation for this iteration: "
                "optimizer result not statistically significant."
            )
            return None, requested_validation_end

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
    optimization_chart_summary = _build_optimization_chart_snippet(top_n)
    chart_summary_with_opt = (
        f"{chart_summary}<br>Opt Params:<br>{optimization_chart_summary}"
        if optimization_chart_summary
        else chart_summary
    )

    create_cumulative_pnl_chart(
        trades=trades,
        initial_balance=initial_balance,
        output_path=str(iter_dir / "validation_pnl.html"),
        title_suffix=chart_summary_with_opt,
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
        title_suffix=chart_summary_with_opt,
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
    if args.monte_carlo_permutations <= 0:
        raise ValueError("monte-carlo-permutations must be > 0")
    if not (0.0 <= args.monte_carlo_p_threshold <= 1.0):
        raise ValueError("monte-carlo-p-threshold must be between 0 and 1")
    if args.monte_carlo_short_step <= 0:
        raise ValueError("monte-carlo-short-step must be > 0")
    if args.monte_carlo_long_step <= 0:
        raise ValueError("monte-carlo-long-step must be > 0")
    if args.monte_carlo_rr_step <= 0:
        raise ValueError("monte-carlo-rr-step must be > 0")

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
            enable_monte_carlo=args.enable_monte_carlo,
            monte_carlo_permutations=args.monte_carlo_permutations,
            monte_carlo_p_threshold=args.monte_carlo_p_threshold,
            monte_carlo_random_seed=args.monte_carlo_random_seed,
            monte_carlo_early_stop=args.monte_carlo_early_stop,
            monte_carlo_short_step=args.monte_carlo_short_step,
            monte_carlo_long_step=args.monte_carlo_long_step,
            monte_carlo_rr_step=args.monte_carlo_rr_step,
        )
        if iteration_summary is not None:
            iterations.append(iteration_summary)
            print_iteration_validation_summary(
                iteration_summary, initial_balance=args.initial_balance
            )

        next_optimization_start = actual_validation_end - timedelta(
            days=DEFAULT_OPTIMIZATION_DAYS
        )
        if next_optimization_start <= cursor:
            next_optimization_start = cursor + timedelta(days=1)

        cursor = next_optimization_start
        iteration += 1

    if not iterations:
        print(
            "\n⚠️ No validation iterations were completed. "
            "All windows were skipped or out of range."
        )
        return

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
            "enable_monte_carlo": args.enable_monte_carlo,
            "monte_carlo_permutations": args.monte_carlo_permutations,
            "monte_carlo_p_threshold": args.monte_carlo_p_threshold,
            "monte_carlo_random_seed": args.monte_carlo_random_seed,
            "monte_carlo_early_stop": args.monte_carlo_early_stop,
            "monte_carlo_short_step": args.monte_carlo_short_step,
            "monte_carlo_long_step": args.monte_carlo_long_step,
            "monte_carlo_rr_step": args.monte_carlo_rr_step,
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
