#!/usr/bin/env python3
"""Monte Carlo permutation significance test for MA optimization."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from run_ma_optimization_kite_parallel import run_parallel_optimization_grid


def _dataframe_to_payload(data: pd.DataFrame) -> Dict:
    return {
        "data": data.to_dict("records"),
        "index": data.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "columns": list(data.columns),
    }


def _payload_to_dataframe(payload: Dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    frame.index = pd.to_datetime(payload["index"])
    return frame


def _best_mean_return_result(
    optimization_results: Dict,
    min_total_trades: int,
) -> Dict:
    best_score = float("-inf")
    best_meta = None
    for rr, rr_results in optimization_results.items():
        short_windows = rr_results.get("short_windows", [])
        long_windows = rr_results.get("long_windows", [])
        trades = rr_results.get("total_trades", [])
        pnls = rr_results.get("total_pnl", [])
        for short_w, long_w, total_trades, total_pnl in zip(
            short_windows, long_windows, trades, pnls
        ):
            if int(total_trades) < min_total_trades:
                continue
            mean_return = float(total_pnl) / max(int(total_trades), 1)
            if mean_return > best_score:
                best_score = mean_return
                best_meta = {
                    "short_window": int(short_w),
                    "long_window": int(long_w),
                    "risk_reward_ratio": float(rr),
                    "total_trades": int(total_trades),
                    "total_pnl": float(total_pnl),
                    "mean_return_per_trade": float(mean_return),
                }
    if best_meta is None:
        return {
            "best_score": -999.0,
            "best_params": None,
        }
    return {
        "best_score": float(best_score),
        "best_params": best_meta,
    }


def _build_synthetic_series_from_permuted_returns(
    data: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    synthetic = data.copy()
    ohlc_cols = [col for col in ("Open", "High", "Low", "Close") if col in data.columns]
    if "Close" not in ohlc_cols:
        raise ValueError("Input data must include 'Close' column for Monte Carlo test.")

    ohlc_prices = data[ohlc_cols].astype(float)
    ohlc_returns = ohlc_prices.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    shuffled = ohlc_returns.to_numpy().copy()
    rng = np.random.default_rng(seed)
    if len(shuffled) > 1:
        shuffled[1:] = rng.permutation(shuffled[1:], axis=0)

    synthetic_ohlc = np.empty_like(shuffled, dtype=float)
    synthetic_ohlc[0] = ohlc_prices.iloc[0].to_numpy(dtype=float)
    for idx in range(1, len(shuffled)):
        synthetic_ohlc[idx] = synthetic_ohlc[idx - 1] * (1.0 + shuffled[idx])

    for col_idx, col in enumerate(ohlc_cols):
        synthetic[col] = synthetic_ohlc[:, col_idx]

    return synthetic


def _permutation_worker(args: Tuple) -> float:
    (
        data_payload,
        permutation_seed,
        short_window_range,
        long_window_range,
        risk_reward_ratios,
        trading_fee,
        min_total_trades,
    ) = args
    data = _payload_to_dataframe(data_payload)
    synthetic = _build_synthetic_series_from_permuted_returns(
        data=data, seed=permutation_seed
    )
    optimization_results = run_parallel_optimization_grid(
        data=synthetic,
        short_window_range=short_window_range,
        long_window_range=long_window_range,
        risk_reward_ratios=risk_reward_ratios,
        trading_fee=trading_fee,
        max_workers=1,
        show_progress=False,
    )
    return _best_mean_return_result(
        optimization_results, min_total_trades=min_total_trades
    )["best_score"]


def run_monte_carlo_permutation_test(
    data: pd.DataFrame,
    short_window_range: Iterable[int],
    long_window_range: Iterable[int],
    risk_reward_ratios: Iterable[float],
    actual_optimization_results: Dict,
    trading_fee: float = 0.0,
    n_permutations: int = 100,
    p_value_threshold: float = 0.05,
    random_seed: int = 42,
    max_workers: Optional[int] = None,
    min_total_trades: int = 1,
    enable_early_stopping: bool = True,
) -> Dict:
    """Run permutation test and return significance diagnostics."""
    short_window_range = list(short_window_range)
    long_window_range = list(long_window_range)
    risk_reward_ratios = list(risk_reward_ratios)

    actual_best = _best_mean_return_result(
        actual_optimization_results,
        min_total_trades=min_total_trades,
    )
    actual_score = actual_best["best_score"]

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    data_payload = _dataframe_to_payload(data)
    seeds = [random_seed + idx for idx in range(n_permutations)]

    shuffled_scores: List[float] = []
    extreme_count = 0
    n_completed = 0
    n_requested = int(n_permutations)
    early_stopped = False
    early_stop_reason = ""

    next_seed_idx = 0
    futures: Dict = {}
    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        initial_submissions = min(max_workers, n_requested)
        for _ in range(initial_submissions):
            args = (
                data_payload,
                seeds[next_seed_idx],
                short_window_range,
                long_window_range,
                risk_reward_ratios,
                trading_fee,
                min_total_trades,
            )
            future = executor.submit(_permutation_worker, args)
            futures[future] = next_seed_idx
            next_seed_idx += 1

        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            stop_now = False
            for future in done:
                futures.pop(future, None)
                score = float(future.result())
                shuffled_scores.append(score)
                n_completed += 1
                if score >= actual_score:
                    extreme_count += 1

                remaining = n_requested - n_completed
                p_lower_bound = extreme_count / max(n_requested, 1)
                p_upper_bound = (extreme_count + remaining) / max(n_requested, 1)
                if enable_early_stopping and p_lower_bound > p_value_threshold:
                    early_stopped = True
                    early_stop_reason = (
                        "cannot_be_significant_anymore"
                    )
                    stop_now = True
                    break
                if enable_early_stopping and p_upper_bound <= p_value_threshold:
                    early_stopped = True
                    early_stop_reason = (
                        "guaranteed_significant"
                    )
                    stop_now = True
                    break

                if next_seed_idx < n_requested:
                    args = (
                        data_payload,
                        seeds[next_seed_idx],
                        short_window_range,
                        long_window_range,
                        risk_reward_ratios,
                        trading_fee,
                        min_total_trades,
                    )
                    next_future = executor.submit(_permutation_worker, args)
                    futures[next_future] = next_seed_idx
                    next_seed_idx += 1

            if stop_now:
                for pending in list(futures.keys()):
                    pending.cancel()
                futures.clear()
                break
    finally:
        executor.shutdown(wait=not early_stopped, cancel_futures=early_stopped)

    remaining = n_requested - n_completed
    p_lower_bound = extreme_count / max(n_requested, 1)
    p_upper_bound = (extreme_count + remaining) / max(n_requested, 1)
    is_significant = bool(p_upper_bound <= p_value_threshold)
    if n_completed == n_requested:
        p_value = p_lower_bound
    elif is_significant:
        p_value = p_upper_bound
    else:
        p_value = p_lower_bound

    return {
        "actual_score": float(actual_score),
        "actual_best_params": actual_best["best_params"],
        "n_permutations": n_requested,
        "n_permutations_completed": int(n_completed),
        "shuffled_scores": shuffled_scores,
        "extreme_count": int(extreme_count),
        "p_value": float(p_value),
        "p_value_lower_bound": float(p_lower_bound),
        "p_value_upper_bound": float(p_upper_bound),
        "p_value_threshold": float(p_value_threshold),
        "is_significant": is_significant,
        "early_stopping_enabled": bool(enable_early_stopping),
        "early_stopped": bool(early_stopped),
        "early_stop_reason": early_stop_reason,
        "score_metric": "best_mean_return_per_trade",
    }
