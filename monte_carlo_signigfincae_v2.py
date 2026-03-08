#!/usr/bin/env python3
"""Faster Monte Carlo significance test for selected MA parameter sets.

This version avoids re-running full-grid optimization on each permutation.
Instead, it validates only the already-selected parameter sets by comparing
real-data mean return per trade vs synthetic shuffled-series scores.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from strategies import MovingAverageCrossover


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


def _build_synthetic_ohlc_from_shuffled_close_returns(
    data: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Shuffle close-to-close returns while preserving bar internal OHLC shape."""
    if "Close" not in data.columns:
        raise ValueError("Input data must include 'Close' column.")
    required = [col for col in ("Open", "High", "Low", "Close") if col in data.columns]

    close = data["Close"].astype(float).to_numpy()
    if len(close) <= 1:
        return data.copy()

    close_returns = np.zeros_like(close, dtype=float)
    close_returns[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] == 0.0, 1.0, close[:-1])

    rng = np.random.default_rng(seed)
    shuffled_returns = close_returns.copy()
    shuffled_returns[1:] = rng.permutation(shuffled_returns[1:])

    synthetic_close = np.empty_like(close, dtype=float)
    synthetic_close[0] = close[0]
    for idx in range(1, len(close)):
        synthetic_close[idx] = synthetic_close[idx - 1] * (1.0 + shuffled_returns[idx])

    synthetic = data.copy()
    for col in required:
        col_values = data[col].astype(float).to_numpy()
        ratios = np.divide(
            col_values,
            close,
            out=np.ones_like(col_values, dtype=float),
            where=close != 0.0,
        )
        synthetic[col] = synthetic_close * ratios

    return synthetic


def _mean_return_per_trade_for_params(
    data: pd.DataFrame,
    short_window: int,
    long_window: int,
    risk_reward_ratio: float,
    trading_fee: float,
    min_total_trades: int,
) -> Tuple[float, Dict]:
    if short_window >= long_window:
        return -999.0, {"total_trades": 0, "total_pnl": -999.0}

    strategy = MovingAverageCrossover(
        short_window=short_window,
        long_window=long_window,
        risk_reward_ratio=risk_reward_ratio,
        trading_fee=trading_fee,
    )
    strategy.generate_signals(data)
    metrics = strategy.get_strategy_metrics()
    total_trades = int(metrics.get("total_trades", 0))
    total_pnl = float(metrics.get("total_pnl", 0.0))

    if total_trades < min_total_trades:
        return -999.0, {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "mean_return_per_trade": -999.0,
        }

    mean_return = total_pnl / max(total_trades, 1)
    return float(mean_return), {
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "mean_return_per_trade": float(mean_return),
    }


def _worker_permutation_scores(args: Tuple) -> List[float]:
    (
        data_payload,
        permutation_seed,
        param_sets,
        trading_fee,
        min_total_trades,
    ) = args
    data = _payload_to_dataframe(data_payload)
    synthetic = _build_synthetic_ohlc_from_shuffled_close_returns(
        data=data,
        seed=permutation_seed,
    )

    scores = []
    for param in param_sets:
        score, _ = _mean_return_per_trade_for_params(
            data=synthetic,
            short_window=int(param["short_window"]),
            long_window=int(param["long_window"]),
            risk_reward_ratio=float(param["risk_reward_ratio"]),
            trading_fee=trading_fee,
            min_total_trades=min_total_trades,
        )
        scores.append(float(score))
    return scores


def run_monte_carlo_param_significance_test_v2(
    data: pd.DataFrame,
    selected_params: Sequence[Dict],
    trading_fee: float = 0.0,
    n_permutations: int = 1000,
    p_value_threshold: float = 0.05,
    random_seed: int = 42,
    max_workers: Optional[int] = None,
    min_total_trades: int = 1,
    enable_early_stopping: bool = True,
) -> Dict:
    """Validate selected params by permutation test on shuffled close returns."""
    param_sets = [
        {
            "short_window": int(item["short_window"]),
            "long_window": int(item["long_window"]),
            "risk_reward_ratio": float(item["risk_reward_ratio"]),
        }
        for item in selected_params
        if all(k in item for k in ("short_window", "long_window", "risk_reward_ratio"))
    ]
    if not param_sets:
        raise ValueError("selected_params is empty or missing required fields.")

    actual_scores = []
    actual_meta = []
    for param in param_sets:
        actual_score, meta = _mean_return_per_trade_for_params(
            data=data,
            short_window=param["short_window"],
            long_window=param["long_window"],
            risk_reward_ratio=param["risk_reward_ratio"],
            trading_fee=trading_fee,
            min_total_trades=min_total_trades,
        )
        actual_scores.append(float(actual_score))
        actual_meta.append({**param, **meta})

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    n_requested = int(n_permutations)
    seeds = [random_seed + idx for idx in range(n_requested)]
    data_payload = _dataframe_to_payload(data)

    shuffled_scores_by_param: List[List[float]] = [[] for _ in param_sets]
    extreme_counts = [0 for _ in param_sets]
    n_completed = 0
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
                param_sets,
                trading_fee,
                min_total_trades,
            )
            future = executor.submit(_worker_permutation_scores, args)
            futures[future] = next_seed_idx
            next_seed_idx += 1

        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            stop_now = False
            for future in done:
                futures.pop(future, None)
                perm_scores = future.result()
                n_completed += 1

                for idx, score in enumerate(perm_scores):
                    shuffled_scores_by_param[idx].append(float(score))
                    if float(score) >= actual_scores[idx]:
                        extreme_counts[idx] += 1

                remaining = n_requested - n_completed
                lower_bounds = [count / max(n_requested, 1) for count in extreme_counts]
                upper_bounds = [
                    (count + remaining) / max(n_requested, 1) for count in extreme_counts
                ]

                if enable_early_stopping:
                    if all(lb > p_value_threshold for lb in lower_bounds):
                        early_stopped = True
                        early_stop_reason = "all_params_cannot_be_significant_anymore"
                        stop_now = True
                        break
                    if all(ub <= p_value_threshold for ub in upper_bounds):
                        early_stopped = True
                        early_stop_reason = "all_params_guaranteed_significant"
                        stop_now = True
                        break

                if next_seed_idx < n_requested:
                    args = (
                        data_payload,
                        seeds[next_seed_idx],
                        param_sets,
                        trading_fee,
                        min_total_trades,
                    )
                    next_future = executor.submit(_worker_permutation_scores, args)
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
    results = []
    significant_indices = []
    best_significant_idx = None
    best_significant_score = float("-inf")

    for idx, param in enumerate(actual_meta):
        lower = extreme_counts[idx] / max(n_requested, 1)
        upper = (extreme_counts[idx] + remaining) / max(n_requested, 1)
        is_significant = bool(upper <= p_value_threshold)
        if n_completed == n_requested:
            p_value = lower
        elif is_significant:
            p_value = upper
        else:
            p_value = lower

        param_result = {
            **param,
            "p_value": float(p_value),
            "p_value_lower_bound": float(lower),
            "p_value_upper_bound": float(upper),
            "is_significant": is_significant,
            "extreme_count": int(extreme_counts[idx]),
            "shuffled_scores": shuffled_scores_by_param[idx],
        }
        results.append(param_result)

        if is_significant:
            significant_indices.append(idx)
            if param["mean_return_per_trade"] > best_significant_score:
                best_significant_score = param["mean_return_per_trade"]
                best_significant_idx = idx

    return {
        "method": "param_significance_v2",
        "n_permutations": n_requested,
        "n_permutations_completed": int(n_completed),
        "p_value_threshold": float(p_value_threshold),
        "early_stopping_enabled": bool(enable_early_stopping),
        "early_stopped": bool(early_stopped),
        "early_stop_reason": early_stop_reason,
        "score_metric": "mean_return_per_trade",
        "param_results": results,
        "significant_param_indices": significant_indices,
        "best_significant_param_index": best_significant_idx,
        "is_significant": len(significant_indices) > 0,
    }
