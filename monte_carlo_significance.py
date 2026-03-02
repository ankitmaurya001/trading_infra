#!/usr/bin/env python3
"""Monte Carlo permutation significance test for MA optimization."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    close = data["Close"].astype(float)

    close_returns = close.pct_change().fillna(0.0).to_numpy()
    shuffled = close_returns.copy()
    rng = np.random.default_rng(seed)
    if len(shuffled) > 1:
        shuffled[1:] = rng.permutation(shuffled[1:])

    synthetic_close = np.empty_like(shuffled, dtype=float)
    synthetic_close[0] = float(close.iloc[0])
    for idx in range(1, len(shuffled)):
        synthetic_close[idx] = synthetic_close[idx - 1] * (1.0 + shuffled[idx])

    # Preserve intrabar relationships by scaling OHLC against original close path.
    scaling = synthetic_close / close.to_numpy(dtype=float)
    for col in ("Open", "High", "Low", "Close"):
        if col in synthetic.columns:
            synthetic[col] = data[col].astype(float).to_numpy() * scaling

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
    worker_args = [
        (
            data_payload,
            seed,
            short_window_range,
            long_window_range,
            risk_reward_ratios,
            trading_fee,
            min_total_trades,
        )
        for seed in seeds
    ]

    shuffled_scores: List[float] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_permutation_worker, args) for args in worker_args]
        for future in as_completed(futures):
            shuffled_scores.append(float(future.result()))

    extreme_count = sum(1 for score in shuffled_scores if score >= actual_score)
    p_value = extreme_count / max(n_permutations, 1)

    return {
        "actual_score": float(actual_score),
        "actual_best_params": actual_best["best_params"],
        "n_permutations": int(n_permutations),
        "shuffled_scores": shuffled_scores,
        "extreme_count": int(extreme_count),
        "p_value": float(p_value),
        "p_value_threshold": float(p_value_threshold),
        "is_significant": bool(p_value <= p_value_threshold),
        "score_metric": "best_mean_return_per_trade",
    }
