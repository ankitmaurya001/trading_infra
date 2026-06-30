#!/usr/bin/env python3
"""Agentic strategy search over registered strategy templates.

The loop is deliberately simple and inspectable:
1. Discover registered strategy definitions.
2. Optimize each candidate on train data.
3. Validate best params out-of-sample.
4. Rank by validation-first score.
5. Mutate the best candidates into tighter search spaces.
6. Save JSON and CSV artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from strategies import STRATEGY_DEFINITIONS, StrategyDefinition
from strategy_optimizer import StrategyOptimizer

PENALTY_SCORE = -999.0


@dataclass
class StrategySearchConfig:
    generations: int = 2
    top_k: int = 3
    train_ratio: float = 0.7
    min_train_trades: int = 10
    min_validation_trades: int = 3
    max_drawdown_threshold: float = 0.5
    sharpe_threshold: float = -999.0
    optimization_metric: str = "composite_score"
    validation_weight: float = 0.7
    train_weight: float = 0.3
    trading_fee: Optional[float] = None


@dataclass
class StrategyCandidate:
    candidate_id: str
    generation: int
    strategy_key: str
    strategy_name: str
    param_ranges: Dict[str, List[Any]]
    parent_id: Optional[str] = None
    notes: str = "initial"


@dataclass
class StrategyEvaluation:
    candidate_id: str
    generation: int
    strategy_key: str
    strategy_name: str
    param_ranges: Dict[str, List[Any]]
    best_params: Optional[Dict[str, Any]]
    train_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    train_score: float
    validation_score: float
    blended_score: float
    status: str
    error: Optional[str] = None

    def flat_row(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "strategy_key": self.strategy_key,
            "strategy_name": self.strategy_name,
            "status": self.status,
            "blended_score": self.blended_score,
            "validation_score": self.validation_score,
            "train_score": self.train_score,
            "validation_total_trades": self.validation_metrics.get("total_trades"),
            "validation_total_pnl": self.validation_metrics.get("total_pnl"),
            "validation_sharpe_ratio": self.validation_metrics.get("sharpe_ratio"),
            "validation_max_drawdown": self.validation_metrics.get("max_drawdown"),
            "validation_win_rate": self.validation_metrics.get("win_rate"),
            "train_total_trades": self.train_metrics.get("total_trades"),
            "train_total_pnl": self.train_metrics.get("total_pnl"),
            "train_sharpe_ratio": self.train_metrics.get("sharpe_ratio"),
            "train_max_drawdown": self.train_metrics.get("max_drawdown"),
            "train_win_rate": self.train_metrics.get("win_rate"),
            "best_params": json.dumps(_to_jsonable(self.best_params), sort_keys=True),
            "error": self.error,
        }


@dataclass
class StrategySearchReport:
    created_at: str
    train_rows: int
    validation_rows: int
    config: Dict[str, Any]
    evaluations: List[StrategyEvaluation]
    best: Optional[StrategyEvaluation]


class StrategySearchAgent:
    """Search, validate, rank, and mutate registered strategies."""

    def __init__(
        self,
        config: Optional[StrategySearchConfig] = None,
        strategy_definitions: Optional[Dict[str, StrategyDefinition]] = None,
    ):
        self.config = config or StrategySearchConfig()
        self.strategy_definitions = strategy_definitions or STRATEGY_DEFINITIONS

    def run(
        self,
        data: pd.DataFrame,
        strategy_keys: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ) -> StrategySearchReport:
        self._validate_data(data)
        train_data, validation_data = self._split_data(data)
        candidates = self._initial_candidates(strategy_keys)
        evaluations: List[StrategyEvaluation] = []

        for generation in range(self.config.generations):
            if verbose:
                print(f"\nGeneration {generation + 1}/{self.config.generations}")
                print(f"Testing {len(candidates)} candidates")

            generation_results = [
                self.evaluate_candidate(candidate, train_data, validation_data)
                for candidate in candidates
            ]
            evaluations.extend(generation_results)

            ranked = self._rank_successful(generation_results)
            if verbose:
                self._print_generation_summary(ranked)

            if generation == self.config.generations - 1 or not ranked:
                break

            candidates = self._mutate_winners(ranked, next_generation=generation + 1)

        best = self._rank_successful(evaluations)[0] if self._rank_successful(evaluations) else None
        return StrategySearchReport(
            created_at=datetime.now().isoformat(),
            train_rows=len(train_data),
            validation_rows=len(validation_data),
            config=asdict(self.config),
            evaluations=evaluations,
            best=best,
        )

    def evaluate_candidate(
        self,
        candidate: StrategyCandidate,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
    ) -> StrategyEvaluation:
        definition = self.strategy_definitions[candidate.strategy_key]

        try:
            optimizer = StrategyOptimizer(
                data=train_data,
                strategy_class=definition.strategy_class,
                param_ranges=candidate.param_ranges,
                optimization_metric=self.config.optimization_metric,
                min_trades=self.config.min_train_trades,
                max_drawdown_threshold=self.config.max_drawdown_threshold,
                sharpe_threshold=self.config.sharpe_threshold,
                run_robustness_analysis=False,
            )
            best_params, train_metrics = optimizer.optimize()
            if not best_params or not train_metrics:
                raise ValueError("No valid optimized parameter set found")

            strategy = definition.create_strategy(best_params)
            signals = strategy.generate_signals(validation_data)
            if signals is None or signals.empty:
                raise ValueError("Strategy returned no validation signals")

            validation_metrics = strategy.get_strategy_metrics()
            train_score = self._score_metrics(train_metrics, self.config.min_train_trades)
            validation_score = self._score_metrics(
                validation_metrics,
                self.config.min_validation_trades,
            )
            blended_score = (
                self.config.validation_weight * validation_score
                + self.config.train_weight * train_score
            )

            return StrategyEvaluation(
                candidate_id=candidate.candidate_id,
                generation=candidate.generation,
                strategy_key=candidate.strategy_key,
                strategy_name=candidate.strategy_name,
                param_ranges=candidate.param_ranges,
                best_params=best_params,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                train_score=train_score,
                validation_score=validation_score,
                blended_score=blended_score,
                status="ok",
            )
        except Exception as exc:
            return StrategyEvaluation(
                candidate_id=candidate.candidate_id,
                generation=candidate.generation,
                strategy_key=candidate.strategy_key,
                strategy_name=candidate.strategy_name,
                param_ranges=candidate.param_ranges,
                best_params=None,
                train_metrics={},
                validation_metrics={},
                train_score=PENALTY_SCORE,
                validation_score=PENALTY_SCORE,
                blended_score=PENALTY_SCORE,
                status="failed",
                error=str(exc),
            )

    def save_report(self, report: StrategySearchReport, out_dir: Path) -> Dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "strategy_search_report.json"
        csv_path = out_dir / "strategy_search_rankings.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(report), f, indent=2)

        rows = [evaluation.flat_row() for evaluation in report.evaluations]
        pd.DataFrame(rows).sort_values(
            by=["blended_score"],
            ascending=False,
        ).to_csv(csv_path, index=False)

        return {"json": json_path, "csv": csv_path}

    def _initial_candidates(
        self,
        strategy_keys: Optional[Iterable[str]],
    ) -> List[StrategyCandidate]:
        keys = list(strategy_keys) if strategy_keys else list(self.strategy_definitions)
        candidates = []
        for key in keys:
            definition = self.strategy_definitions[key]
            ranges = self._prepare_ranges(definition.optimization_param_ranges)
            candidates.append(
                StrategyCandidate(
                    candidate_id=f"g0_{key}",
                    generation=0,
                    strategy_key=key,
                    strategy_name=definition.name,
                    param_ranges=ranges,
                )
            )
        return candidates

    def _prepare_ranges(self, ranges: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        prepared = {name: list(values) for name, values in ranges.items()}
        if self.config.trading_fee is not None and "trading_fee" in prepared:
            prepared["trading_fee"] = [self.config.trading_fee]
        return prepared

    def _mutate_winners(
        self,
        ranked: List[StrategyEvaluation],
        next_generation: int,
    ) -> List[StrategyCandidate]:
        next_candidates = []
        for rank, evaluation in enumerate(ranked[: self.config.top_k], start=1):
            if not evaluation.best_params:
                continue
            definition = self.strategy_definitions[evaluation.strategy_key]
            mutated_ranges = self._mutate_ranges(
                definition.optimization_param_ranges,
                evaluation.best_params,
            )
            mutated_ranges = self._prepare_ranges(mutated_ranges)
            next_candidates.append(
                StrategyCandidate(
                    candidate_id=f"g{next_generation}_{evaluation.strategy_key}_{rank}",
                    generation=next_generation,
                    strategy_key=evaluation.strategy_key,
                    strategy_name=evaluation.strategy_name,
                    param_ranges=mutated_ranges,
                    parent_id=evaluation.candidate_id,
                    notes="mutated around validation winner",
                )
            )
        return next_candidates

    def _mutate_ranges(
        self,
        base_ranges: Dict[str, List[Any]],
        best_params: Dict[str, Any],
    ) -> Dict[str, List[Any]]:
        mutated = {}
        for name, original_values in base_ranges.items():
            values = list(original_values)
            best_value = best_params.get(name)
            if name == "trading_fee" or best_value is None or not _is_number(best_value):
                mutated[name] = values
                continue

            step = self._infer_step(values, best_value)
            lower_bound = min(values) if values else best_value
            upper_bound = max(values) if values else best_value
            raw_values = [best_value - step, best_value, best_value + step]

            if isinstance(best_value, (int, np.integer)):
                new_values = sorted(
                    {
                        int(round(value))
                        for value in raw_values
                        if lower_bound <= value <= upper_bound and value > 0
                    }
                )
            else:
                new_values = sorted(
                    {
                        round(float(value), 6)
                        for value in raw_values
                        if lower_bound <= value <= upper_bound and value > 0
                    }
                )

            mutated[name] = new_values or [best_value]
        return mutated

    def _infer_step(self, values: List[Any], best_value: Any) -> float:
        numeric_values = sorted(float(value) for value in values if _is_number(value))
        if len(numeric_values) >= 2:
            diffs = [
                numeric_values[i] - numeric_values[i - 1]
                for i in range(1, len(numeric_values))
                if numeric_values[i] > numeric_values[i - 1]
            ]
            if diffs:
                return float(np.median(diffs))
        return max(abs(float(best_value)) * 0.2, 1.0)

    def _score_metrics(self, metrics: Dict[str, Any], min_trades: int) -> float:
        total_trades = int(metrics.get("total_trades", 0) or 0)
        max_drawdown = _safe_float(metrics.get("max_drawdown", 1.0), default=1.0)
        sharpe_ratio = _safe_float(metrics.get("sharpe_ratio", 0.0))

        if total_trades < min_trades:
            return PENALTY_SCORE
        if max_drawdown > self.config.max_drawdown_threshold:
            return PENALTY_SCORE
        if sharpe_ratio < self.config.sharpe_threshold:
            return PENALTY_SCORE

        calmar_ratio = _safe_float(metrics.get("calmar_ratio", 0.0))
        profit_factor = _safe_float(metrics.get("profit_factor", 0.0))
        win_rate = _safe_float(metrics.get("win_rate", 0.0))
        geometric_mean_return = _safe_float(metrics.get("geometric_mean_return", 0.0))

        sharpe_score = np.clip((sharpe_ratio + 3.0) / 6.0, 0.0, 1.0)
        calmar_score = np.clip(calmar_ratio / 10.0, 0.0, 1.0)
        profit_factor_score = np.clip(profit_factor / 5.0, 0.0, 1.0)
        gmr_score = np.clip((geometric_mean_return + 0.05) / 0.10, 0.0, 1.0)
        drawdown_score = np.clip(1.0 - max_drawdown, 0.0, 1.0)

        return float(
            0.25 * sharpe_score
            + 0.20 * calmar_score
            + 0.15 * profit_factor_score
            + 0.15 * win_rate
            + 0.15 * gmr_score
            + 0.10 * drawdown_score
        )

    def _split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_index = int(len(data) * self.config.train_ratio)
        split_index = max(1, min(split_index, len(data) - 1))
        return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()

    def _rank_successful(
        self,
        evaluations: List[StrategyEvaluation],
    ) -> List[StrategyEvaluation]:
        successful = [item for item in evaluations if item.status == "ok"]
        return sorted(successful, key=lambda item: item.blended_score, reverse=True)

    def _print_generation_summary(self, ranked: List[StrategyEvaluation]) -> None:
        if not ranked:
            print("No successful candidates in this generation.")
            return

        print("Rank | Strategy | Score | Val PnL | Val Trades | Params")
        print("-----+----------+-------+---------+------------+----------------")
        for rank, result in enumerate(ranked[: self.config.top_k], start=1):
            pnl = _safe_float(result.validation_metrics.get("total_pnl", 0.0))
            trades = result.validation_metrics.get("total_trades", 0)
            print(
                f"{rank:>4} | {result.strategy_key:<8} | "
                f"{result.blended_score:>5.3f} | {pnl:>7.2%} | "
                f"{trades:>10} | {result.best_params}"
            )

    def _validate_data(self, data: pd.DataFrame) -> None:
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")
        if len(data) < 20:
            raise ValueError("Need at least 20 rows for strategy search")


def load_data_from_csv(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    datetime_columns = [
        column
        for column in data.columns
        if column.lower() in {"date", "datetime", "timestamp", "time"}
    ]
    if datetime_columns:
        data[datetime_columns[0]] = pd.to_datetime(data[datetime_columns[0]])
        data = data.set_index(datetime_columns[0])
    elif data.columns[0].startswith("Unnamed"):
        data[data.columns[0]] = pd.to_datetime(data[data.columns[0]])
        data = data.set_index(data.columns[0])
    return data


def fetch_data(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    from data_fetcher import DataFetcher

    fetcher = DataFetcher()
    return fetcher.fetch_data(symbol, start_date, end_date, interval=interval)


def parse_strategy_keys(raw: str) -> Optional[List[str]]:
    if raw.lower() == "all":
        return None
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(keys).difference(STRATEGY_DEFINITIONS))
    if unknown:
        raise ValueError(f"Unknown strategy keys: {unknown}. Known: {sorted(STRATEGY_DEFINITIONS)}")
    return keys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic strategy search")
    parser.add_argument("--data-file", type=Path, help="CSV file with OHLCV data")
    parser.add_argument("--symbol", default="BTC-USD")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--strategies", default="all")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--min-train-trades", type=int, default=10)
    parser.add_argument("--min-validation-trades", type=int, default=3)
    parser.add_argument("--max-drawdown-threshold", type=float, default=0.5)
    parser.add_argument("--sharpe-threshold", type=float, default=-999.0)
    parser.add_argument("--trading-fee", type=float)
    parser.add_argument("--out-dir", type=Path, default=Path("agentic_strategy_search_results"))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    strategy_keys = parse_strategy_keys(args.strategies)

    if args.data_file:
        data = load_data_from_csv(args.data_file)
    else:
        end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = args.start_date or (
            datetime.now() - timedelta(days=120)
        ).strftime("%Y-%m-%d")
        data = fetch_data(args.symbol, start_date, end_date, args.interval)

    config = StrategySearchConfig(
        generations=args.generations,
        top_k=args.top_k,
        train_ratio=args.train_ratio,
        min_train_trades=args.min_train_trades,
        min_validation_trades=args.min_validation_trades,
        max_drawdown_threshold=args.max_drawdown_threshold,
        sharpe_threshold=args.sharpe_threshold,
        trading_fee=args.trading_fee,
    )
    agent = StrategySearchAgent(config=config)
    report = agent.run(data, strategy_keys=strategy_keys, verbose=not args.quiet)
    paths = agent.save_report(report, args.out_dir)

    if report.best:
        print("\nBest strategy found")
        print(f"Strategy: {report.best.strategy_name} ({report.best.strategy_key})")
        print(f"Score: {report.best.blended_score:.4f}")
        print(f"Params: {report.best.best_params}")
        print(f"Validation metrics: {report.best.validation_metrics}")
    else:
        print("\nNo successful strategy candidate found.")

    print(f"Report: {paths['json']}")
    print(f"Rankings: {paths['csv']}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _to_jsonable(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
