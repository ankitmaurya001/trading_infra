#!/usr/bin/env python3
"""LLM-driven strategy research loop.

This script asks an OpenAI-compatible LLM endpoint for strategy ideas, converts
safe JSON proposals into executable strategy definitions, optimizes/validates
them locally, and ranks them against optional registered baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from agentic_strategy_search import (
    StrategySearchAgent,
    StrategySearchConfig,
    fetch_data,
    load_data_from_csv,
)
from strategies import STRATEGY_DEFINITIONS, StrategyDefinition
from strategy_dsl import build_rule_strategy_definition


DEFAULT_SYSTEM_PROMPT = """You are a systematic quantitative strategy researcher.
Return only JSON. Do not return Python code.
You may propose classic, published, or original technical-analysis strategies.
If your runtime has web search, you may search for strategy ideas, but the final
answer must be valid JSON matching the requested schema."""


IDEA_SCHEMA_PROMPT = """
Return JSON with this top-level shape:

{
  "research_notes": "short rationale",
  "strategies": [
    {
      "type": "rule_dsl",
      "strategy_key": "short_unique_name",
      "strategy_name": "Human readable name",
      "hypothesis": "Why this might work",
      "param_ranges": {
        "fast_period": [5, 8, 13],
        "slow_period": [21, 34, 55],
        "risk_reward_ratio": [1.5, 2.0, 2.5],
        "trading_fee": [0.001]
      },
      "constraints": [
        {"op": "lt", "left": "fast_period", "right": "slow_period"}
      ],
      "indicators": [
        {"kind": "ema", "name": "ema_fast", "source": "Close", "period": "fast_period"},
        {"kind": "ema", "name": "ema_slow", "source": "Close", "period": "slow_period"}
      ],
      "long_entry": {"op": "crosses_above", "left": "ema_fast", "right": "ema_slow"},
      "short_entry": {"op": "crosses_below", "left": "ema_fast", "right": "ema_slow"}
    }
  ]
}

Supported indicator kinds:
- sma, ema, rsi, macd, stochastic, atr, cci, roc, zscore,
  rolling_high, rolling_low, rolling_mean, rolling_std, bollinger

Bollinger indicator example:
{"kind": "bollinger", "source": "Close", "period": "bb_period", "std_dev": "bb_std",
 "middle": "bb_mid", "upper": "bb_upper", "lower": "bb_lower"}

Supported condition ops:
- all, any, gt, gte, lt, lte, crosses_above, crosses_below

Supported source/condition tokens:
- OHLCV columns: Open, High, Low, Close, Volume
- indicator names created in indicators
- parameter names from param_ranges
- {"param": "param_name"} or {"value": 70}

Keep each parameter list small. Avoid more than 5 values per parameter.
Always include risk_reward_ratio and trading_fee.
"""


@dataclass
class LLMStrategyResearchResult:
    created_at: str
    prompt_context: Dict[str, Any]
    llm_payload: Dict[str, Any]
    accepted_strategy_keys: List[str]
    rejected_strategies: List[Dict[str, Any]]
    rounds: List[Dict[str, Any]]
    search_report_path: str
    rankings_path: str


class OpenAICompatibleLLMClient:
    """Tiny stdlib client for OpenAI-compatible chat/completions endpoints."""

    def __init__(
        self,
        endpoint_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_seconds: int = 120,
    ):
        self.endpoint_url = endpoint_url
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM endpoint HTTP {exc.code}: {body}") from exc

        response_payload = json.loads(raw)
        content = response_payload["choices"][0]["message"]["content"]
        return parse_llm_json(content)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def build_strategy_research_prompt(
    symbol: str,
    exchange: Optional[str],
    interval: str,
    n_ideas: int,
    allow_web_search: bool,
    extra_context: Optional[str],
    data_summary: Dict[str, Any],
    target_summary: Dict[str, Any],
    previous_feedback: Optional[Dict[str, Any]] = None,
) -> str:
    web_note = (
        "Use web search if your endpoint has that tool available. Look for strategy ideas, market microstructure notes, seasonality, trend/mean-reversion behavior, and risk caveats for this instrument or comparable commodity futures."
        if allow_web_search
        else "Do not rely on live web search; propose from your internal quant knowledge."
    )
    context = extra_context or "No extra context provided."
    registered = ", ".join(sorted(STRATEGY_DEFINITIONS))
    exchange_text = exchange or "not specified"
    feedback_text = (
        json.dumps(previous_feedback, indent=2)
        if previous_feedback
        else "No prior rounds yet."
    )
    return f"""
Suggest {n_ideas} strategy ideas for symbol/context: {symbol}, exchange: {exchange_text}, interval: {interval}.
{web_note}

Registered baselines already available locally: {registered}.
Prefer ideas that are different from simple moving-average crossover.
The local evaluator will optimize parameters, validate out-of-sample, and rank.
You may suggest classic strategies from online research and your own original
ideas, but final strategy definitions must fit the JSON DSL.

Quality target:
{json.dumps(target_summary, indent=2)}

Market data summary available to you:
{json.dumps(data_summary, indent=2)}

Prior research feedback:
{feedback_text}

If prior ideas performed poorly, do not repeat them. Propose meaningfully
different logic, indicators, filters, and parameter ranges.

Extra user context:
{context}

{IDEA_SCHEMA_PROMPT}
"""


def parse_llm_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def load_ideas(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_definitions_from_llm_payload(
    payload: Dict[str, Any],
    include_baselines: bool = True,
) -> tuple[Dict[str, StrategyDefinition], List[Dict[str, Any]]]:
    definitions: Dict[str, StrategyDefinition] = {}
    rejected = []

    if include_baselines:
        definitions.update(STRATEGY_DEFINITIONS)

    for raw_strategy in payload.get("strategies", []):
        try:
            strategy_type = raw_strategy.get("type", "rule_dsl")
            if strategy_type == "registered":
                key = raw_strategy["strategy_key"]
                definition = STRATEGY_DEFINITIONS[key]
                if raw_strategy.get("param_ranges"):
                    definition = StrategyDefinition(
                        key=definition.key,
                        name=definition.name,
                        strategy_class=definition.strategy_class,
                        optimization_param_ranges=raw_strategy["param_ranges"],
                        live_parameter_names=definition.live_parameter_names,
                        optimization_defaults=definition.optimization_defaults,
                    )
                definitions[definition.key] = definition
            elif strategy_type == "rule_dsl":
                definition = build_rule_strategy_definition(raw_strategy)
                definitions[definition.key] = definition
            else:
                raise ValueError(f"Unsupported strategy proposal type: {strategy_type}")
        except Exception as exc:
            rejected.append({"strategy": raw_strategy, "error": str(exc)})

    return definitions, rejected


def run_llm_strategy_research(args: argparse.Namespace) -> LLMStrategyResearchResult:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.llm_rounds <= 0:
        raise ValueError("llm-rounds must be > 0")
    if args.feedback_top_n <= 0:
        raise ValueError("feedback-top-n must be > 0")

    data = load_research_data(args)
    data_summary = summarize_market_data(data, sample_rows=args.llm_sample_rows)
    target_summary = build_target_summary(args)
    round_records = []
    all_rejected = []
    all_accepted_keys = []
    all_llm_payloads = []
    all_ranking_rows = []
    aggregate_feedback = None

    if args.ideas_file:
        ideas_file_payload = load_ideas(args.ideas_file)
        total_rounds = 1
        client = None
        prompt_context = {
            "source": str(args.ideas_file),
            "data_summary": data_summary,
            "target_summary": target_summary,
        }
    else:
        endpoint_url = args.llm_url or os.getenv("LLM_STRATEGY_ENDPOINT")
        model = args.llm_model or os.getenv("LLM_STRATEGY_MODEL", "gpt-4.1")
        api_key = args.llm_api_key or os.getenv("LLM_STRATEGY_API_KEY")
        if not endpoint_url:
            raise ValueError("Provide --llm-url or set LLM_STRATEGY_ENDPOINT, or use --ideas-file")

        client = OpenAICompatibleLLMClient(endpoint_url, model=model, api_key=api_key)
        ideas_file_payload = None
        total_rounds = args.llm_rounds
        prompt_context = {
            "endpoint_url": endpoint_url,
            "model": model,
            "data_summary": data_summary,
            "target_summary": target_summary,
        }
    for round_index in range(1, total_rounds + 1):
        if ideas_file_payload is not None:
            llm_payload = ideas_file_payload
        else:
            user_prompt = build_strategy_research_prompt(
                symbol=args.symbol,
                exchange=args.exchange,
                interval=args.interval,
                n_ideas=args.n_ideas,
                allow_web_search=args.allow_llm_web_search,
                extra_context=args.context,
                data_summary=data_summary,
                target_summary=target_summary,
                previous_feedback=aggregate_feedback,
            )
            prompt_context[f"round_{round_index}_prompt"] = user_prompt
            llm_payload = client.complete_json(DEFAULT_SYSTEM_PROMPT, user_prompt)

        round_dir = out_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        ideas_path = round_dir / "llm_strategy_ideas.json"
        with open(ideas_path, "w", encoding="utf-8") as f:
            json.dump(llm_payload, f, indent=2)

        if args.print_llm_reply:
            print(f"\nLLM reply (round {round_index})")
            print(json.dumps(llm_payload, indent=2))

        definitions, rejected = build_definitions_from_llm_payload(
            llm_payload,
            include_baselines=args.include_baselines,
        )
        all_llm_payloads.append(llm_payload)
        all_rejected.extend(
            {
                "round": round_index,
                **item,
            }
            for item in rejected
        )

        if args.print_llm_reply:
            proposed_keys = [
                item.get("strategy_key", "<missing>")
                for item in llm_payload.get("strategies", [])
            ]
            print(f"\nLLM proposed strategies (round {round_index}): {proposed_keys}")
            print(f"Accepted for testing (round {round_index}): {sorted(definitions)}")
            if rejected:
                print("Rejected strategies:")
                for item in rejected:
                    strategy = item.get("strategy", {})
                    print(f"  - {strategy.get('strategy_key', '<missing>')}: {item.get('error')}")

        if not definitions:
            round_records.append(
                {
                    "round": round_index,
                    "ideas_path": str(ideas_path),
                    "accepted_strategy_keys": [],
                    "rejected_count": len(rejected),
                    "search_report_path": None,
                    "rankings_path": None,
                    "best": None,
                }
            )
            aggregate_feedback = build_round_feedback(None, rejected, args.feedback_top_n, target_summary)
            continue

        search_config = StrategySearchConfig(
            generations=args.generations,
            top_k=args.top_k,
            train_ratio=args.train_ratio,
            min_train_trades=args.min_train_trades,
            min_validation_trades=args.min_validation_trades,
            max_drawdown_threshold=args.max_drawdown_threshold,
            sharpe_threshold=args.sharpe_threshold,
            trading_fee=args.trading_fee,
        )
        search_agent = StrategySearchAgent(
            config=search_config,
            strategy_definitions=definitions,
        )
        print(f"\nLLM research round {round_index}/{total_rounds}")
        search_report = search_agent.run(data, verbose=not args.quiet)
        paths = search_agent.save_report(search_report, round_dir)
        accepted_keys = sorted(definitions)
        all_accepted_keys.extend(accepted_keys)

        for evaluation in search_report.evaluations:
            row = evaluation.flat_row()
            row["llm_round"] = round_index
            all_ranking_rows.append(row)

        round_best = search_report.best.flat_row() if search_report.best else None
        target_evaluation = evaluate_target(search_report.best, target_summary)
        if round_best is not None:
            round_best["target_evaluation"] = target_evaluation
        round_records.append(
            {
                "round": round_index,
                "ideas_path": str(ideas_path),
                "accepted_strategy_keys": accepted_keys,
                "rejected_count": len(rejected),
                "search_report_path": str(paths["json"]),
                "rankings_path": str(paths["csv"]),
                "best": round_best,
                "target_met": target_evaluation["target_met"],
                "target_checks": target_evaluation["checks"],
            }
        )
        aggregate_feedback = build_round_feedback(search_report, rejected, args.feedback_top_n, target_summary)

        if not args.ideas_file and round_index < total_rounds:
            prompt_context[f"round_{round_index}_evaluated_feedback"] = aggregate_feedback

        if args.stop_when_target_met and target_evaluation["target_met"]:
            print(
                "\nTarget met. Stopping early after "
                f"LLM round {round_index}/{total_rounds}."
            )
            break

    if not all_ranking_rows:
        raise ValueError("No valid strategy definitions produced by LLM payloads")

    aggregate_rankings_path = out_dir / "aggregate_strategy_rankings.csv"
    aggregate_df = pd.DataFrame(all_ranking_rows).sort_values(
        by=["blended_score"],
        ascending=False,
    )
    aggregate_df.to_csv(aggregate_rankings_path, index=False)
    aggregate_report_path = out_dir / "aggregate_strategy_search_report.json"
    aggregate_report = {
        "created_at": datetime.now().isoformat(),
        "rounds": round_records,
        "best": aggregate_df.iloc[0].to_dict() if not aggregate_df.empty else None,
        "target": target_summary,
        "target_found": next((item for item in round_records if item.get("target_met")), None),
        "rankings_path": str(aggregate_rankings_path),
    }
    with open(aggregate_report_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(aggregate_report), f, indent=2)

    result = LLMStrategyResearchResult(
        created_at=datetime.now().isoformat(),
        prompt_context=prompt_context,
        llm_payload={"rounds": all_llm_payloads},
        accepted_strategy_keys=sorted(set(all_accepted_keys)),
        rejected_strategies=all_rejected,
        rounds=round_records,
        search_report_path=str(aggregate_report_path),
        rankings_path=str(aggregate_rankings_path),
    )
    result_path = out_dir / "llm_strategy_research_report.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(asdict(result)), f, indent=2)

    best_row = aggregate_df.iloc[0].to_dict() if not aggregate_df.empty else None
    target_found = next((item for item in round_records if item.get("target_met")), None)
    if target_found:
        print("\nTarget-quality strategy found")
        best = target_found["best"]
        print(f"Round: {target_found['round']}")
        print(f"Strategy: {best['strategy_name']} ({best['strategy_key']})")
        print(f"Score: {best['blended_score']:.4f}")
        print(f"Params: {best['best_params']}")
    elif best_row:
        print("\nNo target-quality strategy found. Best attempt")
        print(f"Round: {best_row['llm_round']}")
        print(f"Strategy: {best_row['strategy_name']} ({best_row['strategy_key']})")
        print(f"Score: {best_row['blended_score']:.4f}")
        print(f"Params: {best_row['best_params']}")
    else:
        print("\nNo successful strategy candidate found.")

    print(f"Target: {target_summary}")
    print(f"Accepted strategies: {sorted(set(all_accepted_keys))}")
    if all_rejected:
        print(f"Rejected strategies: {len(all_rejected)}")
    print(f"Research report: {result_path}")
    print(f"Search report: {aggregate_report_path}")
    print(f"Rankings: {aggregate_rankings_path}")
    return result


def build_target_summary(args: argparse.Namespace) -> Dict[str, Any]:
    target_max_drawdown = (
        args.target_max_drawdown
        if args.target_max_drawdown is not None
        else args.max_drawdown_threshold
    )
    target_min_trades = (
        args.target_min_trades
        if args.target_min_trades is not None
        else args.min_validation_trades
    )
    target_train_win_rate = (
        args.target_train_win_rate
        if args.target_train_win_rate is not None
        else args.target_win_rate
    )
    return {
        "validation_win_rate_at_least": args.target_win_rate,
        "validation_total_pnl_at_least": args.target_min_pnl,
        "validation_max_drawdown_at_most": target_max_drawdown,
        "validation_trades_at_least": target_min_trades,
        "require_train_stability": args.target_require_train_stability,
        "train_win_rate_at_least": target_train_win_rate,
        "train_total_pnl_at_least": args.target_min_train_pnl,
        "train_validation_win_rate_gap_at_most": args.target_max_win_rate_gap,
    }


def evaluate_target(evaluation, target_summary: Dict[str, Any]) -> Dict[str, Any]:
    if evaluation is None or getattr(evaluation, "status", None) != "ok":
        return {
            "target_met": False,
            "checks": {
                "status": {
                    "passed": False,
                    "actual": getattr(evaluation, "status", None),
                    "target": "ok",
                }
            },
        }

    train_metrics = evaluation.train_metrics or {}
    validation_metrics = evaluation.validation_metrics or {}
    checks = {}

    def add_check(name: str, actual: float, target: float, passed: bool) -> None:
        checks[name] = {
            "passed": bool(passed),
            "actual": actual,
            "target": target,
        }

    validation_win_rate = safe_metric(validation_metrics, "win_rate")
    validation_total_pnl = safe_metric(validation_metrics, "total_pnl")
    validation_max_drawdown = safe_metric(validation_metrics, "max_drawdown", default=1.0)
    validation_trades = safe_metric(validation_metrics, "total_trades")
    train_win_rate = safe_metric(train_metrics, "win_rate")
    train_total_pnl = safe_metric(train_metrics, "total_pnl")
    train_trades = safe_metric(train_metrics, "total_trades")
    win_rate_gap = abs(train_win_rate - validation_win_rate)

    add_check(
        "validation_win_rate",
        validation_win_rate,
        target_summary["validation_win_rate_at_least"],
        validation_win_rate >= target_summary["validation_win_rate_at_least"],
    )
    add_check(
        "validation_total_pnl",
        validation_total_pnl,
        target_summary["validation_total_pnl_at_least"],
        validation_total_pnl >= target_summary["validation_total_pnl_at_least"],
    )
    add_check(
        "validation_max_drawdown",
        validation_max_drawdown,
        target_summary["validation_max_drawdown_at_most"],
        validation_max_drawdown <= target_summary["validation_max_drawdown_at_most"],
    )
    add_check(
        "validation_total_trades",
        validation_trades,
        target_summary["validation_trades_at_least"],
        validation_trades >= target_summary["validation_trades_at_least"],
    )

    if target_summary["require_train_stability"]:
        add_check(
            "train_win_rate",
            train_win_rate,
            target_summary["train_win_rate_at_least"],
            train_win_rate >= target_summary["train_win_rate_at_least"],
        )
        add_check(
            "train_total_pnl",
            train_total_pnl,
            target_summary["train_total_pnl_at_least"],
            train_total_pnl >= target_summary["train_total_pnl_at_least"],
        )
        add_check(
            "train_total_trades",
            train_trades,
            target_summary["validation_trades_at_least"],
            train_trades >= target_summary["validation_trades_at_least"],
        )
        add_check(
            "train_validation_win_rate_gap",
            win_rate_gap,
            target_summary["train_validation_win_rate_gap_at_most"],
            win_rate_gap <= target_summary["train_validation_win_rate_gap_at_most"],
        )

    target_met = all(item["passed"] for item in checks.values())
    return {
        "target_met": target_met,
        "checks": checks,
    }


def build_round_feedback(
    search_report,
    rejected: List[Dict[str, Any]],
    top_n: int,
    target_summary: Dict[str, Any],
) -> Dict[str, Any]:
    feedback = {
        "quality_target": target_summary,
        "rejected_strategies": [
            {
                "strategy_key": item.get("strategy", {}).get("strategy_key", "<missing>"),
                "error": item.get("error"),
            }
            for item in rejected
        ],
        "top_results": [],
        "weak_results": [],
    }
    if search_report is None:
        return feedback

    ranked = sorted(
        search_report.evaluations,
        key=lambda item: item.blended_score,
        reverse=True,
    )
    for item in ranked[:top_n]:
        feedback["top_results"].append(
            {
                "strategy_key": item.strategy_key,
                "strategy_name": item.strategy_name,
                "status": item.status,
                "blended_score": item.blended_score,
                "validation_pnl": item.validation_metrics.get("total_pnl"),
                "validation_win_rate": item.validation_metrics.get("win_rate"),
                "validation_trades": item.validation_metrics.get("total_trades"),
                "validation_drawdown": item.validation_metrics.get("max_drawdown"),
                "target_evaluation": evaluate_target(item, target_summary),
                "best_params": item.best_params,
                "error": item.error,
            }
        )
    for item in ranked[-top_n:]:
        feedback["weak_results"].append(
            {
                "strategy_key": item.strategy_key,
                "strategy_name": item.strategy_name,
                "status": item.status,
                "blended_score": item.blended_score,
                "validation_pnl": item.validation_metrics.get("total_pnl"),
                "validation_win_rate": item.validation_metrics.get("win_rate"),
                "validation_trades": item.validation_metrics.get("total_trades"),
                "validation_drawdown": item.validation_metrics.get("max_drawdown"),
                "target_evaluation": evaluate_target(item, target_summary),
                "best_params": item.best_params,
                "error": item.error,
            }
        )
    return feedback


def safe_metric(metrics: Dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        value = metrics.get(name, default)
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def load_research_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_file:
        return load_data_from_csv(args.data_file)

    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start_date or (
        datetime.now() - timedelta(days=args.days)
    ).strftime("%Y-%m-%d")

    if args.data_source == "kite":
        from run_ma_mock_validation_majority_kite import fetch_kite_data

        return fetch_kite_data(
            symbol=args.symbol,
            exchange=args.exchange,
            start_date=start_date,
            end_date=end_date,
            interval=args.interval,
        )

    return fetch_data(args.symbol, start_date, end_date, args.interval)


def summarize_market_data(data: pd.DataFrame, sample_rows: int = 0) -> Dict[str, Any]:
    close = data["Close"]
    returns = close.pct_change().dropna()
    summary = {
        "rows": int(len(data)),
        "start": data.index[0].isoformat() if hasattr(data.index[0], "isoformat") else str(data.index[0]),
        "end": data.index[-1].isoformat() if hasattr(data.index[-1], "isoformat") else str(data.index[-1]),
        "columns": list(data.columns),
        "close_min": float(close.min()),
        "close_max": float(close.max()),
        "close_last": float(close.iloc[-1]),
        "mean_return_per_bar": float(returns.mean()) if not returns.empty else 0.0,
        "return_volatility_per_bar": float(returns.std()) if not returns.empty else 0.0,
        "max_single_bar_gain": float(returns.max()) if not returns.empty else 0.0,
        "max_single_bar_loss": float(returns.min()) if not returns.empty else 0.0,
    }
    if sample_rows > 0:
        sample = data.tail(sample_rows).reset_index().astype(str)
        summary["recent_rows"] = sample.to_dict(orient="records")
    return summary


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-driven strategy research")
    parser.add_argument("--llm-url", help="OpenAI-compatible /chat/completions endpoint")
    parser.add_argument("--llm-model", help="Model name for the endpoint")
    parser.add_argument("--llm-api-key", help="API key. Prefer LLM_STRATEGY_API_KEY env var")
    parser.add_argument("--ideas-file", type=Path, help="Use saved LLM ideas JSON instead of calling endpoint")
    parser.add_argument("--allow-llm-web-search", action="store_true")
    parser.add_argument("--n-ideas", type=int, default=5)
    parser.add_argument("--llm-rounds", type=int, default=1)
    parser.add_argument("--feedback-top-n", type=int, default=5)
    parser.add_argument("--context", help="Extra context for the LLM strategy researcher")
    parser.add_argument("--include-baselines", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--data-source", choices=["yahoo", "kite"], default="yahoo")
    parser.add_argument("--symbol", default="BTC-USD")
    parser.add_argument("--exchange", default="MCX")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--min-train-trades", type=int, default=10)
    parser.add_argument("--min-validation-trades", type=int, default=3)
    parser.add_argument("--max-drawdown-threshold", type=float, default=0.5)
    parser.add_argument("--sharpe-threshold", type=float, default=-999.0)
    parser.add_argument("--trading-fee", type=float)
    parser.add_argument("--target-win-rate", type=float, default=0.5)
    parser.add_argument("--target-min-pnl", type=float, default=0.0)
    parser.add_argument("--target-max-drawdown", type=float)
    parser.add_argument("--target-min-trades", type=int)
    parser.add_argument("--target-require-train-stability", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-train-win-rate", type=float)
    parser.add_argument("--target-min-train-pnl", type=float, default=0.0)
    parser.add_argument("--target-max-win-rate-gap", type=float, default=0.25)
    parser.add_argument("--stop-when-target-met", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llm-sample-rows", type=int, default=0)
    parser.add_argument("--print-llm-reply", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("llm_strategy_research_results"))
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_llm_strategy_research(args)


if __name__ == "__main__":
    main()
