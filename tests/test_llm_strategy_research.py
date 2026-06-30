import argparse
import json

import numpy as np
import pandas as pd

from llm_strategy_research import (
    build_definitions_from_llm_payload,
    parse_llm_json,
    run_llm_strategy_research,
)


def _sample_ohlcv(path):
    periods = 120
    index = pd.date_range("2025-01-01 09:15", periods=periods, freq="15min")
    x = np.arange(periods)
    close = 100 + 0.04 * x + 3.5 * np.sin(x / 4.0)
    data = pd.DataFrame(
        {
            "Date": index,
            "Open": close + 0.1,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(periods, 1000),
        }
    )
    data.to_csv(path, index=False)


def _ideas_payload():
    return {
        "research_notes": "Try an EMA crossover proposed by the LLM.",
        "strategies": [
            {
                "type": "rule_dsl",
                "strategy_key": "ema_cross_test",
                "strategy_name": "EMA Cross Test",
                "hypothesis": "Short EMA crossing long EMA captures trend changes.",
                "param_ranges": {
                    "fast_period": [3, 5],
                    "slow_period": [8, 10],
                    "risk_reward_ratio": [1.5],
                    "trading_fee": [0.001],
                },
                "constraints": [
                    {"op": "lt", "left": "fast_period", "right": "slow_period"},
                ],
                "indicators": [
                    {"kind": "ema", "name": "ema_fast", "source": "Close", "period": "fast_period"},
                    {"kind": "ema", "name": "ema_slow", "source": "Close", "period": "slow_period"},
                ],
                "long_entry": {"op": "crosses_above", "left": "ema_fast", "right": "ema_slow"},
                "short_entry": {"op": "crosses_below", "left": "ema_fast", "right": "ema_slow"},
            }
        ],
    }


def test_parse_llm_json_extracts_object_from_markdown():
    parsed = parse_llm_json("```json\n{\"strategies\": []}\n```")

    assert parsed == {"strategies": []}


def test_build_definitions_from_llm_payload_without_baselines():
    definitions, rejected = build_definitions_from_llm_payload(
        _ideas_payload(),
        include_baselines=False,
    )

    assert rejected == []
    assert list(definitions) == ["llm_ema_cross_test"]


def test_run_llm_strategy_research_from_ideas_file(tmp_path):
    ideas_file = tmp_path / "ideas.json"
    data_file = tmp_path / "data.csv"
    ideas_file.write_text(json.dumps(_ideas_payload()), encoding="utf-8")
    _sample_ohlcv(data_file)

    args = argparse.Namespace(
        ideas_file=ideas_file,
        llm_url=None,
        llm_model=None,
        llm_api_key=None,
        allow_llm_web_search=False,
        n_ideas=1,
        llm_rounds=1,
        feedback_top_n=3,
        context=None,
        include_baselines=False,
        data_file=data_file,
        data_source="yahoo",
        symbol="TEST",
        exchange="MCX",
        start_date=None,
        end_date=None,
        days=120,
        interval="15m",
        generations=1,
        top_k=1,
        train_ratio=0.7,
        min_train_trades=0,
        min_validation_trades=0,
        max_drawdown_threshold=1.0,
        sharpe_threshold=-999,
        trading_fee=None,
        target_win_rate=0.0,
        target_min_pnl=-1.0,
        target_max_drawdown=1.0,
        target_min_trades=0,
        target_require_train_stability=False,
        target_train_win_rate=None,
        target_min_train_pnl=0.0,
        target_max_win_rate_gap=1.0,
        stop_when_target_met=True,
        llm_sample_rows=0,
        print_llm_reply=False,
        out_dir=tmp_path / "out",
        quiet=True,
    )

    result = run_llm_strategy_research(args)

    assert result.accepted_strategy_keys == ["llm_ema_cross_test"]
    assert result.rejected_strategies == []
    assert (tmp_path / "out" / "round_01" / "llm_strategy_ideas.json").exists()
    assert (tmp_path / "out" / "llm_strategy_research_report.json").exists()
    assert (tmp_path / "out" / "aggregate_strategy_search_report.json").exists()
    assert (tmp_path / "out" / "aggregate_strategy_rankings.csv").exists()
    assert (tmp_path / "out" / "round_01" / "strategy_search_rankings.csv").exists()
