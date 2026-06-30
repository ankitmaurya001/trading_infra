import numpy as np
import pandas as pd
import pytest

from strategy_dsl import build_rule_strategy_definition, validate_rule_strategy_proposal
from strategy_optimizer import StrategyOptimizer


def _sample_ohlcv(periods: int = 120) -> pd.DataFrame:
    index = pd.date_range("2025-01-01 09:15", periods=periods, freq="15min")
    x = np.arange(periods)
    close = 100 + 0.03 * x + 3.0 * np.sin(x / 3.5)
    open_ = close + 0.1 * np.sin(x / 5.0)
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(periods, 1000),
        },
        index=index,
    )


def _ema_crossover_proposal():
    return {
        "type": "rule_dsl",
        "strategy_key": "ema_rsi_trend",
        "strategy_name": "EMA RSI Trend",
        "hypothesis": "Trend-following crossover filtered by RSI.",
        "param_ranges": {
            "fast_period": [3, 5],
            "slow_period": [8, 10],
            "rsi_period": [6],
            "rsi_long_max": [70],
            "rsi_short_min": [30],
            "risk_reward_ratio": [1.5],
            "trading_fee": [0.001],
        },
        "constraints": [
            {"op": "lt", "left": "fast_period", "right": "slow_period"},
        ],
        "indicators": [
            {"kind": "ema", "name": "ema_fast", "source": "Close", "period": "fast_period"},
            {"kind": "ema", "name": "ema_slow", "source": "Close", "period": "slow_period"},
            {"kind": "rsi", "name": "rsi", "source": "Close", "period": "rsi_period"},
        ],
        "long_entry": {
            "op": "all",
            "conditions": [
                {"op": "crosses_above", "left": "ema_fast", "right": "ema_slow"},
                {"op": "lt", "left": "rsi", "right": "rsi_long_max"},
            ],
        },
        "short_entry": {
            "op": "all",
            "conditions": [
                {"op": "crosses_below", "left": "ema_fast", "right": "ema_slow"},
                {"op": "gt", "left": "rsi", "right": "rsi_short_min"},
            ],
        },
    }


def test_rule_strategy_definition_can_optimize_and_generate_signals():
    definition = build_rule_strategy_definition(_ema_crossover_proposal())
    data = _sample_ohlcv()

    optimizer = StrategyOptimizer(
        data=data.iloc[:80],
        strategy_class=definition.strategy_class,
        param_ranges=definition.optimization_param_ranges,
        min_trades=0,
        sharpe_threshold=-999,
        max_drawdown_threshold=1.0,
        run_robustness_analysis=False,
    )
    best_params, best_metrics = optimizer.optimize()

    strategy = definition.create_strategy(best_params)
    signals = strategy.generate_signals(data.iloc[80:])

    assert best_params is not None
    assert isinstance(best_metrics, dict)
    assert {"Signal", "Position", "Take_Profit", "Stop_Loss"}.issubset(signals.columns)
    assert isinstance(strategy.get_strategy_metrics(), dict)


def test_rule_strategy_proposal_rejects_unsupported_indicator_kind():
    proposal = _ema_crossover_proposal()
    proposal["indicators"][0]["kind"] = "model_written_python"

    with pytest.raises(ValueError, match="Unsupported indicator kind"):
        validate_rule_strategy_proposal(proposal)
