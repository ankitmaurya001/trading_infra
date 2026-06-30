import numpy as np
import pandas as pd
import pytest

from strategies import (
    BaseStrategy,
    MovingAverageCrossover,
    STRATEGY_CLASSES,
    STRATEGY_DEFINITIONS,
    StrategyDefinition,
)
from strategy_manager import StrategyManager
from strategy_optimizer import StrategyOptimizer


def _sample_ohlcv(periods: int = 120) -> pd.DataFrame:
    index = pd.date_range("2025-01-01 09:15", periods=periods, freq="15min")
    x = np.arange(periods)
    close = 100 + 0.04 * x + 3.5 * np.sin(x / 4.0)
    open_ = close + 0.1 * np.sin(x / 3.0)
    high = np.maximum(open_, close) + 0.7
    low = np.minimum(open_, close) - 0.7
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


@pytest.mark.parametrize("strategy_class", STRATEGY_CLASSES)
def test_builtin_strategies_expose_template_definitions(strategy_class):
    definition = strategy_class.get_definition()
    key = strategy_class.STRATEGY_KEY

    assert isinstance(definition, StrategyDefinition)
    assert STRATEGY_DEFINITIONS[key] == definition
    assert definition.key == key
    assert issubclass(definition.strategy_class, BaseStrategy)
    assert definition.optimization_param_ranges
    assert definition.live_parameter_names
    assert set(definition.live_parameter_names).issubset(definition.optimization_param_ranges)


@pytest.mark.parametrize("definition", STRATEGY_DEFINITIONS.values())
def test_strategy_definition_builds_live_strategy_and_generates_required_columns(definition):
    params = definition.default_params()
    assert definition.validate_parameters(params)

    strategy = definition.create_strategy(params)
    signals = strategy.generate_signals(_sample_ohlcv())

    assert isinstance(strategy, BaseStrategy)
    assert signals.shape[0] == 120
    assert {"Signal", "Position", "Take_Profit", "Stop_Loss"}.issubset(signals.columns)
    assert isinstance(strategy.get_strategy_metrics(), dict)


def test_strategy_optimizer_uses_strategy_owned_parameter_validation():
    data = _sample_ohlcv()
    optimizer = StrategyOptimizer(
        data=data,
        strategy_class=MovingAverageCrossover,
        param_ranges={
            "short_window": [10],
            "long_window": [5],
            "risk_reward_ratio": [1.5],
            "trading_fee": [0.001],
        },
        min_trades=0,
        sharpe_threshold=-999,
    )

    best_params, best_metrics = optimizer.optimize()

    assert best_params is None
    assert best_metrics is None
    assert optimizer.results == []
    assert optimizer.failed_runs == []


def test_strategy_manager_uses_definition_for_live_initialization_and_validation():
    manager = StrategyManager()
    manager.set_manual_parameters(
        ma_params={
            "short_window": 3,
            "long_window": 8,
            "risk_reward_ratio": 1.5,
            "trading_fee": 0.001,
        }
    )

    strategies = manager.initialize_strategies(["ma"])

    assert len(strategies) == 1
    assert isinstance(strategies[0], MovingAverageCrossover)
    assert manager.get_strategy_by_name("ma") is strategies[0]
    assert manager.validate_strategy_parameters("ma", manager.get_strategy_parameters("ma"))
    assert not manager.validate_strategy_parameters(
        "ma",
        {
            "short_window": 8,
            "long_window": 3,
            "risk_reward_ratio": 1.5,
            "trading_fee": 0.001,
        },
    )
