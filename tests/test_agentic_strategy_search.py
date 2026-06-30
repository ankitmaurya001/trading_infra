from dataclasses import replace

import numpy as np
import pandas as pd

from agentic_strategy_search import StrategySearchAgent, StrategySearchConfig
from strategies import STRATEGY_DEFINITIONS


def _sample_ohlcv(periods: int = 140) -> pd.DataFrame:
    index = pd.date_range("2025-01-01 09:15", periods=periods, freq="15min")
    x = np.arange(periods)
    close = 100 + 0.03 * x + 4.0 * np.sin(x / 4.0) + 0.8 * np.sin(x / 1.7)
    open_ = close + 0.1 * np.sin(x / 3.0)
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


def _small_strategy_definitions():
    return {
        "ma": replace(
            STRATEGY_DEFINITIONS["ma"],
            optimization_param_ranges={
                "short_window": [3, 5],
                "long_window": [8, 10],
                "risk_reward_ratio": [1.5],
                "trading_fee": [0.001],
            },
        ),
        "macd": replace(
            STRATEGY_DEFINITIONS["macd"],
            optimization_param_ranges={
                "fast_period": [3, 5],
                "slow_period": [8, 10],
                "signal_period": [3],
                "risk_reward_ratio": [1.5],
                "trading_fee": [0.001],
            },
        ),
    }


def test_agentic_strategy_search_runs_generations_and_saves_artifacts(tmp_path):
    config = StrategySearchConfig(
        generations=2,
        top_k=1,
        train_ratio=0.65,
        min_train_trades=0,
        min_validation_trades=0,
        max_drawdown_threshold=1.0,
        sharpe_threshold=-999,
    )
    agent = StrategySearchAgent(
        config=config,
        strategy_definitions=_small_strategy_definitions(),
    )

    report = agent.run(_sample_ohlcv(), verbose=False)
    paths = agent.save_report(report, tmp_path)

    assert report.train_rows == 91
    assert report.validation_rows == 49
    assert report.best is not None
    assert report.best.status == "ok"
    assert len(report.evaluations) >= 3
    assert {item.generation for item in report.evaluations} == {0, 1}
    assert paths["json"].exists()
    assert paths["csv"].exists()


def test_agentic_strategy_search_rejects_unknown_strategy_key():
    agent = StrategySearchAgent(
        config=StrategySearchConfig(generations=1),
        strategy_definitions=_small_strategy_definitions(),
    )

    try:
        agent.run(_sample_ohlcv(), strategy_keys=["missing"], verbose=False)
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected missing strategy key to fail")
