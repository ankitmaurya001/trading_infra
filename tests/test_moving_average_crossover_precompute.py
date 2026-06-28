import numpy as np
import pandas as pd
import pandas.testing as pdt

from strategies import MovingAverageCrossover
from strategy_optimizer import StrategyOptimizer


def _sample_ohlcv(periods: int = 80) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="h")
    base = 100 + np.sin(np.linspace(0, 8 * np.pi, periods)) * 5 + np.linspace(0, 3, periods)
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 1.5,
            "Low": base - 1.5,
            "Close": base,
            "Volume": np.full(periods, 1000),
        },
        index=index,
    )


def test_precomputed_ma_signals_match_standard_generation():
    data = _sample_ohlcv()

    standard_strategy = MovingAverageCrossover(short_window=3, long_window=8, risk_reward_ratio=1.5)
    standard_signals = standard_strategy.generate_signals(data)
    standard_metrics = standard_strategy.get_strategy_metrics()

    cached_strategy = MovingAverageCrossover(short_window=3, long_window=8, risk_reward_ratio=1.5)
    cache = cached_strategy.precompute_indicators(data, [3, 8])
    cached_signals = cached_strategy.generate_signals_from_precomputed(
        data,
        cache["SMA"][3],
        cache["SMA"][8],
        cache["ATR"],
    )
    cached_metrics = cached_strategy.get_strategy_metrics()

    pdt.assert_frame_equal(standard_signals, cached_signals)
    assert standard_metrics == cached_metrics


def test_optimizer_reuses_ma_indicator_cache_for_grid_and_sensitivity_windows():
    data = _sample_ohlcv()
    optimizer = StrategyOptimizer(
        data=data,
        strategy_class=MovingAverageCrossover,
        param_ranges={
            "short_window": [3, 4],
            "long_window": [8, 10],
            "risk_reward_ratio": [1.5],
        },
        min_trades=0,
        sharpe_threshold=-999,
    )

    best_params, best_metrics = optimizer.optimize()

    assert best_params is not None
    assert isinstance(best_metrics, dict)
    assert optimizer._ma_indicator_cache is not None
    assert set(optimizer._ma_indicator_cache["SMA"]) == {3, 4, 8, 10}
    assert "ATR" in optimizer._ma_indicator_cache

    cached_windows_before = set(optimizer._ma_indicator_cache["SMA"])
    top_result = optimizer.get_top_results(1)
    optimizer.analyze_local_sensitivity(top_result, variation_percent=0.5)

    assert cached_windows_before.issubset(set(optimizer._ma_indicator_cache["SMA"]))
