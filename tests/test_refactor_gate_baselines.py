import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import data_fetcher
from data_fetcher import BinanceDataFetcher, DataFetcher, KiteDataFetcher
from strategies import DonchianChannelBreakout, MovingAverageCrossover, RSIStrategy
from strategy_optimizer import StrategyOptimizer


BASELINE_PATH = Path(__file__).parent / "fixtures" / "refactor_gate_baselines.json"

STRATEGY_CASES = {
    "ma": {
        "class": MovingAverageCrossover,
        "ranges": {
            "short_window": [3, 5],
            "long_window": [12, 18],
            "risk_reward_ratio": [1.2],
            "trading_fee": [0.0005],
        },
    },
    "rsi": {
        "class": RSIStrategy,
        "ranges": {
            "period": [8, 14],
            "overbought": [70],
            "oversold": [30],
            "risk_reward_ratio": [1.2, 1.8],
            "trading_fee": [0.0005],
        },
    },
    "donchian": {
        "class": DonchianChannelBreakout,
        "ranges": {
            "channel_period": [5, 8],
            "risk_reward_ratio": [1.2, 1.8],
            "trading_fee": [0.0005],
        },
    },
}

BASELINE_METRICS = [
    "total_trades",
    "win_rate",
    "total_pnl",
    "geometric_mean_return",
    "sharpe_ratio",
    "max_drawdown",
    "profit_factor",
    "calmar_ratio",
]


def _gate_ohlcv(periods: int = 240) -> pd.DataFrame:
    index = pd.date_range("2025-01-01 09:15", periods=periods, freq="15min")
    x = np.arange(periods)
    close = 100 + 0.025 * x + 4.0 * np.sin(x / 4.0) + 1.2 * np.sin(x / 1.7)
    open_ = close + 0.2 * np.sin(x / 3.0)
    high = np.maximum(open_, close) + 0.8 + 0.1 * np.cos(x / 5.0)
    low = np.minimum(open_, close) - 0.8 - 0.1 * np.sin(x / 6.0)
    volume = 1000 + (x % 17) * 11

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def _metric_subset(metrics: dict) -> dict:
    return {name: metrics[name] for name in BASELINE_METRICS}


def _assert_metrics_match(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for name, expected_value in expected.items():
        if isinstance(expected_value, float):
            assert actual[name] == pytest.approx(expected_value, rel=1e-10, abs=1e-10)
        else:
            assert actual[name] == expected_value


def _print_gate_case(title: str, **fields) -> None:
    print(f"\n[refactor-gate] {title}")
    for name, value in fields.items():
        print(f"[refactor-gate]   {name}: {value}")


def test_yahoo_fetcher_cleans_mocked_history_and_exposes_key_points(monkeypatch):
    index = pd.date_range("2025-01-01", periods=30, freq="D")
    raw = pd.DataFrame(
        {
            "Open": np.linspace(100, 130, len(index)),
            "High": np.linspace(101, 131, len(index)),
            "Low": np.linspace(99, 129, len(index)),
            "Close": list(np.linspace(100, 110, 21)) + [140, 111, 112, 113, 114, 115, 116, 117, 118],
            "Volume": np.arange(1000, 1000 + len(index)),
        },
        index=index,
    )

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end, interval):
            assert self.symbol == "TEST"
            assert start == "2025-01-01"
            assert end == "2025-02-01"
            assert interval == "1d"
            return raw

    monkeypatch.setattr(data_fetcher, "YFINANCE_AVAILABLE", True)
    monkeypatch.setattr(data_fetcher, "yf", SimpleNamespace(Ticker=FakeTicker))

    fetched = DataFetcher().fetch_data("TEST", "2025-01-01", "2025-02-01")

    assert not fetched.empty
    assert fetched.index.tz.zone == "Asia/Kolkata"
    assert list(fetched.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Returns",
        "Avg_Daily_Return",
        "Volatility",
        "Key_Point",
    ]
    assert fetched["Returns"].notna().sum() == len(fetched) - 1
    assert fetched["Key_Point"].dtype == bool
    assert DataFetcher().get_key_points().empty

    fetcher = DataFetcher()
    fetcher.data = raw
    cleaned = fetcher.clean_data()
    key_points = fetcher.get_key_points()
    assert cleaned is fetcher.data
    assert not key_points.empty
    assert key_points["Key_Point"].all()


def test_binance_cleaner_normalizes_provider_payload_without_network():
    open_time = pd.Timestamp("2025-01-01 00:00:00").timestamp() * 1000
    rows = []
    for i in range(30):
        close = 100 + i + (15 if i == 21 else 0)
        rows.append(
            [
                int(open_time + i * 60_000),
                str(close - 0.5),
                str(close + 1),
                str(close - 1),
                str(close),
                str(10 + i),
                int(open_time + (i + 1) * 60_000),
                "1000.5",
                str(20 + i),
                "4.2",
                "420.0",
                "0",
            ]
        )

    raw = pd.DataFrame(
        rows,
        columns=[
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close Time",
            "Quote Asset Volume",
            "Number of Trades",
            "Taker Buy Base Asset Volume",
            "Taker Buy Quote Asset Volume",
            "Ignore",
        ],
    )

    fetcher = BinanceDataFetcher.__new__(BinanceDataFetcher)
    cleaned = fetcher._clean_binance_data(raw)

    assert list(cleaned.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Returns",
        "Avg_Daily_Return",
        "Volatility",
        "Key_Point",
    ]
    assert cleaned.index.tz.zone == "Asia/Kolkata"
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        assert pd.api.types.is_numeric_dtype(cleaned[column])
    assert cleaned["Returns"].notna().sum() == len(cleaned) - 1
    assert cleaned["Key_Point"].dtype == bool
    assert fetcher.get_key_points(cleaned).equals(cleaned[cleaned["Key_Point"]])


def test_kite_fetcher_uses_instrument_token_and_cleans_historical_data_without_network():
    dates = pd.date_range("2025-01-01 09:15", periods=30, freq="15min", tz="Asia/Kolkata")
    raw_rows = []
    for i, timestamp in enumerate(dates):
        close = 100 + i + (18 if i == 22 else 0)
        raw_rows.append(
            {
                "date": timestamp.to_pydatetime(),
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000 + i,
            }
        )

    class FakeKite:
        def __init__(self):
            self.historical_calls = []

        def historical_data(self, **kwargs):
            self.historical_calls.append(kwargs)
            return raw_rows

    fake_kite = FakeKite()
    fetcher = KiteDataFetcher.__new__(KiteDataFetcher)
    fetcher.kite = fake_kite
    fetcher.instrument_tokens = {"TATAMOTORS": 884737}
    fetcher.exchange = "NSE"

    cleaned = fetcher.fetch_historical_data(
        "TATAMOTORS",
        "2025-01-01",
        "2025-01-02",
        interval="15minute",
        continuous=True,
    )

    assert fake_kite.historical_calls == [
        {
            "instrument_token": 884737,
            "from_date": "2025-01-01",
            "to_date": "2025-01-02",
            "interval": "15minute",
            "continuous": True,
        }
    ]
    assert list(cleaned.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Returns",
        "Avg_Daily_Return",
        "Volatility",
        "Key_Point",
    ]
    assert cleaned.index.tz.zone == "Asia/Kolkata"
    assert cleaned.index.is_monotonic_increasing
    assert cleaned["Returns"].notna().sum() == len(cleaned) - 1
    assert cleaned["Key_Point"].dtype == bool
    assert fetcher.get_key_points(cleaned).equals(cleaned[cleaned["Key_Point"]])


@pytest.mark.parametrize("strategy_name", ["ma", "rsi", "donchian"])
def test_strategy_optimizer_matches_refactor_gate_baseline(strategy_name):
    baselines = json.loads(BASELINE_PATH.read_text())
    expected = baselines[strategy_name]
    case = STRATEGY_CASES[strategy_name]
    data = _gate_ohlcv()
    train_data = data.iloc[:160]

    optimizer = StrategyOptimizer(
        data=train_data,
        strategy_class=case["class"],
        param_ranges=case["ranges"],
        optimization_metric="composite_score",
        min_trades=1,
        max_drawdown_threshold=1.0,
        sharpe_threshold=-999,
    )

    best_params, best_metrics = optimizer.optimize()
    actual_metrics = _metric_subset(best_metrics)
    _print_gate_case(
        f"{strategy_name} optimizer baseline",
        strategy_class=case["class"].__name__,
        param_ranges=case["ranges"],
        tested_results=len(optimizer.results),
        failed_runs=len(optimizer.failed_runs),
        best_params=best_params,
        actual_train_metrics=actual_metrics,
        expected_train_metrics=expected["train_metrics"],
    )

    assert best_params == expected["best_params"]
    assert len(optimizer.results) == expected["tested_results"]
    assert len(optimizer.failed_runs) == expected["failed_runs"]
    _assert_metrics_match(actual_metrics, expected["train_metrics"])


@pytest.mark.parametrize("strategy_name", ["ma", "rsi", "donchian"])
def test_optimized_strategy_runs_on_future_data_baseline(strategy_name):
    baselines = json.loads(BASELINE_PATH.read_text())
    expected = baselines[strategy_name]
    case = STRATEGY_CASES[strategy_name]
    future_data = _gate_ohlcv().iloc[160:]

    strategy = case["class"](**expected["best_params"])
    signals = strategy.generate_signals(future_data)
    metrics = strategy.get_strategy_metrics()
    signal_counts = {
        str(signal): int(count)
        for signal, count in signals["Signal"].value_counts().sort_index().items()
    }
    actual_metrics = _metric_subset(metrics)
    _print_gate_case(
        f"{strategy_name} future-data baseline",
        strategy_class=case["class"].__name__,
        params=expected["best_params"],
        future_rows=len(future_data),
        signal_counts=signal_counts,
        actual_future_metrics=actual_metrics,
        expected_future_metrics=expected["future_metrics"],
    )

    assert signals.shape[0] == len(future_data)
    assert {"Signal", "Position", "Take_Profit", "Stop_Loss"}.issubset(signals.columns)
    assert signal_counts == expected["future_signal_counts"]
    _assert_metrics_match(actual_metrics, expected["future_metrics"])
