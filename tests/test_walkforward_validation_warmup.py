import pandas as pd

from run_ma_mock_validation_majority_kite import run_majority_vote_validation


def _ohlc_from_closes(index, closes):
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [price + 1 for price in closes],
            "Low": [price - 1 for price in closes],
            "Close": closes,
            "Volume": [100] * len(closes),
        },
        index=index,
    )


def test_validation_start_uses_warmup_candles_for_ma_cross():
    validation_start = pd.Timestamp("2026-03-13 10:45:00")
    index = pd.date_range("2026-03-13 09:15:00", periods=8, freq="15min")
    # The final warmup candle has SMA(2) == SMA(5). The first validation candle
    # jumps high enough to make SMA(2) cross above SMA(5), so validation should
    # enter immediately instead of waiting for five new validation candles.
    data = _ohlc_from_closes(index, [10, 10, 10, 10, 10, 10, 20, 21])

    result = run_majority_vote_validation(
        data=data,
        param_sets=[{"short_window": 2, "long_window": 5, "risk_reward_ratio": 2.0}],
        symbol="TEST",
        verbose=False,
        start_trading_at=validation_start,
    )

    assert result.loc[validation_start, "event"] == "LONG_ENTRY"
    assert pd.notna(result.loc[validation_start, "SMA_short_0"])
    assert pd.notna(result.loc[validation_start, "SMA_long_0"])
    assert not result[result.index < validation_start]["event"].astype(bool).any()


def test_validation_does_not_reenter_on_stale_majority_after_trade_exit():
    index = pd.date_range("2026-03-20 09:15:00", periods=8, freq="5min")
    data = _ohlc_from_closes(index, [10, 10, 10, 9, 8, 7, 8.95, 8.8])

    result = run_majority_vote_validation(
        data=data,
        param_sets=[{"short_window": 2, "long_window": 3, "risk_reward_ratio": 10.0}],
        symbol="TEST",
        verbose=False,
        enable_trailing_stop=True,
        breakeven_activation_r=1.0,
        breakeven_buffer_atr=0.1,
        trailing_activation_r=100.0,
    )

    entries = result[result["event"].isin(["LONG_ENTRY", "SHORT_ENTRY"])]
    assert list(entries["event"]) == ["SHORT_ENTRY"]
    assert result.loc[index[6], "event"] == "EXIT_SL"
    assert result.loc[index[7], "majority_signal"] == -1
    assert result.loc[index[7], "event"] == ""
