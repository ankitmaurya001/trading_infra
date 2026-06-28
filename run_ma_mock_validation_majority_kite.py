#!/usr/bin/env python3
"""
Run MA majority-vote mock validation on Zerodha Kite data.

This is a validation-style runner (similar to run_ma_mock_validation_kite.py),
not an optimization runner. It supports multiple MA parameter sets where each
set maintains its own directional vote until its own TP is reached.
"""

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import os
import time
import webbrowser
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_fetcher import KiteDataFetcher
from strategies import PositionType
import config as cfg

# ============================================================================
# Defaults (validation-style)
# ============================================================================
DEFAULT_SYMBOL = "NATGASMINI26FEBFUT"
DEFAULT_EXCHANGE = "MCX"
DEFAULT_INTERVAL = "15m"
DAYS_TO_VALIDATE = 30
DEFAULT_START_DATE = (datetime.now() - timedelta(days=DAYS_TO_VALIDATE)).strftime(
    "%Y-%m-%d"
)
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")

DEFAULT_PARAM_SETS = [
    {"short_window": 10, "long_window": 170, "risk_reward_ratio": 7.5},
    {"short_window": 5, "long_window": 170, "risk_reward_ratio": 6.5},
    {"short_window": 10, "long_window": 150, "risk_reward_ratio": 6.0},
]
DEFAULT_NUM_LOTS = 1
DEFAULT_LOT_SIZE = 250
DEFAULT_ENABLE_TRAILING_STOP = False
DEFAULT_BREAKEVEN_ACTIVATION_R = 3.0
DEFAULT_BREAKEVEN_BUFFER_ATR = 1
DEFAULT_TRAILING_ACTIVATION_R = DEFAULT_BREAKEVEN_ACTIVATION_R + 1
DEFAULT_TRAILING_ATR_MULTIPLIER = 3.0
# ============================================================================


@dataclass
class ParamSetState:
    short_window: int
    long_window: int
    risk_reward_ratio: float
    state: PositionType = PositionType.NONE
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    activated_at: Optional[int] = None


def map_interval_to_kite(interval: str) -> str:
    interval_mapping = {
        "1m": "minute",
        "3m": "3minute",
        "5m": "5minute",
        "15m": "15minute",
        "30m": "30minute",
        "1h": "hour",
        "2h": "2hour",
        "4h": "4hour",
        "1d": "day",
        "1w": "week",
        "1M": "month",
    }
    return interval_mapping.get(interval, "15minute")


def convert_date_format(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
    except ValueError:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")


def fetch_kite_data(
    symbol: str, exchange: str, start_date: str, end_date: str, interval: str
) -> pd.DataFrame:
    kite_interval = map_interval_to_kite(interval)
    start_date_kite = convert_date_format(start_date)
    end_date_kite = convert_date_format(end_date)

    print(
        f"📊 Fetching validation data: {symbol} [{exchange}] {start_date_kite} -> {end_date_kite} ({kite_interval})"
    )

    kite_fetcher = KiteDataFetcher(credentials=cfg.KITE_CREDENTIALS, exchange=exchange)
    kite_fetcher.authenticate()
    data = kite_fetcher.fetch_historical_data(
        symbol=symbol,
        start_date=start_date_kite,
        end_date=end_date_kite,
        interval=kite_interval,
    )

    if data is None or data.empty:
        raise ValueError("No data fetched from Kite")
    return data


def validate_param_sets(param_sets: List[Dict]) -> List[Dict]:
    if not param_sets:
        raise ValueError("At least one parameter set is required")

    validated = []
    for i, p in enumerate(param_sets):
        short_window = int(p["short_window"])
        long_window = int(p["long_window"])
        rr = float(p["risk_reward_ratio"])

        if short_window < 2:
            raise ValueError(f"param set {i}: short_window must be >= 2")
        if long_window <= short_window:
            raise ValueError(f"param set {i}: long_window must be > short_window")
        if rr <= 0:
            raise ValueError(f"param set {i}: risk_reward_ratio must be > 0")

        validated.append(
            {
                "short_window": short_window,
                "long_window": long_window,
                "risk_reward_ratio": rr,
            }
        )
    return validated


def load_param_sets(
    params_json: Optional[str], params_file: Optional[str]
) -> List[Dict]:
    if params_json:
        return validate_param_sets(json.loads(params_json))
    if params_file:
        with open(params_file, "r", encoding="utf-8") as f:
            return validate_param_sets(json.load(f))
    return validate_param_sets(DEFAULT_PARAM_SETS)


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    tr1 = data["High"] - data["Low"]
    tr2 = (data["High"] - data["Close"].shift()).abs()
    tr3 = (data["Low"] - data["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _fmt_level(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "NA"


def _calculate_pnl_rupees(
    side: str,
    entry_price: float,
    exit_price: float,
    num_lots: int = DEFAULT_NUM_LOTS,
    lot_size: int = DEFAULT_LOT_SIZE,
) -> float:
    price_diff = (
        (exit_price - entry_price) if side == "BUY" else (entry_price - exit_price)
    )
    return price_diff * num_lots * lot_size


def _price_from_row(row: pd.Series, column: str, fallback: float) -> float:
    value = row.get(column)
    if pd.notna(value):
        return float(value)
    return fallback


def _initialize_mfe_fields(trade: Dict, ts, row: pd.Series, price: float) -> None:
    del row
    trade["max_favorable_price"] = price
    trade["max_favorable_time"] = ts
    trade["max_favorable_pnl"] = 0.0
    trade["max_favorable_pnl_rupees"] = 0.0


def _update_mfe_fields(
    trade: Dict, ts, row: pd.Series, price: float, favorable_price_override=None
) -> None:
    favorable_price = (
        float(favorable_price_override)
        if favorable_price_override is not None
        else (
            _price_from_row(row, "High", price)
            if trade["action"] == "BUY"
            else _price_from_row(row, "Low", price)
        )
    )
    existing = trade.get("max_favorable_price", trade["entry_price"])
    is_more_favorable = (
        favorable_price > existing
        if trade["action"] == "BUY"
        else favorable_price < existing
    )
    if is_more_favorable:
        trade["max_favorable_price"] = favorable_price
        trade["max_favorable_time"] = ts
        trade["max_favorable_pnl"] = max(
            0.0,
            (
                (favorable_price - trade["entry_price"]) / trade["entry_price"]
                if trade["action"] == "BUY"
                else (trade["entry_price"] - favorable_price) / trade["entry_price"]
            ),
        )
        trade["max_favorable_pnl_rupees"] = max(
            0.0,
            _calculate_pnl_rupees(
                side=trade["action"],
                entry_price=trade["entry_price"],
                exit_price=favorable_price,
            ),
        )


def _finalize_trade_pnl_fields(trade: Dict, exit_price: float) -> None:
    pnl = (
        (exit_price - trade["entry_price"]) / trade["entry_price"]
        if trade["action"] == "BUY"
        else (trade["entry_price"] - exit_price) / trade["entry_price"]
    )
    trade["pnl"] = pnl
    trade["pnl_rupees"] = _calculate_pnl_rupees(
        side=trade["action"],
        entry_price=trade["entry_price"],
        exit_price=exit_price,
    )
    trade["profit_given_back"] = max(
        0.0, float(trade.get("max_favorable_pnl", 0.0)) - pnl
    )
    trade["profit_given_back_rupees"] = max(
        0.0,
        float(trade.get("max_favorable_pnl_rupees", 0.0)) - trade["pnl_rupees"],
    )


def run_majority_vote_validation(
    data: pd.DataFrame,
    param_sets: List[Dict],
    symbol: str = DEFAULT_SYMBOL,
    initial_balance: float = 10000.0,
    verbose: bool = True,
    mock_delay: float = 0.0,
    stop_on_result: bool = False,
    max_consecutive_losses: int = 5,
    return_stop_metadata: bool = False,
    enable_trailing_stop: bool = DEFAULT_ENABLE_TRAILING_STOP,
    breakeven_activation_r: float = DEFAULT_BREAKEVEN_ACTIVATION_R,
    breakeven_buffer_atr: float = DEFAULT_BREAKEVEN_BUFFER_ATR,
    trailing_activation_r: float = DEFAULT_TRAILING_ACTIVATION_R,
    trailing_atr_multiplier: float = DEFAULT_TRAILING_ATR_MULTIPLIER,
    start_trading_at: Optional[Union[str, date, datetime, pd.Timestamp]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[str]]]:
    df = data.copy()
    trading_start_ts: Optional[pd.Timestamp] = None
    if start_trading_at is not None:
        trading_start_ts = pd.Timestamp(start_trading_at)
    df["ATR"] = calculate_atr(df)

    states: List[ParamSetState] = []
    for idx, p in enumerate(param_sets):
        short_col = f"SMA_short_{idx}"
        long_col = f"SMA_long_{idx}"
        df[short_col] = df["Close"].rolling(window=p["short_window"]).mean()
        df[long_col] = df["Close"].rolling(window=p["long_window"]).mean()

        states.append(
            ParamSetState(
                short_window=p["short_window"],
                long_window=p["long_window"],
                risk_reward_ratio=p["risk_reward_ratio"],
            )
        )

    df["long_votes"] = 0
    df["short_votes"] = 0
    df["majority_signal"] = 0
    df["position"] = 0
    df["trade_take_profit"] = np.nan
    df["trade_stop_loss"] = np.nan
    df["trade_trailing_active"] = False
    df["trade_unrealized_r"] = np.nan
    df["trade_max_favorable_price"] = np.nan
    df["event"] = ""

    current_position = PositionType.NONE
    current_trade_tp: Optional[float] = None
    current_trade_sl: Optional[float] = None
    current_trade_entry_price: Optional[float] = None
    current_trade_entry_atr: Optional[float] = None
    current_trade_entry_balance: Optional[float] = None
    current_trade_side: Optional[str] = None
    current_trade_initial_risk: Optional[float] = None
    current_trade_highest_price: Optional[float] = None
    current_trade_lowest_price: Optional[float] = None
    current_trade_trailing_active = False
    sim_balance = float(initial_balance)
    previous_majority_signal = 0
    min_required_votes = len(states) // 2 + 1

    consecutive_losses = 0
    stop_timestamp: Optional[pd.Timestamp] = None
    stop_reason: Optional[str] = None

    for i in range(1, len(df)):
        realized_trade_pnl: Optional[float] = None
        price = float(df["Close"].iloc[i])
        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])
        atr = float(df["ATR"].iloc[i])
        if np.isnan(atr) or atr <= 0:
            atr = price * 0.01

        current_ts = pd.Timestamp(df.index[i])
        trading_enabled = True
        if trading_start_ts is not None:
            comparable_start_ts = trading_start_ts
            if current_ts.tzinfo is not None and comparable_start_ts.tzinfo is None:
                comparable_start_ts = comparable_start_ts.tz_localize(current_ts.tzinfo)
            elif current_ts.tzinfo is None and comparable_start_ts.tzinfo is not None:
                comparable_start_ts = comparable_start_ts.tz_localize(None)
            trading_enabled = current_ts >= comparable_start_ts

        if not trading_enabled:
            if mock_delay > 0:
                time.sleep(mock_delay)
            continue

        for idx, st in enumerate(states):
            short_col = f"SMA_short_{idx}"
            long_col = f"SMA_long_{idx}"
            prev_short = df[short_col].iloc[i - 1]
            prev_long = df[long_col].iloc[i - 1]
            curr_short = df[short_col].iloc[i]
            curr_long = df[long_col].iloc[i]

            if (
                np.isnan(prev_short)
                or np.isnan(prev_long)
                or np.isnan(curr_short)
                or np.isnan(curr_long)
            ):
                continue

            # Keep S_i vote active until either TP or SL of S_i is hit.
            if st.state == PositionType.LONG:
                hit_tp = st.take_profit is not None and price >= st.take_profit
                hit_sl = st.stop_loss is not None and price <= st.stop_loss
                if hit_tp or hit_sl:
                    if verbose:
                        reason = "TP" if hit_tp else "SL"
                        print(
                            f"🗳️  [{df.index[i]}] Vote {idx + 1} LONG inactive ({reason} hit @ {price:.2f})"
                        )
                    st.state = PositionType.NONE
                    st.take_profit = None
                    st.stop_loss = None
                    st.activated_at = None
            elif st.state == PositionType.SHORT:
                hit_tp = st.take_profit is not None and price <= st.take_profit
                hit_sl = st.stop_loss is not None and price >= st.stop_loss
                if hit_tp or hit_sl:
                    if verbose:
                        reason = "TP" if hit_tp else "SL"
                        print(
                            f"🗳️  [{df.index[i]}] Vote {idx + 1} SHORT inactive ({reason} hit @ {price:.2f})"
                        )
                    st.state = PositionType.NONE
                    st.take_profit = None
                    st.stop_loss = None
                    st.activated_at = None

            if st.state == PositionType.NONE:
                bullish_cross = prev_short <= prev_long and curr_short > curr_long
                bearish_cross = prev_short >= prev_long and curr_short < curr_long

                if bullish_cross:
                    st.state = PositionType.LONG
                    st.stop_loss = price - atr
                    st.take_profit = price + (atr * st.risk_reward_ratio)
                    st.activated_at = i
                    if verbose:
                        print(
                            f"🗳️  [{df.index[i]}] Vote {idx + 1} ACTIVE LONG | ATR={atr:.2f} SL={st.stop_loss:.2f} TP={st.take_profit:.2f}"
                        )
                elif bearish_cross:
                    st.state = PositionType.SHORT
                    st.stop_loss = price + atr
                    st.take_profit = price - (atr * st.risk_reward_ratio)
                    st.activated_at = i
                    if verbose:
                        print(
                            f"🗳️  [{df.index[i]}] Vote {idx + 1} ACTIVE SHORT | ATR={atr:.2f} SL={st.stop_loss:.2f} TP={st.take_profit:.2f}"
                        )

        long_votes = sum(1 for s in states if s.state == PositionType.LONG)
        short_votes = sum(1 for s in states if s.state == PositionType.SHORT)
        df.loc[df.index[i], "long_votes"] = long_votes
        df.loc[df.index[i], "short_votes"] = short_votes

        majority_signal = 0
        if long_votes >= min_required_votes and long_votes > short_votes:
            majority_signal = 1
        elif short_votes >= min_required_votes and short_votes > long_votes:
            majority_signal = -1
        df.loc[df.index[i], "majority_signal"] = majority_signal

        majority_changed = majority_signal != previous_majority_signal
        if verbose and majority_changed:
            label = (
                "LONG"
                if majority_signal == 1
                else "SHORT" if majority_signal == -1 else "NONE"
            )
            print(
                f"🧮 [{df.index[i]}] Majority -> {label} (long_votes={long_votes}, short_votes={short_votes}, required={min_required_votes})"
            )

        latest_long_vote = max(
            (
                s
                for s in states
                if s.state == PositionType.LONG and s.activated_at is not None
            ),
            key=lambda s: s.activated_at,
            default=None,
        )
        latest_short_vote = max(
            (
                s
                for s in states
                if s.state == PositionType.SHORT and s.activated_at is not None
            ),
            key=lambda s: s.activated_at,
            default=None,
        )

        # Entry-only model:
        # - Open trade only when flat and a new strict majority appears.
        # - Once in trade, ignore majority changes; exit only via TP/SL.
        if current_position == PositionType.NONE and majority_changed:
            if majority_signal == 1:
                df.loc[df.index[i], "event"] = "LONG_ENTRY"
                current_position = PositionType.LONG
                current_trade_tp = (
                    latest_long_vote.take_profit if latest_long_vote else None
                )
                current_trade_sl = (
                    latest_long_vote.stop_loss if latest_long_vote else None
                )
                if verbose:
                    print(
                        f"🔄 [{df.index[i]}] Majority Vote - BUY 1.0000 {symbol} @ ₹{price:.2f}"
                    )
                    print(f"💰 Balance Before Entry: ₹{sim_balance:.2f}")
                    print(f"📊 ATR: ₹{atr:.2f}")
                    print(f"🛑 Stop Loss: ₹{_fmt_level(current_trade_sl)}")
                    print(f"🎯 Take Profit: ₹{_fmt_level(current_trade_tp)}")
                current_trade_entry_price = price
                current_trade_entry_atr = atr
                current_trade_entry_balance = sim_balance
                current_trade_side = "BUY"
                current_trade_initial_risk = (
                    abs(price - current_trade_sl)
                    if current_trade_sl is not None
                    else atr
                )
                current_trade_highest_price = high
                current_trade_lowest_price = low
                current_trade_trailing_active = False
            elif majority_signal == -1:
                df.loc[df.index[i], "event"] = "SHORT_ENTRY"
                current_position = PositionType.SHORT
                current_trade_tp = (
                    latest_short_vote.take_profit if latest_short_vote else None
                )
                current_trade_sl = (
                    latest_short_vote.stop_loss if latest_short_vote else None
                )
                if verbose:
                    print(
                        f"🔄 [{df.index[i]}] Majority Vote - SELL 1.0000 {symbol} @ ₹{price:.2f}"
                    )
                    print(f"💰 Balance Before Entry: ₹{sim_balance:.2f}")
                    print(f"📊 ATR: ₹{atr:.2f}")
                    print(f"🛑 Stop Loss: ₹{_fmt_level(current_trade_sl)}")
                    print(f"🎯 Take Profit: ₹{_fmt_level(current_trade_tp)}")
                current_trade_entry_price = price
                current_trade_entry_atr = atr
                current_trade_entry_balance = sim_balance
                current_trade_side = "SELL"
                current_trade_initial_risk = (
                    abs(current_trade_sl - price)
                    if current_trade_sl is not None
                    else atr
                )
                current_trade_highest_price = high
                current_trade_lowest_price = low
                current_trade_trailing_active = False

        # If the trade has moved enough in our favour, protect it with a
        # breakeven move first and then a Chandelier-style ATR trailing stop.
        if (
            enable_trailing_stop
            and current_position != PositionType.NONE
            and current_trade_entry_price is not None
            and current_trade_initial_risk is not None
            and current_trade_initial_risk > 0
        ):
            current_trade_highest_price = (
                high
                if current_trade_highest_price is None
                else max(current_trade_highest_price, high)
            )
            current_trade_lowest_price = (
                low
                if current_trade_lowest_price is None
                else min(current_trade_lowest_price, low)
            )

            if current_position == PositionType.LONG:
                unrealized_r = (
                    price - current_trade_entry_price
                ) / current_trade_initial_risk
                if unrealized_r >= breakeven_activation_r:
                    breakeven_sl = current_trade_entry_price + (
                        breakeven_buffer_atr * atr
                    )
                    if current_trade_sl is None or breakeven_sl > current_trade_sl:
                        if verbose:
                            print(
                                f"🛡️  [{df.index[i]}] LONG SL moved to breakeven+buffer @ {breakeven_sl:.2f}"
                            )
                        current_trade_sl = breakeven_sl
                if unrealized_r >= trailing_activation_r:
                    current_trade_trailing_active = True
                    trailing_sl = current_trade_highest_price - (
                        trailing_atr_multiplier * atr
                    )
                    if current_trade_sl is None or trailing_sl > current_trade_sl:
                        if verbose:
                            print(
                                f"🛡️  [{df.index[i]}] LONG ATR trailing SL moved @ {trailing_sl:.2f} "
                                f"(high_since_entry={current_trade_highest_price:.2f}, ATR={atr:.2f})"
                            )
                        current_trade_sl = trailing_sl
            elif current_position == PositionType.SHORT:
                unrealized_r = (
                    current_trade_entry_price - price
                ) / current_trade_initial_risk
                if unrealized_r >= breakeven_activation_r:
                    breakeven_sl = current_trade_entry_price - (
                        breakeven_buffer_atr * atr
                    )
                    if current_trade_sl is None or breakeven_sl < current_trade_sl:
                        if verbose:
                            print(
                                f"🛡️  [{df.index[i]}] SHORT SL moved to breakeven+buffer @ {breakeven_sl:.2f}"
                            )
                        current_trade_sl = breakeven_sl
                if unrealized_r >= trailing_activation_r:
                    current_trade_trailing_active = True
                    trailing_sl = current_trade_lowest_price + (
                        trailing_atr_multiplier * atr
                    )
                    if current_trade_sl is None or trailing_sl < current_trade_sl:
                        if verbose:
                            print(
                                f"🛡️  [{df.index[i]}] SHORT ATR trailing SL moved @ {trailing_sl:.2f} "
                                f"(low_since_entry={current_trade_lowest_price:.2f}, ATR={atr:.2f})"
                            )
                        current_trade_sl = trailing_sl

        # If in a trade, enforce TP/SL exits using levels captured at entry.
        if current_position == PositionType.LONG:
            if current_trade_tp is not None and price >= current_trade_tp:
                df.loc[df.index[i], "event"] = "EXIT_TP"
                if verbose:
                    print(f"🟢 [{df.index[i]}] Majority Vote - LONG EXIT (TP)")
                current_position = PositionType.NONE
                if (
                    current_trade_entry_price is not None
                    and current_trade_entry_balance is not None
                ):
                    pnl = (
                        price - current_trade_entry_price
                    ) / current_trade_entry_price
                    realized_trade_pnl = pnl
                    pnl_rupees = _calculate_pnl_rupees(
                        side="BUY",
                        entry_price=current_trade_entry_price,
                        exit_price=price,
                    )
                    new_balance = current_trade_entry_balance + pnl_rupees
                    if verbose:
                        print("🔍 PnL Calculation Debug:")
                        print(f"   side: {current_trade_side}")
                        print(f"   entry_price: {current_trade_entry_price:.2f}")
                        print(f"   exit_price: {price:.2f}")
                        print(
                            f"   ✅ PnL%: ({price:.2f} - {current_trade_entry_price:.2f}) / {current_trade_entry_price:.2f} = {pnl:.6f}"
                        )
                        print(
                            f"✅ [{df.index[i]}] Majority Vote - CLOSED {current_trade_side} position"
                        )
                        print(f"📈 PnL: {pnl * 100:.2f}% (₹{pnl_rupees:+.2f})")
                        print(
                            f"💰 Balance: ₹{current_trade_entry_balance:.2f} -> ₹{new_balance:.2f} | Entry ATR: ₹{_fmt_level(current_trade_entry_atr)} Exit ATR: ₹{atr:.2f}"
                        )
                    sim_balance = new_balance
                current_trade_tp = None
                current_trade_sl = None
                current_trade_entry_price = None
                current_trade_entry_atr = None
                current_trade_entry_balance = None
                current_trade_side = None
                current_trade_initial_risk = None
                current_trade_highest_price = None
                current_trade_lowest_price = None
                current_trade_trailing_active = False
            elif current_trade_sl is not None and price <= current_trade_sl:
                exit_event = (
                    "EXIT_TRAILING_SL" if current_trade_trailing_active else "EXIT_SL"
                )
                df.loc[df.index[i], "event"] = exit_event
                if verbose:
                    reason = "TRAILING SL" if current_trade_trailing_active else "SL"
                    print(f"🟢 [{df.index[i]}] Majority Vote - LONG EXIT ({reason})")
                current_position = PositionType.NONE
                if (
                    current_trade_entry_price is not None
                    and current_trade_entry_balance is not None
                ):
                    pnl = (
                        price - current_trade_entry_price
                    ) / current_trade_entry_price
                    realized_trade_pnl = pnl
                    pnl_rupees = _calculate_pnl_rupees(
                        side="BUY",
                        entry_price=current_trade_entry_price,
                        exit_price=price,
                    )
                    new_balance = current_trade_entry_balance + pnl_rupees
                    if verbose:
                        print("🔍 PnL Calculation Debug:")
                        print(f"   side: {current_trade_side}")
                        print(f"   entry_price: {current_trade_entry_price:.2f}")
                        print(f"   exit_price: {price:.2f}")
                        print(
                            f"   ✅ PnL%: ({price:.2f} - {current_trade_entry_price:.2f}) / {current_trade_entry_price:.2f} = {pnl:.6f}"
                        )
                        print(
                            f"✅ [{df.index[i]}] Majority Vote - CLOSED {current_trade_side} position"
                        )
                        print(f"📈 PnL: {pnl * 100:.2f}% (₹{pnl_rupees:+.2f})")
                        print(
                            f"💰 Balance: ₹{current_trade_entry_balance:.2f} -> ₹{new_balance:.2f} | Entry ATR: ₹{_fmt_level(current_trade_entry_atr)} Exit ATR: ₹{atr:.2f}"
                        )
                    sim_balance = new_balance
                current_trade_tp = None
                current_trade_sl = None
                current_trade_entry_price = None
                current_trade_entry_atr = None
                current_trade_entry_balance = None
                current_trade_side = None
                current_trade_initial_risk = None
                current_trade_highest_price = None
                current_trade_lowest_price = None
                current_trade_trailing_active = False
        elif current_position == PositionType.SHORT:
            if current_trade_tp is not None and price <= current_trade_tp:
                df.loc[df.index[i], "event"] = "EXIT_TP"
                if verbose:
                    print(f"🟢 [{df.index[i]}] Majority Vote - SHORT EXIT (TP)")
                current_position = PositionType.NONE
                if (
                    current_trade_entry_price is not None
                    and current_trade_entry_balance is not None
                ):
                    pnl = (
                        current_trade_entry_price - price
                    ) / current_trade_entry_price
                    realized_trade_pnl = pnl
                    pnl_rupees = _calculate_pnl_rupees(
                        side="SELL",
                        entry_price=current_trade_entry_price,
                        exit_price=price,
                    )
                    new_balance = current_trade_entry_balance + pnl_rupees
                    if verbose:
                        print("🔍 PnL Calculation Debug:")
                        print(f"   side: {current_trade_side}")
                        print(f"   entry_price: {current_trade_entry_price:.2f}")
                        print(f"   exit_price: {price:.2f}")
                        print(
                            f"   ✅ PnL%: ({current_trade_entry_price:.2f} - {price:.2f}) / {current_trade_entry_price:.2f} = {pnl:.6f}"
                        )
                        print(
                            f"✅ [{df.index[i]}] Majority Vote - CLOSED {current_trade_side} position"
                        )
                        print(f"📈 PnL: {pnl * 100:.2f}% (₹{pnl_rupees:+.2f})")
                        print(
                            f"💰 Balance: ₹{current_trade_entry_balance:.2f} -> ₹{new_balance:.2f} | Entry ATR: ₹{_fmt_level(current_trade_entry_atr)} Exit ATR: ₹{atr:.2f}"
                        )
                    sim_balance = new_balance
                current_trade_tp = None
                current_trade_sl = None
                current_trade_entry_price = None
                current_trade_entry_atr = None
                current_trade_entry_balance = None
                current_trade_side = None
                current_trade_initial_risk = None
                current_trade_highest_price = None
                current_trade_lowest_price = None
                current_trade_trailing_active = False
            elif current_trade_sl is not None and price >= current_trade_sl:
                exit_event = (
                    "EXIT_TRAILING_SL" if current_trade_trailing_active else "EXIT_SL"
                )
                df.loc[df.index[i], "event"] = exit_event
                if verbose:
                    reason = "TRAILING SL" if current_trade_trailing_active else "SL"
                    print(f"🟢 [{df.index[i]}] Majority Vote - SHORT EXIT ({reason})")
                current_position = PositionType.NONE
                if (
                    current_trade_entry_price is not None
                    and current_trade_entry_balance is not None
                ):
                    pnl = (
                        current_trade_entry_price - price
                    ) / current_trade_entry_price
                    realized_trade_pnl = pnl
                    pnl_rupees = _calculate_pnl_rupees(
                        side="SELL",
                        entry_price=current_trade_entry_price,
                        exit_price=price,
                    )
                    new_balance = current_trade_entry_balance + pnl_rupees
                    if verbose:
                        print("🔍 PnL Calculation Debug:")
                        print(f"   side: {current_trade_side}")
                        print(f"   entry_price: {current_trade_entry_price:.2f}")
                        print(f"   exit_price: {price:.2f}")
                        print(
                            f"   ✅ PnL%: ({current_trade_entry_price:.2f} - {price:.2f}) / {current_trade_entry_price:.2f} = {pnl:.6f}"
                        )
                        print(
                            f"✅ [{df.index[i]}] Majority Vote - CLOSED {current_trade_side} position"
                        )
                        print(f"📈 PnL: {pnl * 100:.2f}% (₹{pnl_rupees:+.2f})")
                        print(
                            f"💰 Balance: ₹{current_trade_entry_balance:.2f} -> ₹{new_balance:.2f} | Entry ATR: ₹{_fmt_level(current_trade_entry_atr)} Exit ATR: ₹{atr:.2f}"
                        )
                    sim_balance = new_balance
                current_trade_tp = None
                current_trade_sl = None
                current_trade_entry_price = None
                current_trade_entry_atr = None
                current_trade_entry_balance = None
                current_trade_side = None
                current_trade_initial_risk = None
                current_trade_highest_price = None
                current_trade_lowest_price = None
                current_trade_trailing_active = False

        df.loc[df.index[i], "position"] = current_position.value
        if current_trade_tp is not None:
            df.loc[df.index[i], "trade_take_profit"] = current_trade_tp
        if current_trade_sl is not None:
            df.loc[df.index[i], "trade_stop_loss"] = current_trade_sl
        if (
            current_position != PositionType.NONE
            and current_trade_entry_price is not None
        ):
            if current_position == PositionType.LONG:
                max_favorable_price = current_trade_highest_price
                if current_trade_initial_risk and current_trade_initial_risk > 0:
                    df.loc[df.index[i], "trade_unrealized_r"] = (
                        price - current_trade_entry_price
                    ) / current_trade_initial_risk
            else:
                max_favorable_price = current_trade_lowest_price
                if current_trade_initial_risk and current_trade_initial_risk > 0:
                    df.loc[df.index[i], "trade_unrealized_r"] = (
                        current_trade_entry_price - price
                    ) / current_trade_initial_risk
            if max_favorable_price is not None:
                df.loc[df.index[i], "trade_max_favorable_price"] = max_favorable_price
            df.loc[df.index[i], "trade_trailing_active"] = current_trade_trailing_active

        if stop_on_result and realized_trade_pnl is not None:
            if realized_trade_pnl > 0:
                stop_timestamp = pd.Timestamp(df.index[i])
                stop_reason = "winning_trade_reached"
                break

            consecutive_losses += 1
            if consecutive_losses >= max_consecutive_losses:
                stop_timestamp = pd.Timestamp(df.index[i])
                stop_reason = f"{max_consecutive_losses}_consecutive_losses"
                break

        if mock_delay > 0:
            time.sleep(mock_delay)

        previous_majority_signal = majority_signal

    if stop_timestamp is not None:
        df = df[df.index <= stop_timestamp].copy()

    if return_stop_metadata:
        return df, stop_timestamp, stop_reason
    return df


def extract_trade_history(df: pd.DataFrame, initial_balance: float) -> pd.DataFrame:
    trades: List[Dict] = []
    open_trade: Optional[Dict] = None
    running_balance = initial_balance

    for ts, row in df.iterrows():
        price = float(row["Close"])
        event = str(row.get("event", ""))
        if open_trade is not None and event not in {"EXIT_TP", "EXIT_SL"}:
            _update_mfe_fields(open_trade, ts, row, price)

        if not event:
            continue

        tp = row.get("trade_take_profit")
        sl = row.get("trade_stop_loss")
        atr = row.get("ATR")
        action: Optional[str] = None

        if event == "LONG_ENTRY":
            action = "BUY"
        elif event == "SHORT_ENTRY":
            action = "SELL"

        if action is not None:
            open_trade = {
                "entry_time": ts,
                "entry_price": price,
                "action": action,
                "take_profit": float(tp) if pd.notna(tp) else None,
                "stop_loss": float(sl) if pd.notna(sl) else None,
                "entry_atr": float(atr) if pd.notna(atr) else None,
                "status": "open",
                "balance_before": running_balance,
            }
            _initialize_mfe_fields(open_trade, ts, row, price)
            continue

        if (
            event in {"EXIT_TP", "EXIT_SL", "EXIT_TRAILING_SL"}
            and open_trade is not None
        ):
            open_trade["exit_time"] = ts
            open_trade["exit_price"] = price
            open_trade["exit_atr"] = float(atr) if pd.notna(atr) else None
            open_trade["status"] = (
                "tp_hit"
                if event == "EXIT_TP"
                else "trailing_sl_hit" if event == "EXIT_TRAILING_SL" else "sl_hit"
            )
            pnl = (
                (price - open_trade["entry_price"]) / open_trade["entry_price"]
                if open_trade["action"] == "BUY"
                else (open_trade["entry_price"] - price) / open_trade["entry_price"]
            )
            open_trade["pnl"] = pnl
            open_trade["pnl_rupees"] = _calculate_pnl_rupees(
                side=open_trade["action"],
                entry_price=open_trade["entry_price"],
                exit_price=price,
            )
            running_balance += open_trade["pnl_rupees"]
            open_trade["balance_after"] = running_balance
            trades.append(open_trade)
            open_trade = None

    if open_trade is not None:
        final_price = float(df["Close"].iloc[-1])
        final_ts = df.index[-1]
        final_row = df.iloc[-1]
        _update_mfe_fields(open_trade, final_ts, final_row, final_price)
        open_trade["exit_time"] = final_ts
        open_trade["exit_price"] = final_price
        final_atr = df["ATR"].iloc[-1]
        open_trade["exit_atr"] = float(final_atr) if pd.notna(final_atr) else None
        open_trade["status"] = "closed_end"
        _finalize_trade_pnl_fields(open_trade, exit_price=final_price)
        running_balance += open_trade["pnl_rupees"]
        open_trade["balance_after"] = running_balance
        trades.append(open_trade)

    return pd.DataFrame(trades)


def calculate_performance_metrics(trades: pd.DataFrame, initial_balance: float) -> Dict:
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "final_balance": initial_balance,
            "max_drawdown": 0.0,
            "losing_sl_trades": 0,
            "losing_sl_with_positive_mfe": 0,
            "losing_sl_avg_max_favorable_pnl": 0.0,
            "losing_sl_best_max_favorable_pnl": 0.0,
            "losing_sl_total_profit_given_back_rupees": 0.0,
        }

    wins = (trades["pnl"] > 0).sum()
    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades else 0.0

    final_balance = float(trades["balance_after"].iloc[-1])
    total_pnl = (final_balance - initial_balance) / initial_balance

    equity = pd.Series([initial_balance] + trades["balance_after"].tolist())
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    losing_trades = trades[trades["pnl"] < 0].copy()
    losing_sl_trades = losing_trades[losing_trades["status"] == "sl_hit"].copy()
    if "max_favorable_pnl" in losing_sl_trades.columns and not losing_sl_trades.empty:
        positive_mfe = losing_sl_trades[losing_sl_trades["max_favorable_pnl"] > 0]
        losing_sl_count = int(len(losing_sl_trades))
        losing_sl_with_positive_mfe = int(len(positive_mfe))
        losing_sl_avg_max_favorable_pnl = float(
            losing_sl_trades["max_favorable_pnl"].mean()
        )
        losing_sl_best_max_favorable_pnl = float(
            losing_sl_trades["max_favorable_pnl"].max()
        )
        losing_sl_total_profit_given_back_rupees = float(
            losing_sl_trades.get(
                "profit_given_back_rupees", pd.Series(dtype=float)
            ).sum()
        )
    else:
        losing_sl_count = int(len(losing_sl_trades))
        losing_sl_with_positive_mfe = 0
        losing_sl_avg_max_favorable_pnl = 0.0
        losing_sl_best_max_favorable_pnl = 0.0
        losing_sl_total_profit_given_back_rupees = 0.0

    return {
        "total_trades": total_trades,
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "final_balance": final_balance,
        "max_drawdown": max_drawdown,
        "losing_sl_trades": losing_sl_count,
        "losing_sl_with_positive_mfe": losing_sl_with_positive_mfe,
        "losing_sl_avg_max_favorable_pnl": losing_sl_avg_max_favorable_pnl,
        "losing_sl_best_max_favorable_pnl": losing_sl_best_max_favorable_pnl,
        "losing_sl_total_profit_given_back_rupees": losing_sl_total_profit_given_back_rupees,
    }


def create_cumulative_pnl_chart(
    trades: pd.DataFrame,
    initial_balance: float,
    output_path: str,
    title_suffix: Optional[str] = None,
    auto_open: bool = False,
) -> Optional[str]:
    if trades.empty:
        print("⚠️  No closed trades to plot")
        return None

    closed_trades = trades.sort_values("exit_time").reset_index(drop=True)
    cumulative_pnl = (
        (closed_trades["balance_after"] - initial_balance) / initial_balance * 100
    )

    first_entry = closed_trades["entry_time"].iloc[0]
    plot_times = [first_entry] + closed_trades["exit_time"].tolist()
    plot_pnls = [0.0] + cumulative_pnl.tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_times,
            y=plot_pnls,
            mode="lines+markers",
            name="Cumulative PnL",
            line=dict(color="#2E86AB", width=3),
            marker=dict(size=8),
            hovertemplate="<b>Time:</b> %{x}<br><b>Cumulative PnL:</b> %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    chart_title = "Majority-vote validation cumulative PnL (%)"
    if title_suffix:
        chart_title = f"{chart_title}<br><sup>{title_suffix}</sup>"

    fig.update_layout(
        title=chart_title,
        xaxis_title="Time",
        yaxis_title="PnL (%)",
        template="plotly_white",
        hovermode="x unified",
        height=700,
    )
    fig.write_html(output_path)
    print(f"📊 Cumulative PnL chart saved: {output_path}")
    if auto_open:
        abs_chart = os.path.abspath(output_path)
        print(f"🌐 Opening chart: {abs_chart}")
        try:
            webbrowser.open(f"file://{abs_chart}")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"   Please open manually: {abs_chart}")
    return output_path


def _normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    normalized = df.copy()
    lower_to_actual = {str(c).strip().lower(): c for c in normalized.columns}
    rename_map = {}
    for target in ["open", "high", "low", "close"]:
        actual = lower_to_actual.get(target)
        if actual is not None and actual != target:
            rename_map[actual] = target
    if rename_map:
        normalized = normalized.rename(columns=rename_map)
    return normalized


def create_ohlc_trade_chart(
    ohlc: pd.DataFrame,
    trades: pd.DataFrame,
    param_sets: List[Dict],
    symbol: str,
    exchange: str,
    output_path: str,
    title_suffix: Optional[str] = None,
    auto_open: bool = False,
    display_start_at: Optional[Union[str, date, datetime, pd.Timestamp]] = None,
) -> Optional[str]:
    if ohlc is None or ohlc.empty:
        print("⚠️  No OHLC data to plot")
        return None

    chart_df = _normalize_ohlc_columns(ohlc).sort_index()
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(chart_df.columns))
    if missing:
        print(f"⚠️  Missing OHLC columns for chart: {missing}")
        return None

    if param_sets:
        short_palette = [
            "#1f77b4",
            "#17becf",
            "#2ca02c",
            "#9467bd",
            "#e377c2",
            "#8c564b",
            "#7f7f7f",
            "#bcbd22",
        ]
        long_palette = [
            "#ff7f0e",
            "#d62728",
            "#8c564b",
            "#7f7f7f",
            "#bcbd22",
            "#e377c2",
            "#9467bd",
            "#17becf",
        ]
        for idx, p in enumerate(param_sets):
            short_window = int(p["short_window"])
            long_window = int(p["long_window"])
            short_col = f"short_ma_{idx}"
            long_col = f"long_ma_{idx}"
            legend_group = f"ma_set_{idx}"
            set_label = f"Set {idx + 1} MA (S{short_window}/L{long_window})"
            chart_df[short_col] = chart_df["close"].rolling(short_window).mean()
            chart_df[long_col] = chart_df["close"].rolling(long_window).mean()

    if display_start_at is not None:
        display_start_ts = pd.Timestamp(display_start_at)
        if not chart_df.empty:
            first_chart_ts = pd.Timestamp(chart_df.index[0])
            if first_chart_ts.tzinfo is not None and display_start_ts.tzinfo is None:
                display_start_ts = display_start_ts.tz_localize(first_chart_ts.tzinfo)
            elif first_chart_ts.tzinfo is None and display_start_ts.tzinfo is not None:
                display_start_ts = display_start_ts.tz_localize(None)
        chart_df = chart_df[chart_df.index >= display_start_ts]

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="OHLC",
        )
    )

    if param_sets:
        for idx, p in enumerate(param_sets):
            short_window = int(p["short_window"])
            long_window = int(p["long_window"])
            short_col = f"short_ma_{idx}"
            long_col = f"long_ma_{idx}"
            legend_group = f"ma_set_{idx}"
            set_label = f"Set {idx + 1} MA (S{short_window}/L{long_window})"

            fig.add_trace(
                go.Scatter(
                    x=chart_df.index,
                    y=chart_df[short_col],
                    mode="lines",
                    name=set_label,
                    legendgroup=legend_group,
                    showlegend=True,
                    line=dict(width=1.4, color=short_palette[idx % len(short_palette)]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=chart_df.index,
                    y=chart_df[long_col],
                    mode="lines",
                    name=f"{set_label} Long",
                    legendgroup=legend_group,
                    showlegend=False,
                    line=dict(
                        width=1.4,
                        color=long_palette[idx % len(long_palette)],
                        dash="dot",
                    ),
                )
            )

    if trades is not None and not trades.empty:
        entries = trades.copy()
        entries["entry_time"] = pd.to_datetime(entries["entry_time"], errors="coerce")
        entries["exit_time"] = pd.to_datetime(entries["exit_time"], errors="coerce")

        long_entries = entries[entries["action"] == "BUY"]
        short_entries = entries[entries["action"] == "SELL"]

        for subset, name, color, marker_symbol in [
            (long_entries, "Long Entry", "#2ca02c", "triangle-up"),
            (short_entries, "Short Entry", "#d62728", "triangle-down"),
        ]:
            if subset.empty:
                continue
            hover = (
                "<b>"
                + name
                + "</b><br>"
                + "Entry: "
                + subset["entry_time"].astype(str)
                + "<br>Price: "
                + subset["entry_price"].map(lambda x: f"{x:.2f}")
                + "<br>ATR: "
                + subset["entry_atr"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
                + "<br>TP: "
                + subset["take_profit"].map(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "NA"
                )
                + "<br>SL: "
                + subset["stop_loss"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
            )
            fig.add_trace(
                go.Scatter(
                    x=subset["entry_time"],
                    y=subset["entry_price"],
                    mode="markers",
                    name=name,
                    marker=dict(symbol=marker_symbol, size=10, color=color),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        exits = entries[entries["exit_time"].notna()].copy()
        if not exits.empty:
            hover = (
                "<b>Trade Exit</b><br>"
                + "Exit: "
                + exits["exit_time"].astype(str)
                + "<br>Price: "
                + exits["exit_price"].map(lambda x: f"{x:.2f}")
                + "<br>Status: "
                + exits["status"].astype(str)
                + "<br>Entry ATR: "
                + exits["entry_atr"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
                + "<br>Exit ATR: "
                + exits["exit_atr"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
                + "<br>PnL: "
                + exits["pnl"].map(lambda x: f"{x * 100:.2f}%")
                + "<br>PnL (Rs): "
                + exits["pnl_rupees"].map(lambda x: f"{x:+,.2f}")
            )
            if "max_favorable_pnl" in exits.columns:
                hover = (
                    hover
                    + "<br>Max favorable: "
                    + exits["max_favorable_pnl"].map(lambda x: f"{x * 100:.2f}%")
                    + "<br>Max favorable (Rs): "
                    + exits["max_favorable_pnl_rupees"].map(lambda x: f"{x:+,.2f}")
                )
            if "profit_given_back" in exits.columns:
                hover = (
                    hover
                    + "<br>Profit given back: "
                    + exits["profit_given_back"].map(lambda x: f"{x * 100:.2f}%")
                )
            fig.add_trace(
                go.Scatter(
                    x=exits["exit_time"],
                    y=exits["exit_price"],
                    mode="markers",
                    name="Trade Exit",
                    marker=dict(symbol="x", size=10, color="#9467bd"),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    chart_title = f"{symbol} ({exchange}) Majority-vote mock validation"
    if title_suffix:
        chart_title = f"{chart_title}<br><sup>{title_suffix}</sup>"

    fig.update_layout(
        title=chart_title,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(groupclick="togglegroup"),
        xaxis_rangeslider_visible=True,
        height=850,
    )
    fig.write_html(output_path, config={"scrollZoom": True, "displayModeBar": True})
    print(f"📊 OHLC+Trade chart saved: {output_path}")
    if auto_open:
        abs_chart = os.path.abspath(output_path)
        print(f"🌐 Opening chart: {abs_chart}")
        try:
            webbrowser.open(f"file://{abs_chart}")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"   Please open manually: {abs_chart}")
    return output_path


def summarize(df: pd.DataFrame) -> None:
    events = df[df["event"] != ""][
        ["Close", "long_votes", "short_votes", "majority_signal", "position", "event"]
    ]

    print("\n📌 Majority-vote validation summary")
    print("=" * 60)
    print(f"Total bars: {len(df)}")
    print(f"Majority LONG bars: {(df['majority_signal'] == 1).sum()}")
    print(f"Majority SHORT bars: {(df['majority_signal'] == -1).sum()}")
    print(f"No-majority bars: {(df['majority_signal'] == 0).sum()}")
    print(f"Position events: {len(events)}")
    if not events.empty:
        print("\nLast 15 events:")
        print(events.tail(15).to_string())


def summarize_trades(trades: pd.DataFrame, metrics: Dict) -> None:
    print("\n💼 Trade summary")
    print("=" * 60)
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Win rate: {metrics['win_rate']:.2%}")
    print(f"Total PnL: {metrics['total_pnl']:.2%}")
    print(f"Final balance: ₹{metrics['final_balance']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2%}")

    if not trades.empty:
        print("\nLast 15 closed trades:")
        cols = [
            "entry_time",
            "exit_time",
            "action",
            "entry_price",
            "exit_price",
            "entry_atr",
            "exit_atr",
            "status",
            "pnl",
            "pnl_rupees",
            "balance_before",
            "balance_after",
        ]
        print(trades[cols].tail(15).to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MA majority-vote mock validation with Kite data"
    )
    p.add_argument("--symbol", default=None, help=f"Symbol (default: {DEFAULT_SYMBOL})")
    p.add_argument(
        "--exchange", default=None, help=f"Exchange (default: {DEFAULT_EXCHANGE})"
    )
    p.add_argument(
        "--start",
        default=None,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START_DATE})",
    )
    p.add_argument(
        "--end", default=None, help=f"End date YYYY-MM-DD (default: {DEFAULT_END_DATE})"
    )
    p.add_argument(
        "--interval",
        default=None,
        help=f"Interval e.g. 15m (default: {DEFAULT_INTERVAL})",
    )
    p.add_argument("--params", help="JSON array string of parameter sets")
    p.add_argument("--params-file", help="Path to JSON file containing parameter sets")
    p.add_argument("--output-csv", default="ma_majority_mock_validation_kite.csv")
    p.add_argument(
        "--trades-csv", default="ma_majority_mock_validation_kite_trades.csv"
    )
    p.add_argument("--pnl-chart", default="ma_majority_mock_validation_kite_pnl.html")
    p.add_argument(
        "--ohlc-chart", default="ma_majority_mock_validation_kite_ohlc_trades.html"
    )
    p.add_argument(
        "--mock-delay",
        type=float,
        default=0.0,
        help="Seconds between bars for live-like simulation",
    )
    p.add_argument("--quiet", action="store_true", help="Reduce live simulation logs")
    p.add_argument(
        "--no-open-chart", action="store_true", help="Do not auto-open OHLC chart"
    )
    p.add_argument("--initial-balance", type=float, default=10000.0)
    p.add_argument(
        "--enable-trailing-stop",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_TRAILING_STOP,
        help="Enable breakeven plus ATR/Chandelier trailing stop during validation.",
    )
    p.add_argument(
        "--breakeven-activation-r",
        type=float,
        default=DEFAULT_BREAKEVEN_ACTIVATION_R,
        help="Move SL to breakeven after this many initial-risk units in profit.",
    )
    p.add_argument(
        "--breakeven-buffer-atr",
        type=float,
        default=DEFAULT_BREAKEVEN_BUFFER_ATR,
        help="ATR buffer beyond entry when moving the SL to breakeven.",
    )
    p.add_argument(
        "--trailing-activation-r",
        type=float,
        default=DEFAULT_TRAILING_ACTIVATION_R,
        help="Start ATR/Chandelier trailing after this many initial-risk units in profit.",
    )
    p.add_argument(
        "--trailing-atr-multiplier",
        type=float,
        default=DEFAULT_TRAILING_ATR_MULTIPLIER,
        help="ATR multiplier used for the Chandelier trailing stop distance.",
    )
    p.add_argument("--out", default="ma_mock_results_kite", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol if args.symbol is not None else DEFAULT_SYMBOL
    exchange = args.exchange if args.exchange is not None else DEFAULT_EXCHANGE
    start_date = args.start if args.start is not None else DEFAULT_START_DATE
    end_date = args.end if args.end is not None else DEFAULT_END_DATE
    interval = args.interval if args.interval is not None else DEFAULT_INTERVAL
    param_sets = load_param_sets(args.params, args.params_file)
    if args.breakeven_activation_r < 0:
        raise ValueError("breakeven-activation-r must be >= 0")
    if args.breakeven_buffer_atr < 0:
        raise ValueError("breakeven-buffer-atr must be >= 0")
    if args.trailing_activation_r < 0:
        raise ValueError("trailing-activation-r must be >= 0")
    if args.trailing_atr_multiplier <= 0:
        raise ValueError("trailing-atr-multiplier must be > 0")

    print("🚀 MA Majority-vote MOCK VALIDATION (Kite)")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Exchange: {exchange}")
    print(f"Start: {start_date}")
    print(f"End: {end_date}")
    print(f"Interval: {interval}")
    print(f"Param sets ({len(param_sets)}): {param_sets}")
    print(
        "Trailing stop: "
        f"{args.enable_trailing_stop} "
        f"(BE={args.breakeven_activation_r}R + {args.breakeven_buffer_atr} ATR, "
        f"trail={args.trailing_activation_r}R / {args.trailing_atr_multiplier} ATR)"
    )
    print("🚦 Running live-style mock simulation...")

    data = fetch_kite_data(symbol, exchange, start_date, end_date, interval)
    result = run_majority_vote_validation(
        data,
        param_sets,
        symbol=symbol,
        initial_balance=args.initial_balance,
        verbose=not args.quiet,
        mock_delay=args.mock_delay,
        enable_trailing_stop=args.enable_trailing_stop,
        breakeven_activation_r=args.breakeven_activation_r,
        breakeven_buffer_atr=args.breakeven_buffer_atr,
        trailing_activation_r=args.trailing_activation_r,
        trailing_atr_multiplier=args.trailing_atr_multiplier,
    )
    summarize(result)

    os.makedirs(args.out, exist_ok=True)
    result_csv_path = os.path.join(args.out, args.output_csv)
    trades_csv_path = os.path.join(args.out, args.trades_csv)
    pnl_chart_path = os.path.join(args.out, args.pnl_chart)
    ohlc_chart_path = os.path.join(args.out, args.ohlc_chart)

    result.to_csv(result_csv_path)
    print(f"\n💾 Saved detailed validation output to: {result_csv_path}")

    trades = extract_trade_history(result, initial_balance=args.initial_balance)
    trades.to_csv(trades_csv_path, index=False)
    print(f"💾 Saved trade history to: {trades_csv_path}")

    metrics = calculate_performance_metrics(
        trades, initial_balance=args.initial_balance
    )
    summarize_trades(trades, metrics)
    create_cumulative_pnl_chart(
        trades, initial_balance=args.initial_balance, output_path=pnl_chart_path
    )
    out_chart = create_ohlc_trade_chart(
        ohlc=data,
        trades=trades,
        param_sets=param_sets,
        symbol=symbol,
        exchange=exchange,
        output_path=ohlc_chart_path,
    )
    if out_chart and not args.no_open_chart:
        abs_chart = os.path.abspath(out_chart)
        print(f"🌐 Opening chart: {abs_chart}")
        try:
            webbrowser.open(f"file://{abs_chart}")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"   Please open manually: {abs_chart}")


if __name__ == "__main__":
    main()
