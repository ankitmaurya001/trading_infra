#!/usr/bin/env python3
"""
Run MA majority-vote mock validation on Zerodha Kite data.

This is a validation-style runner (similar to run_ma_mock_validation_kite.py),
not an optimization runner. It supports multiple MA parameter sets where each
set maintains its own directional vote until its own TP is reached.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
import time
import webbrowser
from typing import Dict, List, Optional

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
    {"short_window": 4, "long_window": 184, "risk_reward_ratio": 7.0},
    {"short_window": 12, "long_window": 50, "risk_reward_ratio": 7.0},
]
DEFAULT_NUM_LOTS = 1
DEFAULT_LOT_SIZE = 250
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
    price_diff = (exit_price - entry_price) if side == "BUY" else (entry_price - exit_price)
    return price_diff * num_lots * lot_size


def run_majority_vote_validation(
    data: pd.DataFrame,
    param_sets: List[Dict],
    symbol: str = DEFAULT_SYMBOL,
    initial_balance: float = 10000.0,
    verbose: bool = True,
    mock_delay: float = 0.0,
) -> pd.DataFrame:
    df = data.copy()
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
    df["event"] = ""

    current_position = PositionType.NONE
    current_trade_tp: Optional[float] = None
    current_trade_sl: Optional[float] = None
    current_trade_entry_price: Optional[float] = None
    current_trade_entry_atr: Optional[float] = None
    current_trade_entry_balance: Optional[float] = None
    current_trade_side: Optional[str] = None
    sim_balance = float(initial_balance)
    last_majority_signal = 0
    min_required_votes = len(states) // 2 + 1

    for i in range(1, len(df)):
        price = float(df["Close"].iloc[i])
        atr = float(df["ATR"].iloc[i])
        if np.isnan(atr) or atr <= 0:
            atr = price * 0.01

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

        if verbose and majority_signal != last_majority_signal:
            label = "LONG" if majority_signal == 1 else "SHORT" if majority_signal == -1 else "NONE"
            print(
                f"🧮 [{df.index[i]}] Majority -> {label} (long_votes={long_votes}, short_votes={short_votes}, required={min_required_votes})"
            )
            last_majority_signal = majority_signal

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
        # - Open trade only when flat and a strict majority appears.
        # - Once in trade, ignore majority changes; exit only via TP/SL.
        if current_position == PositionType.NONE:
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
                    print(
                        f"📊 ATR: ₹{atr:.2f}"
                    )
                    print(f"🛑 Stop Loss: ₹{_fmt_level(current_trade_sl)}")
                    print(f"🎯 Take Profit: ₹{_fmt_level(current_trade_tp)}")
                current_trade_entry_price = price
                current_trade_entry_atr = atr
                current_trade_entry_balance = sim_balance
                current_trade_side = "BUY"
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
                    print(
                        f"📊 ATR: ₹{atr:.2f}"
                    )
                    print(f"🛑 Stop Loss: ₹{_fmt_level(current_trade_sl)}")
                    print(f"🎯 Take Profit: ₹{_fmt_level(current_trade_tp)}")
                current_trade_entry_price = price
                current_trade_entry_atr = atr
                current_trade_entry_balance = sim_balance
                current_trade_side = "SELL"

        # If in a trade, enforce TP/SL exits using levels captured at entry.
        if current_position == PositionType.LONG:
            if current_trade_tp is not None and price >= current_trade_tp:
                df.loc[df.index[i], "event"] = "EXIT_TP"
                if verbose:
                    print(f"🟢 [{df.index[i]}] Majority Vote - LONG EXIT (TP)")
                current_position = PositionType.NONE
                if current_trade_entry_price is not None and current_trade_entry_balance is not None:
                    pnl = (price - current_trade_entry_price) / current_trade_entry_price
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
            elif current_trade_sl is not None and price <= current_trade_sl:
                df.loc[df.index[i], "event"] = "EXIT_SL"
                if verbose:
                    print(f"🟢 [{df.index[i]}] Majority Vote - LONG EXIT (SL)")
                current_position = PositionType.NONE
                if current_trade_entry_price is not None and current_trade_entry_balance is not None:
                    pnl = (price - current_trade_entry_price) / current_trade_entry_price
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
        elif current_position == PositionType.SHORT:
            if current_trade_tp is not None and price <= current_trade_tp:
                df.loc[df.index[i], "event"] = "EXIT_TP"
                if verbose:
                    print(f"🟢 [{df.index[i]}] Majority Vote - SHORT EXIT (TP)")
                current_position = PositionType.NONE
                if current_trade_entry_price is not None and current_trade_entry_balance is not None:
                    pnl = (current_trade_entry_price - price) / current_trade_entry_price
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
            elif current_trade_sl is not None and price >= current_trade_sl:
                df.loc[df.index[i], "event"] = "EXIT_SL"
                if verbose:
                    print(f"🟢 [{df.index[i]}] Majority Vote - SHORT EXIT (SL)")
                current_position = PositionType.NONE
                if current_trade_entry_price is not None and current_trade_entry_balance is not None:
                    pnl = (current_trade_entry_price - price) / current_trade_entry_price
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

        df.loc[df.index[i], "position"] = current_position.value
        if current_trade_tp is not None:
            df.loc[df.index[i], "trade_take_profit"] = current_trade_tp
        if current_trade_sl is not None:
            df.loc[df.index[i], "trade_stop_loss"] = current_trade_sl

        if mock_delay > 0:
            time.sleep(mock_delay)

    return df


def extract_trade_history(df: pd.DataFrame, initial_balance: float) -> pd.DataFrame:
    trades: List[Dict] = []
    open_trade: Optional[Dict] = None
    running_balance = initial_balance

    for ts, row in df.iterrows():
        event = str(row.get("event", ""))
        if not event:
            continue

        price = float(row["Close"])
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
            continue

        if event in {"EXIT_TP", "EXIT_SL"} and open_trade is not None:
            open_trade["exit_time"] = ts
            open_trade["exit_price"] = price
            open_trade["exit_atr"] = float(atr) if pd.notna(atr) else None
            open_trade["status"] = "tp_hit" if event == "EXIT_TP" else "sl_hit"
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
        open_trade["exit_time"] = final_ts
        open_trade["exit_price"] = final_price
        final_atr = df["ATR"].iloc[-1]
        open_trade["exit_atr"] = float(final_atr) if pd.notna(final_atr) else None
        open_trade["status"] = "closed_end"
        pnl = (
            (final_price - open_trade["entry_price"]) / open_trade["entry_price"]
            if open_trade["action"] == "BUY"
            else (open_trade["entry_price"] - final_price) / open_trade["entry_price"]
        )
        open_trade["pnl"] = pnl
        open_trade["pnl_rupees"] = _calculate_pnl_rupees(
            side=open_trade["action"],
            entry_price=open_trade["entry_price"],
            exit_price=final_price,
        )
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

    return {
        "total_trades": total_trades,
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "final_balance": final_balance,
        "max_drawdown": max_drawdown,
    }


def create_cumulative_pnl_chart(
    trades: pd.DataFrame, initial_balance: float, output_path: str
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
    fig.update_layout(
        title="Majority-vote validation cumulative PnL (%)",
        xaxis_title="Time",
        yaxis_title="PnL (%)",
        template="plotly_white",
        hovermode="x unified",
        height=700,
    )
    fig.write_html(output_path)
    print(f"📊 Cumulative PnL chart saved: {output_path}")
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
        p = param_sets[0]
        short_window = int(p["short_window"])
        long_window = int(p["long_window"])
        chart_df["short_ma"] = chart_df["close"].rolling(short_window).mean()
        chart_df["long_ma"] = chart_df["close"].rolling(long_window).mean()
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["short_ma"],
                mode="lines",
                name=f"Short MA ({short_window})",
                line=dict(width=1.8, color="#1f77b4"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["long_ma"],
                mode="lines",
                name=f"Long MA ({long_window})",
                line=dict(width=1.8, color="#ff7f0e"),
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
                "<b>" + name + "</b><br>"
                + "Entry: " + subset["entry_time"].astype(str)
                + "<br>Price: " + subset["entry_price"].map(lambda x: f"{x:.2f}")
                + "<br>ATR: " + subset["entry_atr"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
                + "<br>TP: " + subset["take_profit"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
                + "<br>SL: " + subset["stop_loss"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
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
                + "Exit: " + exits["exit_time"].astype(str)
                + "<br>Price: " + exits["exit_price"].map(lambda x: f"{x:.2f}")
                + "<br>Status: " + exits["status"].astype(str)
                + "<br>Entry ATR: " + exits["entry_atr"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
                + "<br>Exit ATR: " + exits["exit_atr"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
                + "<br>PnL: " + exits["pnl"].map(lambda x: f"{x * 100:.2f}%")
                + "<br>PnL (Rs): " + exits["pnl_rupees"].map(lambda x: f"{x:+,.2f}")
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

    fig.update_layout(
        title=f"{symbol} ({exchange}) Majority-vote mock validation",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
        height=850,
    )
    fig.write_html(output_path, config={"scrollZoom": True, "displayModeBar": True})
    print(f"📊 OHLC+Trade chart saved: {output_path}")
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

    print("🚀 MA Majority-vote MOCK VALIDATION (Kite)")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Exchange: {exchange}")
    print(f"Start: {start_date}")
    print(f"End: {end_date}")
    print(f"Interval: {interval}")
    print(f"Param sets ({len(param_sets)}): {param_sets}")
    print("🚦 Running live-style mock simulation...")

    data = fetch_kite_data(symbol, exchange, start_date, end_date, interval)
    result = run_majority_vote_validation(
        data,
        param_sets,
        symbol=symbol,
        initial_balance=args.initial_balance,
        verbose=not args.quiet,
        mock_delay=args.mock_delay,
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
