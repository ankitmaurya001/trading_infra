"""Safe strategy DSL for LLM-proposed strategy ideas.

The DSL lets an LLM propose indicator/rule based strategies as JSON. We convert
those ideas into `BaseStrategy` subclasses without executing model-written code.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies import BaseStrategy, PositionType, Signal, StrategyDefinition, Trade


REQUIRED_SIGNAL_COLUMNS = {"Signal", "Position", "Take_Profit", "Stop_Loss"}
SUPPORTED_INDICATORS = {
    "atr",
    "cci",
    "macd",
    "roc",
    "sma",
    "ema",
    "rsi",
    "rolling_high",
    "rolling_low",
    "rolling_mean",
    "rolling_std",
    "bollinger",
    "stochastic",
    "zscore",
}
SUPPORTED_CONDITION_OPS = {
    "all",
    "any",
    "gt",
    "gte",
    "lt",
    "lte",
    "crosses_above",
    "crosses_below",
}


class RuleBasedStrategy(BaseStrategy):
    """Base class for runtime strategy classes produced from LLM JSON."""

    SPEC: Dict[str, Any] = {}
    STRATEGY_KEY = "rule_dsl"
    STRATEGY_NAME = "Rule DSL Strategy"
    OPTIMIZATION_PARAM_RANGES: Dict[str, List[Any]] = {}
    LIVE_PARAMETER_NAMES: Tuple[str, ...] = ()

    def __init__(self, risk_reward_ratio: float = 2.0, trading_fee: float = 0.0, **params):
        super().__init__(self.STRATEGY_NAME, risk_reward_ratio, trading_fee=trading_fee)
        self.params = dict(params)
        self.params["risk_reward_ratio"] = risk_reward_ratio
        self.params["trading_fee"] = trading_fee

    @classmethod
    def validate_parameters(cls, params: Dict[str, Any]) -> bool:
        if not all(name in params for name in cls.LIVE_PARAMETER_NAMES):
            return False

        for indicator in cls.SPEC.get("indicators", []):
            kind = indicator.get("kind")
            period = _resolve_param_or_literal(indicator.get("period"), params)
            if period is not None and period <= 0:
                return False
            std_dev = _resolve_param_or_literal(indicator.get("std_dev"), params)
            if std_dev is not None and std_dev <= 0:
                return False
            if kind == "macd":
                fast_period = _resolve_param_or_literal(indicator.get("fast_period"), params)
                slow_period = _resolve_param_or_literal(indicator.get("slow_period"), params)
                signal_period = _resolve_param_or_literal(indicator.get("signal_period"), params)
                if (
                    fast_period is None
                    or slow_period is None
                    or signal_period is None
                    or fast_period <= 0
                    or slow_period <= 0
                    or signal_period <= 0
                    or fast_period >= slow_period
                ):
                    return False
            if kind == "stochastic":
                signal_period = _resolve_param_or_literal(indicator.get("signal_period"), params)
                if signal_period is None or signal_period <= 0:
                    return False

        for constraint in cls.SPEC.get("constraints", []):
            if not _constraint_matches(constraint, params):
                return False

        return True

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = self._add_indicators(df)
        df["ATR"] = self.calculate_atr(df)
        df["Signal"] = Signal.HOLD.value
        df["Position"] = 0
        df["Take_Profit"] = np.nan
        df["Stop_Loss"] = np.nan

        long_entry = self.SPEC.get("long_entry")
        short_entry = self.SPEC.get("short_entry")
        if not long_entry and not short_entry:
            raise ValueError("Rule strategy must define long_entry or short_entry")

        current_position = PositionType.NONE

        for i in range(1, len(df)):
            current_price = df["Close"].iloc[i]
            current_atr = df["ATR"].iloc[i]
            if np.isnan(current_atr):
                continue

            if (
                long_entry
                and current_position == PositionType.NONE
                and self._condition_matches(df, i, long_entry)
            ):
                df.loc[df.index[i], "Signal"] = Signal.LONG_ENTRY.value
                current_position = PositionType.LONG
                take_profit, stop_loss = self.calculate_trade_levels(
                    current_price,
                    PositionType.LONG,
                    current_atr,
                )
                df.loc[df.index[i], "Take_Profit"] = take_profit
                df.loc[df.index[i], "Stop_Loss"] = stop_loss
                self.active_trade = Trade(
                    entry_date=df.index[i],
                    entry_price=current_price,
                    position_type=PositionType.LONG,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                )
            elif (
                short_entry
                and current_position == PositionType.NONE
                and self._condition_matches(df, i, short_entry)
            ):
                df.loc[df.index[i], "Signal"] = Signal.SHORT_ENTRY.value
                current_position = PositionType.SHORT
                take_profit, stop_loss = self.calculate_trade_levels(
                    current_price,
                    PositionType.SHORT,
                    current_atr,
                )
                df.loc[df.index[i], "Take_Profit"] = take_profit
                df.loc[df.index[i], "Stop_Loss"] = stop_loss
                self.active_trade = Trade(
                    entry_date=df.index[i],
                    entry_price=current_price,
                    position_type=PositionType.SHORT,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                )

            if self.active_trade:
                current_position = self._maybe_close_trade(df, i, current_price, current_position)

            df.loc[df.index[i], "Position"] = current_position.value
            if self.active_trade:
                df.loc[df.index[i], "Take_Profit"] = self.active_trade.take_profit
                df.loc[df.index[i], "Stop_Loss"] = self.active_trade.stop_loss

        return df

    def generate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self._add_indicators(data.copy())
        df["Signal"] = Signal.HOLD.value
        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        for indicator in self.SPEC.get("indicators", []):
            kind = indicator["kind"]
            source = indicator.get("source", "Close")
            period = _resolve_param_or_literal(indicator.get("period"), self.params)
            name = indicator.get("name", kind)

            if source not in df.columns:
                raise ValueError(f"Unknown indicator source column: {source}")

            if kind == "sma":
                df[name] = df[source].rolling(window=int(period)).mean()
            elif kind == "ema":
                df[name] = df[source].ewm(span=int(period), adjust=False).mean()
            elif kind == "rsi":
                df[name] = _calculate_rsi(df[source], int(period))
            elif kind == "atr":
                df[name] = _calculate_atr(df, int(period))
            elif kind == "cci":
                df[name] = _calculate_cci(df, int(period))
            elif kind == "macd":
                fast_period = _resolve_param_or_literal(indicator.get("fast_period"), self.params)
                slow_period = _resolve_param_or_literal(indicator.get("slow_period"), self.params)
                signal_period = _resolve_param_or_literal(indicator.get("signal_period"), self.params)
                macd_name = indicator.get("macd", f"{name}_macd")
                signal_name = indicator.get("signal", f"{name}_signal")
                histogram_name = indicator.get("histogram", f"{name}_histogram")
                ema_fast = df[source].ewm(span=int(fast_period), adjust=False).mean()
                ema_slow = df[source].ewm(span=int(slow_period), adjust=False).mean()
                df[macd_name] = ema_fast - ema_slow
                df[signal_name] = df[macd_name].ewm(span=int(signal_period), adjust=False).mean()
                df[histogram_name] = df[macd_name] - df[signal_name]
            elif kind == "roc":
                df[name] = df[source].pct_change(periods=int(period))
            elif kind == "rolling_high":
                df[name] = df[source].rolling(window=int(period)).max()
            elif kind == "rolling_low":
                df[name] = df[source].rolling(window=int(period)).min()
            elif kind == "rolling_mean":
                df[name] = df[source].rolling(window=int(period)).mean()
            elif kind == "rolling_std":
                df[name] = df[source].rolling(window=int(period)).std()
            elif kind == "bollinger":
                middle = indicator.get("middle", f"{name}_middle")
                upper = indicator.get("upper", f"{name}_upper")
                lower = indicator.get("lower", f"{name}_lower")
                std_dev = _resolve_param_or_literal(indicator.get("std_dev"), self.params)
                rolling_mean = df[source].rolling(window=int(period)).mean()
                rolling_std = df[source].rolling(window=int(period)).std()
                df[middle] = rolling_mean
                df[upper] = rolling_mean + rolling_std * float(std_dev)
                df[lower] = rolling_mean - rolling_std * float(std_dev)
            elif kind == "stochastic":
                k_name = indicator.get("k", f"{name}_k")
                d_name = indicator.get("d", f"{name}_d")
                signal_period = _resolve_param_or_literal(indicator.get("signal_period"), self.params)
                low_min = df["Low"].rolling(window=int(period)).min()
                high_max = df["High"].rolling(window=int(period)).max()
                df[k_name] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-8)
                df[d_name] = df[k_name].rolling(window=int(signal_period)).mean()
            elif kind == "zscore":
                rolling_mean = df[source].rolling(window=int(period)).mean()
                rolling_std = df[source].rolling(window=int(period)).std()
                df[name] = (df[source] - rolling_mean) / (rolling_std + 1e-8)
            else:
                raise ValueError(f"Unsupported indicator kind: {kind}")

        return df

    def _condition_matches(self, df: pd.DataFrame, i: int, condition: Dict[str, Any]) -> bool:
        op = condition.get("op")
        if op == "all":
            return all(self._condition_matches(df, i, item) for item in condition.get("conditions", []))
        if op == "any":
            return any(self._condition_matches(df, i, item) for item in condition.get("conditions", []))

        left = condition.get("left")
        right = condition.get("right")
        left_now = self._value_at(df, i, left)
        right_now = self._value_at(df, i, right)
        if pd.isna(left_now) or pd.isna(right_now):
            return False

        if op == "gt":
            return left_now > right_now
        if op == "gte":
            return left_now >= right_now
        if op == "lt":
            return left_now < right_now
        if op == "lte":
            return left_now <= right_now
        if op == "crosses_above":
            left_prev = self._value_at(df, i - 1, left)
            right_prev = self._value_at(df, i - 1, right)
            if pd.isna(left_prev) or pd.isna(right_prev):
                return False
            return left_prev <= right_prev and left_now > right_now
        if op == "crosses_below":
            left_prev = self._value_at(df, i - 1, left)
            right_prev = self._value_at(df, i - 1, right)
            if pd.isna(left_prev) or pd.isna(right_prev):
                return False
            return left_prev >= right_prev and left_now < right_now

        raise ValueError(f"Unsupported condition op: {op}")

    def _value_at(self, df: pd.DataFrame, i: int, token: Any) -> float:
        if isinstance(token, dict):
            if "param" in token:
                return float(self.params[token["param"]])
            if "value" in token:
                return float(token["value"])
            raise ValueError(f"Unsupported value token: {token}")
        if isinstance(token, (int, float, np.integer, np.floating)):
            return float(token)
        if isinstance(token, str):
            if token in self.params:
                return float(self.params[token])
            if token in df.columns:
                return float(df[token].iloc[i])
        raise ValueError(f"Unknown condition token: {token}")

    def _maybe_close_trade(
        self,
        df: pd.DataFrame,
        i: int,
        current_price: float,
        current_position: PositionType,
    ) -> PositionType:
        trade = self.active_trade
        if trade is None:
            return current_position

        if trade.position_type == PositionType.LONG:
            if current_price >= trade.take_profit:
                exit_signal = Signal.LONG_EXIT
                status = "tp_hit"
            elif current_price <= trade.stop_loss:
                exit_signal = Signal.LONG_EXIT
                status = "sl_hit"
            else:
                return current_position
            position_type = PositionType.LONG
        else:
            if current_price <= trade.take_profit:
                exit_signal = Signal.SHORT_EXIT
                status = "tp_hit"
            elif current_price >= trade.stop_loss:
                exit_signal = Signal.SHORT_EXIT
                status = "sl_hit"
            else:
                return current_position
            position_type = PositionType.SHORT

        df.loc[df.index[i], "Signal"] = exit_signal.value
        trade.exit_date = df.index[i]
        trade.exit_price = current_price
        trade.pnl = self.calculate_pnl_with_fees(
            trade.entry_price,
            current_price,
            position_type,
        )
        trade.status = status
        self.trades.append(trade)
        self.active_trade = None
        return PositionType.NONE


def build_rule_strategy_definition(proposal: Dict[str, Any]) -> StrategyDefinition:
    """Build a StrategyDefinition from a validated LLM rule proposal."""
    spec = validate_rule_strategy_proposal(proposal)
    strategy_key = spec["strategy_key"]
    strategy_name = spec["strategy_name"]
    param_ranges = spec["param_ranges"]
    live_parameter_names = tuple(param_ranges.keys())

    class_name = _to_class_name(strategy_key)
    strategy_class = type(
        class_name,
        (RuleBasedStrategy,),
        {
            "SPEC": spec,
            "STRATEGY_KEY": strategy_key,
            "STRATEGY_NAME": strategy_name,
            "OPTIMIZATION_PARAM_RANGES": param_ranges,
            "LIVE_PARAMETER_NAMES": live_parameter_names,
            "__module__": __name__,
        },
    )

    return StrategyDefinition(
        key=strategy_key,
        name=strategy_name,
        strategy_class=strategy_class,
        optimization_param_ranges=param_ranges,
        live_parameter_names=live_parameter_names,
    )


def validate_rule_strategy_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    required = {"strategy_key", "strategy_name", "param_ranges", "indicators"}
    missing = required.difference(proposal)
    if missing:
        raise ValueError(f"Rule proposal missing required fields: {sorted(missing)}")

    spec = dict(proposal)
    spec["strategy_key"] = _safe_key(str(spec["strategy_key"]))
    spec["strategy_name"] = str(spec["strategy_name"])
    spec["param_ranges"] = _validate_param_ranges(spec["param_ranges"])
    if "risk_reward_ratio" not in spec["param_ranges"]:
        spec["param_ranges"]["risk_reward_ratio"] = [1.5, 2.0, 2.5, 3.0]
    if "trading_fee" not in spec["param_ranges"]:
        spec["param_ranges"]["trading_fee"] = [0.001]

    indicator_names = set()
    for indicator in spec["indicators"]:
        kind = indicator.get("kind")
        if kind not in SUPPORTED_INDICATORS:
            raise ValueError(f"Unsupported indicator kind: {kind}")
        if kind not in {"bollinger", "macd", "stochastic"}:
            name = indicator.get("name")
            if not name:
                raise ValueError(f"{kind} indicators must define name")
            indicator_names.add(name)
        elif kind == "bollinger":
            for output_name in ("middle", "upper", "lower"):
                value = indicator.get(output_name)
                if value:
                    indicator_names.add(value)
        elif kind == "macd":
            name = indicator.get("name", "macd")
            indicator_names.update(
                {
                    indicator.get("macd", f"{name}_macd"),
                    indicator.get("signal", f"{name}_signal"),
                    indicator.get("histogram", f"{name}_histogram"),
                }
            )
        elif kind == "stochastic":
            name = indicator.get("name", "stoch")
            indicator_names.update(
                {
                    indicator.get("k", f"{name}_k"),
                    indicator.get("d", f"{name}_d"),
                }
            )

    if not spec.get("long_entry") and not spec.get("short_entry"):
        raise ValueError("Rule proposal must define long_entry or short_entry")
    for side in ("long_entry", "short_entry"):
        if spec.get(side):
            _validate_condition(spec[side])

    for constraint in spec.get("constraints", []):
        _validate_constraint(constraint)

    return spec


def _validate_param_ranges(param_ranges: Dict[str, Any]) -> Dict[str, List[Any]]:
    if not isinstance(param_ranges, dict) or not param_ranges:
        raise ValueError("param_ranges must be a non-empty object")

    validated = {}
    for name, values in param_ranges.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Parameter {name} must have a non-empty list of values")
        if len(values) > 25:
            raise ValueError(f"Parameter {name} has too many values; keep search space bounded")
        validated[name] = values
    return validated


def _validate_condition(condition: Dict[str, Any]) -> None:
    op = condition.get("op")
    if op not in SUPPORTED_CONDITION_OPS:
        raise ValueError(f"Unsupported condition op: {op}")
    if op in {"all", "any"}:
        items = condition.get("conditions", [])
        if not items:
            raise ValueError(f"{op} condition requires nested conditions")
        for item in items:
            _validate_condition(item)
        return
    if "left" not in condition or "right" not in condition:
        raise ValueError(f"{op} condition requires left and right")


def _validate_constraint(constraint: Dict[str, Any]) -> None:
    if constraint.get("op") not in {"lt", "lte", "gt", "gte"}:
        raise ValueError(f"Unsupported constraint op: {constraint.get('op')}")
    if "left" not in constraint or "right" not in constraint:
        raise ValueError("Constraint requires left and right")


def _constraint_matches(constraint: Dict[str, Any], params: Dict[str, Any]) -> bool:
    left = _resolve_param_or_literal(constraint["left"], params)
    right = _resolve_param_or_literal(constraint["right"], params)
    op = constraint["op"]
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    return False


def _resolve_param_or_literal(value: Any, params: Dict[str, Any]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        if "param" in value:
            return float(params[value["param"]])
        if "value" in value:
            return float(value["value"])
    if isinstance(value, str):
        if value in params:
            return float(params[value])
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()


def _calculate_cci(df: pd.DataFrame, period: int) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))),
        raw=True,
    )
    return (typical_price - sma) / (0.015 * mean_deviation + 1e-8)


def _safe_key(value: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    if not key:
        raise ValueError("strategy_key cannot be empty")
    if not key.startswith("llm_"):
        key = f"llm_{key}"
    return key


def _to_class_name(strategy_key: str) -> str:
    return "".join(part.capitalize() for part in strategy_key.split("_")) + "Strategy"
