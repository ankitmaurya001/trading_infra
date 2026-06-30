# Strategy Template

This document is the agent-readable contract for adding a new trading strategy.
Use it with `strategies.py`, `strategy_optimizer.py`, `strategy_manager.py`, and
`agentic_strategy_search.py`.

## Required Shape

Every strategy must:

1. Subclass `BaseStrategy`.
2. Define a unique `STRATEGY_KEY`.
3. Define a human-readable `STRATEGY_NAME`.
4. Define `OPTIMIZATION_PARAM_RANGES` for grid search.
5. Define `LIVE_PARAMETER_NAMES` for params required to instantiate the live strategy.
6. Implement `validate_parameters(cls, params)`.
7. Implement `generate_signals(self, data)`.
8. Return a signal DataFrame with these columns:
   - `Signal`
   - `Position`
   - `Take_Profit`
   - `Stop_Loss`
9. Add the class to `STRATEGY_CLASSES` in `strategies.py`.
10. Add or update tests when the strategy has custom behavior.

## Minimal Skeleton

```python
class MyStrategy(BaseStrategy):
    STRATEGY_KEY = "my_strategy"
    STRATEGY_NAME = "My Strategy"
    OPTIMIZATION_PARAM_RANGES = {
        "lookback": [10, 20, 30],
        "risk_reward_ratio": [1.5, 2.0, 2.5],
        "trading_fee": [0.001],
    }
    LIVE_PARAMETER_NAMES = (
        "lookback",
        "risk_reward_ratio",
        "trading_fee",
    )

    def __init__(
        self,
        lookback: int = 20,
        risk_reward_ratio: float = 2.0,
        trading_fee: float = 0.0,
    ):
        super().__init__(self.STRATEGY_NAME, risk_reward_ratio, trading_fee=trading_fee)
        self.lookback = lookback

    @classmethod
    def validate_parameters(cls, params: Dict[str, Any]) -> bool:
        return params.get("lookback", 0) > 0

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["ATR"] = self.calculate_atr(df)
        df["Signal"] = Signal.HOLD.value
        df["Position"] = 0
        df["Take_Profit"] = np.nan
        df["Stop_Loss"] = np.nan

        # Add strategy-specific entry and exit logic here.
        return df
```

After adding the class:

```python
STRATEGY_CLASSES = (
    MovingAverageCrossover,
    RSIStrategy,
    DonchianChannelBreakout,
    MyStrategy,
)
```

`STRATEGY_DEFINITIONS` is built automatically from `STRATEGY_CLASSES`.

## Optimization Contract

`StrategyOptimizer` creates strategy instances from `OPTIMIZATION_PARAM_RANGES`.
Invalid combinations must be rejected by `validate_parameters`.

Examples:

- Moving average: `short_window < long_window`
- RSI: `oversold < overbought`
- Donchian: `channel_period > 0`

Keep ranges small enough for grid search, or add a strategy-specific optimizer
path if the search space is large.

## Live Contract

`StrategyManager.initialize_strategies` uses `StrategyDefinition.create_strategy`
to instantiate live strategies from optimized or manual params.

Every name in `LIVE_PARAMETER_NAMES` must:

- exist in `OPTIMIZATION_PARAM_RANGES`
- be accepted by the strategy constructor
- be present in optimized/manual params before live trading starts

## Signal Contract

`generate_signals(data)` must:

- accept OHLCV data with `Open`, `High`, `Low`, `Close`, and `Volume`
- preserve the input row count
- use `Signal` enum values
- update `Position` consistently
- append closed `Trade` objects to `self.trades`
- leave metrics available through `get_strategy_metrics()`

Use `calculate_trade_levels`, `calculate_pnl_with_fees`, and `calculate_atr`
from `BaseStrategy` instead of duplicating that logic.

## Tests To Run

Run the contract tests after adding or changing a strategy:

```bash
pytest tests/test_strategy_template_contract.py -q
```

Run the agentic search loop after the strategy is registered:

```bash
python agentic_strategy_search.py --data-file path/to/ohlcv.csv --generations 2 --top-k 3
```

The search loop automatically discovers everything in `STRATEGY_DEFINITIONS`,
optimizes each candidate on train data, validates out-of-sample, mutates the
best candidates into tighter search ranges, and writes:

- `agentic_strategy_search_results/strategy_search_report.json`
- `agentic_strategy_search_results/strategy_search_rankings.csv`

Run the full configured suite before handoff:

```bash
pytest -q
```
