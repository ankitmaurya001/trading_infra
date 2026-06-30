# LLM Strategy Research Loop

This workflow lets an external LLM propose strategy ideas while the local repo
does the optimization, validation, comparison, and artifact writing.

The LLM does not write executable Python. It returns JSON in a constrained
strategy DSL. Local code converts valid proposals into `BaseStrategy` classes,
runs `StrategyOptimizer`, validates out-of-sample, mutates top candidates, and
ranks the results.

## Run With An LLM Endpoint

Use an OpenAI-compatible `/chat/completions` endpoint:

`--data-file` is optional. It is useful for reproducible offline research, but
for MCX/Kite workflows the script can fetch broker data directly.

```bash
export LLM_STRATEGY_ENDPOINT="https://your-llm-endpoint/v1/chat/completions"
export LLM_STRATEGY_MODEL="your-model"
export LLM_STRATEGY_API_KEY="your-key"

python llm_strategy_research.py \
  --data-source kite \
  --symbol NATGASMINI26JULFUT \
  --exchange MCX \
  --interval 5m \
  --days 120 \
  --n-ideas 5 \
  --llm-rounds 20 \
  --generations 2 \
  --top-k 3 \
  --trading-fee 0.001 \
  --target-win-rate 0.50 \
  --target-min-pnl 0.0 \
  --target-max-drawdown 0.25 \
  --target-min-trades 30 \
  --target-require-train-stability \
  --print-llm-reply \
  --context "Prefer non-MA strategies suitable for MCX commodity futures."
```

If the endpoint supports web search, allow it in the prompt:

```bash
python llm_strategy_research.py \
  --data-source kite \
  --symbol NATGASMINI26JULFUT \
  --exchange MCX \
  --interval 5m \
  --allow-llm-web-search \
  --n-ideas 5
```

The prompt includes a compact market-data summary: row count, date range, close
range, recent return statistics, and available columns. If you also want to
share a few recent bars with the LLM, pass:

```bash
--llm-sample-rows 20
```

Keep sample rows small; the local evaluator, not the LLM, does the real testing.

`--n-ideas` controls how many strategy ideas the LLM should propose per round.
`--llm-rounds` controls how many proposal/test/feedback rounds to run. After
each round, the script feeds the optimizer/validation results back to the LLM
and asks it to avoid weak ideas and propose different or improved logic.

The loop stops early when the best candidate meets the target gates. By default,
the target is validation win rate at least 50%, validation PnL at least 0,
drawdown under `--max-drawdown-threshold`, and matching train-side stability.
You can tune the target:

```bash
--target-win-rate 0.50
--target-min-pnl 0.0
--target-max-drawdown 0.25
--target-min-trades 30
--target-require-train-stability
--target-max-win-rate-gap 0.25
```

To force all LLM rounds to run even after a target-quality strategy is found:

```bash
--no-stop-when-target-met
```

To print the full LLM JSON reply plus accepted/rejected strategies before
optimization starts, pass:

```bash
--print-llm-reply
```

By default, only LLM-proposed strategies are tested. To compare against local
baselines such as MA, RSI, Donchian, Bollinger, and MACD, pass:

```bash
--include-baselines
```

## Run Offline With Saved Ideas

```bash
python llm_strategy_research.py \
  --ideas-file strategy_ideas.json \
  --data-file path/to/ohlcv.csv \
  --generations 2 \
  --top-k 3
```

## Outputs

The default output directory is `llm_strategy_research_results/`.

It writes:

- `llm_strategy_research_report.json`: LLM prompt/source plus accepted/rejected ideas
- `aggregate_strategy_search_report.json`: best result and per-round summaries
- `aggregate_strategy_rankings.csv`: flat ranking table across all LLM rounds
- `round_XX/llm_strategy_ideas.json`: raw LLM proposal payload per round
- `round_XX/strategy_search_report.json`: full optimizer/validation report per round
- `round_XX/strategy_search_rankings.csv`: rankings per round

## Proposal Schema

The LLM should return:

```json
{
  "research_notes": "short rationale",
  "strategies": [
    {
      "type": "rule_dsl",
      "strategy_key": "ema_rsi_trend",
      "strategy_name": "EMA RSI Trend",
      "hypothesis": "Trend-following crossover filtered by RSI.",
      "param_ranges": {
        "fast_period": [5, 8, 13],
        "slow_period": [21, 34, 55],
        "rsi_period": [10, 14],
        "rsi_long_max": [65, 70],
        "rsi_short_min": [30, 35],
        "risk_reward_ratio": [1.5, 2.0, 2.5],
        "trading_fee": [0.001]
      },
      "constraints": [
        {"op": "lt", "left": "fast_period", "right": "slow_period"}
      ],
      "indicators": [
        {"kind": "ema", "name": "ema_fast", "source": "Close", "period": "fast_period"},
        {"kind": "ema", "name": "ema_slow", "source": "Close", "period": "slow_period"},
        {"kind": "rsi", "name": "rsi", "source": "Close", "period": "rsi_period"}
      ],
      "long_entry": {
        "op": "all",
        "conditions": [
          {"op": "crosses_above", "left": "ema_fast", "right": "ema_slow"},
          {"op": "lt", "left": "rsi", "right": "rsi_long_max"}
        ]
      },
      "short_entry": {
        "op": "all",
        "conditions": [
          {"op": "crosses_below", "left": "ema_fast", "right": "ema_slow"},
          {"op": "gt", "left": "rsi", "right": "rsi_short_min"}
        ]
      }
    }
  ]
}
```

Supported indicator kinds:

- `sma`
- `ema`
- `rsi`
- `macd`
- `stochastic`
- `atr`
- `cci`
- `roc`
- `zscore`
- `rolling_high`
- `rolling_low`
- `rolling_mean`
- `rolling_std`
- `bollinger`

Supported condition ops:

- `all`
- `any`
- `gt`
- `gte`
- `lt`
- `lte`
- `crosses_above`
- `crosses_below`

## Safety Boundary

The LLM can suggest strategy logic, parameter ranges, and hypotheses. It cannot
ship arbitrary code into the evaluator. Any idea outside the DSL is rejected and
recorded in `llm_strategy_research_report.json`.
