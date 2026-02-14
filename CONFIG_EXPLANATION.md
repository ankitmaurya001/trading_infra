# Configuration Flags Explanation

## Overview

The Kite trading engine uses **one flag** to control behavior:

1. **`live_trading`** - Controls **order execution** (virtual vs real orders)

**Note**: `mock_mode` has been removed. This engine always uses **live data** from Kite Connect API. For backtesting, use separate scripts like `run_ma_mock_validation_kite.py`.

## Flag Details

### `live_trading` (Order Execution)

Controls whether orders are actually placed:

- **`live_trading: false`** - **Virtual Trading** (Default - Safe)
  - Strategy signals are generated
  - Trades are simulated/virtual
  - No real orders placed on exchange
  - Safe for testing strategies with live data
  - PnL is calculated but not real

- **`live_trading: true`** - **Live Trading** (Real Orders)
  - Real orders are placed on Kite Connect
  - Requires sufficient margin
  - Real money at risk
  - Use with extreme caution!

## Trading Modes

| live_trading | Data Source | Order Execution | Use Case |
|--------------|-------------|-----------------|----------|
| `false` | Live (Kite API) | Virtual | Testing with real-time data (recommended) |
| `true` | Live (Kite API) | Real | Live trading (real money) |

**Note**: This engine always uses live data from Kite Connect API. For backtesting with historical data, use separate scripts.

## What About `trading_mode`?

The `trading_mode` field in the config file is **NOT used** by the Kite trading engine. It's a leftover from the Binance trading engine implementation and can be ignored or removed.

The Binance engine uses `trading_mode` to distinguish between:
- `testnet` - Binance testnet
- `live` - Binance live trading
- `mock` - Mock/paper trading

But for Kite, we use the simpler two-flag system:
- `mock_mode` - for data source
- `live_trading` - for order execution

## Recommended Workflow

### Step 1: Backtest (Use Separate Script)
Use `run_ma_mock_validation_kite.py` or similar scripts for backtesting with historical data.

### Step 2: Test with Live Data (live_trading: false)
```json
{
    "live_trading": false
}
```
Test your strategy with real-time data but no real orders.

### Step 3: Live Trading (live_trading: true)
```json
{
    "live_trading": true
}
```
**Only enable this when you're confident!**

## Example Configurations

### Safe Testing Configuration
```json
{
    "live_trading": false,
    "symbol": "GOLDM25DECFUT",
    "interval": "15minute"
}
```
- Uses live data from Kite API
- No real orders
- Perfect for testing

### Live Trading Configuration
```json
{
    "live_trading": true,
    "symbol": "GOLDM25DECFUT",
    "interval": "15minute"
}
```
- Uses live data from Kite API
- Places real orders
- **Use with caution!**

**Note**: For backtesting, use separate scripts like `run_ma_mock_validation_kite.py`

## Summary

- **`live_trading`** = Whether orders are real or virtual
- **Data Source** = Always live data from Kite Connect API
- **Backtesting** = Use separate scripts (e.g., `run_ma_mock_validation_kite.py`)
- **`trading_mode`** = Not used for Kite (can be ignored)

