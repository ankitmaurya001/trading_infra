# Kite MCX Commodity Trading Guide

This guide explains how to run the Kite trading engine and test the broker APIs.

## Configuration

### Configuration Flags

The trading engine uses **two independent flags**:

1. **`mock_mode`** - Controls **data source** (historical vs live data)
2. **`live_trading`** - Controls **order execution** (virtual vs real orders)

#### `mock_mode` (Data Source)

- **`mock_mode: true`** - **Backtest Mode**
  - Uses historical data from past dates
  - Processes data sequentially
  - No real-time data fetching
  - Market hours don't matter

- **`mock_mode: false`** - **Live Data Mode**
  - Fetches real-time data from Kite Connect API
  - Processes new candles as they close
  - Requires market to be open (MCX: 9:00 AM - 11:55 PM IST)

#### `live_trading` (Order Execution)

- **`live_trading: false`** - **Virtual Trading** (Default - Safe)
  - Strategy signals are generated
  - Trades are simulated/virtual
  - No real orders placed
  - Safe for testing

- **`live_trading: true`** - **Live Trading** (Real Orders)
  - Real orders are placed on Kite Connect
  - Requires sufficient margin
  - **Use with extreme caution!**

**Note**: The `trading_mode` field in config is NOT used for Kite trading engine (it's a leftover from Binance implementation).

## Running the Trading Engine

### Basic Usage

```bash
# Run with default config (trading_config_kite.json)
python kite_trading_engine.py

# Run with custom config file
python kite_trading_engine.py --config my_config.json
```

### Configuration File Structure

Edit `trading_config_kite.json`:

```json
{
    "symbol": "GOLDM25DECFUT",           // MCX commodity symbol
    "interval": "15minute",              // Data interval
    "polling_frequency": 60,             // Polling frequency in seconds
    "initial_balance": 10000,            // Virtual balance (for tracking)
    "max_leverage": 10,                  // Maximum leverage
    "max_loss_percent": 2.0,             // Max loss per trade (%)
    "live_trading": false,               // Place real orders (true) or virtual (false)
    "enabled_strategies": ["ma"],        // Strategies to use
    "commodity_trading": {
        "margin_buffer_percent": 20,     // Extra margin buffer (%)
        "margin_check_interval_seconds": 300,  // Margin check frequency
        "margin_alert_threshold_percent": 150,  // Alert if margin < 150% of required
        "margin_critical_threshold_percent": 110,  // Exit if margin < 110% of required
        "use_gtt_for_stop_loss": true   // Use GTT for overnight stop-loss
    }
}
```

### Mode Combinations

| mock_mode | live_trading | Data Source | Order Execution | Use Case |
|-----------|--------------|-------------|-----------------|----------|
| `true` | `false` | Historical | Virtual | Backtesting strategies |
| `false` | `false` | Live | Virtual | Testing with real-time data (recommended) |
| `false` | `true` | Live | Real | Live trading (real money) |

## Testing the Broker APIs

### Run the Test Script

```bash
# Test all broker APIs (dry run - no real orders)
python test_kite_broker.py
```

### What the Test Script Does

The test script (`test_kite_broker.py`) tests:

1. **Authentication** - Verifies Kite Connect login
2. **Broker Initialization** - Tests broker setup
3. **Margin Checking** - Gets available/used margins
4. **Lot Size Retrieval** - Gets lot size for symbol
5. **Price Retrieval** - Gets current LTP
6. **Symbol Information** - Gets symbol details (lot size, tick size, etc.)
7. **Position Retrieval** - Gets current positions
8. **Open Orders** - Gets pending orders
9. **GTT Orders** - Gets active GTT orders
10. **Margin Calculation** - Calculates required margin
11. **Order Placement (Dry Run)** - Tests order placement logic (no real orders)
12. **GTT Placement (Dry Run)** - Tests GTT order logic (no real orders)

### Enabling Real Order Placement in Tests

To actually place orders in the test script, uncomment the code in:
- `test_place_order_dry_run()` function
- `test_place_gtt_dry_run()` function

**⚠️ WARNING**: Only do this if you want to place real orders!

## Step-by-Step: First Time Setup

### 1. Configure Credentials

Ensure `config.py` has your Kite credentials:
```python
KITE_CREDENTIALS = {
    "username": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD",
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",
    "totp_key": "YOUR_TOTP_KEY"
}
KITE_EXCHANGE = "MCX"
```

### 2. Test APIs First

```bash
# Run test script to verify everything works
python test_kite_broker.py
```

### 3. Configure Trading

Edit `trading_config_kite.json`:
- Set your symbol (e.g., `"GOLDM25DECFUT"`)
- Set `"live_trading": false` for virtual trading
- Configure your strategies

### 4. Run in Virtual Mode

```bash
# Start with virtual trading (safe)
python kite_trading_engine.py
```

Monitor the logs to see:
- Strategy signals generated
- Virtual trades executed
- No real orders placed

### 5. Enable Live Trading (When Ready)

**⚠️ IMPORTANT**: Only enable live trading after thorough testing!

1. Ensure you have sufficient margin
2. Start with small position sizes
3. Monitor closely
4. Set `"live_trading": true` in config
5. Run: `python kite_trading_engine.py`

## Features

### Margin Management

- **Pre-Entry Check**: Validates margin before placing orders
- **Continuous Monitoring**: Background thread checks margins every 5 minutes
- **Auto-Exit**: Exits positions if margin falls below critical threshold
- **Alerts**: Warns when margin is below alert threshold

### Stop-Loss with GTT

- **Overnight Protection**: GTT orders persist across trading sessions
- **Automatic Triggering**: Triggers when price hits stop-loss level
- **Direction-Aware**: Handles both LONG and SHORT positions

### Take-Profit Monitoring

- **Candle-Based**: Checks take-profit when new candles close
- **Automatic Execution**: Places market order when target is hit
- **GTT Cleanup**: Deletes GTT stop-loss before placing take-profit order

### Position Management

- **Lot-Based Trading**: Automatically uses 1 lot per position
- **Position Syncing**: Syncs with Kite positions
- **PnL Tracking**: Real-time profit/loss calculation

## Safety Features

1. **Default Virtual Mode**: `live_trading` defaults to `false`
2. **Margin Checks**: Validates margin before every order
3. **Risk Limits**: Configurable max loss per trade
4. **Auto-Exit**: Exits positions if margin is critically low
5. **GTT Protection**: Stop-loss persists overnight

## Troubleshooting

### Authentication Errors

- Check credentials in `config.py`
- Verify TOTP key is correct
- Ensure API key has trading permissions

### Margin Errors

- Check available margin: Run test script
- Reduce position size
- Increase `margin_buffer_percent` in config

### Order Placement Errors

- Verify symbol exists on MCX
- Check market hours (MCX: 9:00 AM - 11:55 PM IST)
- Ensure sufficient margin

### GTT Order Errors

- Verify stop-loss price is valid
- Check if GTT limit reached (max 50 GTTs)
- Ensure position exists before placing GTT

## Logs

Logs are saved in:
- `logs/{symbol}_{timestamp}_{mode}/trading_engine.log`
- `logs/{symbol}_{timestamp}_{mode}/trades.csv`
- `logs/{symbol}_{timestamp}_{mode}/status.json`

## Example Workflow

```bash
# 1. Test APIs
python test_kite_broker.py

# 2. Run in virtual mode with live data
# Edit config: "live_trading": false (default)
python kite_trading_engine.py

# 3. Monitor logs and verify signals

# 4. When ready, enable live trading
# Edit config: "live_trading": true
python kite_trading_engine.py
```

## Important Notes

- **Always test in virtual mode first**
- **Start with small positions**
- **Monitor margin requirements**
- **GTT orders count towards your GTT limit (max 50)**
- **Market hours matter**: MCX is open 9:00 AM - 11:55 PM IST (weekdays)
- **Overnight positions require sufficient margin**

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Run test script to verify API connectivity
3. Check Kite Connect API documentation: https://kite.trade/docs/

