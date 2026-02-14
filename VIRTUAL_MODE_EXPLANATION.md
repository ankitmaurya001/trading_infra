# Virtual Trading Mode Explanation

## What is Virtual Trading?

When `live_trading: false`, the system operates in **Virtual Trading Mode**. This mode provides a realistic simulation of live trading while keeping your capital safe.

## What Virtual Mode DOES

### ✅ Still Connects to Kite
- Authenticates with Kite Connect API
- Fetches real-time market data
- Gets current prices, margins, and account information

### ✅ Validates Everything
- **Margin Checks**: Validates if you have sufficient margin before "placing" trades
- **Lot Size Validation**: Checks lot sizes for commodities
- **Price Validation**: Uses real market prices from Kite
- **Market Hours**: Respects market hours (MCX: 9:00 AM - 11:55 PM IST)

### ✅ Tracks Everything Internally
- **Position Tracking**: Maintains internal position records
- **PnL Calculation**: Calculates profit/loss based on real prices
- **Stop Loss/Take Profit**: Calculates and monitors levels (but doesn't place GTT orders)
- **Trade History**: Logs all virtual trades
- **Performance Metrics**: Tracks win rate, Sharpe ratio, etc.

### ✅ Margin Monitoring
- Continuously monitors margin requirements
- Alerts if margin would be insufficient
- Shows what would happen in live trading

## What Virtual Mode DOES NOT Do

### ❌ No Real Orders
- Does NOT place orders on Kite Connect
- Does NOT create GTT stop-loss orders
- Does NOT modify your actual positions
- Does NOT use your real capital

### ❌ No Real Risk
- No money at risk
- No actual positions opened
- No broker fees charged
- No margin actually used

## How It Works

```
┌─────────────────────────────────────────┐
│  Strategy generates signal (BUY/SELL)   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Check margin (via Kite API)            │ ← Still checks real margin
│  Validate lot size (via Kite API)       │ ← Still validates
│  Get current price (via Kite API)       │ ← Still gets real price
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Calculate stop-loss & take-profit      │ ← Still calculates
│  Create virtual trade record            │ ← Tracks internally
│  Update internal PnL                    │ ← Tracks performance
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  ❌ SKIP: place_order() call            │ ← Does NOT place order
│  ❌ SKIP: place_gtt_order() call        │ ← Does NOT place GTT
└─────────────────────────────────────────┘
```

## Example Flow

### Virtual Mode (`live_trading: false`)

1. **Strategy Signal**: "BUY GOLDM25DECFUT"
2. **Margin Check**: ✅ Checks real margin via Kite API
   - Available: ₹50,000
   - Required: ₹10,000
   - Result: ✅ Sufficient margin
3. **Price Check**: ✅ Gets real LTP from Kite: ₹65,000
4. **Lot Size**: ✅ Gets real lot size: 1
5. **Trade Execution**: 
   - Creates virtual trade record
   - Calculates stop-loss: ₹64,000
   - Calculates take-profit: ₹66,000
   - Updates internal PnL tracker
   - ❌ Does NOT call `kite.place_order()`
   - ❌ Does NOT call `kite.place_gtt()`
6. **Monitoring**: 
   - Monitors real price changes
   - Updates virtual PnL
   - Checks if take-profit/stop-loss would be hit
   - ❌ Does NOT place exit orders

### Live Mode (`live_trading: true`)

Same as above, but:
- ✅ Calls `kite.place_order()` - Real order placed
- ✅ Calls `kite.place_gtt()` - Real GTT stop-loss placed
- ✅ Real money at risk
- ✅ Real positions opened

## Benefits of Virtual Mode

1. **Safe Testing**: Test strategies with real market data without risk
2. **Margin Validation**: Verify you have sufficient margin before going live
3. **Realistic Simulation**: Uses actual prices, margins, and market conditions
4. **Performance Tracking**: See how your strategy would perform
5. **Debugging**: Identify issues before risking real capital

## When to Use Virtual Mode

- ✅ Testing new strategies
- ✅ Validating margin requirements
- ✅ Understanding strategy behavior
- ✅ Debugging trading logic
- ✅ Learning the system

## When to Switch to Live Mode

Only switch to `live_trading: true` when:
- ✅ Strategy tested thoroughly in virtual mode
- ✅ You understand the risks
- ✅ You have sufficient margin
- ✅ You're ready to risk real capital

## Code Flow

```python
# In trading_engine.py
if self.broker and self.use_broker:  # use_broker = live_trading
    # Place real orders
    broker_order = self.broker.place_order(...)
    gtt_order = self.broker.place_gtt_order(...)
else:
    # Virtual mode - skip order placement
    # But still track trade internally
    pass

# Trade is always tracked internally regardless of mode
trade = {
    'status': 'open',
    'entry_price': price,
    'take_profit': take_profit,
    'stop_loss': stop_loss,
    # ... tracked in memory
}
```

## Summary

**Virtual Mode = Real Data + Real Validation + Virtual Execution**

- ✅ Real connection to Kite
- ✅ Real margin checks
- ✅ Real price data
- ✅ Real validation
- ❌ No real orders
- ❌ No real risk

This gives you the confidence that your strategy will work in live trading, without risking capital.

