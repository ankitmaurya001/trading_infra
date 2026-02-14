# How to Enable Commodity Trading and Transfer Funds

## Problem

If you see:
- `"enabled": false` in commodity margins
- Available margin: ₹0.00
- But you have funds in your account

This means:
1. **Commodity trading is not enabled** in your Zerodha account, OR
2. **Funds are in equity segment** and need to be transferred to commodity segment

## Solution

### Step 1: Enable Commodity Trading

1. **Login to Kite Web**: https://kite.zerodha.com
2. **Go to**: Settings → Account → Product Enablement
3. **Enable**: Commodity Trading (MCX)
4. **Complete**: Any required documentation/KYC if prompted

### Step 2: Transfer Funds to Commodity Segment

1. **Go to**: Funds → Commodity
2. **Click**: "Transfer Funds"
3. **Select**: From Equity → To Commodity
4. **Enter Amount**: Transfer the amount you want to use for trading
5. **Confirm**: Transfer

### Step 3: Verify

After enabling and transferring:

1. **Check in Kite Web**: Funds → Commodity should show your balance
2. **Run test script**: `python test_kite_broker.py`
3. **Verify**: 
   - `enabled: true`
   - Available margin should show your transferred amount

## Important Notes

- **Separate Segments**: Equity and Commodity are separate segments in Zerodha
- **Fund Transfer**: Funds need to be explicitly transferred between segments
- **Margin Requirements**: Commodity margins are typically higher than equity
- **Actual Margin**: Use `order_margins()` API to get real margin requirements (not estimates)

## API Response Structure

When commodity is enabled and has funds:
```json
{
  "enabled": true,
  "net": 19500.0,  // Your available balance
  "available": {
    "cash": 19500.0,
    "live_balance": 19500.0,
    ...
  },
  "utilised": {
    "debits": 0,
    "span": 0,
    ...
  }
}
```

## Quick Check

Run the test script to see:
- If commodity is enabled
- Where your funds are (equity vs commodity)
- Actual margin requirements

```bash
python test_kite_broker.py
```

The script will now show:
- ✅ Commodity Trading Enabled: YES/NO
- 💡 Found funds in Equity segment (if applicable)
- Instructions on how to fix

