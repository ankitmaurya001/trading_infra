# PnL Calculation Fix: Using Lot Size for Commodity Futures

## The Problem 🐛

The dollar PnL calculation was **incorrect** for commodity futures trading. Here's what was wrong:

### Before Fix (WRONG ❌)

```python
# In trading_engine.py, line 564 (OLD CODE)
dollar_pnl = (price - trade['entry_price']) * trade['quantity']
```

**Issue:**
- `trade['quantity']` = **1** (number of lots)
- For NATGASMINI: PnL = (₹276.80 - ₹264.50) × **1** = ₹12.30 ❌
- **This is WRONG!** It should be ₹3,075 (₹12.30 × 250 MMBTU)

### Why It Was Wrong

For commodity futures:
- **`quantity`** = Number of lots (typically 1)
- **`lot_size`** = Contract size per lot (e.g., 250 MMBTU for NATGASMINI)
- **PnL** = Price change × **lot_size** × quantity

The code was using `quantity` (1 lot) instead of `lot_size` (250 MMBTU)!

---

## The Fix ✅

### After Fix (CORRECT ✅)

```python
# In trading_engine.py, lines 565-587 (NEW CODE)
lot_size_for_pnl = trade.get('lot_size')
if lot_size_for_pnl and lot_size_for_pnl > 0:
    # Commodity futures: PnL = price_change × lot_size × quantity
    quantity_for_pnl = trade.get('quantity', 1)
    if trade['action'] == 'BUY':
        dollar_pnl = (price - trade['entry_price']) * lot_size_for_pnl * quantity_for_pnl
    else:
        dollar_pnl = (trade['entry_price'] - price) * lot_size_for_pnl * quantity_for_pnl
else:
    # For non-commodity brokers (e.g., Binance), quantity is already in base units
    if trade['action'] == 'BUY':
        dollar_pnl = (price - trade['entry_price']) * trade['quantity']
    else:
        dollar_pnl = (trade['entry_price'] - price) * trade['quantity']
```

**Now:**
- For NATGASMINI: PnL = (₹276.80 - ₹264.50) × **250** × 1 = ₹3,075 ✅
- **This is CORRECT!**

---

## What Changed

### 1. Store `lot_size` in Trade Record

**Location:** `trading_engine.py`, line ~496

```python
trade = {
    # ... other fields ...
    'lot_size': lot_size  # Store lot_size for commodity futures
}
```

### 2. Get `lot_size` During Trade Execution

**Location:** `trading_engine.py`, lines ~363-366

```python
# Get and store lot_size for commodity futures
lot_size = None
if is_kite_broker and actual_lot_margin and actual_lot_margin > 0:
    try:
        lot_info = self.broker.get_symbol_filters(self.symbol)
        lot_size = lot_info.get('lot_size', 1)
```

### 3. Use `lot_size` in PnL Calculation

**Location:** `trading_engine.py`, lines 565-587

- Checks if `lot_size` exists in trade record
- If yes (commodity futures): Uses `lot_size × quantity` for PnL
- If no (other brokers): Uses `quantity` directly (backward compatible)

---

## Example Calculation

### NATGASMINI26FEBFUT Trade

**Trade Details:**
- Entry Price: ₹264.50
- Exit Price: ₹276.80
- Lot Size: 250 MMBTU
- Quantity: 1 lot

**Before Fix (WRONG):**
```
dollar_pnl = (276.80 - 264.50) × 1 = ₹12.30 ❌
```

**After Fix (CORRECT):**
```
dollar_pnl = (276.80 - 264.50) × 250 × 1 = ₹3,075 ✅
```

---

## Backward Compatibility

The fix is **backward compatible**:
- ✅ For commodity futures (Kite): Uses `lot_size` correctly
- ✅ For other brokers (Binance, etc.): Uses `quantity` directly (no change)
- ✅ Old trades without `lot_size`: Falls back to using `quantity` (works but may be less accurate)

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **PnL Calculation** | `price_change × quantity` | `price_change × lot_size × quantity` |
| **For NATGASMINI** | ₹12.30 ❌ | ₹3,075 ✅ |
| **lot_size Stored?** | No | Yes |
| **Backward Compatible?** | N/A | Yes ✅ |

**The fix ensures correct PnL calculation for commodity futures by using `lot_size` instead of just `quantity`!** 🎯

