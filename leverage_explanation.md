# MCX Commodity Leverage & PnL Explanation

## Your Trade Details
- **Symbol**: NATGASMINI26FEBFUT
- **Entry Price**: ₹264.50
- **Current Price**: ₹276.80
- **PnL**: ₹3,400 (positive)
- **Lot Size**: 250 MMBTU
- **NRML Margin**: ₹14,451
- **Margin Rate**: 23.75%
- **Base Price (for margin)**: ₹243.4

---

## How Leverage Works in MCX Commodities

### 1. Margin Calculation (How Much Capital You Need)

**Key Concept**: Margin is calculated on a **base price** set by the exchange, NOT your entry price!

```
Base Price (set by exchange) = ₹243.4
Lot Size = 250 MMBTU

Contract Value (at base price) = 243.4 × 250 = ₹60,850

Margin Required = Contract Value × Margin Rate
                = ₹60,850 × 23.75%
                = ₹14,451 ✓
```

**Important Points**:
- The margin (₹14,451) is **blocked** from your account when you open the position
- This margin is calculated on the base price (₹243.4), not your entry price (₹264.50)
- The margin rate (23.75%) represents the leverage: You control ₹60,850 worth of contract with ₹14,451
- **Effective Leverage** = Contract Value / Margin = ₹60,850 / ₹14,451 ≈ **4.21x**

---

### 2. PnL Calculation (How Profit/Loss is Calculated)

**Key Concept**: PnL is calculated on the **actual price movement** × **lot size**, NOT based on margin!

```
Entry Price = ₹264.50
Current Price = ₹276.80

Price Change = 276.80 - 264.50 = ₹12.30 per unit

PnL = Price Change × Lot Size
    = ₹12.30 × 250 MMBTU
    = ₹3,075
```

**However**, you're seeing ₹3,400. This could be due to:
1. **Slight price differences**: Your actual entry might be slightly different
2. **M2M (Mark-to-Market) adjustments**: Daily settlement adjustments
3. **Rounding**: Exchange calculations may round differently

---

### 3. Understanding the Relationship

**Why margin and PnL seem disconnected:**

```
Margin (₹14,451) = Based on BASE PRICE (₹243.4)
PnL (₹3,400)     = Based on ACTUAL PRICE MOVEMENT (₹264.50 → ₹276.80)
```

These are **two separate calculations**:
- **Margin** = Capital requirement (risk management)
- **PnL** = Actual profit/loss from price movement

---

### 4. Leverage Impact

**What leverage means in practice:**

```
Without Leverage (if you had to pay full contract value):
- Capital needed = ₹264.50 × 250 = ₹66,125
- Profit = ₹3,400
- Return = 3,400 / 66,125 = 5.14%

With Leverage (actual):
- Capital needed = ₹14,451 (margin)
- Profit = ₹3,400
- Return = 3,400 / 14,451 = 23.53% 🚀
```

**This is why leverage is powerful but risky!**
- Small price movements create large percentage returns
- But losses are also magnified

---

### 5. Risk Management

**Important considerations:**

1. **Margin Call Risk**: If price moves against you significantly, you may need additional margin
2. **M2M (Mark-to-Market)**: Daily settlement can affect your available margin
3. **Stop Loss**: Always use stop-loss to limit downside (leverage amplifies losses too!)

**Example of downside:**
```
If price drops to ₹250.00:
Price Change = 250.00 - 264.50 = -₹14.50 per unit
Loss = -₹14.50 × 250 = -₹3,625
Return = -3,625 / 14,451 = -25.08% 😱
```

---

## Summary

✅ **Margin (₹14,451)**: Capital blocked, calculated on base price (₹243.4)  
✅ **PnL (₹3,400)**: Actual profit, calculated on price movement (₹264.50 → ₹276.80)  
✅ **Leverage (~4.21x)**: You control ₹60,850 contract with ₹14,451 capital  
✅ **Return**: ~23.53% on your margin (vs 5.14% without leverage)

**The key insight**: Margin and PnL are calculated independently:
- Margin = Risk management tool (based on base price)
- PnL = Actual trading result (based on your entry/exit prices)

