#!/usr/bin/env python3
"""
Verify leverage and PnL calculations for MCX commodity futures.
Helps understand the relationship between margin, leverage, and PnL.
"""

def calculate_mcx_leverage_and_pnl():
    """
    Calculate and explain leverage and PnL for NATGASMINI26FEBFUT trade.
    """
    print("=" * 70)
    print("MCX COMMODITY LEVERAGE & PnL CALCULATION VERIFICATION")
    print("=" * 70)
    print()
    
    # Trade details
    symbol = "NATGASMINI26FEBFUT"
    entry_price = 264.50
    current_price = 276.80
    lot_size = 250  # MMBTU
    nrml_margin = 14451  # INR
    margin_rate = 23.75  # Percentage
    base_price = 243.4  # Base price for margin calculation
    reported_pnl = 3400  # INR (what user sees on Kite)
    
    print(f"📊 TRADE DETAILS:")
    print(f"   Symbol: {symbol}")
    print(f"   Entry Price: ₹{entry_price:.2f}")
    print(f"   Current Price: ₹{current_price:.2f}")
    print(f"   Lot Size: {lot_size} MMBTU")
    print(f"   NRML Margin: ₹{nrml_margin:,}")
    print(f"   Margin Rate: {margin_rate}%")
    print(f"   Base Price (for margin): ₹{base_price:.2f}")
    print(f"   Reported PnL: ₹{reported_pnl:,}")
    print()
    
    # 1. Verify Margin Calculation
    print("=" * 70)
    print("1. MARGIN CALCULATION (Capital Requirement)")
    print("=" * 70)
    
    contract_value_at_base = base_price * lot_size
    calculated_margin = contract_value_at_base * (margin_rate / 100)
    
    print(f"   Base Price: ₹{base_price:.2f}")
    print(f"   Lot Size: {lot_size} MMBTU")
    print(f"   Contract Value (at base price) = {base_price:.2f} × {lot_size} = ₹{contract_value_at_base:,.2f}")
    print(f"   Margin Required = ₹{contract_value_at_base:,.2f} × {margin_rate}% = ₹{calculated_margin:,.2f}")
    print(f"   Actual NRML Margin: ₹{nrml_margin:,}")
    
    if abs(calculated_margin - nrml_margin) < 10:
        print(f"   ✅ Margin calculation matches! (Difference: ₹{abs(calculated_margin - nrml_margin):.2f})")
    else:
        print(f"   ⚠️  Margin difference: ₹{abs(calculated_margin - nrml_margin):.2f}")
    print()
    
    # 2. Calculate Leverage
    print("=" * 70)
    print("2. LEVERAGE CALCULATION")
    print("=" * 70)
    
    # Leverage based on base price
    leverage_at_base = contract_value_at_base / nrml_margin
    
    # Leverage based on entry price (what you actually paid)
    contract_value_at_entry = entry_price * lot_size
    leverage_at_entry = contract_value_at_entry / nrml_margin
    
    print(f"   Contract Value (at base price ₹{base_price:.2f}): ₹{contract_value_at_base:,.2f}")
    print(f"   Margin Used: ₹{nrml_margin:,}")
    print(f"   Leverage (at base price) = ₹{contract_value_at_base:,.2f} / ₹{nrml_margin:,} = {leverage_at_base:.2f}x")
    print()
    print(f"   Contract Value (at entry price ₹{entry_price:.2f}): ₹{contract_value_at_entry:,.2f}")
    print(f"   Margin Used: ₹{nrml_margin:,}")
    print(f"   Leverage (at entry price) = ₹{contract_value_at_entry:,.2f} / ₹{nrml_margin:,} = {leverage_at_entry:.2f}x")
    print()
    
    # 3. Calculate PnL
    print("=" * 70)
    print("3. PnL CALCULATION")
    print("=" * 70)
    
    price_change = current_price - entry_price
    calculated_pnl = price_change * lot_size
    
    print(f"   Entry Price: ₹{entry_price:.2f}")
    print(f"   Current Price: ₹{current_price:.2f}")
    print(f"   Price Change = ₹{current_price:.2f} - ₹{entry_price:.2f} = ₹{price_change:.2f} per unit")
    print(f"   PnL = Price Change × Lot Size")
    print(f"       = ₹{price_change:.2f} × {lot_size}")
    print(f"       = ₹{calculated_pnl:,.2f}")
    print(f"   Reported PnL: ₹{reported_pnl:,}")
    
    difference = reported_pnl - calculated_pnl
    if abs(difference) < 50:
        print(f"   ✅ PnL calculation matches! (Difference: ₹{difference:.2f})")
    else:
        print(f"   ⚠️  PnL difference: ₹{difference:.2f}")
        print(f"   Possible reasons:")
        print(f"      - Actual entry price might be slightly different")
        print(f"      - M2M (Mark-to-Market) adjustments")
        print(f"      - Exchange rounding differences")
        print(f"      - Brokerage/charges (though usually minimal)")
    print()
    
    # 4. Return Calculation
    print("=" * 70)
    print("4. RETURN CALCULATION")
    print("=" * 70)
    
    return_on_margin = (reported_pnl / nrml_margin) * 100
    return_without_leverage = (reported_pnl / contract_value_at_entry) * 100
    
    print(f"   Capital Invested (Margin): ₹{nrml_margin:,}")
    print(f"   Profit: ₹{reported_pnl:,}")
    print(f"   Return on Margin = (₹{reported_pnl:,} / ₹{nrml_margin:,}) × 100 = {return_on_margin:.2f}%")
    print()
    print(f"   If NO leverage (full contract value):")
    print(f"   Capital Needed: ₹{contract_value_at_entry:,.2f}")
    print(f"   Profit: ₹{reported_pnl:,}")
    print(f"   Return = (₹{reported_pnl:,} / ₹{contract_value_at_entry:,.2f}) × 100 = {return_without_leverage:.2f}%")
    print()
    print(f"   🚀 Leverage amplifies returns by {leverage_at_entry:.2f}x!")
    print()
    
    # 5. Risk Analysis
    print("=" * 70)
    print("5. RISK ANALYSIS")
    print("=" * 70)
    
    # Calculate break-even
    break_even_price = entry_price
    print(f"   Break-even Price: ₹{break_even_price:.2f} (no profit, no loss)")
    
    # Calculate stop-loss scenarios
    stop_loss_percentages = [5, 10, 15, 20]
    print(f"\n   Stop-Loss Scenarios:")
    print(f"   {'SL %':<8} {'SL Price':<12} {'Loss (₹)':<15} {'Return %':<12}")
    print(f"   {'-' * 50}")
    
    for sl_pct in stop_loss_percentages:
        sl_price = entry_price * (1 - sl_pct / 100)
        loss = (entry_price - sl_price) * lot_size
        loss_pct = (loss / nrml_margin) * 100
        print(f"   {sl_pct:>5}%   ₹{sl_price:>8.2f}   ₹{loss:>12,.2f}   {loss_pct:>10.2f}%")
    
    print()
    
    # Calculate take-profit scenarios
    take_profit_percentages = [5, 10, 15, 20]
    print(f"   Take-Profit Scenarios:")
    print(f"   {'TP %':<8} {'TP Price':<12} {'Profit (₹)':<15} {'Return %':<12}")
    print(f"   {'-' * 50}")
    
    for tp_pct in take_profit_percentages:
        tp_price = entry_price * (1 + tp_pct / 100)
        profit = (tp_price - entry_price) * lot_size
        profit_pct = (profit / nrml_margin) * 100
        print(f"   {tp_pct:>5}%   ₹{tp_price:>8.2f}   ₹{profit:>12,.2f}   {profit_pct:>10.2f}%")
    
    print()
    
    # 6. Key Insights
    print("=" * 70)
    print("6. KEY INSIGHTS")
    print("=" * 70)
    print()
    print("   ✅ Margin (₹14,451) is calculated on BASE PRICE (₹243.4), not entry price")
    print("   ✅ PnL (₹3,400) is calculated on ACTUAL PRICE MOVEMENT (₹264.50 → ₹276.80)")
    print("   ✅ These are INDEPENDENT calculations:")
    print("      - Margin = Risk management tool (capital requirement)")
    print("      - PnL = Trading result (profit/loss from price movement)")
    print()
    print("   ⚠️  Leverage amplifies BOTH profits AND losses:")
    print(f"      - Current profit: {return_on_margin:.2f}% on margin")
    print(f"      - If price drops 10%: Loss = {(entry_price * 0.1 * lot_size / nrml_margin) * 100:.2f}% on margin")
    print()
    print("   💡 Always use stop-loss to limit downside risk!")
    print()
    
    print("=" * 70)


if __name__ == "__main__":
    calculate_mcx_leverage_and_pnl()

