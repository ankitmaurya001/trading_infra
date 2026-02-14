"""
Detailed explanation of the correct PnL calculation method
"""

def explain_pnl_calculation(trade_pnls, initial_capital=100):
    """
    Explain the PnL calculation step by step
    """
    print(f"Starting capital: ${initial_capital}")
    print(f"Trade percentages: {[f'{p*100:.1f}%' for p in trade_pnls]}")
    print("\n" + "="*50)
    
    current_capital = initial_capital
    total_multiplier = 1.0
    
    print(f"{'Trade':<6} {'Pct':<8} {'Multiplier':<12} {'Capital':<12} {'Explanation'}")
    print("-" * 60)
    
    for i, pnl in enumerate(trade_pnls, 1):
        multiplier = 1 + pnl
        new_capital = current_capital * multiplier
        total_multiplier *= multiplier
        
        print(f"{i:<6} {pnl*100:>6.1f}% {multiplier:>10.3f} ${new_capital:>10.2f} ", end="")
        
        if pnl > 0:
            print(f"${current_capital:.2f} × {multiplier:.3f} = ${new_capital:.2f}")
        else:
            print(f"${current_capital:.2f} × {multiplier:.3f} = ${new_capital:.2f}")
        
        current_capital = new_capital
    
    print("-" * 60)
    final_pnl = total_multiplier - 1.0
    final_capital = initial_capital * total_multiplier
    
    print(f"\nFinal Results:")
    print(f"Total multiplier: {total_multiplier:.4f}")
    print(f"Final capital: ${final_capital:.2f}")
    print(f"Total PnL: {final_pnl:.4f} ({final_pnl*100:.2f}%)")
    
    # Compare with incorrect method
    incorrect_pnl = sum(trade_pnls)
    print(f"\nComparison:")
    print(f"Incorrect method (sum): {incorrect_pnl:.4f} ({incorrect_pnl*100:.2f}%)")
    print(f"Correct method (compound): {final_pnl:.4f} ({final_pnl*100:.2f}%)")
    print(f"Difference: {abs(final_pnl - incorrect_pnl):.4f} ({abs(final_pnl - incorrect_pnl)*100:.2f}%)")
    
    return final_pnl

def demonstrate_compounding_effect():
    """Demonstrate why compounding matters"""
    print("=== COMPOUNDING EFFECT DEMONSTRATION ===\n")
    
    # Example 1: Consistent gains
    print("Example 1: Two 10% gains")
    print("Incorrect thinking: 10% + 10% = 20%")
    print("Reality: 1.10 × 1.10 = 1.21 = 21%")
    explain_pnl_calculation([0.10, 0.10])
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Loss and recovery
    print("Example 2: 50% loss followed by 50% gain")
    print("Incorrect thinking: -50% + 50% = 0% (break even)")
    print("Reality: 0.50 × 1.50 = 0.75 = -25% (still down!)")
    explain_pnl_calculation([-0.50, 0.50])
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Volatile trading
    print("Example 3: Volatile trading sequence")
    print("Trades: +5%, -2%, +8%, -3%, +12%")
    explain_pnl_calculation([0.05, -0.02, 0.08, -0.03, 0.12])

def show_mathematical_formula():
    """Show the mathematical formula"""
    print("\n=== MATHEMATICAL FORMULA ===")
    print("For trades with returns r₁, r₂, r₃, ..., rₙ:")
    print("Total Return = (1 + r₁) × (1 + r₂) × (1 + r₃) × ... × (1 + rₙ) - 1")
    print("\nIn Python:")
    print("total_multiplier = 1.0")
    print("for pnl in trade_pnls:")
    print("    total_multiplier *= (1 + pnl)")
    print("total_return = total_multiplier - 1.0")
    
    print("\nWhy this works:")
    print("• Each (1 + pnl) represents the multiplier for that trade")
    print("• Multiplying them together gives the cumulative effect")
    print("• Subtracting 1 converts back to percentage form")

if __name__ == "__main__":
    print("PNL CALCULATION EXPLANATION")
    print("=" * 50)
    
    demonstrate_compounding_effect()
    show_mathematical_formula()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS:")
    print("1. Percentages compound, they don't add")
    print("2. A 50% loss requires a 100% gain to break even")
    print("3. Volatility hurts returns (geometric mean < arithmetic mean)")
    print("4. The correct method accounts for the order of trades")
    print("5. Small losses have a bigger impact than small gains") 