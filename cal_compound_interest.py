import matplotlib.pyplot as plt

# Inputs
principal = float(input("Enter principal amount: "))
rate = float(input("Enter annual interest rate (%): "))
time = int(input("Enter time (in years): "))
n = int(input("Enter number of times interest is compounded per year: "))

# Convert rate
r = rate / 100

amounts = []
pnls = []
periods = []

current_amount = principal

total_periods = n * time

for i in range(1, total_periods + 1):
    current_amount = current_amount * (1 + r / n)
    pnl = current_amount - principal

    periods.append(i)
    amounts.append(current_amount)
    pnls.append(pnl)

# Final results
print(f"\nFinal Amount: ₹{current_amount:.2f}")
print(f"Total Profit (PnL): ₹{pnls[-1]:.2f}")

# Plot PnL
plt.figure()
plt.plot(periods, pnls)
plt.xlabel("Compounding Period")
plt.ylabel("Profit (PnL)")
plt.title("Compound Interest Growth (PnL Over Time)")
plt.grid(True)
plt.show()
