# 🏥 Strategy Health Monitor

A quantitative framework for **Strategy Lifecycle Management** - knowing when to stop trading a strategy and when to re-optimize parameters.

## 📋 Table of Contents

- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [7 Mathematical Signals](#7-mathematical-signals)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Interpreting Results](#interpreting-results)
- [Integration Examples](#integration-examples)
- [Best Practices](#best-practices)

---

## The Problem

You run a backtest with parameters (Short=4, Long=58, RR=6.0) and see:

```
Cumulative PnL: +10% → +16% → -5% → +13% → -17%
```

**Questions that arise:**
1. When should I have stopped trading these parameters?
2. How do I know the strategy has "broken" vs normal drawdown?
3. When should I re-run optimization to find new parameters?

**This module answers these questions mathematically.**

---

## The Solution

The `StrategyHealthMonitor` analyzes your trade history and provides:

| Output | Description |
|--------|-------------|
| **Health Score** | 0-100 score (100 = perfect, 0 = terrible) |
| **Status** | HEALTHY → WARNING → CRITICAL → STOPPED |
| **Should Stop** | Boolean - stop trading immediately? |
| **Should Re-optimize** | Boolean - run parameter optimization? |
| **Recommendation** | Human-readable action to take |

---

## 7 Mathematical Signals

### 1️⃣ Maximum Drawdown Circuit Breaker

The simplest and most important signal - if you've lost too much from peak, stop.

**Formula:**
```
Drawdown = (Peak Equity - Current Equity) / Peak Equity
```

**Thresholds:**
| Drawdown | Status | Action |
|----------|--------|--------|
| < 10% | ✅ Healthy | Continue |
| 10-15% | ⚠️ Warning | Monitor closely |
| 15-20% | 🚨 Critical | Consider stopping |
| > 20% | 🛑 Stopped | Stop immediately |

---

### 2️⃣ Rolling Performance Metrics

Analyzes the last N trades (default: 20) for key metrics:

**Win Rate:**
```
Win Rate = Winning Trades / Total Trades
```
- Critical if < 35%

**Rolling Sharpe Ratio:**
```
Sharpe = (Mean Return / Std Dev) × √252
```
- Critical if < 0.3 (or negative)

**Profit Factor:**
```
Profit Factor = Sum of Wins / |Sum of Losses|
```
- Critical if < 0.8 (losing money)

---

### 3️⃣ Equity Curve Moving Average (Meta-Strategy)

**Concept:** Apply a trend filter to your own performance. Trade only when your equity is above its moving average.

```
Signal = Current Equity vs MA(10) of Equity
```

| Condition | Status |
|-----------|--------|
| Equity > MA | ✅ Healthy trend |
| Equity < MA | ⚠️ Warning |
| Equity < MA by 5%+ | 🚨 STOP SIGNAL |

**Why it works:** If your equity is consistently below its MA, you're in a performance downtrend - the parameters no longer fit the market.

---

### 4️⃣ CUSUM Change Detection

**Statistical Process Control** technique from manufacturing - detects when cumulative deviations become statistically significant.

**Formula:**
```
S_t = max(0, S_{t-1} + z_t)

where z_t = (x_t - μ_baseline) / σ_baseline
```

**Interpretation:**
- Compares recent performance vs. first half of trades (baseline)
- Large negative CUSUM = performance has significantly degraded
- Threshold: |CUSUM| > 4 standard deviations

---

### 5️⃣ Time Under Water Analysis

Tracks how long the strategy has been in drawdown (consecutive trades below peak equity).

| Trades Underwater | Status |
|-------------------|--------|
| < 8 | ✅ Normal fluctuation |
| 8-15 | ⚠️ Extended drawdown |
| > 15 | 🚨 Critical - likely regime change |

**Extended time underwater suggests the market regime has changed.**

---

### 6️⃣ Regime Change Detection

Statistical test comparing recent performance distribution to historical.

**Formula (Z-test):**
```
z = (x̄_recent - x̄_historical) / (σ / √n)
```

| Z-Score | Interpretation |
|---------|----------------|
| z > -1 | No change detected |
| -2 < z < -1 | Possible degradation |
| z < -2 | Regime change detected (95% confidence) |

---

### 7️⃣ Performance Trend Analysis

Linear regression on cumulative P&L to detect downward trends.

```
Slope = Linear regression coefficient of cumulative returns
```

| Normalized Slope | Status |
|------------------|--------|
| > -0.2 | ✅ Acceptable |
| -0.5 to -0.2 | ⚠️ Declining |
| < -0.5 | 🚨 Strong downtrend |

---

## Quick Start

### Basic Usage

```python
from strategy_health_monitor import StrategyHealthMonitor

# Create monitor
monitor = StrategyHealthMonitor()

# Analyze trade history (DataFrame with 'pnl', 'status', 'exit_time' columns)
report = monitor.analyze(trade_history_df, initial_balance=10000)

# Print full report
print(report)

# Check actions
if report.should_stop:
    print("STOP TRADING!")
if report.should_reoptimize:
    print("Run re-optimization!")
```

### Quick Functions

```python
from strategy_health_monitor import (
    should_reoptimize, 
    should_stop_trading, 
    get_health_score
)

# One-liner checks
stop, reason = should_stop_trading(trade_history_df)
reopt, reason = should_reoptimize(trade_history_df)
score = get_health_score(trade_history_df)  # 0-100

print(f"Health Score: {score}/100")
print(f"Stop Trading: {stop}")
print(f"Re-optimize: {reopt}")
```

### Generate Visual Report

```python
from strategy_health_monitor import create_health_monitor_chart

create_health_monitor_chart(
    trade_history=trade_history_df,
    initial_balance=10000,
    output_path="health_report.html"
)
```

---

## Configuration

Customize thresholds based on your risk tolerance:

```python
monitor = StrategyHealthMonitor(
    # Drawdown thresholds
    max_drawdown_warning=0.10,      # 10% = warning
    max_drawdown_critical=0.15,     # 15% = critical
    max_drawdown_stop=0.20,         # 20% = stop
    
    # Rolling metrics (last N trades)
    min_rolling_sharpe=0.3,         # Minimum Sharpe
    min_rolling_win_rate=0.35,      # Minimum win rate
    min_rolling_profit_factor=0.8,  # Minimum profit factor
    rolling_window=20,              # Trades to analyze
    
    # Equity curve MA
    equity_ma_period=10,            # MA period
    
    # Time under water
    max_underwater_trades=15,       # Max consecutive trades in DD
    
    # CUSUM sensitivity
    cusum_threshold=4.0,            # Std devs for alert
    
    # Regime change
    regime_change_threshold=2.0,    # Z-score threshold
)
```

### Aggressive Settings (Stop Earlier)

```python
monitor = StrategyHealthMonitor(
    max_drawdown_stop=0.10,         # Stop at 10% DD
    min_rolling_win_rate=0.40,      # Higher win rate required
    max_underwater_trades=10,       # Less patience
)
```

### Conservative Settings (More Patience)

```python
monitor = StrategyHealthMonitor(
    max_drawdown_stop=0.30,         # Allow 30% DD
    min_rolling_win_rate=0.25,      # Lower win rate OK
    max_underwater_trades=25,       # More patience
)
```

---

## Interpreting Results

### Sample Output

```
============================================================
📊 STRATEGY HEALTH REPORT
============================================================
Overall Status: 🚨 CRITICAL
Confidence: 75.0%

📋 Individual Signals:
  ⚠️ Maximum Drawdown: 0.1694 (threshold: 0.1500)
     → Drawdown (16.9%) at critical level - consider stopping
  🚨 Rolling Win Rate: 0.1579 (threshold: 0.3500)
     → Rolling win rate (15.8%) below minimum (35.0%)
  🚨 Equity Curve MA: -0.0823 (threshold: -0.0500)
     → Equity -8.2% below MA, 6 periods below - STOP SIGNAL
  ✅ Time Under Water: 8.0000 (threshold: 15.0000)
     → Underwater for 8 trades - acceptable
  🚨 CUSUM Change Detection: -5.2341 (threshold: -4.0000)
     → CUSUM detected significant performance drop
  ⚠️ Regime Change: -1.8234 (threshold: -2.0000)
     → Possible regime shift - recent underperformance

🎯 Recommendation: 🚨 STOP TRADING - Multiple critical signals. 
   Run re-optimization to find new parameters.
   Stop Trading: Yes
   Re-optimize: Yes
============================================================
```

### Status Decision Matrix

| Signals | Overall Status | Action |
|---------|----------------|--------|
| Any STOPPED | 🛑 STOPPED | Stop immediately |
| 2+ CRITICAL | 🚨 CRITICAL | Stop + Re-optimize |
| 1 CRITICAL | 🚨 CRITICAL | Re-optimize |
| 3+ WARNING | ⚠️ WARNING | Schedule re-optimization |
| 1-2 WARNING | ⚠️ WARNING | Monitor closely |
| All HEALTHY | ✅ HEALTHY | Continue trading |

---

## Integration Examples

### With Mock Validation Script

The `run_ma_mock_validation_kite.py` script automatically includes health monitoring:

```bash
python run_ma_mock_validation_kite.py
```

Output includes:
```
================================================================================
🏥 STRATEGY HEALTH MONITORING
================================================================================

📊 Analyzing health for param set 1:
   Parameters: {'short_window': 4, 'long_window': 58, 'risk_reward_ratio': 6.0}

[Full health report...]

📋 SUMMARY RECOMMENDATIONS
================================================================================

🚨 Param Set 1 (S=4, L=58, RR=6.0):
   Health Score: 45.2/100
   Status: CRITICAL
   ⛔ ACTION: STOP TRADING with these parameters
   🔄 ACTION: Run re-optimization to find new parameters
```

### In Live Trading Loop

```python
from strategy_health_monitor import StrategyHealthMonitor

monitor = StrategyHealthMonitor()

# After each trade closes
def on_trade_closed(trade_history):
    report = monitor.analyze(trade_history)
    
    if report.should_stop:
        # Immediately close all positions
        close_all_positions()
        send_alert("Strategy stopped - health critical")
        
    elif report.should_reoptimize:
        # Continue trading but schedule re-optimization
        schedule_reoptimization()
        send_alert("Re-optimization needed")
```

### Periodic Health Check

```python
import schedule

def daily_health_check():
    trades = load_recent_trades(days=30)
    score = get_health_score(trades)
    
    if score < 50:
        send_telegram_alert(f"⚠️ Strategy health low: {score}/100")
    
schedule.every().day.at("18:00").do(daily_health_check)
```

---

## Best Practices

### 1. Don't Over-Optimize Thresholds

The default thresholds are based on common industry practices. Resist the urge to fine-tune them based on past data - that's curve fitting.

### 2. Use Multiple Signals

Don't rely on a single metric. The power of this system is in combining multiple independent signals.

### 3. Act on CRITICAL, Monitor on WARNING

- **CRITICAL signals** = Take action (stop or re-optimize)
- **WARNING signals** = Increase monitoring, prepare for action

### 4. Re-optimization Frequency

| Market Condition | Re-optimization Frequency |
|------------------|---------------------------|
| Stable/Trending | Monthly |
| Volatile | Weekly |
| After Critical Signal | Immediately |

### 5. Keep Backup Parameters

Always have 2-3 sets of validated parameters ready to switch to if current set fails.

### 6. Log Everything

Keep records of:
- When health signals triggered
- What action was taken
- Outcome of that action

This builds institutional knowledge for future improvements.

---

## Mathematical Reference

### Formulas Summary

| Metric | Formula |
|--------|---------|
| Drawdown | `(Peak - Current) / Peak` |
| Win Rate | `Wins / Total Trades` |
| Sharpe Ratio | `(Mean - Rf) / Std × √252` |
| Profit Factor | `Σ Wins / |Σ Losses|` |
| CUSUM | `S_t = max(0, S_{t-1} + z_t)` |
| Regime Z-Score | `(x̄_recent - x̄_hist) / (σ/√n)` |

### Statistical Basis

- **CUSUM**: Based on Page's cumulative sum test (1954)
- **Regime Detection**: Standard two-sample z-test
- **Drawdown**: Industry standard risk metric
- **Equity MA**: Popularized by van Tharp and Kaufman

---

## Files

| File | Description |
|------|-------------|
| `strategy_health_monitor.py` | Main module with all classes and functions |
| `run_ma_mock_validation_kite.py` | Example integration with backtesting |

---

## FAQ

**Q: How many trades do I need for reliable signals?**
A: Minimum 10 trades, ideally 30+ for statistical significance.

**Q: Can I use this for any strategy?**
A: Yes! It works on any strategy that produces a trade history with P&L.

**Q: Should I trust CRITICAL signals absolutely?**
A: CRITICAL signals are strong indicators but use judgment. Check if there's an obvious external cause (news event, exchange issue) before taking action.

**Q: How do I reduce false positives?**
A: Increase thresholds (more conservative) or require more signals to trigger action.

---

## Author

Trading Infrastructure Team

## License

MIT License - Use freely in your trading systems.

