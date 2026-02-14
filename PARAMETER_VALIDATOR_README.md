# 🔍 Parameter Validator

**Proactive Parameter Drift Detection** - Validates that your current strategy parameters are still optimal by running optimization on recent data.

## 🎯 The Problem It Solves

Traditional health monitoring is **reactive** - it detects problems after losses occur. Parameter validation is **proactive** - it detects when parameters have drifted from optimal **before** performance degrades.

### Example Scenario

```
Day 1-30: Optimize → Best params: Short=4, Long=58, RR=6.0
Day 31-60: Trade with these params, PnL = +8%
Day 61: Run parameter validation
  - New optimal: Short=8, Long=45, RR=4.5
  - Distance = 13.7 (above threshold)
  - Performance gap: New params would give +12% vs current +8%
  → ALERT: Parameters have drifted, re-optimize!

Day 62: Re-optimize on recent data → Get new params
Day 63+: Trade with new params
```

This catches the issue **early**, before the +8% turns into -17%.

---

## 🚀 Quick Start

### Basic Usage

```python
from parameter_validator import validate_ma_parameters

# Your current parameters
current_params = {
    'short_window': 4,
    'long_window': 58,
    'risk_reward_ratio': 6.0
}

# Validate (runs optimization on last 30 days)
result = validate_ma_parameters(
    current_params=current_params,
    symbol='SILVERMIC26FEBFUT',
    interval='15m',
    exchange='MCX',
    validation_frequency_days=7,  # Weekly
    data_window_days=30           # 30 days of recent data
)

# Check results
if result.should_reoptimize:
    print(f"⚠️ {result.alert_message}")
    print(f"New optimal: {result.new_optimal_params}")
else:
    print("✅ Parameters still optimal")
```

### Command Line

```bash
python run_parameter_validation.py \
    --symbol SILVERMIC26FEBFUT \
    --exchange MCX \
    --interval 15m \
    --short 4 \
    --long 58 \
    --rr 6.0
```

---

## 📊 How It Works

### 1. Fetch Recent Data
- Fetches last N days of data (default: 30 days)
- Uses same data source as your trading (Kite Connect)

### 2. Run Full Grid Search
- Runs complete optimization on recent data
- Tests all parameter combinations
- Finds new optimal parameters

### 3. Compare Parameters
- Calculates **Euclidean distance** between current and new optimal
- Normalizes parameters to 0-1 range for fair comparison
- Calculates **performance gap** (how much better new params would be)

### 4. Generate Alert
- **None**: Parameters still optimal (distance < 3)
- **Monitor**: Some drift (distance 3-7)
- **Warning**: Significant drift (distance 7-12)
- **Critical**: Major drift (distance > 12)

---

## ⚙️ Configuration

### Frequency & Data Window

```python
validator = ParameterValidator(
    validation_frequency_days=7,   # Run weekly
    data_window_days=30             # Use last 30 days
)
```

### Distance Thresholds

```python
validator = ParameterValidator(
    distance_threshold_monitor=3.0,    # Monitor if distance > 3
    distance_threshold_warning=7.0,   # Warning if distance > 7
    distance_threshold_critical=12.0, # Critical if distance > 12
)
```

### Performance Gap Threshold

```python
validator = ParameterValidator(
    performance_gap_threshold=0.05  # Alert if new params 5% better
)
```

---

## 📋 Validation Result

The `ValidationResult` object contains:

| Field | Description |
|-------|-------------|
| `current_params` | Your current parameters |
| `new_optimal_params` | New optimal parameters from recent data |
| `parameter_distance` | Euclidean distance (0 = identical, higher = more different) |
| `performance_gap` | How much better new params would be (as %) |
| `should_reoptimize` | Boolean - should you re-optimize? |
| `alert_level` | 'none', 'monitor', 'warning', 'critical' |
| `alert_message` | Human-readable alert message |
| `current_params_performance` | Performance metrics for current params |
| `new_optimal_performance` | Performance metrics for new optimal params |
| `stability_score` | How stable optimal params are (0-1) |

---

## 🔄 Integration with Health Monitor

Parameter validation **complements** health monitoring:

| System | When It Triggers | What It Detects |
|--------|------------------|-----------------|
| **Health Monitor** | After losses occur | Performance degradation |
| **Parameter Validator** | Before losses occur | Parameter drift |

**Best Practice**: Use both together:
1. **Weekly**: Run parameter validation (proactive)
2. **Daily**: Check health monitor (reactive backup)

---

## 📅 Scheduling

### Manual (Weekly)

```bash
# Add to cron (runs every Monday at 9 AM)
0 9 * * 1 cd /path/to/trading_infra && python run_parameter_validation.py --symbol SILVERMIC26FEBFUT --exchange MCX --interval 15m --short 4 --long 58 --rr 6.0
```

### Python Script

```python
import schedule
from parameter_validator import validate_ma_parameters

def weekly_validation():
    result = validate_ma_parameters(
        current_params={'short_window': 4, 'long_window': 58, 'risk_reward_ratio': 6.0},
        symbol='SILVERMIC26FEBFUT',
        interval='15m',
        exchange='MCX'
    )
    
    if result.should_reoptimize:
        send_alert(result.alert_message)

# Run every Monday at 9 AM
schedule.every().monday.at("09:00").do(weekly_validation)
```

---

## 🎯 Decision Matrix

| Distance | Performance Gap | Action |
|----------|----------------|--------|
| < 3 | < 2% | ✅ Continue - params still optimal |
| 3-7 | 2-5% | 📊 Monitor - some drift detected |
| 7-12 | 5-10% | ⚠️ Warning - schedule re-optimization |
| > 12 | > 10% | 🚨 Critical - re-optimize immediately |

---

## 💡 Best Practices

### 1. Run Weekly
- Weekly validation catches drift early
- Daily is too frequent (noise)
- Monthly is too infrequent (missed opportunities)

### 2. Use 30-Day Window
- 30 days balances recency vs. statistical significance
- Too short (< 15 days) = noisy results
- Too long (> 60 days) = includes outdated market conditions

### 3. Don't Auto-Switch
- Always **alert user** instead of auto-switching
- User should review and decide
- May want to wait for confirmation trade

### 4. Track Validation History
- Save all validation results
- Track how often parameters drift
- Identify patterns (e.g., drift more in volatile periods)

### 5. Combine with Health Monitor
- Parameter validation = proactive
- Health monitor = reactive backup
- Use both for comprehensive monitoring

---

## 🔧 Advanced Usage

### Custom Parameter Ranges

```python
result = validator.validate_parameters(
    current_params=current_params,
    symbol='SILVERMIC26FEBFUT',
    interval='15m',
    short_window_range=list(range(4, 15)),  # Custom range
    long_window_range=list(range(40, 70)),  # Custom range
    risk_reward_range=[4.0, 4.5, 5.0, 5.5, 6.0]  # Custom range
)
```

### Save Results

```python
validator.save_validation_result(
    result,
    output_dir="validation_results"
)
```

### Check If Should Run

```python
last_validation = datetime(2025, 1, 1)
if validator.should_run_validation(last_validation):
    result = validator.validate_parameters(...)
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `parameter_validator.py` | Main module with ParameterValidator class |
| `run_parameter_validation.py` | Command-line script to run validation |
| `validation_results/` | Directory where results are saved (JSON) |

---

## 🐛 Troubleshooting

### "Kite data fetcher not initialized"
- Check `config.py` has valid Kite credentials
- Ensure Kite Connect is authenticated

### "No data fetched"
- Check symbol name is correct
- Verify exchange is correct (NSE, BSE, MCX)
- Ensure date range has data

### "Optimization failed"
- May need more data (increase `data_window_days`)
- Check parameter ranges are valid
- Ensure short_window < long_window

---

## 📚 Example Output

```
================================================================================
🔍 PARAMETER VALIDATION
================================================================================
Current params: {'short_window': 4, 'long_window': 58, 'risk_reward_ratio': 6.0}
Symbol: SILVERMIC26FEBFUT, Exchange: MCX, Interval: 15m
Data window: 30 days
📥 Fetching validation data: SILVERMIC26FEBFUT from 2025-01-01 to 2025-01-31
✅ Fetched 1248 data points
🔍 Running optimization on 1248 data points...
   Parameter ranges: 17 short × 61 long × 11 RR
✅ Optimization complete
   Best params: {'short_window': 8, 'long_window': 45, 'risk_reward_ratio': 4.5}
   Best Sharpe: 1.234
   Best PnL: 12.34%
📊 Evaluating current parameters...
================================================================================
📋 VALIDATION RESULTS
================================================================================
⚠️ WARNING: Parameters have drifted
   Distance: 8.45 (threshold: 7.0)
   Performance gap: 6.78%
   Current: Short=4, Long=58, RR=6.0
   New optimal: Short=8, Long=45, RR=4.5
   💡 RECOMMENDATION: Schedule re-optimization soon
================================================================================
```

---

## 🎓 Key Concepts

### Parameter Distance
- Euclidean distance in normalized parameter space
- 0 = identical parameters
- Higher = more different
- Normalized so all parameters weighted equally

### Performance Gap
- Difference in expected returns between current and new optimal
- Positive = new params would be better
- Negative = current params are still better (rare)

### Stability Score
- Measures how clustered top N optimal parameters are
- High stability (0.8+) = market conditions stable
- Low stability (< 0.5) = market conditions changing rapidly

---

## 📝 License

MIT License - Use freely in your trading systems.

