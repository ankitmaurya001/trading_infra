# Kite Comprehensive Strategy Validation

A comprehensive strategy validation pipeline specifically designed for Indian markets using Kite Connect API. This system combines parameter optimization with robust backtesting for NSE, BSE, and MCX markets.

## 🚀 Features

### Market Support
- **NSE (National Stock Exchange)**: Equity stocks with 9:15 AM - 3:30 PM IST trading hours
- **BSE (Bombay Stock Exchange)**: Equity stocks with 9:15 AM - 3:30 PM IST trading hours  
- **MCX (Multi Commodity Exchange)**: Commodities with 9:00 AM - 11:30 PM IST trading hours

### Validation Pipeline
- **Data Fetching**: Real-time historical data from Kite Connect
- **Market Hours Filtering**: Automatic filtering for Indian market hours and weekdays
- **Train-Test Split**: Proper data splitting with configurable ratios
- **Parameter Optimization**: Grid search optimization on training data
- **Mock Trading**: Realistic backtesting on test data
- **Top N Validation**: Test multiple parameter sets for robustness
- **Performance Metrics**: Comprehensive risk-adjusted metrics

### Strategies Supported
- **Moving Average Crossover**: Short/long window optimization
- **RSI Strategy**: Period, overbought/oversold level optimization
- **Donchian Channel**: Channel period optimization

## 📋 Prerequisites

1. **Kite Connect API Access**: Active Kite Connect account with API access
2. **Credentials**: Update `config.py` with your Kite credentials
3. **Python Dependencies**: All required packages installed
4. **Market Hours**: Run during market hours for live data (optional for historical data)

## 🔧 Setup

### 1. Update Credentials
Edit `config.py` with your Kite Connect credentials:

```python
KITE_CREDENTIALS = {
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "access_token": "your_access_token"  # Optional, can be generated
}

KITE_EXCHANGE = "NSE"  # or "BSE", "MCX"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Quick Start

#### NSE Stock Validation
```bash
# Full validation for NSE stocks
python run_kite_comprehensive_validation.py

# Quick validation (7 days, MA strategy only)
python run_kite_comprehensive_validation.py --quick
```

#### MCX Commodity Validation
```bash
# MCX commodity validation
python run_kite_comprehensive_validation.py --mcx
```

### Advanced Usage

#### Custom Validation
```python
from kite_comprehensive_strategy_validation import create_kite_validator_for_exchange

# Create NSE validator
validator = create_kite_validator_for_exchange("NSE")

# Fetch and split data
train_data, test_data = validator.fetch_and_split_data(
    symbol="TATAMOTORS",
    start_date="2024-01-01",
    end_date="2024-12-31",
    interval="15minute",
    train_ratio=0.7
)

# Optimize strategies
optimization_results = validator.optimize_strategies_on_train_data(
    strategies_to_optimize=['ma', 'rsi', 'donchian']
)

# Validate on test data
validation_results = validator.validate_strategies_on_test_data(
    strategies_to_validate=['ma', 'rsi', 'donchian'],
    top_n_params=5
)

# Compare and get recommendations
recommendations = validator.compare_strategies()
live_setup = validator.get_live_trading_setup()

# Save results
validator.save_results("my_validation_results")
```

## 📊 Configuration Options

### Exchange-Specific Settings

#### NSE/BSE (Equity Markets)
```python
validator = create_kite_validator_for_exchange("NSE", 
    max_leverage=10.0,      # Higher leverage for equities
    max_loss_percent=2.0,   # 2% max loss per trade
    trading_fee=0.0003      # 0.03% Kite charges
)
```

#### MCX (Commodity Markets)
```python
validator = create_kite_validator_for_exchange("MCX",
    max_leverage=5.0,       # Lower leverage for commodities
    max_loss_percent=1.5,   # Stricter risk management
    trading_fee=0.0003      # 0.03% Kite charges
)
```

### Data Parameters
- **Symbol**: Exchange-specific symbol (e.g., "TATAMOTORS", "RELIANCE", "COPPER25OCTFUT")
- **Interval**: "15minute", "1hour", "1day"
- **Train Ratio**: 0.7 (70% training, 30% testing)
- **Random Shift**: 1.2 (20% overlap for robustness)

## 📈 Output and Results

### Generated Files
```
kite_validation_results/
├── optimization_results_YYYYMMDD_HHMMSS.json
├── validation_results_YYYYMMDD_HHMMSS.json
├── recommendations_YYYYMMDD_HHMMSS.json
├── train_data_YYYYMMDD_HHMMSS.csv
├── test_data_YYYYMMDD_HHMMSS.csv
└── kite_market_info.json
```

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Calmar Ratio**: Return vs max drawdown
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Maximum peak-to-trough decline
- **Total PnL**: Overall profit/loss percentage
- **Total Trades**: Number of trades executed

### Strategy Comparison
```
Strategy        Param Set  Sharpe   Calmar   Win Rate   Max DD   Total PnL  Trades  
----------------------------------------------------------------------------------------------------
ma              Set 1      0.279    636735.000 41.18%     9.43%    49.64%     17      
ma              Set 2      0.629    1526476662.351 58.33%     6.35%    496.39%    12      
ma              Set 3      0.420    232129957.881 46.67%     7.55%    441.32%    15      
```

## 🎯 Examples

### Example 1: NSE Stock Validation
```bash
python kite_validation_example.py --nse
```

### Example 2: MCX Commodity Validation
```bash
python kite_validation_example.py --mcx
```

### Example 3: Custom Validation
```bash
python kite_validation_example.py --custom
```

## 🔍 Troubleshooting

### Common Issues

#### Authentication Errors
```
❌ Authentication failed: Invalid credentials
```
**Solution**: Check your Kite Connect credentials in `config.py`

#### No Data Fetched
```
❌ No data fetched for TATAMOTORS
```
**Solutions**:
- Verify symbol is correct (e.g., "TATAMOTORS" not "TATAMOTORS.NS")
- Check if market is open (9:15 AM - 3:30 PM IST for NSE/BSE)
- Ensure date range has trading days

#### Market Hours Filtering
```
❌ No data remaining after filtering for market hours
```
**Solutions**:
- Check if date range includes weekdays
- Verify market hours configuration
- For MCX, ensure extended hours are considered

### Debug Mode
Enable debug logging for detailed information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Integration with Live Trading

### Using Validation Results
```python
# Get the best strategy setup
live_setup = validator.get_live_trading_setup()

# Use with KiteTradingEngine
from kite_trading_engine import KiteTradingEngine

engine = KiteTradingEngine("kite_trading_config.json")
engine.config.update({
    "enabled_strategies": [live_setup['strategy_key']],
    f"{live_setup['strategy_key']}_params": live_setup['parameters']
})
```

### Configuration File
```json
{
    "symbol": "TATAMOTORS",
    "interval": "15minute",
    "enabled_strategies": ["ma"],
    "ma_params": {
        "short_window": 5,
        "long_window": 60,
        "risk_reward_ratio": 3.5,
        "trading_fee": 0.0003
    }
}
```

## 🎯 Best Practices

### 1. Data Quality
- Use sufficient historical data (30+ days minimum)
- Ensure data includes various market conditions
- Filter for market hours only

### 2. Parameter Optimization
- Test multiple parameter ranges
- Use top N validation for robustness
- Consider market-specific parameters

### 3. Risk Management
- Set appropriate max loss percentages
- Use conservative leverage for commodities
- Monitor drawdowns closely

### 4. Validation
- Always validate on out-of-sample data
- Test multiple parameter sets
- Consider robustness over pure performance

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Kite Connect API documentation
3. Verify market hours and trading days
4. Test with sample data first

## 🔄 Updates

The system automatically handles:
- Market hours filtering
- Timezone conversions (IST)
- Weekday-only trading
- Exchange-specific configurations
- Real-time data fetching during market hours

---

**Happy Trading! 📈**
