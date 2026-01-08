#!/usr/bin/env python3
"""
Parameter Validator - Continuous Parameter Drift Detection

This module validates that current strategy parameters are still optimal by:
1. Running optimization on recent data
2. Comparing current params vs new optimal params
3. Detecting parameter drift
4. Alerting user when re-optimization is needed

Usage:
    validator = ParameterValidator(
        validation_frequency_days=7,  # Weekly
        data_window_days=30
    )
    
    result = validator.validate_parameters(
        current_params={'short_window': 4, 'long_window': 58, 'risk_reward_ratio': 6.0},
        symbol='SILVERMIC26FEBFUT',
        exchange='MCX',
        interval='15m'
    )
    
    if result['should_reoptimize']:
        print(f"⚠️ Parameters have drifted! {result['alert_message']}")
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import json
from dataclasses import dataclass, asdict

from strategy_optimizer import StrategyOptimizer
from strategies import MovingAverageCrossover
from data_fetcher import KiteDataFetcher
from ma_3d_optimization_visualizer import MAOptimization3DVisualizer
import config as cfg

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of parameter validation"""
    validation_date: str
    current_params: Dict[str, Any]
    new_optimal_params: Dict[str, Any]
    parameter_distance: float
    performance_gap: float
    should_reoptimize: bool
    alert_level: str  # 'none', 'monitor', 'warning', 'critical'
    alert_message: str
    current_params_performance: Dict[str, float]
    new_optimal_performance: Dict[str, float]
    stability_score: float  # 0-1, how stable are optimal params
    validation_data_period: str
    
    def to_dict(self):
        return asdict(self)


class ParameterValidator:
    """
    Validates that current strategy parameters are still optimal.
    
    Runs periodic optimization on recent data and compares results to current parameters.
    """
    
    def __init__(
        self,
        validation_frequency_days: int = 7,  # Weekly by default
        data_window_days: int = 30,  # 30 days of recent data
        distance_threshold_monitor: float = 3.0,  # Monitor if distance > 3
        distance_threshold_warning: float = 7.0,  # Warning if distance > 7
        distance_threshold_critical: float = 12.0,  # Critical if distance > 12
        performance_gap_threshold: float = 0.1,  # 10% performance gap triggers alert
        trading_fee: float = 0.0,
        exchange: str = "MCX"
    ):
        """
        Initialize parameter validator.
        
        Args:
            validation_frequency_days: How often to run validation (default: 7 = weekly)
            data_window_days: How much recent data to use for validation (default: 30)
            distance_threshold_monitor: Parameter distance threshold for monitoring
            distance_threshold_warning: Parameter distance threshold for warning
            distance_threshold_critical: Parameter distance threshold for critical alert
            performance_gap_threshold: Performance gap % to trigger alert
            trading_fee: Trading fee for optimization
            exchange: Exchange name (NSE, BSE, MCX)
        """
        self.validation_frequency_days = validation_frequency_days
        self.data_window_days = data_window_days
        self.distance_threshold_monitor = distance_threshold_monitor
        self.distance_threshold_warning = distance_threshold_warning
        self.distance_threshold_critical = distance_threshold_critical
        self.performance_gap_threshold = performance_gap_threshold
        self.trading_fee = trading_fee
        self.exchange = exchange
        
        # Store optimization ranges for distance calculation (set during run_optimization)
        self._last_short_range = None
        self._last_long_range = None
        self._last_rr_range = None
        
        # Initialize Kite data fetcher
        self.kite_fetcher = None
        self._initialize_kite_fetcher()
    
    def _initialize_kite_fetcher(self):
        """Initialize Kite data fetcher"""
        try:
            self.kite_fetcher = KiteDataFetcher(
                credentials=cfg.KITE_CREDENTIALS,
                exchange=self.exchange
            )
            self.kite_fetcher.authenticate()
            logger.info("✅ Kite data fetcher initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Kite fetcher: {e}")
            self.kite_fetcher = None
    
    def map_interval_to_kite(self, interval: str) -> str:
        """Map common interval formats to Kite Connect format"""
        interval_mapping = {
            '1m': 'minute',
            '3m': '3minute',
            '5m': '5minute',
            '15m': '15minute',
            '30m': '30minute',
            '1h': 'hour',
            '2h': '2hour',
            '4h': '4hour',
            '1d': 'day',
            '1w': 'week',
            '1M': 'month'
        }
        return interval_mapping.get(interval, '15minute')
    
    def fetch_recent_data(
        self,
        symbol: str,
        interval: str,
        days: int = None
    ) -> pd.DataFrame:
        """
        Fetch recent data for validation.
        
        Args:
            symbol: Trading symbol
            interval: Data interval (e.g., '15m')
            days: Number of days to fetch (uses self.data_window_days if None)
            
        Returns:
            DataFrame with OHLCV data
        """
        if days is None:
            days = self.data_window_days
        
        if self.kite_fetcher is None:
            raise ValueError("Kite data fetcher not initialized. Cannot fetch data.")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        kite_interval = self.map_interval_to_kite(interval)
        
        logger.info(f"📥 Fetching validation data: {symbol} from {start_date_str} to {end_date_str}")
        
        data = self.kite_fetcher.fetch_historical_data(
            symbol=symbol,
            start_date=start_date_str,
            end_date=end_date_str,
            interval=kite_interval
        )
        
        if data.empty:
            raise ValueError(f"No data fetched for {symbol}")
        
        logger.info(f"✅ Fetched {len(data)} data points")
        return data
    
    def calculate_parameter_distance(
        self,
        current_params: Dict[str, Any],
        new_params: Dict[str, Any],
        short_window_range: List[int] = None,
        long_window_range: List[int] = None,
        risk_reward_range: List[float] = None
    ) -> float:
        """
        Calculate Euclidean distance between current and new parameters.
        
        Normalizes parameters to 0-1 range for fair comparison.
        Uses the actual ranges from optimization (or stored ranges if not provided).
        
        Args:
            current_params: Current parameter dict
            new_params: New optimal parameter dict
            short_window_range: Range of short windows used in optimization (optional)
            long_window_range: Range of long windows used in optimization (optional)
            risk_reward_range: Range of risk-reward ratios used in optimization (optional)
            
        Returns:
            Distance metric (higher = more different)
        """
        # Use provided ranges, or fall back to stored ranges from last optimization, or defaults
        if short_window_range is None:
            short_window_range = self._last_short_range or [5, 10, 15, 20, 25, 30]
        if long_window_range is None:
            long_window_range = self._last_long_range or [20, 30, 40, 50, 60, 70]
        if risk_reward_range is None:
            risk_reward_range = self._last_rr_range or [4.0, 4.5, 5.0, 5.5, 6.0]
        
        # Calculate min/max from actual ranges
        param_ranges = {
            'short_window': (min(short_window_range), max(short_window_range)),
            'long_window': (min(long_window_range), max(long_window_range)),
            'risk_reward_ratio': (min(risk_reward_range), max(risk_reward_range))
        }
        
        distance_squared = 0.0
        
        for param_name in ['short_window', 'long_window', 'risk_reward_ratio']:
            if param_name not in current_params or param_name not in new_params:
                continue
            
            current_val = current_params[param_name]
            new_val = new_params[param_name]
            
            # Normalize to 0-1 range using actual optimization ranges
            param_min, param_max = param_ranges[param_name]
            if param_max == param_min:
                # Avoid division by zero if range has only one value
                continue
            current_norm = (current_val - param_min) / (param_max - param_min)
            new_norm = (new_val - param_min) / (param_max - param_min)
            
            # Add to distance
            distance_squared += (current_norm - new_norm) ** 2
        
        return np.sqrt(distance_squared)
    
    def run_optimization(
        self,
        data: pd.DataFrame,
        short_window_range: List[int] = None,
        long_window_range: List[int] = None,
        risk_reward_range: List[float] = None,
        use_neighborhood_aware: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Run full grid search optimization on data.
        Uses neighborhood-aware scoring to find robust optimal parameters.
        
        Args:
            data: OHLCV data
            short_window_range: Range of short windows to test
            long_window_range: Range of long windows to test
            risk_reward_range: Range of risk-reward ratios to test
            use_neighborhood_aware: If True, use neighborhood-aware scoring (recommended)
            
        Returns:
            Tuple of (best_parameters, best_metrics)
        """
        # Default ranges if not provided (matching optimization script)
        if short_window_range is None:
            short_window_range = [5, 10, 15, 20, 25, 30]  # Matches run_ma_optimization_kite.py
        if long_window_range is None:
            long_window_range = [20, 30, 40, 50, 60, 70]  # Matches run_ma_optimization_kite.py
        if risk_reward_range is None:
            risk_reward_range = [4.0, 4.5, 5.0, 5.5, 6.0]  # Matches run_ma_optimization_kite.py
        
        # Store ranges for distance calculation
        self._last_short_range = short_window_range
        self._last_long_range = long_window_range
        self._last_rr_range = risk_reward_range
        
        logger.info(f"🔍 Running optimization on {len(data)} data points...")
        logger.info(f"   Parameter ranges: {len(short_window_range)} short × {len(long_window_range)} long × {len(risk_reward_range)} RR")
        logger.info(f"   Using neighborhood-aware scoring: {use_neighborhood_aware}")
        
        if use_neighborhood_aware:
            # Use visualizer for neighborhood-aware scoring
            import tempfile
            import shutil
            temp_output_dir = tempfile.mkdtemp(prefix="param_validation_")
            try:
                visualizer = MAOptimization3DVisualizer(
                    data,
                    trading_fee=self.trading_fee,
                    auto_open=False,
                    output_dir=temp_output_dir  # Temporary directory for validation
                )
                
                # Run optimization grid
                results = visualizer.run_optimization_grid(
                    short_window_range, long_window_range, risk_reward_range
                )
                
                # Calculate neighborhood-aware scores
                neighborhood_recommendations = visualizer.find_optimal_parameters_neighborhood_aware(
                    metric='composite_score'
                )
                
                if neighborhood_recommendations and 'overall_best' in neighborhood_recommendations:
                    overall_best = neighborhood_recommendations['overall_best']
                    best_params = {
                        'short_window': overall_best['short_window'],
                        'long_window': overall_best['long_window'],
                        'risk_reward_ratio': overall_best['risk_reward_ratio'],
                        'trading_fee': self.trading_fee
                    }
                    
                    # Get metrics for the best neighborhood-aware params
                    best_metrics = self.evaluate_parameters(data, best_params)
                    
                    logger.info(f"✅ Optimization complete (neighborhood-aware)")
                    logger.info(f"   Best neighborhood-aware params: {best_params}")
                    logger.info(f"   Neighborhood-aware score: {overall_best.get('neighborhood_aware_score', 0):.3f}")
                    logger.info(f"   Original score: {overall_best.get('original_score', 0):.3f}")
                    logger.info(f"   Best Sharpe: {best_metrics.get('sharpe_ratio', 0):.3f}")
                    logger.info(f"   Best PnL: {best_metrics.get('total_pnl', 0):.2%}")
                    
                    # Cleanup temp directory
                    try:
                        shutil.rmtree(temp_output_dir)
                    except:
                        pass
                    
                    return best_params, best_metrics
                else:
                    # Fallback to raw optimization if neighborhood-aware fails
                    logger.warning("⚠️  Neighborhood-aware scoring failed, falling back to raw optimization")
                    use_neighborhood_aware = False
            except Exception as e:
                logger.warning(f"⚠️  Neighborhood-aware scoring error: {e}, falling back to raw optimization")
                use_neighborhood_aware = False
            finally:
                # Cleanup temp directory
                try:
                    shutil.rmtree(temp_output_dir)
                except:
                    pass
        
        if not use_neighborhood_aware:
            # Fallback: Use raw optimization
            param_ranges = {
                'short_window': short_window_range,
                'long_window': long_window_range,
                'risk_reward_ratio': risk_reward_range,
                'trading_fee': [self.trading_fee]
            }
            
            optimizer = StrategyOptimizer(
                data=data,
                strategy_class=MovingAverageCrossover,
                param_ranges=param_ranges,
                optimization_metric="composite_score",
                min_trades=5,  # Lower threshold for validation (less data)
                max_drawdown_threshold=0.5,
                sharpe_threshold=0.0  # Lower threshold for validation
            )
            
            best_params, best_metrics = optimizer.optimize()
            
            logger.info(f"✅ Optimization complete (raw)")
            logger.info(f"   Best params: {best_params}")
            logger.info(f"   Best Sharpe: {best_metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"   Best PnL: {best_metrics.get('total_pnl', 0):.2%}")
            
            return best_params, best_metrics
    
    def evaluate_parameters(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate specific parameters on data.
        
        Args:
            data: OHLCV data
            params: Parameter dict to evaluate
            
        Returns:
            Performance metrics dict
        """
        # Ensure trading_fee is in params
        eval_params = dict(params)
        if 'trading_fee' not in eval_params:
            eval_params['trading_fee'] = self.trading_fee
        
        optimizer = StrategyOptimizer(
            data=data,
            strategy_class=MovingAverageCrossover,
            param_ranges={k: [v] for k, v in eval_params.items()},
            optimization_metric="composite_score",
            min_trades=1,
            max_drawdown_threshold=1.0,
            sharpe_threshold=-999
        )
        
        # Run single evaluation
        _, metrics = optimizer.optimize()
        
        return metrics
    
    def calculate_stability_score(
        self,
        optimizer: StrategyOptimizer,
        top_n: int = 5
    ) -> float:
        """
        Calculate stability score based on how clustered top N results are.
        
        If top results are similar → high stability (good)
        If top results are spread out → low stability (bad)
        
        Args:
            optimizer: StrategyOptimizer instance with results
            top_n: Number of top results to analyze
            
        Returns:
            Stability score 0-1 (1 = very stable, 0 = unstable)
        """
        try:
            top_results = optimizer.get_top_results(top_n)
            if len(top_results) < 2:
                return 0.5  # Neutral if not enough data
            
            # Extract parameters
            short_windows = [r['parameters']['short_window'] for r in top_results]
            long_windows = [r['parameters']['long_window'] for r in top_results]
            rr_ratios = [r['parameters']['risk_reward_ratio'] for r in top_results]
            
            # Calculate coefficient of variation (std/mean) for each parameter
            def cv(values):
                if np.mean(values) == 0:
                    return 1.0  # High variation if mean is 0
                return np.std(values) / abs(np.mean(values))
            
            cv_short = cv(short_windows)
            cv_long = cv(long_windows)
            cv_rr = cv(rr_ratios)
            
            # Average CV (lower = more stable)
            avg_cv = (cv_short + cv_long + cv_rr) / 3
            
            # Convert to stability score (0-1, higher = more stable)
            # CV of 0 = perfect stability, CV of 0.5+ = unstable
            stability = max(0, 1 - (avg_cv * 2))
            
            return stability
        except Exception as e:
            logger.warning(f"Could not calculate stability: {e}")
            return 0.5
    
    def validate_parameters(
        self,
        current_params: Dict[str, Any],
        symbol: str,
        interval: str = "15m",
        exchange: str = None,
        short_window_range: List[int] = None,
        long_window_range: List[int] = None,
        risk_reward_range: List[float] = None
    ) -> ValidationResult:
        """
        Validate current parameters by running optimization on recent data.
        
        Args:
            current_params: Current parameter dict with keys: short_window, long_window, risk_reward_ratio
            symbol: Trading symbol
            interval: Data interval
            exchange: Exchange name (overrides default)
            short_window_range: Custom short window range for optimization
            long_window_range: Custom long window range for optimization
            risk_reward_range: Custom risk-reward range for optimization
            
        Returns:
            ValidationResult with drift analysis
        """
        if exchange:
            self.exchange = exchange
            self._initialize_kite_fetcher()
        
        logger.info("=" * 80)
        logger.info("🔍 PARAMETER VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Current params: {current_params}")
        logger.info(f"Symbol: {symbol}, Exchange: {self.exchange}, Interval: {interval}")
        logger.info(f"Data window: {self.data_window_days} days")
        
        # Fetch recent data
        try:
            data = self.fetch_recent_data(symbol, interval)
        except Exception as e:
            logger.error(f"❌ Failed to fetch data: {e}")
            raise
        
        # Run optimization
        try:
            new_optimal_params, new_optimal_metrics = self.run_optimization(
                data,
                short_window_range=short_window_range,
                long_window_range=long_window_range,
                risk_reward_range=risk_reward_range
            )
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            raise
        
        # Evaluate current parameters on same data
        logger.info("📊 Evaluating current parameters...")
        current_metrics = self.evaluate_parameters(data, current_params)
        
        # Calculate parameter distance (pass the ranges used in optimization)
        param_distance = self.calculate_parameter_distance(
            current_params, 
            new_optimal_params,
            short_window_range=short_window_range,
            long_window_range=long_window_range,
            risk_reward_range=risk_reward_range
        )
        
        # Calculate performance gap
        current_pnl = current_metrics.get('total_pnl', 0)
        new_optimal_pnl = new_optimal_metrics.get('total_pnl', 0)
        performance_gap = new_optimal_pnl - current_pnl
        
        # Determine alert level
        alert_level = 'none'
        should_reoptimize = False
        alert_message = ""
        
        if param_distance >= self.distance_threshold_critical or performance_gap >= self.performance_gap_threshold * 2:
            alert_level = 'critical'
            should_reoptimize = True
            alert_message = (
                f"🚨 CRITICAL: Parameters have significantly drifted!\n"
                f"   Distance: {param_distance:.2f} (threshold: {self.distance_threshold_critical})\n"
                f"   Performance gap: {performance_gap:.2%} (new optimal would give {performance_gap:.2%} better returns)\n"
                f"   Current: Short={current_params['short_window']}, Long={current_params['long_window']}, RR={current_params['risk_reward_ratio']}\n"
                f"   New optimal: Short={new_optimal_params['short_window']}, Long={new_optimal_params['long_window']}, RR={new_optimal_params['risk_reward_ratio']}\n"
                f"   ⚠️ RECOMMENDATION: Re-optimize immediately!"
            )
        elif param_distance >= self.distance_threshold_warning or performance_gap >= self.performance_gap_threshold:
            alert_level = 'warning'
            should_reoptimize = True
            alert_message = (
                f"⚠️ WARNING: Parameters have drifted\n"
                f"   Distance: {param_distance:.2f} (threshold: {self.distance_threshold_warning})\n"
                f"   Performance gap: {performance_gap:.2%}\n"
                f"   Current: Short={current_params['short_window']}, Long={current_params['long_window']}, RR={current_params['risk_reward_ratio']}\n"
                f"   New optimal: Short={new_optimal_params['short_window']}, Long={new_optimal_params['long_window']}, RR={new_optimal_params['risk_reward_ratio']}\n"
                f"   💡 RECOMMENDATION: Schedule re-optimization soon"
            )
        elif param_distance >= self.distance_threshold_monitor:
            alert_level = 'monitor'
            should_reoptimize = False
            alert_message = (
                f"📊 MONITOR: Some parameter drift detected\n"
                f"   Distance: {param_distance:.2f} (threshold: {self.distance_threshold_monitor})\n"
                f"   Performance gap: {performance_gap:.2%}\n"
                f"   Current params still acceptable, but monitor closely"
            )
        else:
            alert_level = 'none'
            should_reoptimize = False
            alert_message = (
                f"✅ Parameters still optimal\n"
                f"   Distance: {param_distance:.2f} (very close to optimal)\n"
                f"   Performance gap: {performance_gap:.2%}\n"
                f"   No action needed"
            )
        
        # Calculate stability (requires optimizer instance, simplified here)
        stability_score = 0.7  # Default, could be improved with optimizer access
        
        # Create result
        result = ValidationResult(
            validation_date=datetime.now().isoformat(),
            current_params=current_params,
            new_optimal_params=new_optimal_params,
            parameter_distance=param_distance,
            performance_gap=performance_gap,
            should_reoptimize=should_reoptimize,
            alert_level=alert_level,
            alert_message=alert_message,
            current_params_performance=current_metrics,
            new_optimal_performance=new_optimal_metrics,
            stability_score=stability_score,
            validation_data_period=f"{data.index[0]} to {data.index[-1]}"
        )
        
        logger.info("=" * 80)
        logger.info("📋 VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(alert_message)
        logger.info("=" * 80)
        
        return result
    
    def should_run_validation(self, last_validation_date: datetime = None) -> bool:
        """
        Check if validation should be run based on frequency.
        
        Args:
            last_validation_date: Date of last validation (None = always run)
            
        Returns:
            True if validation should run
        """
        if last_validation_date is None:
            return True
        
        days_since = (datetime.now() - last_validation_date).days
        return days_since >= self.validation_frequency_days
    
    def save_validation_result(
        self,
        result: ValidationResult,
        output_dir: str = "validation_results"
    ) -> str:
        """
        Save validation result to JSON file.
        
        Args:
            result: ValidationResult to save
            output_dir: Directory to save results
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info(f"💾 Validation result saved: {filepath}")
        return filepath


# ============================================================================
# Quick Usage Functions
# ============================================================================

def validate_ma_parameters(
    current_params: Dict[str, Any],
    symbol: str,
    interval: str = "15m",
    exchange: str = "MCX",
    validation_frequency_days: int = 7,
    data_window_days: int = 30,
    **kwargs
) -> ValidationResult:
    """
    Quick function to validate MA parameters.
    
    Args:
        current_params: Dict with short_window, long_window, risk_reward_ratio
        symbol: Trading symbol
        interval: Data interval
        exchange: Exchange name
        validation_frequency_days: How often to validate (default: 7 = weekly)
        data_window_days: Recent data window (default: 30 days)
        **kwargs: Additional args for ParameterValidator
        
    Returns:
        ValidationResult
    """
    validator = ParameterValidator(
        validation_frequency_days=validation_frequency_days,
        data_window_days=data_window_days,
        exchange=exchange,
        **kwargs
    )
    
    return validator.validate_parameters(
        current_params=current_params,
        symbol=symbol,
        interval=interval,
        exchange=exchange
    )


if __name__ == "__main__":
    # Demo
    print("🔍 Parameter Validator Demo")
    print("=" * 50)
    
    # Example current parameters
    current_params = {
        'short_window': 4,
        'long_window': 58,
        'risk_reward_ratio': 6.0
    }
    
    validator = ParameterValidator(
        validation_frequency_days=7,
        data_window_days=30
    )
    
    try:
        result = validator.validate_parameters(
            current_params=current_params,
            symbol='SILVERMIC26FEBFUT',
            interval='15m',
            exchange='MCX'
        )
        
        print("\n" + result.alert_message)
        print(f"\nShould re-optimize: {result.should_reoptimize}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

