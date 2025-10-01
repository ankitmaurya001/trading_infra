import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from itertools import product
import warnings
import logging
from strategies import BaseStrategy, MovingAverageCrossover, RSIStrategy, DonchianChannelBreakout, PositionType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    A robust strategy optimizer that uses multiple metrics to avoid overfitting
    and select the best hyperparameters.
    """
    
    # Class constants for better maintainability
    PENALTY_SCORE = -999
    EPSILON = 1e-8  # Small value to prevent division by zero
    
    def __init__(self, data: pd.DataFrame, strategy_class: type, 
                 param_ranges: Dict[str, List], 
                 optimization_metric: str = "composite_score",
                 min_trades: int = 10,
                 max_drawdown_threshold: float = 0.5,
                 sharpe_threshold: float = 0.5):
        """
        Initialize the strategy optimizer.
        
        Args:
            data: OHLCV data for backtesting
            strategy_class: The strategy class to optimize
            param_ranges: Dictionary of parameter names and their possible values
            optimization_metric: Metric to optimize for ("composite_score", "sharpe_ratio", "calmar_ratio", etc.)
            min_trades: Minimum number of trades required for valid results
            max_drawdown_threshold: Maximum acceptable drawdown
            sharpe_threshold: Minimum acceptable Sharpe ratio
        """
        self.data = data
        self.strategy_class = strategy_class
        self.param_ranges = param_ranges
        self.optimization_metric = optimization_metric
        self.min_trades = min_trades
        self.max_drawdown_threshold = max_drawdown_threshold
        self.sharpe_threshold = sharpe_threshold
        self.results = []
        self.failed_runs = []  # Track failed configurations for debugging
        
    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate a composite score that balances multiple metrics to avoid overfitting.
        
        Args:
            metrics: Dictionary of strategy metrics
            
        Returns:
            float: Composite score (higher is better)
        """
        # Extract key metrics with defensive defaults
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        calmar_ratio = metrics.get('calmar_ratio', 0)
        profit_factor = metrics.get('profit_factor', 0)
        win_rate = metrics.get('win_rate', 0)
        max_drawdown = metrics.get('max_drawdown', 1)
        total_trades = metrics.get('total_trades', 0)
        geometric_mean_return = metrics.get('geometric_mean_return', 0)
        
        # Log missing critical metrics
        missing_metrics = []
        for metric in ['sharpe_ratio', 'calmar_ratio', 'profit_factor', 'win_rate', 'max_drawdown', 'total_trades']:
            if metric not in metrics:
                missing_metrics.append(metric)
        
        if missing_metrics:
            logger.warning(f"Missing metrics in strategy results: {missing_metrics}")
        
        # Penalize insufficient trades
        if total_trades < self.min_trades:
            logger.info(f"Insufficient trades: {total_trades} < {self.min_trades}")
            return self.PENALTY_SCORE
        
        # Penalize excessive drawdown
        if max_drawdown > self.max_drawdown_threshold:
            logger.info(f"Excessive drawdown: {max_drawdown:.4f} > {self.max_drawdown_threshold}")
            return self.PENALTY_SCORE
        
        # Penalize poor risk-adjusted returns
        if sharpe_ratio < self.sharpe_threshold:
            #logger.info(f"Poor Sharpe ratio: {sharpe_ratio:.4f} < {self.sharpe_threshold}")
            return self.PENALTY_SCORE
        
        # Defensive handling of infinite values
        if np.isinf(profit_factor) or np.isnan(profit_factor):
            profit_factor = 0
        if np.isinf(calmar_ratio) or np.isnan(calmar_ratio):
            calmar_ratio = 0
        if np.isinf(sharpe_ratio) or np.isnan(sharpe_ratio):
            sharpe_ratio = 0
        
        # Calculate composite score with weights
        # These weights can be adjusted based on your preferences
        score = (
            0.25 * sharpe_ratio +           # Risk-adjusted returns
            0.20 * calmar_ratio +           # Return vs drawdown
            0.15 * profit_factor +          # Profit efficiency
            0.15 * win_rate +               # Consistency
            0.15 * geometric_mean_return +  # Compound growth
            0.10 * (1 - max_drawdown)       # Drawdown penalty
        )
        
        return score
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameter combinations to avoid invalid configurations.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            bool: True if parameters are valid
        """
        # Example validations for different strategies
        if self.strategy_class == MovingAverageCrossover:
            short_window = params.get('short_window', 0)
            long_window = params.get('long_window', 0)
            return short_window < long_window
        
        elif self.strategy_class == RSIStrategy:
            overbought = params.get('overbought', 70)
            oversold = params.get('oversold', 30)
            return overbought > oversold
        
        return True
    
    def _reset_strategy_state(self, strategy: BaseStrategy) -> None:
        """
        Reset strategy internal state to ensure clean runs.
        
        Args:
            strategy: Strategy instance to reset
        """
        strategy.trades = []
        strategy.active_trade = None
        strategy.current_position = PositionType.NONE
        strategy.signals = None
    
    def optimize(self, n_jobs: int = 1) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            n_jobs: Number of parallel jobs (1 for sequential processing)
            
        Returns:
            Tuple of (best_parameters, best_metrics)
        """
        logger.info(f"Starting optimization for {self.strategy_class.__name__}...")
        logger.info(f"Parameter ranges: {self.param_ranges}")
        
        # Generate all parameter combinations
        param_names = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())
        param_combinations = list(product(*param_values))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations...")
        
        best_score = -float('inf')
        best_params = None
        best_metrics = None
        
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            # Skip invalid parameter combinations
            if not self.validate_parameters(params):
                logger.debug(f"Skipping invalid parameters: {params}")
                continue
            
            try:
                # Create fresh strategy instance for each test
                strategy = self.strategy_class(**params)
                
                # Ensure clean state
                self._reset_strategy_state(strategy)
                
                # Generate signals and get metrics
                signals_df = strategy.generate_signals(self.data)
                metrics = strategy.get_strategy_metrics()
                
                # Validate that signals were generated
                if signals_df is None or len(signals_df) == 0:
                    raise ValueError("Strategy failed to generate signals")
                
                # Calculate optimization score
                if self.optimization_metric == "composite_score":
                    score = self.calculate_composite_score(metrics)
                else:
                    score = metrics.get(self.optimization_metric, -float('inf'))
                    if np.isinf(score) or np.isnan(score):
                        score = -float('inf')
                
                # Store results
                result = {
                    'parameters': params,
                    'metrics': metrics,
                    'score': score,
                    'signals_df_shape': signals_df.shape if signals_df is not None else None
                }
                self.results.append(result)
                
                # Update best if better
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(param_combinations)} combinations...")
                    
            except Exception as e:
                error_msg = f"Error with parameters {params}: {str(e)}"
                logger.warning(error_msg)
                
                # Store failed run for debugging
                failed_run = {
                    'parameters': params,
                    'metrics': {},
                    'score': self.PENALTY_SCORE,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                self.failed_runs.append(failed_run)
                continue
        
        logger.info(f"Optimization complete! Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Failed runs: {len(self.failed_runs)}")
        
        # Calculate local sensitivity analysis and add robustness info to best_metrics
        if len(self.results) >= 5:
            try:
                # Get top 5 results for robustness analysis
                top_results = self.get_top_results(5)
                self._local_sensitivity_results = self.analyze_local_sensitivity(top_results, variation_percent=[0.0375, 0.01875], robustness_selection='average')
                
                # Find the most robust result
                most_robust_result = None
                best_robustness = float('inf')
                
                for result_key, analysis in self._local_sensitivity_results.items():
                    overall_robustness = analysis['param_variations'].get('overall_robustness', float('inf'))
                    if overall_robustness < best_robustness:
                        best_robustness = overall_robustness
                        most_robust_result = analysis
                
                if most_robust_result and best_robustness != float('inf'):
                    # Add robustness info to best_metrics
                    best_metrics['most_robust_params'] = most_robust_result['base_params']
                    best_metrics['most_robust_score'] = most_robust_result['base_score']
                    best_metrics['most_robust_robustness'] = best_robustness
                    
                    # Calculate robustness for the best parameters
                    best_params_robustness = None
                    for result_key, analysis in self._local_sensitivity_results.items():
                        if analysis['base_params'] == best_params:
                            best_params_robustness = analysis['param_variations'].get('overall_robustness', float('inf'))
                            break
                    
                    if best_params_robustness is not None:
                        best_metrics['robustness'] = best_params_robustness
                    
                    logger.info(f"Most robust parameters: {most_robust_result['base_params']} (robustness: {best_robustness:.4f})")
                    if best_params_robustness is not None:
                        logger.info(f"Best parameters robustness: {best_params_robustness:.4f}")
                    
            except Exception as e:
                logger.warning(f"Could not calculate robustness: {e}")
                self._local_sensitivity_results = None
        else:
            self._local_sensitivity_results = None
        
        return best_params, best_metrics
    
    def get_top_results(self, n: int = 10) -> List[Dict]:
        """
        Get the top N results from optimization.
        
        Args:
            n: Number of top results to return
            
        Returns:
            List of top results sorted by score
        """
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:n]
    
    def analyze_parameter_sensitivity(self, param_name: str) -> Dict[str, float]:
        """
        Analyze how sensitive the strategy is to a specific parameter.
        
        Note: This analysis does not account for parameter interactions.
        It calculates average scores per parameter value in isolation.
        
        Args:
            param_name: Name of the parameter to analyze
            
        Returns:
            Dictionary mapping parameter values to average scores
        """
        param_scores = {}
        
        for result in self.results:
            param_value = result['parameters'].get(param_name)
            if param_value is not None:
                if param_value not in param_scores:
                    param_scores[param_value] = []
                param_scores[param_value].append(result['score'])
        
        # Calculate average scores
        avg_scores = {}
        for param_value, scores in param_scores.items():
            # Filter out penalty scores for better analysis
            valid_scores = [s for s in scores if s > self.PENALTY_SCORE]
            if valid_scores:
                #avg_scores[param_value] = np.mean(valid_scores)
                avg_scores[param_value] = np.median(valid_scores)
            else:
                avg_scores[param_value] = self.PENALTY_SCORE
        
        return avg_scores
    
    def analyze_local_sensitivity(self, top_results: List[Dict], variation_percent: Union[float, List[float]] = 0.15, robustness_selection: str = 'average') -> Dict:
        """
        Analyze local sensitivity around the top parameter combinations.
        Tests parameter variations within ±variation_percent of optimal values.
        
        Args:
            top_results: List of top optimization results
            variation_percent: Single percentage or list of percentages to test
                               (e.g., 0.15 for ±15%). If a list is provided,
                               robustness will be computed for each level.
            robustness_selection: How to compute overall robustness when multiple
                                  variations are provided. Options: 'primary'
                                  (first value), 'average' (mean across all),
                                  'max' (worst case), 'min' (best case).
            
        Returns:
            Dictionary with robustness analysis for each top result
        """
        local_sensitivity_results = {}

        # Normalize variation percents to a list for unified processing
        if isinstance(variation_percent, (int, float)):
            variation_percents: List[float] = [float(variation_percent)]
        else:
            variation_percents = [float(v) for v in variation_percent]
        
        for i, result in enumerate(top_results):
            base_params = result['parameters'].copy()
            base_score = result['score']
            
            print(f"\n🔍 LOCAL SENSITIVITY ANALYSIS - Result #{i+1}")
            print(f"Base Score: {base_score:.4f}")
            print(f"Base Parameters: {base_params}")
            print("-" * 60)
            
            # Test variations for each parameter
            param_variations = {}
            robustness_scores_by_variation = {vp: [] for vp in variation_percents}
            
            for param_name, base_value in base_params.items():
                if param_name == 'trading_fee':  # Skip trading_fee as it's usually fixed
                    continue
                    
                print(f"\n📊 Testing {param_name} variations around {base_value}:")
                
                # Compute per-variation scores and robustness
                by_variation = {}
                for vp in variation_percents:
                    # Generate test values around the base value for this variation percent
                    if isinstance(base_value, int):
                        variation_range = max(1, int(base_value * vp))
                        test_values = [
                            max(1, base_value - variation_range),
                            base_value,
                            base_value + variation_range
                        ]
                    else:
                        variation_range = base_value * vp
                        test_values = [
                            max(0.1, base_value - variation_range),
                            base_value,
                            base_value + variation_range
                        ]
                    # Remove duplicates and sort
                    test_values = sorted(list(set(test_values)))

                    param_scores = {}
                    for test_value in test_values:
                        # Create test parameters
                        test_params = base_params.copy()
                        test_params[param_name] = test_value

                        # Validate parameters
                        if not self.validate_parameters(test_params):
                            print(f"  {test_value}: INVALID (skipped)")
                            continue

                        # Run strategy with test parameters
                        try:
                            strategy = self.strategy_class(**test_params)
                            self._reset_strategy_state(strategy)

                            # Run backtest using generate_signals method
                            signals_data = strategy.generate_signals(self.data)

                            # Calculate metrics
                            metrics = strategy.get_strategy_metrics()
                            score = self.calculate_composite_score(metrics)

                            param_scores[test_value] = score
                            print(f"  {test_value}: {score:.4f}")

                        except Exception as e:
                            print(f"  {test_value}: ERROR - {str(e)}")
                            param_scores[test_value] = self.PENALTY_SCORE

                    # Calculate robustness metrics for this parameter and variation
                    valid_scores = [s for s in param_scores.values() if s > self.PENALTY_SCORE]
                    if valid_scores:
                        score_variance = np.var(valid_scores)
                        score_std = np.std(valid_scores)
                        min_score = min(valid_scores)
                        max_score = max(valid_scores)
                        robustness = score_std / (base_score + self.EPSILON)

                        by_variation[vp] = {
                            'scores': param_scores,
                            'variance': score_variance,
                            'std_dev': score_std,
                            'min_score': min_score,
                            'max_score': max_score,
                            'robustness': robustness
                        }
                        robustness_scores_by_variation[vp].append(robustness)
                    else:
                        print(f"  ❌ No valid scores for {param_name} at variation {vp}")

                # Attach per-parameter, per-variation details
                param_variations[param_name] = {
                    'by_variation': by_variation
                }

                # For backward compatibility, print summary for the first variation only
                first_vp = variation_percents[0]
                if first_vp in by_variation:
                    rb = by_variation[first_vp]['robustness']
                    min_s = by_variation[first_vp]['min_score']
                    max_s = by_variation[first_vp]['max_score']
                    print(f"  📈 Robustness (@±{first_vp:.4f}): {rb:.4f} (lower = more robust)")
                    print(f"  📊 Score range (@±{first_vp:.4f}): {min_s:.4f} - {max_s:.4f}")
            
            # Overall robustness for this result
            overall_by_variation = {}
            for vp, rb_list in robustness_scores_by_variation.items():
                if rb_list:
                    overall_by_variation[vp] = np.mean(rb_list)
                else:
                    overall_by_variation[vp] = float('inf')

            # Compute overall robustness according to selection method
            first_vp = variation_percents[0]
            if robustness_selection == 'primary' or len(variation_percents) == 1:
                overall_robustness_value = overall_by_variation.get(first_vp, float('inf'))
            elif robustness_selection == 'average':
                vals = [overall_by_variation[vp] for vp in variation_percents]
                overall_robustness_value = float(np.mean(vals)) if len(vals) > 0 else float('inf')
            elif robustness_selection == 'max':
                vals = [overall_by_variation[vp] for vp in variation_percents]
                overall_robustness_value = float(np.max(vals)) if len(vals) > 0 else float('inf')
            elif robustness_selection == 'min':
                vals = [overall_by_variation[vp] for vp in variation_percents]
                overall_robustness_value = float(np.min(vals)) if len(vals) > 0 else float('inf')
            else:
                overall_robustness_value = overall_by_variation.get(first_vp, float('inf'))

            param_variations['overall_robustness'] = overall_robustness_value
            param_variations['overall_robustness_by_variation'] = overall_by_variation

            print(f"\n🎯 Overall Robustness by variation: " + ", ".join([f"±{vp:.4f}: {overall_by_variation[vp]:.4f}" for vp in variation_percents]))
            if robustness_selection != 'primary' and len(variation_percents) > 1:
                print(f"➡️ Using '{robustness_selection}' across variations for final robustness: {overall_robustness_value:.4f}")
            
            local_sensitivity_results[f"result_{i+1}"] = {
                'base_params': base_params,
                'base_score': base_score,
                'param_variations': param_variations
            }
        
        return local_sensitivity_results
    
    def get_failed_runs_summary(self) -> Dict[str, int]:
        """
        Get a summary of failed runs by error type.
        
        Returns:
            Dictionary mapping error types to counts
        """
        error_counts = {}
        for failed_run in self.failed_runs:
            error_type = failed_run.get('error_type', 'Unknown')
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts
    
    def print_optimization_summary(self):
        """Print a summary of the optimization results."""
        if not self.results:
            logger.warning("No optimization results available.")
            return
        
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        # Overall statistics
        scores = [r['score'] for r in self.results]
        valid_scores = [s for s in scores if s > self.PENALTY_SCORE]
        
        print(f"Total combinations tested: {len(self.results)}")
        print(f"Successful runs: {len(valid_scores)}")
        print(f"Failed runs: {len(self.failed_runs)}")
        
        if valid_scores:
            print(f"Best score: {max(valid_scores):.4f}")
            print(f"Worst valid score: {min(valid_scores):.4f}")
            print(f"Average valid score: {np.mean(valid_scores):.4f}")
            print(f"Score std dev: {np.std(valid_scores):.4f}")
        
        # Failed runs summary
        if self.failed_runs:
            print(f"\nFAILED RUNS SUMMARY:")
            print("-" * 60)
            error_summary = self.get_failed_runs_summary()
            for error_type, count in error_summary.items():
                print(f"{error_type}: {count} failures")
        
        # Top 5 results
        print("\nTOP 5 RESULTS:")
        print("-" * 60)
        top_results = self.get_top_results(5)
        for i, result in enumerate(top_results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   Parameters: {result['parameters']}")
            metrics = result['metrics']
            print(f"   Sharpe: {metrics.get('sharpe_ratio', 'N/A'):.3f}, "
                  f"Calmar: {metrics.get('calmar_ratio', 'N/A'):.3f}, "
                  f"Win Rate: {metrics.get('win_rate', 'N/A'):.2%}")
            print()
        
        # Parameter sensitivity analysis
        print("PARAMETER SENSITIVITY:")
        print("-" * 60)
        for param_name in self.param_ranges.keys():
            sensitivity = self.analyze_parameter_sensitivity(param_name)
            if sensitivity:
                best_value = max(sensitivity.items(), key=lambda x: x[1])
                print(f"{param_name}: Best value = {best_value[0]} (avg score: {best_value[1]:.4f})")
        
        # Local sensitivity analysis around top results (use cached results)
        if self._local_sensitivity_results is not None:
            print("\n" + "="*60)
            print("LOCAL SENSITIVITY ANALYSIS")
            print("="*60)
            print("Parameter robustness analysis (using cached results)...")
            
            local_sensitivity = self._local_sensitivity_results
        else:
            print("\n" + "="*60)
            print("LOCAL SENSITIVITY ANALYSIS")
            print("="*60)
            print("Testing parameter robustness around top combinations...")
            
            top_results = self.get_top_results(5)
            local_sensitivity = self.analyze_local_sensitivity(top_results, variation_percent=[0.0375, 0.01875], robustness_selection='average')
        
        # Summary of robustness
        print("\n" + "="*60)
        print("ROBUSTNESS SUMMARY")
        print("="*60)
        robustness_ranking = []
        
        # If we have multi-variation robustness, display a compact table first
        # Use keys from the first result to derive the variation percents displayed
        any_result_key = next(iter(local_sensitivity)) if local_sensitivity else None
        variation_columns: List[float] = []
        primary_variation: Optional[float] = None
        if any_result_key is not None:
            any_result = local_sensitivity[any_result_key]
            overall_by_var = any_result['param_variations'].get('overall_robustness_by_variation')
            if isinstance(overall_by_var, dict) and overall_by_var:
                # Preserve insertion order as constructed during analysis
                variation_columns = list(overall_by_var.keys())
                primary_variation = variation_columns[0]

        if variation_columns:
            # Header
            header = f"{'Result':<8} {'Base Score':<12} " + " ".join([f"±{vp:<8.4f}" for vp in variation_columns])
            print(header)
            print("-" * max(70, len(header)))
            # Rows
            for result_key, analysis in local_sensitivity.items():
                base_score = analysis['base_score']
                overall_by_var = analysis['param_variations'].get('overall_robustness_by_variation', {})
                row_values = [f"{overall_by_var.get(vp, float('inf')):<8.4f}" for vp in variation_columns]
                print(f"{result_key:<8} {base_score:<12.4f} " + " ".join(row_values))
            if primary_variation is not None:
                print(f"\nFinal ranking uses primary variation ±{primary_variation:.4f}.")

        for result_key, analysis in local_sensitivity.items():
            overall_robustness = analysis['param_variations'].get('overall_robustness', float('inf'))
            base_score = analysis['base_score']
            robustness_ranking.append((result_key, base_score, overall_robustness))
        
        # Sort by robustness (lower is better)
        robustness_ranking.sort(key=lambda x: x[2])
        
        print(f"{'Rank':<4} {'Result':<8} {'Base Score':<12} {'Robustness':<12} {'Recommendation'}")
        print("-" * 70)
        
        for i, (result_key, base_score, robustness) in enumerate(robustness_ranking, 1):
            if robustness == float('inf'):
                recommendation = "❌ Not robust"
            elif robustness < 0.1:
                recommendation = "🟢 Very robust"
            elif robustness < 0.2:
                recommendation = "🟡 Moderately robust"
            else:
                recommendation = "🔴 Not robust"
            
            print(f"{i:<4} {result_key:<8} {base_score:<12.4f} {robustness:<12.4f} {recommendation}")
        
        print("\n💡 Interpretation:")
        print("• Robustness < 0.1: Very stable parameters (recommended)")
        print("• Robustness 0.1-0.2: Moderately stable (acceptable)")
        print("• Robustness > 0.2: Sensitive to parameter changes (use with caution)")
        
        return local_sensitivity

def optimize_moving_average_crossover(data: pd.DataFrame, 
                                    short_window_range: List[int] = [5, 10, 15, 20, 25],
                                    long_window_range: List[int] = [30, 40, 50, 60, 70],
                                    risk_reward_range: List[float] = [1.5, 2.0, 2.5, 3.0],
                                    trading_fee: float = 0.001,
                                    sharpe_threshold: float = 0.1) -> Tuple[Dict, Dict]:
    """
    Optimize Moving Average Crossover strategy parameters.
    
    Args:
        data: OHLCV data
        short_window_range: Range of short window values to test
        long_window_range: Range of long window values to test
        risk_reward_range: Range of risk-reward ratios to test
        trading_fee: Trading fee as decimal (e.g., 0.001 for 0.1%)
        
    Returns:
        Tuple of (best_parameters, best_metrics)
    """
    # Ensure trading_fee is in decimal format (same as app)
    if trading_fee > 1.0:  # If passed as percentage, convert to decimal
        trading_fee = trading_fee / 100
    
    logger.info(f"MA Crossover Optimization - Trading fee: {trading_fee} (decimal)")
    
    param_ranges = {
        'short_window': short_window_range,
        'long_window': long_window_range,
        'risk_reward_ratio': risk_reward_range,
        'trading_fee': [trading_fee]
    }
    
    optimizer = StrategyOptimizer(
        data=data,
        strategy_class=MovingAverageCrossover,
        param_ranges=param_ranges,
        optimization_metric="composite_score",
        min_trades=10,
        max_drawdown_threshold=0.4,
        sharpe_threshold=sharpe_threshold
    )
    
    best_params, best_metrics = optimizer.optimize()
    optimizer.print_optimization_summary()
    
    return best_params, best_metrics

def optimize_rsi_strategy(data: pd.DataFrame,
                         period_range: List[int] = [10, 14, 20, 30],
                         overbought_range: List[float] = [65, 70, 75, 80],
                         oversold_range: List[float] = [20, 25, 30, 35],
                         risk_reward_range: List[float] = [1.5, 2.0, 2.5, 3.0],
                         trading_fee: float = 0.001,
                         sharpe_threshold: float = 0.1) -> Tuple[Dict, Dict]:
    """
    Optimize RSI strategy parameters.
    
    Args:
        data: OHLCV data
        period_range: Range of RSI periods to test
        overbought_range: Range of overbought levels to test
        oversold_range: Range of oversold levels to test
        risk_reward_range: Range of risk-reward ratios to test
        trading_fee: Trading fee as decimal (e.g., 0.001 for 0.1%)
        
    Returns:
        Tuple of (best_parameters, best_metrics)
    """
    # Ensure trading_fee is in decimal format (same as app)
    if trading_fee > 1.0:  # If passed as percentage, convert to decimal
        trading_fee = trading_fee / 100
    
    logger.info(f"RSI Optimization - Trading fee: {trading_fee} (decimal)")
    
    param_ranges = {
        'period': period_range,
        'overbought': overbought_range,
        'oversold': oversold_range,
        'risk_reward_ratio': risk_reward_range,
        'trading_fee': [trading_fee]
    }
    
    optimizer = StrategyOptimizer(
        data=data,
        strategy_class=RSIStrategy,
        param_ranges=param_ranges,
        optimization_metric="composite_score",
        min_trades=10,
        max_drawdown_threshold=0.4,
        sharpe_threshold=sharpe_threshold
    )
    
    best_params, best_metrics = optimizer.optimize()
    optimizer.print_optimization_summary()
    
    return best_params, best_metrics

def optimize_donchian_channel(data: pd.DataFrame,
                             channel_period_range: List[int] = [10, 15, 20, 25, 30],
                             risk_reward_range: List[float] = [1.5, 2.0, 2.5, 3.0],
                             trading_fee: float = 0.001,
                             sharpe_threshold: float = 0.1) -> Tuple[Dict, Dict]:
    """
    Optimize Donchian Channel Breakout strategy parameters.
    
    Args:
        data: OHLCV data
        channel_period_range: Range of channel periods to test
        risk_reward_range: Range of risk-reward ratios to test
        trading_fee: Trading fee as decimal (e.g., 0.001 for 0.1%)
        
    Returns:
        Tuple of (best_parameters, best_metrics)
    """
    # Ensure trading_fee is in decimal format (same as app)
    if trading_fee > 1.0:  # If passed as percentage, convert to decimal
        trading_fee = trading_fee / 100
    
    logger.info(f"Donchian Channel Optimization - Trading fee: {trading_fee} (decimal)")
    
    param_ranges = {
        'channel_period': channel_period_range,
        'risk_reward_ratio': risk_reward_range,
        'trading_fee': [trading_fee]
    }
    
    optimizer = StrategyOptimizer(
        data=data,
        strategy_class=DonchianChannelBreakout,
        param_ranges=param_ranges,
        optimization_metric="composite_score",
        min_trades=10,
        max_drawdown_threshold=0.4,
        sharpe_threshold=sharpe_threshold
    )
    
    best_params, best_metrics = optimizer.optimize()
    optimizer.print_optimization_summary()
    
    return best_params, best_metrics 