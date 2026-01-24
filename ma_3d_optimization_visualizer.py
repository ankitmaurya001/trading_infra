#!/usr/bin/env python3
"""
3D Visualization for Moving Average Crossover Optimization
This script creates 3D plots showing how the optimization score varies across
different combinations of short_window and long_window for each risk_reward_ratio.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any
import warnings
import os
from scipy import stats
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
warnings.filterwarnings('ignore')

# Global default for optimal-region percentile threshold
DEFAULT_OPTIMAL_PERCENTILE = 50.0  # Top 20% are considered "optimal"

from data_fetcher import BinanceDataFetcher
from strategies import MovingAverageCrossover
from strategy_optimizer import StrategyOptimizer
import config as cfg

# Import neighborhood-aware scoring constants
NEIGHBORHOOD_RADIUS = cfg.NEIGHBORHOOD_RADIUS
NEIGHBORHOOD_RADIUS_MULTIPLIER = getattr(cfg, 'NEIGHBORHOOD_RADIUS_MULTIPLIER', 1.5)
DISTANCE_WEIGHT_POWER = cfg.DISTANCE_WEIGHT_POWER
OWN_SCORE_WEIGHT = cfg.OWN_SCORE_WEIGHT
NEIGHBORHOOD_WEIGHT = cfg.NEIGHBORHOOD_WEIGHT
NEGATIVE_PENALTY_WEIGHT = cfg.NEGATIVE_PENALTY_WEIGHT


def calculate_dynamic_neighborhood_radius(x_values: np.ndarray, y_values: np.ndarray, 
                                          multiplier: float = None) -> Tuple[float, dict]:
    """
    Dynamically calculate the neighborhood radius based on grid step sizes.
    
    This ensures the radius is always appropriate for the given parameter grid,
    automatically adapting to include immediate neighbors regardless of grid spacing.
    
    Args:
        x_values: Array of x coordinates (e.g., short_window values)
        y_values: Array of y coordinates (e.g., long_window values)
        multiplier: How many "steps" to include (default from config)
                   1.5 = immediate diagonal neighbors
                   2.0 = next layer of neighbors
    
    Returns:
        Tuple of (calculated_radius, details_dict)
    """
    if multiplier is None:
        multiplier = NEIGHBORHOOD_RADIUS_MULTIPLIER
    
    # Get unique sorted values
    x_unique = np.sort(np.unique(x_values))
    y_unique = np.sort(np.unique(y_values))
    
    # Calculate ranges
    x_range = x_unique.max() - x_unique.min() if len(x_unique) > 1 else 1.0
    y_range = y_unique.max() - y_unique.min() if len(y_unique) > 1 else 1.0
    
    # Calculate minimum step sizes (smallest gap between adjacent grid points)
    if len(x_unique) > 1:
        x_steps = np.diff(x_unique)
        x_min_step = np.min(x_steps)
        x_avg_step = np.mean(x_steps)
    else:
        x_min_step = x_range
        x_avg_step = x_range
    
    if len(y_unique) > 1:
        y_steps = np.diff(y_unique)
        y_min_step = np.min(y_steps)
        y_avg_step = np.mean(y_steps)
    else:
        y_min_step = y_range
        y_avg_step = y_range
    
    # Normalize step sizes to 0-1 range
    x_step_norm = x_min_step / (x_range + 1e-10)
    y_step_norm = y_min_step / (y_range + 1e-10)
    
    # Calculate diagonal distance to nearest neighbor in normalized space
    # This is the Euclidean distance to a diagonal neighbor
    min_diagonal_dist = np.sqrt(x_step_norm**2 + y_step_norm**2)
    
    # Also calculate orthogonal distance (horizontal/vertical neighbor)
    min_orthogonal_dist = min(x_step_norm, y_step_norm)
    
    # Set radius to include neighbors based on multiplier
    # multiplier=1.0 would just barely include orthogonal neighbors
    # multiplier=1.5 ensures we capture all 8 immediate neighbors (including diagonal)
    # multiplier=2.0 captures the next ring of neighbors
    dynamic_radius = min_diagonal_dist * multiplier
    
    # Ensure radius is reasonable (not too small or too large)
    dynamic_radius = max(dynamic_radius, 0.05)  # At least 5% of range
    dynamic_radius = min(dynamic_radius, 0.5)   # At most 50% of range
    
    details = {
        'x_range': x_range,
        'y_range': y_range,
        'x_min_step': x_min_step,
        'y_min_step': y_min_step,
        'x_step_normalized': x_step_norm,
        'y_step_normalized': y_step_norm,
        'min_orthogonal_dist': min_orthogonal_dist,
        'min_diagonal_dist': min_diagonal_dist,
        'multiplier': multiplier,
        'calculated_radius': dynamic_radius,
        'x_unique_count': len(x_unique),
        'y_unique_count': len(y_unique),
    }
    
    return dynamic_radius, details

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MAOptimization3DVisualizer:
    """
    Creates 3D visualizations for Moving Average Crossover optimization.
    """
    
    def __init__(self, data: pd.DataFrame, trading_fee: float = 0.0, auto_open: bool = False, output_dir: str = "ma_optimization_plots", percentile_threshold: float = DEFAULT_OPTIMAL_PERCENTILE):
        """
        Initialize the 3D visualizer.
        
        Args:
            data: OHLCV data for backtesting
            trading_fee: Trading fee as decimal
            auto_open: Whether to auto-open plots in browser
            output_dir: Directory to save plots
        """
        # Normalize column names to expected OHLCV schema to avoid KeyErrors like 'Close'
        normalized = data.copy()
        try:
            lower_to_actual = {str(c).lower(): c for c in normalized.columns}
            rename_map = {}
            if 'open' in lower_to_actual:
                rename_map[lower_to_actual['open']] = 'Open'
            if 'high' in lower_to_actual:
                rename_map[lower_to_actual['high']] = 'High'
            if 'low' in lower_to_actual:
                rename_map[lower_to_actual['low']] = 'Low'
            if 'close' in lower_to_actual:
                rename_map[lower_to_actual['close']] = 'Close'
            if 'volume' in lower_to_actual:
                rename_map[lower_to_actual['volume']] = 'Volume'
            if rename_map:
                normalized = normalized.rename(columns=rename_map)
        except Exception:
            pass
        self.data = normalized
        self.trading_fee = trading_fee
        self.results = {}
        self.auto_open = auto_open
        self.output_dir = output_dir
        self.percentile_threshold = percentile_threshold
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate_ma_combination(self, short_window: int, long_window: int, 
                               risk_reward_ratio: float) -> Dict[str, Any]:
        """
        Evaluate a single MA combination and return metrics.
        
        Args:
            short_window: Short moving average window
            long_window: Long moving average window
            risk_reward_ratio: Risk-reward ratio
            
        Returns:
            Dictionary containing strategy metrics
        """
        try:
            # Validate parameters
            if short_window >= long_window:
                return {'composite_score': -999, 'sharpe_ratio': -999, 'total_pnl': -999, 'total_trades': 0}
            
            # Create strategy instance
            strategy = MovingAverageCrossover(
                short_window=short_window,
                long_window=long_window,
                risk_reward_ratio=risk_reward_ratio,
                trading_fee=self.trading_fee
            )
            
            # Generate signals
            signals = strategy.generate_signals(self.data)
            
            # Get strategy metrics
            metrics = strategy.get_strategy_metrics()
            
            # Calculate composite score (same as in StrategyOptimizer)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            calmar_ratio = metrics.get('calmar_ratio', 0)
            profit_factor = metrics.get('profit_factor', 0)
            win_rate = metrics.get('win_rate', 0)
            geometric_mean_return = metrics.get('geometric_mean_return', 0)
            max_drawdown = metrics.get('max_drawdown', 0)
            
            # Handle NaN and infinite values
            if np.isinf(profit_factor) or np.isnan(profit_factor):
                profit_factor = 0
            if np.isinf(calmar_ratio) or np.isnan(calmar_ratio):
                calmar_ratio = 0
            if np.isinf(sharpe_ratio) or np.isnan(sharpe_ratio):
                sharpe_ratio = 0
            
            # CRITICAL BUG FIX: Normalize metrics to comparable scales (0-1 range)
            # Without normalization, calmar_ratio can dominate (can be 1000+)
            # while win_rate is only 0-1
            
            # Normalize sharpe_ratio: typical range -3 to +3, map to 0-1
            sharpe_normalized = np.clip((sharpe_ratio + 3) / 6, 0, 1)
            
            # Normalize calmar_ratio: typical range 0 to 10, map to 0-1
            # Clamp extreme values to prevent domination
            calmar_normalized = np.clip(calmar_ratio / 10, 0, 1)
            
            # Normalize profit_factor: typical range 0 to 3, map to 0-1
            # Values above 3 are excellent but shouldn't dominate
            profit_factor_normalized = np.clip(profit_factor / 3, 0, 1)
            
            # win_rate is already 0-1
            win_rate_normalized = np.clip(win_rate, 0, 1)
            
            # Normalize geometric_mean_return: typical range -0.05 to +0.05, map to 0-1
            gmr_normalized = np.clip((geometric_mean_return + 0.05) / 0.10, 0, 1)
            
            # max_drawdown is 0-1, invert so lower drawdown = higher score
            drawdown_score = np.clip(1 - max_drawdown, 0, 1)
            
            # Calculate composite score with normalized metrics
            composite_score = (
                0.25 * sharpe_normalized +           # Risk-adjusted returns
                0.20 * calmar_normalized +           # Return vs drawdown
                0.15 * profit_factor_normalized +    # Profit efficiency
                0.15 * win_rate_normalized +         # Consistency
                0.15 * gmr_normalized +              # Compound growth
                0.10 * drawdown_score                # Drawdown penalty
            )
            
            return {
                'composite_score': composite_score,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'profit_factor': profit_factor,
                'win_rate': win_rate,
                'total_pnl': metrics.get('total_pnl', 0),
                'total_trades': metrics.get('total_trades', 0),
                'max_drawdown': max_drawdown,
                'geometric_mean_return': geometric_mean_return
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating MA combination ({short_window}, {long_window}, {risk_reward_ratio}): {e}")
            return {'composite_score': -999, 'sharpe_ratio': -999, 'total_pnl': -999, 'total_trades': 0}
    
    def run_optimization_grid(self, short_window_range: List[int], 
                             long_window_range: List[int], 
                             risk_reward_ratios: List[float]) -> Dict:
        """
        Run optimization for all combinations and store results.
        
        Args:
            short_window_range: List of short window values to test
            long_window_range: List of long window values to test
            risk_reward_ratios: List of risk-reward ratios to test
            
        Returns:
            Dictionary containing results for each risk_reward_ratio
        """
        print("🔍 Running optimization grid for 3D visualization...")
        print(f"   Short windows: {short_window_range}")
        print(f"   Long windows: {long_window_range}")
        print(f"   Risk-reward ratios: {risk_reward_ratios}")
        
        total_combinations = len(short_window_range) * len(long_window_range) * len(risk_reward_ratios)
        print(f"   Total combinations to test: {total_combinations}")
        
        results = {}
        
        for rr in risk_reward_ratios:
            print(f"\n📊 Processing risk_reward_ratio = {rr}")
            rr_results = {
                'short_windows': [],
                'long_windows': [],
                'composite_score': [],
                'sharpe_ratio': [],
                'total_pnl': [],
                'total_trades': [],
                'win_rate': [],
                'max_drawdown': []
            }
            
            for short_w in short_window_range:
                for long_w in long_window_range:
                    if short_w < long_w:  # Valid combination
                        metrics = self.evaluate_ma_combination(short_w, long_w, rr)
                        
                        rr_results['short_windows'].append(short_w)
                        rr_results['long_windows'].append(long_w)
                        rr_results['composite_score'].append(metrics['composite_score'])
                        rr_results['sharpe_ratio'].append(metrics['sharpe_ratio'])
                        rr_results['total_pnl'].append(metrics['total_pnl'])
                        rr_results['total_trades'].append(metrics['total_trades'])
                        rr_results['win_rate'].append(metrics['win_rate'])
                        rr_results['max_drawdown'].append(metrics['max_drawdown'])
            
            results[rr] = rr_results
            print(f"   ✅ Completed {len(rr_results['short_windows'])} valid combinations")
        
        self.results = results
        return results
    
    def calculate_neighborhood_aware_scores(self, metric: str = 'composite_score', 
                                           neighborhood_radius: float = None,
                                           own_score_weight: float = None,
                                           neighborhood_weight: float = None,
                                           negative_penalty_weight: float = None,
                                           distance_weight_power: float = None) -> Dict:
        """
        Calculate neighborhood-aware composite scores for each point.
        
        This method assigns each point a score based on:
        1. Its own score
        2. The quality of its neighborhood (average of nearby high-scoring points, weighted by distance)
        3. Penalty for negative scores nearby
        4. Reward for being in a smooth, high-scoring region
        
        Args:
            metric: Metric to use for scoring ('composite_score', 'sharpe_ratio', etc.)
            neighborhood_radius: Maximum distance to consider for neighborhood (in normalized space, 0-1 range)
                                 A radius of 0.2 means 20% of the parameter range in normalized space.
                                 This adapts automatically to different parameter ranges.
                                 Recommended values:
                                 - 0.1-0.15: Very tight neighborhood (few neighbors, very local)
                                 - 0.2-0.3: Moderate neighborhood (default: 0.2)
                                 - 0.4-0.5: Large neighborhood (many neighbors, broader region)
                                 
                                 Example: If short_window ranges 16-34 (range=18) and radius=0.2,
                                 it considers points within ~3.6 units in short_window direction.
                                 This ensures both dimensions contribute equally regardless of their ranges.
            own_score_weight: Weight for the point's own score (0-1)
            neighborhood_weight: Weight for neighborhood average score (0-1)
            negative_penalty_weight: Weight for penalty from negative scores nearby (0-1)
            distance_weight_power: Power for distance weighting (higher = closer neighbors weighted much more)
                                  Uses 1/distance^power. Default 2.0 means squared inverse distance.
            
        Returns:
            Dictionary with neighborhood-aware scores and best parameters for each risk_reward_ratio
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return {}
        
        # Use config defaults if not specified
        if neighborhood_radius is None:
            neighborhood_radius = NEIGHBORHOOD_RADIUS
        if own_score_weight is None:
            own_score_weight = OWN_SCORE_WEIGHT
        if neighborhood_weight is None:
            neighborhood_weight = NEIGHBORHOOD_WEIGHT
        if negative_penalty_weight is None:
            negative_penalty_weight = NEGATIVE_PENALTY_WEIGHT
        if distance_weight_power is None:
            distance_weight_power = DISTANCE_WEIGHT_POWER
        
        # Track if we're using auto radius
        use_auto_radius = (neighborhood_radius == "auto" or 
                          (isinstance(neighborhood_radius, str) and neighborhood_radius.lower() == "auto"))
        
        neighborhood_scores = {}
        
        for rr, data in self.results.items():
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) < 2:
                continue
            
            # Filter out invalid scores
            valid_mask = z > -999
            if not np.any(valid_mask):
                continue
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            z_valid = z[valid_mask]
            
            # Calculate neighborhood-aware scores
            # We use NORMALIZED distance for neighbor selection to ensure both dimensions are treated equally
            neighborhood_aware_scores = []
            neighborhood_details = []
            
            # Normalize coordinates (scale each dimension to 0-1 range)
            # This ensures both short_window and long_window contribute equally
            x_range = x_valid.max() - x_valid.min()
            y_range = y_valid.max() - y_valid.min()
            x_norm = (x_valid - x_valid.min()) / (x_range + 1e-10)
            y_norm = (y_valid - y_valid.min()) / (y_range + 1e-10)
            
            # Calculate dynamic radius if auto mode is enabled
            if use_auto_radius:
                actual_radius, radius_details = calculate_dynamic_neighborhood_radius(x_valid, y_valid)
                if rr == list(self.results.keys())[0]:  # Print only once
                    print(f"\n📐 AUTO NEIGHBORHOOD RADIUS CALCULATION:")
                    print(f"   Grid: {radius_details['x_unique_count']} x {radius_details['y_unique_count']} points")
                    print(f"   X step (normalized): {radius_details['x_step_normalized']:.4f}")
                    print(f"   Y step (normalized): {radius_details['y_step_normalized']:.4f}")
                    print(f"   Min diagonal distance: {radius_details['min_diagonal_dist']:.4f}")
                    print(f"   Multiplier: {radius_details['multiplier']}")
                    print(f"   → Calculated radius: {actual_radius:.4f}")
            else:
                actual_radius = neighborhood_radius
            
            for i in range(len(x_valid)):
                # Calculate Euclidean distances in NORMALIZED space (0-1 range)
                # This creates a circle in normalized space, which adapts to parameter ranges
                distances_norm = np.sqrt((x_norm - x_norm[i])**2 + (y_norm - y_norm[i])**2)
                
                # Also calculate original distances for reporting
                distances_orig = np.sqrt((x_valid - x_valid[i])**2 + (y_valid - y_valid[i])**2)
                
                # Find neighbors within radius in NORMALIZED space
                # actual_radius is either auto-calculated or from config (fraction of parameter range, 0-1)
                neighbor_mask = (distances_norm <= actual_radius) & (distances_norm > 0)  # Exclude self
                
                if not np.any(neighbor_mask):
                    # No neighbors, use own score only
                    neighborhood_aware_scores.append(z_valid[i])
                    neighborhood_details.append({
                        'own_score': z_valid[i],
                        'neighborhood_avg': z_valid[i],
                        'neighborhood_std': 0.0,
                        'negative_count': 0,
                        'positive_count': 0,
                        'neighbor_count': 0,
                        'final_score': z_valid[i],
                        'avg_neighbor_distance': 0.0
                    })
                    continue
                
                neighbor_scores = z_valid[neighbor_mask]
                neighbor_distances = distances_norm[neighbor_mask]
                
                # Calculate distance-weighted average (closer neighbors have MUCH more weight)
                # Use inverse distance weighting with power: weight = 1 / (distance^power)
                # Higher power means closer points get exponentially more weight
                # distance_weight_power=2.0 means squared inverse distance (strong preference for close neighbors)
                # distance_weight_power=1.0 means linear inverse distance (moderate preference)
                # distance_weight_power=3.0 means cubic inverse distance (very strong preference)
                weights = 1.0 / (neighbor_distances ** distance_weight_power + 1e-10)
                
                # Normalize weights to sum to 1 for proper weighted average
                weights_sum = np.sum(weights)
                if weights_sum > 0:
                    weights = weights / weights_sum
                
                weighted_avg = np.average(neighbor_scores, weights=weights)
                
                # Calculate neighborhood statistics
                neighborhood_std = np.std(neighbor_scores)
                negative_count = np.sum(neighbor_scores < 0)
                positive_count = np.sum(neighbor_scores > 0)
                neighbor_count = len(neighbor_scores)
                
                # Calculate components
                own_score = z_valid[i]
                
                # Neighborhood bonus: reward high average scores nearby
                # Normalize by the max score to keep it in reasonable range
                max_score = np.max(z_valid) if len(z_valid) > 0 else 1.0
                neighborhood_bonus = (weighted_avg / max_score) * neighborhood_weight if max_score > 0 else 0
                
                # Smoothness bonus: reward low standard deviation (smooth region)
                # Lower std = smoother = better
                smoothness_bonus = (1.0 / (1.0 + neighborhood_std)) * 0.1 if neighborhood_std > 0 else 0.1
                
                # Negative penalty: penalize if there are negative scores nearby
                negative_penalty = (negative_count / neighbor_count) * negative_penalty_weight if neighbor_count > 0 else 0
                
                # Positive bonus: reward if surrounded by positive scores
                positive_bonus = (positive_count / neighbor_count) * 0.05 if neighbor_count > 0 else 0
                
                # Calculate final neighborhood-aware score
                # Normalize own score to 0-1 range for fair weighting
                own_score_norm = (own_score / max_score) * own_score_weight if max_score > 0 else 0
                
                final_score = (
                    own_score_norm +
                    neighborhood_bonus +
                    smoothness_bonus -
                    negative_penalty +
                    positive_bonus
                )
                
                # Scale back to original score range for interpretability
                # But keep the relative improvements from neighborhood
                scaled_score = own_score + (final_score - own_score_norm) * max_score
                
                neighborhood_aware_scores.append(scaled_score)
                neighborhood_details.append({
                    'own_score': own_score,
                    'neighborhood_avg': weighted_avg,
                    'neighborhood_std': neighborhood_std,
                    'negative_count': negative_count,
                    'positive_count': positive_count,
                    'neighbor_count': neighbor_count,
                    'final_score': scaled_score,
                    'avg_neighbor_distance': float(np.mean(neighbor_distances)) if len(neighbor_distances) > 0 else 0.0
                })
            
            # Find best point based on neighborhood-aware score
            best_idx = np.argmax(neighborhood_aware_scores)
            
            neighborhood_scores[rr] = {
                'short_windows': x_valid.tolist(),
                'long_windows': y_valid.tolist(),
                'original_scores': z_valid.tolist(),
                'neighborhood_aware_scores': neighborhood_aware_scores,
                'neighborhood_details': neighborhood_details,
                'best_idx': int(best_idx),
                'best_short': int(x_valid[best_idx]),
                'best_long': int(y_valid[best_idx]),
                'best_original_score': float(z_valid[best_idx]),
                'best_neighborhood_score': float(neighborhood_aware_scores[best_idx]),
                'best_details': neighborhood_details[best_idx]
            }
        
        return neighborhood_scores
    
    def find_optimal_parameters_neighborhood_aware(self, metric: str = 'composite_score',
                                                  neighborhood_radius: float = None,
                                                  own_score_weight: float = None,
                                                  neighborhood_weight: float = None,
                                                  negative_penalty_weight: float = None,
                                                  distance_weight_power: float = None) -> Dict:
        """
        Find optimal parameters using neighborhood-aware scoring.
        
        This is the main method to call for automated parameter selection.
        It finds the best parameters considering both individual scores and neighborhood quality.
        
        Args:
            metric: Metric to use for scoring
            neighborhood_radius: Maximum distance for neighborhood consideration
            own_score_weight: Weight for point's own score
            neighborhood_weight: Weight for neighborhood average
            negative_penalty_weight: Weight for negative score penalty
            distance_weight_power: Power for distance weighting (higher = closer neighbors weighted much more)
            
        Returns:
            Dictionary with best parameters for each risk_reward_ratio and overall best
        """
        neighborhood_scores = self.calculate_neighborhood_aware_scores(
            metric=metric,
            neighborhood_radius=neighborhood_radius,
            own_score_weight=own_score_weight,
            neighborhood_weight=neighborhood_weight,
            negative_penalty_weight=negative_penalty_weight,
            distance_weight_power=distance_weight_power
        )
        
        if not neighborhood_scores:
            return {}
        
        # Find overall best across all risk_reward_ratios
        overall_best = None
        overall_best_score = -np.inf
        
        results_summary = {}
        
        for rr, scores_data in neighborhood_scores.items():
            best_score = scores_data['best_neighborhood_score']
            results_summary[rr] = {
                'short_window': scores_data['best_short'],
                'long_window': scores_data['best_long'],
                'risk_reward_ratio': rr,
                'original_score': scores_data['best_original_score'],
                'neighborhood_aware_score': best_score,
                'neighborhood_details': scores_data['best_details']
            }
            
            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best = results_summary[rr]
        
        return {
            'by_risk_reward_ratio': results_summary,
            'overall_best': overall_best,
            'all_scores': neighborhood_scores
        }
    
    def print_neighborhood_aware_recommendations(self, metric: str = 'composite_score',
                                                neighborhood_radius: float = None,
                                                distance_weight_power: float = None) -> None:
        """
        Print recommendations based on neighborhood-aware scoring.
        
        Args:
            metric: Metric to use for scoring
            neighborhood_radius: Maximum distance for neighborhood consideration (uses config default if None)
            distance_weight_power: Power for distance weighting (uses config default if None)
        """
        # Use config defaults if not specified
        if neighborhood_radius is None:
            neighborhood_radius = NEIGHBORHOOD_RADIUS
        if distance_weight_power is None:
            distance_weight_power = DISTANCE_WEIGHT_POWER
            
        recommendations = self.find_optimal_parameters_neighborhood_aware(
            metric=metric,
            neighborhood_radius=neighborhood_radius,
            distance_weight_power=distance_weight_power
        )
        
        if not recommendations:
            print("❌ No recommendations available.")
            return
        
        print("\n" + "="*80)
        print("NEIGHBORHOOD-AWARE PARAMETER RECOMMENDATIONS")
        print("="*80)
        print(f"\n📊 Scoring Method: Neighborhood-Aware Composite Score")
        print(f"   - Considers point's own score")
        print(f"   - Rewards high-scoring neighborhoods (distance-weighted)")
        print(f"   - Closer neighbors have MUCH more weight (1/distance^{distance_weight_power})")
        print(f"   - Penalizes negative scores nearby")
        print(f"   - Rewards smooth, consistent regions")
        if neighborhood_radius == "auto" or (isinstance(neighborhood_radius, str) and neighborhood_radius.lower() == "auto"):
            print(f"   - Neighborhood radius: AUTO (dynamically calculated based on grid spacing)")
            print(f"   - Radius multiplier: {NEIGHBORHOOD_RADIUS_MULTIPLIER}x diagonal step distance")
        else:
            print(f"   - Neighborhood radius: {neighborhood_radius} (normalized, 0-1 range, {neighborhood_radius*100:.0f}% of parameter range)")
        print(f"   - Distance weight power: {distance_weight_power} (higher = stronger preference for close neighbors)")
        
        print("\n" + "-"*80)
        print("BEST PARAMETERS BY RISK-REWARD RATIO:")
        print("-"*80)
        
        for rr, result in recommendations['by_risk_reward_ratio'].items():
            details = result['neighborhood_details']
            print(f"\n🎯 Risk-Reward Ratio: {rr}")
            print(f"   Recommended Parameters:")
            print(f"      Short Window: {result['short_window']}")
            print(f"      Long Window: {result['long_window']}")
            print(f"   Scores:")
            print(f"      Original Score: {result['original_score']:.4f}")
            print(f"      Neighborhood-Aware Score: {result['neighborhood_aware_score']:.4f}")
            print(f"   Neighborhood Quality:")
            print(f"      Neighbors: {details.get('neighbor_count', 0)}")
            print(f"      Positive Neighbors: {details.get('positive_count', 0)}")
            print(f"      Negative Neighbors: {details.get('negative_count', 0)}")
            print(f"      Neighborhood Avg Score: {details.get('neighborhood_avg', details.get('own_score', 0)):.4f}")
            print(f"      Neighborhood Std Dev: {details.get('neighborhood_std', 0.0):.4f} (lower = smoother)")
            if 'avg_neighbor_distance' in details:
                print(f"      Avg Neighbor Distance: {details['avg_neighbor_distance']:.4f}")
        
        if recommendations.get('overall_best'):
            best = recommendations['overall_best']
            print("\n" + "="*80)
            print("🌟 OVERALL BEST PARAMETERS (Across All Risk-Reward Ratios):")
            print("="*80)
            print(f"   Short Window: {best['short_window']}")
            print(f"   Long Window: {best['long_window']}")
            print(f"   Risk-Reward Ratio: {best['risk_reward_ratio']}")
            print(f"   Original Score: {best['original_score']:.4f}")
            print(f"   Neighborhood-Aware Score: {best['neighborhood_aware_score']:.4f}")
            print(f"   Neighborhood Quality: {best['neighborhood_details']}")
            print("="*80)
    
    def print_top_neighborhood_aware_points(self, metric: str = 'composite_score',
                                            top_n: int = 5,
                                            neighborhood_radius: float = None,
                                            distance_weight_power: float = None) -> None:
        """
        Print the top N points based on neighborhood-aware scores for each risk-reward ratio.
        
        Args:
            metric: Metric to use for scoring
            top_n: Number of top points to print (default: 5)
            neighborhood_radius: Maximum distance for neighborhood consideration (uses config default if None)
            distance_weight_power: Power for distance weighting (uses config default if None)
        """
        # Use config defaults if not specified
        if neighborhood_radius is None:
            neighborhood_radius = NEIGHBORHOOD_RADIUS
        if distance_weight_power is None:
            distance_weight_power = DISTANCE_WEIGHT_POWER
            
        neighborhood_scores = self.calculate_neighborhood_aware_scores(
            metric=metric,
            neighborhood_radius=neighborhood_radius,
            distance_weight_power=distance_weight_power
        )
        
        if not neighborhood_scores:
            print("❌ No neighborhood-aware scores available.")
            return
        
        print("\n" + "="*80)
        print(f"TOP {top_n} NEIGHBORHOOD-AWARE POINTS BY RISK-REWARD RATIO")
        print("="*80)
        
        for rr, scores_data in neighborhood_scores.items():
            # Get all points with their scores
            short_windows = scores_data['short_windows']
            long_windows = scores_data['long_windows']
            original_scores = scores_data['original_scores']
            neighborhood_scores_list = scores_data['neighborhood_aware_scores']
            details_list = scores_data['neighborhood_details']
            
            # Create list of tuples: (neighborhood_score, original_score, short, long, details, index)
            points = list(zip(
                neighborhood_scores_list,
                original_scores,
                short_windows,
                long_windows,
                details_list,
                range(len(short_windows))
            ))
            
            # Sort by neighborhood-aware score (descending)
            points.sort(key=lambda x: x[0], reverse=True)
            
            # Print top N
            print(f"\n🎯 Risk-Reward Ratio: {rr}")
            print("-" * 80)
            print(f"{'Rank':<6} {'Short':<8} {'Long':<8} {'Orig Score':<12} {'NA Score':<12} {'Neighbors':<12} {'Pos/Neg':<12}")
            print("-" * 80)
            
            for rank, (na_score, orig_score, short, long, details, idx) in enumerate(points[:top_n], 1):
                pos_count = details.get('positive_count', 0)
                neg_count = details.get('negative_count', 0)
                neighbor_count = details.get('neighbor_count', 0)
                pos_neg_str = f"{pos_count}/{neg_count}" if neighbor_count > 0 else "N/A"
                
                print(f"{rank:<6} {short:<8} {long:<8} {orig_score:<12.4f} {na_score:<12.4f} {neighbor_count:<12} {pos_neg_str:<12}")
            
            # Show additional details for top point
            if points:
                top_point = points[0]
                top_details = top_point[4]
                print(f"\n   📊 Top Point Details:")
                print(f"      Short Window: {top_point[2]}, Long Window: {top_point[3]}")
                print(f"      Original Score: {top_point[1]:.4f}")
                print(f"      Neighborhood-Aware Score: {top_point[0]:.4f}")
                print(f"      Neighborhood Avg Score: {top_details.get('neighborhood_avg', 0):.4f}")
                print(f"      Neighborhood Std Dev: {top_details.get('neighborhood_std', 0):.4f}")
                print(f"      Neighbors: {top_details.get('neighbor_count', 0)}")
                print(f"      Positive Neighbors: {top_details.get('positive_count', 0)}")
                print(f"      Negative Neighbors: {top_details.get('negative_count', 0)}")
                if 'avg_neighbor_distance' in top_details:
                    print(f"      Avg Neighbor Distance: {top_details['avg_neighbor_distance']:.4f}")
        
        print("\n" + "="*80)
    
    def print_overall_top_neighborhood_aware_points(self, metric: str = 'composite_score',
                                                    top_n: int = 5,
                                                    neighborhood_radius: float = None,
                                                    distance_weight_power: float = None) -> List[Dict]:
        """
        Print and return the overall top N neighborhood-aware points across ALL risk-reward ratios.
        
        This aggregates all parameter combinations from all RR ratios and ranks them by NA score,
        helping identify the most robust parameters regardless of RR setting.
        
        Args:
            metric: Metric to use for scoring
            top_n: Number of top points to show
            neighborhood_radius: Maximum distance for neighborhood consideration
            distance_weight_power: Power for distance weighting
            
        Returns:
            List of top N points with their details
        """
        # Use config defaults if not specified
        if neighborhood_radius is None:
            neighborhood_radius = NEIGHBORHOOD_RADIUS
        if distance_weight_power is None:
            distance_weight_power = DISTANCE_WEIGHT_POWER
        
        # Calculate NA scores for all RR ratios
        na_scores = self.calculate_neighborhood_aware_scores(
            metric=metric,
            neighborhood_radius=neighborhood_radius,
            distance_weight_power=distance_weight_power
        )
        
        if not na_scores:
            print("❌ No neighborhood-aware scores available.")
            return []
        
        # Aggregate all points across all RR ratios
        all_points = []
        
        for rr, data in na_scores.items():
            short_windows = data['short_windows']
            long_windows = data['long_windows']
            orig_scores = data['original_scores']
            na_score_list = data['neighborhood_aware_scores']
            details_list = data['neighborhood_details']
            
            for i in range(len(short_windows)):
                all_points.append({
                    'risk_reward_ratio': rr,
                    'short_window': short_windows[i],
                    'long_window': long_windows[i],
                    'original_score': orig_scores[i],
                    'na_score': na_score_list[i],
                    'details': details_list[i]
                })
        
        # Sort by NA score (descending)
        all_points.sort(key=lambda x: x['na_score'], reverse=True)
        
        # Get top N
        top_points = all_points[:top_n]
        
        # Print results
        print("\n" + "="*80)
        print(f"🏆 OVERALL TOP {top_n} NEIGHBORHOOD-AWARE PARAMETERS (Across All RR Ratios)")
        print("="*80)
        print(f"\n📊 Ranked by Neighborhood-Aware Score (higher = better & more robust)")
        print(f"   Total parameter combinations analyzed: {len(all_points)}")
        
        print("\n" + "-"*100)
        print(f"{'Rank':<6} {'Short':<8} {'Long':<8} {'RR':<8} {'Orig Score':<12} {'NA Score':<12} {'Neighbors':<10} {'Pos/Neg':<10} {'Std Dev':<10}")
        print("-"*100)
        
        for rank, point in enumerate(top_points, 1):
            details = point['details']
            neighbor_count = details.get('neighbor_count', 0)
            pos_count = details.get('positive_count', 0)
            neg_count = details.get('negative_count', 0)
            std_dev = details.get('neighborhood_std', 0)
            pos_neg_str = f"{pos_count}/{neg_count}" if neighbor_count > 0 else "N/A"
            
            print(f"{rank:<6} {point['short_window']:<8} {point['long_window']:<8} {point['risk_reward_ratio']:<8} "
                  f"{point['original_score']:<12.4f} {point['na_score']:<12.4f} {neighbor_count:<10} "
                  f"{pos_neg_str:<10} {std_dev:<10.4f}")
        
        # Show detailed breakdown for #1
        if top_points:
            best = top_points[0]
            details = best['details']
            
            print("\n" + "="*80)
            print("🥇 RECOMMENDED PARAMETERS (Best Overall)")
            print("="*80)
            print(f"\n   📈 Parameter Settings:")
            print(f"      Short Window: {best['short_window']}")
            print(f"      Long Window: {best['long_window']}")
            print(f"      Risk-Reward Ratio: {best['risk_reward_ratio']}")
            
            print(f"\n   📊 Performance Scores:")
            print(f"      Original Composite Score: {best['original_score']:.4f}")
            print(f"      Neighborhood-Aware Score: {best['na_score']:.4f}")
            
            print(f"\n   🏘️  Neighborhood Quality:")
            print(f"      Total Neighbors: {details.get('neighbor_count', 0)}")
            print(f"      Positive Neighbors: {details.get('positive_count', 0)}")
            print(f"      Negative Neighbors: {details.get('negative_count', 0)}")
            print(f"      Neighborhood Avg Score: {details.get('neighborhood_avg', 0):.4f}")
            print(f"      Neighborhood Std Dev: {details.get('neighborhood_std', 0):.4f} (lower = more stable)")
            if 'avg_neighbor_distance' in details:
                print(f"      Avg Neighbor Distance: {details['avg_neighbor_distance']:.4f}")
            
            # Check if this point appears in multiple RR ratios' top results
            similar_params = [p for p in all_points 
                           if p['short_window'] == best['short_window'] 
                           and p['long_window'] == best['long_window']]
            if len(similar_params) > 1:
                print(f"\n   🔄 Cross-RR Consistency:")
                print(f"      This parameter combo appears in {len(similar_params)} RR ratios:")
                for p in similar_params:
                    print(f"         RR={p['risk_reward_ratio']}: NA Score={p['na_score']:.4f}")
        
        # Show parameter frequency analysis
        print("\n" + "-"*80)
        print("📊 PARAMETER FREQUENCY IN TOP RESULTS:")
        print("-"*80)
        
        # Count frequency of short/long windows in top results
        from collections import Counter
        short_counts = Counter([p['short_window'] for p in top_points])
        long_counts = Counter([p['long_window'] for p in top_points])
        rr_counts = Counter([p['risk_reward_ratio'] for p in top_points])
        
        print(f"   Most common Short Windows: {dict(short_counts.most_common(3))}")
        print(f"   Most common Long Windows: {dict(long_counts.most_common(3))}")
        print(f"   Most common RR Ratios: {dict(rr_counts.most_common(3))}")
        
        print("\n" + "="*80)
        
        return top_points
    
    def create_3d_plots(self, metric: str = 'composite_score') -> None:
        """
        Create interactive 3D plots for each risk_reward_ratio using Plotly.
        
        Args:
            metric: Metric to plot ('composite_score', 'sharpe_ratio', 'total_pnl', etc.)
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return
        
        risk_reward_ratios = list(self.results.keys())
        n_plots = len(risk_reward_ratios)
        
        # Calculate subplot grid
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            specs=[[{'type': 'scatter3d'} for _ in range(n_cols)] for _ in range(n_rows)],
            subplot_titles=[f'Risk-Reward Ratio = {rr}' for rr in risk_reward_ratios],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, rr in enumerate(risk_reward_ratios):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            data = self.results[rr]
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) > 0:
                # Create hover text with detailed information
                hover_text = []
                for j in range(len(x)):
                    hover_info = (
                        f"Short Window: {x[j]}<br>"
                        f"Long Window: {y[j]}<br>"
                        f"{metric.replace('_', ' ').title()}: {z[j]:.4f}<br>"
                        f"Sharpe Ratio: {data['sharpe_ratio'][j]:.4f}<br>"
                        f"Total PnL: {data['total_pnl'][j]:.2%}<br>"
                        f"Win Rate: {data['win_rate'][j]:.2%}<br>"
                        f"Total Trades: {data['total_trades'][j]}<br>"
                        f"Max Drawdown: {data['max_drawdown'][j]:.2%}"
                    )
                    hover_text.append(hover_info)
                
                # Add scatter plot
                scatter = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=z,
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title=metric.replace('_', ' ').title()) if i == 0 else None,
                        showscale=True if i == 0 else False
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    name=f'RR={rr}',
                    showlegend=False
                )
                
                fig.add_trace(scatter, row=row, col=col)
                
                # Find and highlight the best point
                if len(z) > 0 and not np.all(np.isnan(z)):
                    best_idx = np.nanargmax(z)
                    best_hover = (
                        f"<b>BEST POINT</b><br>"
                        f"Short Window: {x[best_idx]}<br>"
                        f"Long Window: {y[best_idx]}<br>"
                        f"{metric.replace('_', ' ').title()}: {z[best_idx]:.4f}<br>"
                        f"Sharpe Ratio: {data['sharpe_ratio'][best_idx]:.4f}<br>"
                        f"Total PnL: {data['total_pnl'][best_idx]:.2%}<br>"
                        f"Win Rate: {data['win_rate'][best_idx]:.2%}<br>"
                        f"Total Trades: {data['total_trades'][best_idx]}<br>"
                        f"Max Drawdown: {data['max_drawdown'][best_idx]:.2%}"
                    )
                    
                    best_scatter = go.Scatter3d(
                        x=[x[best_idx]], y=[y[best_idx]], z=[z[best_idx]],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='diamond',
                            opacity=1.0
                        ),
                        text=[best_hover],
                        hovertemplate='%{text}<extra></extra>',
                        name=f'Best RR={rr}',
                        showlegend=False
                    )
                    
                    fig.add_trace(best_scatter, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=f'Moving Average Crossover Optimization - {metric.replace("_", " ").title()}',
            title_x=0.5,
            height=300 * n_rows,
            width=400 * n_cols,
            showlegend=False
        )
        
        # Update axes labels
        for i in range(n_plots):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.update_scenes(
                xaxis_title="Short Window",
                yaxis_title="Long Window", 
                zaxis_title=metric.replace('_', ' ').title(),
                row=row, col=col
            )
        
        # Save as HTML file
        filename = os.path.join(self.output_dir, f"ma_3d_plots_{metric.replace('_', '_')}.html")
        plot(fig, filename=filename, auto_open=self.auto_open)
        print(f"📁 3D plots saved as: {filename}")
        if self.auto_open:
            print(f"🌐 Opening in browser...")
    
    def create_individual_3d_plots(self, metric: str = 'composite_score') -> None:
        """
        Create individual 3D plots for each risk_reward_ratio (one plot per ratio).
        
        Args:
            metric: Metric to plot
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return
        
        for rr, data in self.results.items():
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) == 0:
                continue
            
            # Create hover text with detailed information
            hover_text = []
            for j in range(len(x)):
                hover_info = (
                    f"Short Window: {x[j]}<br>"
                    f"Long Window: {y[j]}<br>"
                    f"{metric.replace('_', ' ').title()}: {z[j]:.4f}<br>"
                    f"Sharpe Ratio: {data['sharpe_ratio'][j]:.4f}<br>"
                    f"Total PnL: {data['total_pnl'][j]:.2%}<br>"
                    f"Win Rate: {data['win_rate'][j]:.2%}<br>"
                    f"Total Trades: {data['total_trades'][j]}<br>"
                    f"Max Drawdown: {data['max_drawdown'][j]:.2%}<br>"
                    f"Risk-Reward Ratio: {rr}"
                )
                hover_text.append(hover_info)
            
            # Create scatter plot
            scatter = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=10,
                    color=z,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title=metric.replace('_', ' ').title())
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name=f'Data Points'
            )
            
            # Find and highlight the best point
            best_idx = np.nanargmax(z)
            best_hover = (
                f"<b>BEST POINT</b><br>"
                f"Short Window: {x[best_idx]}<br>"
                f"Long Window: {y[best_idx]}<br>"
                f"{metric.replace('_', ' ').title()}: {z[best_idx]:.4f}<br>"
                f"Sharpe Ratio: {data['sharpe_ratio'][best_idx]:.4f}<br>"
                f"Total PnL: {data['total_pnl'][best_idx]:.2%}<br>"
                f"Win Rate: {data['win_rate'][best_idx]:.2%}<br>"
                f"Total Trades: {data['total_trades'][best_idx]}<br>"
                f"Max Drawdown: {data['max_drawdown'][best_idx]:.2%}<br>"
                f"Risk-Reward Ratio: {rr}"
            )
            
            best_scatter = go.Scatter3d(
                x=[x[best_idx]], y=[y[best_idx]], z=[z[best_idx]],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='diamond',
                    opacity=1.0
                ),
                text=[best_hover],
                hovertemplate='%{text}<extra></extra>',
                name=f'Best Point'
            )
            
            # Create figure
            fig = go.Figure(data=[scatter, best_scatter])
            
            # Update layout
            fig.update_layout(
                title=f'MA Optimization - Risk-Reward Ratio {rr} - {metric.replace("_", " ").title()}',
                scene=dict(
                    xaxis_title="Short Window",
                    yaxis_title="Long Window",
                    zaxis_title=metric.replace('_', ' ').title()
                ),
                width=800,
                height=600
            )
            
            # Save as HTML file
            filename = os.path.join(self.output_dir, f"ma_individual_3d_plot_rr_{rr}_{metric.replace('_', '_')}.html")
            plot(fig, filename=filename, auto_open=self.auto_open)
            print(f"📁 Individual 3D plot for RR={rr} saved as: {filename}")
    
    def create_2d_heatmaps(self, metric: str = 'composite_score') -> None:
        """
        Create interactive 2D heatmaps for each risk_reward_ratio using Plotly.
        
        Args:
            metric: Metric to plot
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return
        
        risk_reward_ratios = list(self.results.keys())
        n_plots = len(risk_reward_ratios)
        
        # Calculate subplot grid
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f'Risk-Reward Ratio = {rr}' for rr in risk_reward_ratios],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, rr in enumerate(risk_reward_ratios):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            data = self.results[rr]
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) > 0:
                # Create pivot table for heatmap
                df_heatmap = pd.DataFrame({
                    'short_window': x,
                    'long_window': y,
                    metric: z
                })
                
                pivot_table = df_heatmap.pivot(index='long_window', columns='short_window', values=metric)
                
                # Create hover text
                hover_text = []
                for j in range(len(pivot_table.index)):
                    row_text = []
                    for k in range(len(pivot_table.columns)):
                        value = pivot_table.iloc[j, k]
                        if not np.isnan(value):
                            hover_info = (
                                f"Short Window: {pivot_table.columns[k]}<br>"
                                f"Long Window: {pivot_table.index[j]}<br>"
                                f"{metric.replace('_', ' ').title()}: {value:.4f}<br>"
                                f"Risk-Reward Ratio: {rr}"
                            )
                            row_text.append(hover_info)
                        else:
                            row_text.append("")
                    hover_text.append(row_text)
                
                # Create heatmap
                heatmap = go.Heatmap(
                    z=pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    colorscale='Viridis',
                    text=hover_text,
                    texttemplate="%{text}",
                    hovertemplate='%{text}<extra></extra>',
                    showscale=True if i == 0 else False,
                    colorbar=dict(title=metric.replace('_', ' ').title()) if i == 0 else None
                )
                
                fig.add_trace(heatmap, row=row, col=col)
                
                # Find and highlight the best point
                best_idx = np.nanargmax(z)
                best_short = x[best_idx]
                best_long = y[best_idx]
                
                # Find position in pivot table
                short_pos = list(pivot_table.columns).index(best_short)
                long_pos = list(pivot_table.index).index(best_long)
                
                # Add annotation for best point
                fig.add_annotation(
                    x=best_short, y=best_long,
                    text="★ BEST",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    ax=0, ay=-30,
                    font=dict(color="red", size=12),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title=f'Moving Average Crossover Optimization - {metric.replace("_", " ").title()} Heatmaps',
            title_x=0.5,
            height=300 * n_rows,
            width=400 * n_cols
        )
        
        # Update axes labels
        for i in range(n_plots):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.update_xaxes(title_text="Short Window", row=row, col=col)
            fig.update_yaxes(title_text="Long Window", row=row, col=col)
        
        # Save as HTML file
        filename = os.path.join(self.output_dir, f"ma_2d_heatmaps_{metric.replace('_', '_')}.html")
        plot(fig, filename=filename, auto_open=self.auto_open)
        print(f"📁 2D heatmaps saved as: {filename}")
        print(f"🌐 Opening in browser...")
    
    def create_summary_plot(self, metric: str = 'composite_score') -> None:
        """
        Create a single comprehensive plot showing all risk-reward ratios with different colors.
        
        Args:
            metric: Metric to plot
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return
        
        fig = go.Figure()
        
        # Color palette for different risk-reward ratios
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        
        for i, (rr, data) in enumerate(self.results.items()):
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) == 0:
                continue
            
            # Create hover text
            hover_text = []
            for j in range(len(x)):
                hover_info = (
                    f"Risk-Reward Ratio: {rr}<br>"
                    f"Short Window: {x[j]}<br>"
                    f"Long Window: {y[j]}<br>"
                    f"{metric.replace('_', ' ').title()}: {z[j]:.4f}<br>"
                    f"Sharpe Ratio: {data['sharpe_ratio'][j]:.4f}<br>"
                    f"Total PnL: {data['total_pnl'][j]:.2%}<br>"
                    f"Win Rate: {data['win_rate'][j]:.2%}<br>"
                    f"Total Trades: {data['total_trades'][j]}<br>"
                    f"Max Drawdown: {data['max_drawdown'][j]:.2%}"
                )
                hover_text.append(hover_info)
            
            # Add scatter plot for this risk-reward ratio
            scatter = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name=f'RR={rr}',
                showlegend=True
            )
            
            fig.add_trace(scatter)
            
            # Find and highlight the best point for this ratio
            if len(z) > 0 and not np.all(np.isnan(z)):
                best_idx = np.nanargmax(z)
                best_hover = (
                    f"<b>BEST POINT (RR={rr})</b><br>"
                    f"Short Window: {x[best_idx]}<br>"
                    f"Long Window: {y[best_idx]}<br>"
                    f"{metric.replace('_', ' ').title()}: {z[best_idx]:.4f}<br>"
                    f"Sharpe Ratio: {data['sharpe_ratio'][best_idx]:.4f}<br>"
                    f"Total PnL: {data['total_pnl'][best_idx]:.2%}<br>"
                    f"Win Rate: {data['win_rate'][best_idx]:.2%}<br>"
                    f"Total Trades: {data['total_trades'][best_idx]}<br>"
                    f"Max Drawdown: {data['max_drawdown'][best_idx]:.2%}"
                )
                
                best_scatter = go.Scatter3d(
                    x=[x[best_idx]], y=[y[best_idx]], z=[z[best_idx]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=colors[i % len(colors)],
                        symbol='diamond',
                        opacity=1.0
                    ),
                    text=[best_hover],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Best RR={rr}',
                    showlegend=True
                )
                
                fig.add_trace(best_scatter)
        
        # Update layout
        fig.update_layout(
            title=f'MA Optimization Summary - All Risk-Reward Ratios - {metric.replace("_", " ").title()}',
            scene=dict(
                xaxis_title="Short Window",
                yaxis_title="Long Window",
                zaxis_title=metric.replace('_', ' ').title()
            ),
            width=1000,
            height=800
        )
        
        # Save as HTML file
        filename = os.path.join(self.output_dir, f"ma_summary_plot_{metric.replace('_', '_')}.html")
        plot(fig, filename=filename, auto_open=self.auto_open)
        print(f"📁 Summary plot saved as: {filename}")
        print(f"🌐 Opening in browser...")
    
    def create_distribution_contour_plots(self, metric: str = 'composite_score') -> None:
        """
        Create contour plots showing the distribution of scores across parameter space.
        Uses Gaussian Process regression to interpolate between data points.
        
        Args:
            metric: Metric to plot
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return
        
        risk_reward_ratios = list(self.results.keys())
        n_plots = len(risk_reward_ratios)
        
        # Calculate subplot grid
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f'Risk-Reward Ratio = {rr}' for rr in risk_reward_ratios],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, rr in enumerate(risk_reward_ratios):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            data = self.results[rr]
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) < 4:  # Need at least 4 points for meaningful interpolation
                print(f"⚠️  Not enough data points for RR={rr} (need at least 4, have {len(x)})")
                continue
            
            # Create a regular grid for contour plotting
            short_range = np.linspace(x.min(), x.max(), 50)
            long_range = np.linspace(y.min(), y.max(), 50)
            X_grid, Y_grid = np.meshgrid(short_range, long_range)
            
            # Use Gaussian Process regression for smooth interpolation
            try:
                # Prepare data for GP
                X_train = np.column_stack([x, y])
                y_train = z
                
                # Define GP kernel
                kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
                gp.fit(X_train, y_train)
                
                # Predict on grid
                X_grid_flat = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                Z_pred, Z_std = gp.predict(X_grid_flat, return_std=True)
                Z_grid = Z_pred.reshape(X_grid.shape)
                Z_std_grid = Z_std.reshape(X_grid.shape)
                
                # Create contour plot
                contour = go.Contour(
                    x=short_range,
                    y=long_range,
                    z=Z_grid,
                    colorscale='Viridis',
                    showscale=True if i == 0 else False,
                    colorbar=dict(title=metric.replace('_', ' ').title()) if i == 0 else None,
                    name=f'Score Distribution'
                )
                
                fig.add_trace(contour, row=row, col=col)
                
                # Add original data points
                scatter = go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=z,
                        colorscale='Viridis',
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    text=[f"Short: {x[j]}, Long: {y[j]}, Score: {z[j]:.3f}" for j in range(len(x))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Data Points',
                    showlegend=False
                )
                
                fig.add_trace(scatter, row=row, col=col)
                
                # Highlight best point
                best_idx = np.argmax(z)
                best_scatter = go.Scatter(
                    x=[x[best_idx]], y=[y[best_idx]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='diamond',
                        opacity=1.0
                    ),
                    text=[f"BEST: Short={x[best_idx]}, Long={y[best_idx]}, Score={z[best_idx]:.3f}"],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Best Point',
                    showlegend=False
                )
                
                fig.add_trace(best_scatter, row=row, col=col)
                
                # Add confidence intervals (optional)
                if len(x) >= 10:  # Only for sufficient data
                    # Add 1-sigma confidence interval
                    upper_bound = Z_grid + Z_std_grid
                    lower_bound = Z_grid - Z_std_grid
                    
                    # Create confidence interval traces
                    conf_upper = go.Contour(
                        x=short_range, y=long_range, z=upper_bound,
                        colorscale='Reds', opacity=0.3,
                        showscale=False, name=f'Upper Bound',
                        line=dict(width=0)
                    )
                    conf_lower = go.Contour(
                        x=short_range, y=long_range, z=lower_bound,
                        colorscale='Blues', opacity=0.3,
                        showscale=False, name=f'Lower Bound',
                        line=dict(width=0)
                    )
                    
                    fig.add_trace(conf_upper, row=row, col=col)
                    fig.add_trace(conf_lower, row=row, col=col)
                
            except Exception as e:
                print(f"⚠️  Error creating contour for RR={rr}: {e}")
                # Fallback to simple scatter plot
                scatter = go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=z,
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=[f"Short: {x[j]}, Long: {y[j]}, Score: {z[j]:.3f}" for j in range(len(x))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Data Points (No Interpolation)',
                    showlegend=False
                )
                fig.add_trace(scatter, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=f'MA Optimization Score Distribution - {metric.replace("_", " ").title()}',
            title_x=0.5,
            height=300 * n_rows,
            width=400 * n_cols
        )
        
        # Update axes labels
        for i in range(n_plots):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.update_xaxes(title_text="Short Window", row=row, col=col)
            fig.update_yaxes(title_text="Long Window", row=row, col=col)
        
        # Save as HTML file
        filename = os.path.join(self.output_dir, f"ma_distribution_contours_{metric.replace('_', '_')}.html")
        plot(fig, filename=filename, auto_open=self.auto_open)
        print(f"📁 Distribution contour plots saved as: {filename}")
        print(f"🌐 Opening in browser...")
    
    def analyze_optimal_regions(self, metric: str = 'composite_score', 
                              percentile_threshold: float = None) -> Dict:
        """
        Analyze optimal regions in the parameter space using distribution analysis.
        
        Args:
            metric: Metric to analyze
            percentile_threshold: Percentile threshold for defining "optimal" regions
            
        Returns:
            Dictionary containing optimal region analysis
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return {}
        
        optimal_regions = {}
        
        # Determine percentile to use
        pct = self.percentile_threshold if percentile_threshold is None else percentile_threshold

        for rr, data in self.results.items():
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) < 4:
                continue
            
            # Calculate threshold for "optimal" scores
            threshold = np.percentile(z, pct)
            optimal_mask = z >= threshold
            optimal_x = x[optimal_mask]
            optimal_y = y[optimal_mask]
            optimal_z = z[optimal_mask]
            
            # Calculate statistics for optimal region
            optimal_stats = {
                'threshold': threshold,
                'n_optimal_points': len(optimal_x),
                'optimal_short_range': (optimal_x.min(), optimal_x.max()) if len(optimal_x) > 0 else (0, 0),
                'optimal_long_range': (optimal_y.min(), optimal_y.max()) if len(optimal_y) > 0 else (0, 0),
                'mean_optimal_score': optimal_z.mean() if len(optimal_z) > 0 else 0,
                'std_optimal_score': optimal_z.std() if len(optimal_z) > 0 else 0,
                'best_score': z.max(),
                'best_short': x[np.argmax(z)],
                'best_long': y[np.argmax(z)]
            }
            
            # Fit 2D Gaussian to optimal points
            if len(optimal_x) >= 3:
                try:
                    # Calculate mean and covariance
                    optimal_points = np.column_stack([optimal_x, optimal_y])
                    mean = np.mean(optimal_points, axis=0)
                    cov = np.cov(optimal_points.T)
                    
                    # Add regularization to ensure positive definite
                    regularization = 0.1 * np.eye(cov.shape[0])
                    cov_regularized = cov + regularization
                    
                    # Calculate 95% confidence ellipse
                    from scipy.stats import chi2
                    chi2_val = chi2.ppf(0.95, df=2)
                    eigenvals, eigenvecs = np.linalg.eigh(cov_regularized)
                    order = eigenvals.argsort()[::-1]
                    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
                    
                    # Ellipse parameters
                    width, height = 2 * np.sqrt(chi2_val * eigenvals)
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    
                    optimal_stats.update({
                        'gaussian_mean': mean,
                        'gaussian_cov': cov_regularized,
                        'confidence_ellipse': {
                            'center': mean,
                            'width': width,
                            'height': height,
                            'angle': angle
                        }
                    })
                except Exception as e:
                    print(f"⚠️  Error fitting Gaussian for RR={rr}: {e}")
                    # Fallback: create a simple Gaussian based on data range
                    mean = np.array([optimal_x.mean(), optimal_y.mean()])
                    std_short = (optimal_x.max() - optimal_x.min()) / 4
                    std_long = (optimal_y.max() - optimal_y.min()) / 4
                    cov_fallback = np.array([[std_short**2, 0], [0, std_long**2]])
                    
                    optimal_stats.update({
                        'gaussian_mean': mean,
                        'gaussian_cov': cov_fallback,
                        'confidence_ellipse': {
                            'center': mean,
                            'width': 2 * std_short,
                            'height': 2 * std_long,
                            'angle': 0
                        }
                    })
            
            optimal_regions[rr] = optimal_stats
        
        return optimal_regions
    
    def create_optimal_regions_plot(self, metric: str = 'composite_score', 
                                  percentile_threshold: float = None) -> None:
        """
        Create a plot highlighting optimal regions in parameter space.
        
        Args:
            metric: Metric to plot
            percentile_threshold: Percentile threshold for defining "optimal" regions
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return
        
        # Analyze optimal regions
        pct = self.percentile_threshold if percentile_threshold is None else percentile_threshold
        optimal_regions = self.analyze_optimal_regions(metric, pct)
        
        risk_reward_ratios = list(self.results.keys())
        n_plots = len(risk_reward_ratios)
        
        # Calculate subplot grid
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f'Risk-Reward Ratio = {rr}' for rr in risk_reward_ratios],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, rr in enumerate(risk_reward_ratios):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            data = self.results[rr]
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) == 0:
                continue
            
            # Get optimal region info
            if rr in optimal_regions:
                opt_info = optimal_regions[rr]
                threshold = opt_info['threshold']
                optimal_mask = z >= threshold
                optimal_x = x[optimal_mask]
                optimal_y = y[optimal_mask]
                optimal_z = z[optimal_mask]
            else:
                threshold = np.percentile(z, pct)
                optimal_mask = z >= threshold
                optimal_x = x[optimal_mask]
                optimal_y = y[optimal_mask]
                optimal_z = z[optimal_mask]
            
            # Plot all points (non-optimal in light color)
            non_optimal_mask = ~optimal_mask
            if np.any(non_optimal_mask):
                nx = x[non_optimal_mask]
                ny = y[non_optimal_mask]
                nz = z[non_optimal_mask]
                scatter_all = go.Scatter(
                    x=nx, y=ny,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='lightgray',
                        opacity=0.5
                    ),
                    text=[f"Short: {int(nx[j])}, Long: {int(ny[j])}, Score: {nz[j]:.3f}" for j in range(len(nx))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Non-optimal',
                    showlegend=False
                )
                fig.add_trace(scatter_all, row=row, col=col)
            
            # Plot optimal points
            if len(optimal_x) > 0:
                scatter_optimal = go.Scatter(
                    x=optimal_x, y=optimal_y,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=optimal_z,
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=[f"Short: {optimal_x[j]}, Long: {optimal_y[j]}, Score: {optimal_z[j]:.3f}" for j in range(len(optimal_x))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Optimal Region',
                    showlegend=False
                )
                fig.add_trace(scatter_optimal, row=row, col=col)
            
            # Add confidence ellipse if available
            if rr in optimal_regions and 'confidence_ellipse' in optimal_regions[rr]:
                ellipse_info = optimal_regions[rr]['confidence_ellipse']
                center = ellipse_info['center']
                width = ellipse_info['width']
                height = ellipse_info['height']
                angle = ellipse_info['angle']
                
                # Create ellipse points
                t = np.linspace(0, 2*np.pi, 100)
                ellipse_x = center[0] + (width/2) * np.cos(t) * np.cos(np.radians(angle)) - (height/2) * np.sin(t) * np.sin(np.radians(angle))
                ellipse_y = center[1] + (width/2) * np.cos(t) * np.sin(np.radians(angle)) + (height/2) * np.sin(t) * np.cos(np.radians(angle))
                
                ellipse = go.Scatter(
                    x=ellipse_x, y=ellipse_y,
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name=f'95% Confidence',
                    showlegend=False
                )
                fig.add_trace(ellipse, row=row, col=col)
            
            # Highlight best point (original scoring)
            best_idx = np.argmax(z)
            best_scatter = go.Scatter(
                x=[x[best_idx]], y=[y[best_idx]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='diamond',
                    opacity=1.0
                ),
                text=[f"BEST (Original): Short={x[best_idx]}, Long={y[best_idx]}, Score={z[best_idx]:.3f}"],
                hovertemplate='%{text}<extra></extra>',
                name=f'Best Point (Original)',
                showlegend=False
            )
            fig.add_trace(best_scatter, row=row, col=col)
            
            # Highlight neighborhood-aware best point
            try:
                neighborhood_scores = self.calculate_neighborhood_aware_scores(metric=metric)
                if rr in neighborhood_scores:
                    na_data = neighborhood_scores[rr]
                    na_best_short = na_data['best_short']
                    na_best_long = na_data['best_long']
                    na_best_score = na_data['best_neighborhood_score']
                    na_original_score = na_data['best_original_score']
                    na_details = na_data['best_details']
                    
                    # Check if this point exists in the current plot data
                    point_exists = False
                    for j in range(len(x)):
                        if x[j] == na_best_short and y[j] == na_best_long:
                            point_exists = True
                            break
                    
                    if point_exists:
                        na_best_scatter = go.Scatter(
                            x=[na_best_short], y=[na_best_long],
                            mode='markers',
                            marker=dict(
                                size=18,
                                color='gold',
                                symbol='star',
                                opacity=1.0,
                                line=dict(width=2, color='black')
                            ),
                            text=[f"🌟 NEIGHBORHOOD-AWARE BEST: Short={na_best_short}, Long={na_best_long}<br>"
                                  f"Original Score: {na_original_score:.3f}<br>"
                                  f"Neighborhood Score: {na_best_score:.3f}<br>"
                                  f"Neighbors: {na_details['neighbor_count']}<br>"
                                  f"Positive Neighbors: {na_details['positive_count']}<br>"
                                  f"Negative Neighbors: {na_details['negative_count']}<br>"
                                  f"Neighborhood Avg: {na_details['neighborhood_avg']:.3f}<br>"
                                  f"Neighborhood Std: {na_details['neighborhood_std']:.3f}"],
                            hovertemplate='%{text}<extra></extra>',
                            name=f'🌟 Neighborhood-Aware Best',
                            showlegend=False
                        )
                        fig.add_trace(na_best_scatter, row=row, col=col)
            except Exception as e:
                # Silently fail if neighborhood scoring not available
                pass
        
        # Update layout
        fig.update_layout(
            title=f'MA Optimization - Optimal Regions (Top {100-pct}%) - {metric.replace("_", " ").title()}',
            title_x=0.5,
            height=300 * n_rows,
            width=400 * n_cols
        )
        
        # Update axes labels
        for i in range(n_plots):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.update_xaxes(title_text="Short Window", row=row, col=col)
            fig.update_yaxes(title_text="Long Window", row=row, col=col)
        
        # Save as HTML file
        filename = os.path.join(self.output_dir, f"ma_optimal_regions_{metric.replace('_', '_')}.html")
        plot(fig, filename=filename, auto_open=self.auto_open)
        print(f"📁 Optimal regions plot saved as: {filename}")
        print(f"🌐 Opening in browser...")
        
        # Print optimal regions summary
        print(f"\n📊 OPTIMAL REGIONS ANALYSIS (Top {100-percentile_threshold}%)")
        print("="*60)
        for rr, info in optimal_regions.items():
            print(f"\nRisk-Reward Ratio {rr}:")
            print(f"  Threshold Score: {info['threshold']:.4f}")
            print(f"  Optimal Points: {info['n_optimal_points']}")
            print(f"  Optimal Short Range: {info['optimal_short_range'][0]:.0f} - {info['optimal_short_range'][1]:.0f}")
            print(f"  Optimal Long Range: {info['optimal_long_range'][0]:.0f} - {info['optimal_long_range'][1]:.0f}")
            print(f"  Mean Optimal Score: {info['mean_optimal_score']:.4f}")
            print(f"  Best Score: {info['best_score']:.4f} at ({info['best_short']}, {info['best_long']})")
    
    def create_3d_gaussian_surface_plots(self, metric: str = 'composite_score') -> None:
        """
        Create 3D Gaussian surface plots showing the optimization landscape.
        Uses Gaussian Process regression to create smooth 3D surfaces.
        
        Args:
            metric: Metric to plot
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return
        
        risk_reward_ratios = list(self.results.keys())
        n_plots = len(risk_reward_ratios)
        
        # Calculate subplot grid
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            specs=[[{'type': 'scatter3d'} for _ in range(n_cols)] for _ in range(n_rows)],
            subplot_titles=[f'Risk-Reward Ratio = {rr}' for rr in risk_reward_ratios],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, rr in enumerate(risk_reward_ratios):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            data = self.results[rr]
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) < 4:  # Need at least 4 points for meaningful interpolation
                print(f"⚠️  Not enough data points for RR={rr} (need at least 4, have {len(x)})")
                continue
            
            # Create a regular grid for surface plotting
            short_range = np.linspace(x.min(), x.max(), 30)
            long_range = np.linspace(y.min(), y.max(), 30)
            X_grid, Y_grid = np.meshgrid(short_range, long_range)
            
            # Use Gaussian Process regression for smooth interpolation
            try:
                # Prepare data for GP
                X_train = np.column_stack([x, y])
                y_train = z
                
                # Define GP kernel
                kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
                gp.fit(X_train, y_train)
                
                # Predict on grid
                X_grid_flat = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                Z_pred, Z_std = gp.predict(X_grid_flat, return_std=True)
                Z_grid = Z_pred.reshape(X_grid.shape)
                Z_std_grid = Z_std.reshape(X_grid.shape)
                
                # Create 3D surface plot
                surface = go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_grid,
                    colorscale='Viridis',
                    opacity=0.8,
                    name=f'Score Surface',
                    showscale=True if i == 0 else False,
                    colorbar=dict(title=metric.replace('_', ' ').title()) if i == 0 else None
                )
                
                fig.add_trace(surface, row=row, col=col)
                
                # Add original data points
                scatter = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=z,
                        colorscale='Viridis',
                        opacity=0.9,
                        line=dict(width=1, color='white')
                    ),
                    text=[f"Short: {x[j]}, Long: {y[j]}, Score: {z[j]:.3f}" for j in range(len(x))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Data Points',
                    showlegend=False
                )
                
                fig.add_trace(scatter, row=row, col=col)
                
                # Highlight best point
                best_idx = np.argmax(z)
                best_scatter = go.Scatter3d(
                    x=[x[best_idx]], y=[y[best_idx]], z=[z[best_idx]],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='diamond',
                        opacity=1.0
                    ),
                    text=[f"BEST: Short={x[best_idx]}, Long={y[best_idx]}, Score={z[best_idx]:.3f}"],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Best Point',
                    showlegend=False
                )
                
                fig.add_trace(best_scatter, row=row, col=col)
                
                # Add uncertainty surface (optional, for sufficient data)
                if len(x) >= 10:
                    # Create uncertainty surface
                    uncertainty_surface = go.Surface(
                        x=X_grid,
                        y=Y_grid,
                        z=Z_grid + Z_std_grid,
                        colorscale='Reds',
                        opacity=0.3,
                        name=f'Upper Bound',
                        showscale=False
                    )
                    
                    fig.add_trace(uncertainty_surface, row=row, col=col)
                
            except Exception as e:
                print(f"⚠️  Error creating 3D surface for RR={rr}: {e}")
                # Fallback to simple scatter plot
                scatter = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=z,
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=[f"Short: {x[j]}, Long: {y[j]}, Score: {z[j]:.3f}" for j in range(len(x))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Data Points (No Surface)',
                    showlegend=False
                )
                fig.add_trace(scatter, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=f'MA Optimization - 3D Gaussian Surface - {metric.replace("_", " ").title()}',
            title_x=0.5,
            height=400 * n_rows,
            width=500 * n_cols
        )
        
        # Update axes labels
        for i in range(n_plots):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.update_scenes(
                xaxis_title="Short Window",
                yaxis_title="Long Window",
                zaxis_title=metric.replace('_', ' ').title(),
                row=row, col=col
            )
        
        # Save as HTML file
        filename = os.path.join(self.output_dir, f"ma_3d_gaussian_surfaces_{metric.replace('_', '_')}.html")
        plot(fig, filename=filename, auto_open=self.auto_open)
        print(f"📁 3D Gaussian surface plots saved as: {filename}")
        print(f"🌐 Opening in browser...")
    
    def create_combined_3d_plot(self, metric: str = 'composite_score') -> None:
        """
        Create a single 3D plot combining all risk-reward ratios with different colors.
        Shows the complete optimization landscape as a 3D Gaussian surface.
        
        Args:
            metric: Metric to plot
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return
        
        fig = go.Figure()
        
        # Color palette for different risk-reward ratios (using valid Plotly colorscales)
        colorscales = ['blues', 'greens', 'oranges', 'purples', 'teal', 'pinkyl', 'gray', 'viridis']
        
        for i, (rr, data) in enumerate(self.results.items()):
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) < 4:
                continue
            
            # Create a regular grid for this risk-reward ratio
            short_range = np.linspace(x.min(), x.max(), 20)
            long_range = np.linspace(y.min(), y.max(), 20)
            X_grid, Y_grid = np.meshgrid(short_range, long_range)
            
            try:
                # Use Gaussian Process regression
                X_train = np.column_stack([x, y])
                y_train = z
                
                kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
                gp.fit(X_train, y_train)
                
                # Predict on grid
                X_grid_flat = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                Z_pred = gp.predict(X_grid_flat)
                Z_grid = Z_pred.reshape(X_grid.shape)
                
                # Create 3D surface for this risk-reward ratio
                surface = go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_grid,
                    colorscale=colorscales[i % len(colorscales)],
                    opacity=0.6,
                    name=f'RR={rr} Surface',
                    showlegend=True
                )
                
                fig.add_trace(surface)
                
                # Add data points
                scatter = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=z,
                        colorscale=colorscales[i % len(colorscales)],
                        opacity=0.8
                    ),
                    text=[f"RR={rr}, Short: {x[j]}, Long: {y[j]}, Score: {z[j]:.3f}" for j in range(len(x))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'RR={rr} Points',
                    showlegend=True
                )
                
                fig.add_trace(scatter)
                
                # Highlight best point for this ratio
                best_idx = np.argmax(z)
                best_scatter = go.Scatter3d(
                    x=[x[best_idx]], y=[y[best_idx]], z=[z[best_idx]],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='diamond',
                        opacity=1.0
                    ),
                    text=[f"BEST RR={rr}: Short={x[best_idx]}, Long={y[best_idx]}, Score={z[best_idx]:.3f}"],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Best RR={rr}',
                    showlegend=True
                )
                
                fig.add_trace(best_scatter)
                
            except Exception as e:
                print(f"⚠️  Error creating surface for RR={rr}: {e}")
                # Fallback to scatter plot
                scatter = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=z,
                        colorscale=colorscales[i % len(colorscales)],
                        opacity=0.8
                    ),
                    text=[f"RR={rr}, Short: {x[j]}, Long: {y[j]}, Score: {z[j]:.3f}" for j in range(len(x))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'RR={rr} (No Surface)',
                    showlegend=True
                )
                fig.add_trace(scatter)
        
        # Update layout
        fig.update_layout(
            title=f'MA Optimization - Combined 3D Gaussian Landscape - {metric.replace("_", " ").title()}',
            scene=dict(
                xaxis_title="Short Window",
                yaxis_title="Long Window",
                zaxis_title=metric.replace('_', ' ').title()
            ),
            width=1000,
            height=800
        )
        
        # Save as HTML file
        filename = os.path.join(self.output_dir, f"ma_combined_3d_gaussian_{metric.replace('_', '_')}.html")
        plot(fig, filename=filename, auto_open=self.auto_open)
        print(f"📁 Combined 3D Gaussian plot saved as: {filename}")
        print(f"🌐 Opening in browser...")
    
    def create_3d_gaussian_bell_curves(self, metric: str = 'composite_score', 
                                     percentile_threshold: float = None) -> None:
        """
        Create actual 3D Gaussian bell curves fitted to optimal parameter regions.
        These show the probability density distribution of optimal parameters.
        
        Args:
            metric: Metric to analyze
            percentile_threshold: Percentile threshold for defining "optimal" regions
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return
        
        # Analyze optimal regions first
        pct = self.percentile_threshold if percentile_threshold is None else percentile_threshold
        optimal_regions = self.analyze_optimal_regions(metric, pct)
        
        risk_reward_ratios = list(self.results.keys())
        n_plots = len(risk_reward_ratios)
        
        # Calculate subplot grid
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            specs=[[{'type': 'scatter3d'} for _ in range(n_cols)] for _ in range(n_rows)],
            subplot_titles=[f'Risk-Reward Ratio = {rr}' for rr in risk_reward_ratios],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, rr in enumerate(risk_reward_ratios):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            data = self.results[rr]
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) < 3:
                continue
            
            # Get optimal region info
            if rr in optimal_regions and 'gaussian_mean' in optimal_regions[rr]:
                opt_info = optimal_regions[rr]
                mean = opt_info['gaussian_mean']
                cov = opt_info['gaussian_cov']
                
                # Add regularization to ensure positive definite covariance
                regularization = 0.1 * np.eye(cov.shape[0])
                cov_regularized = cov + regularization
                
                # Create a grid for the Gaussian surface
                short_range = np.linspace(x.min(), x.max(), 40)
                long_range = np.linspace(y.min(), y.max(), 40)
                X_grid, Y_grid = np.meshgrid(short_range, long_range)
                
                try:
                    # Calculate 2D Gaussian probability density
                    pos = np.dstack((X_grid, Y_grid))
                    rv = stats.multivariate_normal(mean, cov_regularized, allow_singular=True)
                    Z_gaussian = rv.pdf(pos)
                except Exception as e:
                    print(f"⚠️  Error creating Gaussian for RR={rr}: {e}")
                    # Fallback: create a simple Gaussian centered at the mean
                    # with standard deviations based on data range
                    std_short = (x.max() - x.min()) / 4
                    std_long = (y.max() - y.min()) / 4
                    cov_fallback = np.array([[std_short**2, 0], [0, std_long**2]])
                    rv = stats.multivariate_normal(mean, cov_fallback)
                    Z_gaussian = rv.pdf(pos)
                
                # Normalize to 0-1 range for better visualization
                Z_gaussian = Z_gaussian / Z_gaussian.max()
                
                # Create 3D Gaussian bell curve surface
                gaussian_surface = go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_gaussian,
                    colorscale='Viridis',
                    opacity=0.8,
                    name=f'Gaussian PDF',
                    showscale=True if i == 0 else False,
                    colorbar=dict(title='Probability Density') if i == 0 else None
                )
                
                fig.add_trace(gaussian_surface, row=row, col=col)
                
                # Add original data points
                scatter = go.Scatter3d(
                    x=x, y=y, z=np.zeros_like(z),  # Project onto z=0 plane
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=z,
                        colorscale='Viridis',
                        opacity=0.9,
                        line=dict(width=1, color='white')
                    ),
                    text=[f"Short: {x[j]}, Long: {y[j]}, Score: {z[j]:.3f}" for j in range(len(x))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Data Points',
                    showlegend=False
                )
                
                fig.add_trace(scatter, row=row, col=col)
                
                # Highlight optimal points (projected onto the Gaussian)
                threshold = opt_info['threshold']
                optimal_mask = z >= threshold
                optimal_x = x[optimal_mask]
                optimal_y = y[optimal_mask]
                optimal_z = z[optimal_mask]
                
                if len(optimal_x) > 0:
                    # Project optimal points onto the Gaussian surface
                    optimal_pos = np.column_stack([optimal_x, optimal_y])
                    optimal_gaussian_z = rv.pdf(optimal_pos)
                    if np.isscalar(optimal_gaussian_z):
                        optimal_gaussian_z = optimal_gaussian_z / Z_gaussian.max()
                    else:
                        optimal_gaussian_z = optimal_gaussian_z / Z_gaussian.max()
                    
                    optimal_scatter = go.Scatter3d(
                        x=optimal_x, y=optimal_y, z=optimal_gaussian_z,
                        mode='markers',
                        marker=dict(
                            size=10,
                            color='red',
                            opacity=1.0
                        ),
                        text=[f"Optimal: Short={optimal_x[j]}, Long={optimal_y[j]}, Score={optimal_z[j]:.3f}" for j in range(len(optimal_x))],
                        hovertemplate='%{text}<extra></extra>',
                        name=f'Optimal Points',
                        showlegend=False
                    )
                    
                    fig.add_trace(optimal_scatter, row=row, col=col)
                
                # Highlight best point
                best_idx = np.argmax(z)
                best_x, best_y, best_z = x[best_idx], y[best_idx], z[best_idx]
                best_pos = np.array([[best_x, best_y]])
                best_gaussian_z = rv.pdf(best_pos) / Z_gaussian.max()
                if np.isscalar(best_gaussian_z):
                    best_gaussian_z = best_gaussian_z
                else:
                    best_gaussian_z = best_gaussian_z[0]
                
                best_scatter = go.Scatter3d(
                    x=[best_x], y=[best_y], z=[best_gaussian_z],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='diamond',
                        opacity=1.0
                    ),
                    text=[f"BEST: Short={best_x}, Long={best_y}, Score={best_z:.3f}"],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Best Point',
                    showlegend=False
                )
                
                fig.add_trace(best_scatter, row=row, col=col)
                
                # Add confidence ellipses as wireframes
                if 'confidence_ellipse' in opt_info:
                    ellipse_info = opt_info['confidence_ellipse']
                    center = ellipse_info['center']
                    width = ellipse_info['width']
                    height = ellipse_info['height']
                    angle = ellipse_info['angle']
                    
                    # Create ellipse points
                    t = np.linspace(0, 2*np.pi, 100)
                    ellipse_x = center[0] + (width/2) * np.cos(t) * np.cos(np.radians(angle)) - (height/2) * np.sin(t) * np.sin(np.radians(angle))
                    ellipse_y = center[1] + (width/2) * np.cos(t) * np.sin(np.radians(angle)) + (height/2) * np.sin(t) * np.cos(np.radians(angle))
                    ellipse_z = np.zeros_like(ellipse_x)  # Project onto z=0
                    
                    ellipse_wireframe = go.Scatter3d(
                        x=ellipse_x, y=ellipse_y, z=ellipse_z,
                        mode='lines',
                        line=dict(color='red', width=3),
                        name=f'95% Confidence',
                        showlegend=False
                    )
                    
                    fig.add_trace(ellipse_wireframe, row=row, col=col)
                
            else:
                # Fallback: show data points without Gaussian
                scatter = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=z,
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=[f"Short: {x[j]}, Long: {y[j]}, Score: {z[j]:.3f}" for j in range(len(x))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Data Points (No Gaussian)',
                    showlegend=False
                )
                fig.add_trace(scatter, row=row, col=col)
        
        # Calculate overall stability metrics
        stability_info = []
        for i, rr in enumerate(risk_reward_ratios):
            if rr in optimal_regions and 'gaussian_mean' in optimal_regions[rr]:
                data = self.results[rr]
                x = np.array(data['short_windows'])
                y = np.array(data['long_windows'])
                z = np.array(data[metric])
                
                if len(x) > 0:
                    best_idx = np.argmax(z)
                    best_x, best_y = x[best_idx], y[best_idx]
                    best_pos = np.array([[best_x, best_y]])
                    
                    opt_info = optimal_regions[rr]
                    mean = opt_info['gaussian_mean']
                    cov = opt_info['gaussian_cov']
                    
                    # Distance from mean (in standard deviations)
                    diff = best_pos - mean
                    mahalanobis_distance = np.sqrt(diff @ np.linalg.inv(cov) @ diff.T)[0, 0]
                    
                    # Standard deviation of the distribution
                    eigenvals = np.linalg.eigvals(cov)
                    avg_std = np.sqrt(np.mean(eigenvals))
                    
                    # Stability score
                    stability_score = avg_std / (1 + mahalanobis_distance)
                    
                    # Win rate, pnl and number of trades at best point
                    best_wr = float(data['win_rate'][best_idx]) if 'win_rate' in data else np.nan
                    best_pnl = float(data['total_pnl'][best_idx]) if 'total_pnl' in data else np.nan
                    best_n = int(data['total_trades'][best_idx]) if 'total_trades' in data else 0
                    
                    stability_info.append(
                        f"RR {rr}: Dist={mahalanobis_distance:.2f}σ, Std={avg_std:.2f}, Stability={stability_score:.3f}, WR={best_wr:.1%}, PnL={best_pnl:.1%}, N={best_n}"
                    )
        
        # Create title with stability information
        title = f'MA Optimization - 3D Gaussian Bell Curves (Probability Density) - {metric.replace("_", " ").title()}'
        if stability_info:
            title += '<br><br>Stability/Perf: ' + ' | '.join(stability_info[:3])  # Show first 3 for readability
        
        # Update layout (larger for readability)
        fig.update_layout(
            title=title,
            title_x=0.5,
            title_font_size=18,
            height=500 * n_rows,
            width=650 * n_cols,
            margin=dict(l=20, r=20, t=90, b=20)
        )
        
        # Update axes labels
        for i in range(n_plots):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.update_scenes(
                xaxis_title="Short Window",
                yaxis_title="Long Window",
                zaxis_title="Probability Density",
                row=row, col=col
            )
        
        # Save as HTML file
        filename = os.path.join(self.output_dir, f"ma_3d_gaussian_bell_curves_{metric.replace('_', '_')}.html")
        plot(fig, filename=filename, auto_open=self.auto_open)
        print(f"📁 3D Gaussian bell curves saved as: {filename}")
        print(f"🌐 Opening in browser...")
        
        print(f"\n📊 GAUSSIAN BELL CURVE INTERPRETATION:")
        print("="*60)
        print("• Bell curves show probability density of optimal parameters")
        print("• Peak = most likely optimal parameter combination")
        print("• Width = parameter sensitivity (narrow = sensitive, wide = robust)")
        print("• Red points = actual optimal data points")
        print("• Red diamond = best performing point")
        print("• Red ellipse = 95% confidence region")
    
    def analyze_parameter_robustness(self, metric: str = 'composite_score') -> dict:
        """
        Analyze parameter robustness across different risk-reward ratios.
        
        Args:
            metric: Metric to analyze
            
        Returns:
            Dictionary with robustness analysis results
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return {}
        
        print(f"\n🔍 PARAMETER ROBUSTNESS ANALYSIS")
        print("=" * 60)
        
        # Collect all parameter combinations and their scores
        all_params = []
        for rr, data in self.results.items():
            for i in range(len(data['short_windows'])):
                all_params.append({
                    'risk_reward_ratio': rr,
                    'short_window': data['short_windows'][i],
                    'long_window': data['long_windows'][i],
                    'score': data[metric][i],
                    'sharpe_ratio': data['sharpe_ratio'][i],
                    'total_pnl': data['total_pnl'][i],
                    'win_rate': data['win_rate'][i]
                })
        
        df = pd.DataFrame(all_params)
        
        # Find parameters that appear in top 20% across multiple risk-reward ratios
        robust_params = {}
        for rr in df['risk_reward_ratio'].unique():
            rr_data = df[df['risk_reward_ratio'] == rr]
            # Use the same global/instance percentile for robustness (convert percent → quantile)
            threshold = rr_data['score'].quantile(self.percentile_threshold / 100.0)
            top_params = rr_data[rr_data['score'] >= threshold]
            
            for _, row in top_params.iterrows():
                param_key = f"{row['short_window']}_{row['long_window']}"
                if param_key not in robust_params:
                    robust_params[param_key] = {
                        'short_window': row['short_window'],
                        'long_window': row['long_window'],
                        'appearances': 0,
                        'scores': [],
                        'risk_reward_ratios': [],
                        'avg_score': 0,
                        'max_score': 0,
                        'min_score': float('inf')
                    }
                
                robust_params[param_key]['appearances'] += 1
                robust_params[param_key]['scores'].append(row['score'])
                robust_params[param_key]['risk_reward_ratios'].append(rr)
                robust_params[param_key]['max_score'] = max(robust_params[param_key]['max_score'], row['score'])
                robust_params[param_key]['min_score'] = min(robust_params[param_key]['min_score'], row['score'])
        
        # Calculate average scores
        for param_key in robust_params:
            robust_params[param_key]['avg_score'] = np.mean(robust_params[param_key]['scores'])
        
        # Sort by robustness (appearances) and average score
        robust_list = sorted(robust_params.items(), 
                           key=lambda x: (x[1]['appearances'], x[1]['avg_score']), 
                           reverse=True)
        
        print(f"📊 ROBUST PARAMETER COMBINATIONS (appearing in top 20% across multiple RR ratios):")
        print("-" * 60)
        
        for i, (param_key, data) in enumerate(robust_list[:10]):  # Top 10
            print(f"{i+1:2d}. Short={int(data['short_window']):2d}, Long={int(data['long_window']):2d} | "
                  f"Appearances: {int(data['appearances']):2d} | "
                  f"Avg Score: {data['avg_score']:8.2f} | "
                  f"Score Range: {data['min_score']:6.2f}-{data['max_score']:8.2f}")
            print(f"    Risk-Reward Ratios: {data['risk_reward_ratios']}")
        
        return {
            'robust_parameters': robust_list,
            'total_combinations': len(robust_params),
            'dataframe': df
        }
    
    def create_parameter_recommendations(self, metric: str = 'composite_score') -> dict:
        """
        Create comprehensive parameter recommendations based on multiple criteria.
        
        Args:
            metric: Metric to analyze
            
        Returns:
            Dictionary with parameter recommendations
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return {}
        
        print(f"\n🎯 PARAMETER RECOMMENDATIONS")
        print("=" * 60)
        
        # Get robustness analysis
        robustness = self.analyze_parameter_robustness(metric)
        
        # Find best parameters for each risk-reward ratio
        best_by_rr = {}
        for rr, data in self.results.items():
            best_idx = np.argmax(data[metric])
            best_by_rr[rr] = {
                'short_window': data['short_windows'][best_idx],
                'long_window': data['long_windows'][best_idx],
                'score': data[metric][best_idx],
                'sharpe_ratio': data['sharpe_ratio'][best_idx],
                'total_pnl': data['total_pnl'][best_idx],
                'win_rate': data['win_rate'][best_idx],
                'total_trades': data['total_trades'][best_idx]
            }
        
        # Overall best parameter (highest score across all RR ratios)
        all_scores = []
        for rr, data in self.results.items():
            for i, score in enumerate(data[metric]):
                all_scores.append({
                    'risk_reward_ratio': rr,
                    'short_window': data['short_windows'][i],
                    'long_window': data['long_windows'][i],
                    'score': score,
                    'sharpe_ratio': data['sharpe_ratio'][i],
                    'total_pnl': data['total_pnl'][i],
                    'win_rate': data['win_rate'][i],
                    'total_trades': data['total_trades'][i]
                })
        
        overall_best = max(all_scores, key=lambda x: x['score'])
        
        print(f"🏆 OVERALL BEST PARAMETERS (Highest Score):")
        print(f"   Short Window: {overall_best['short_window']}")
        print(f"   Long Window: {overall_best['long_window']}")
        print(f"   Risk-Reward Ratio: {overall_best['risk_reward_ratio']}")
        print(f"   Composite Score: {overall_best['score']:.2f}")
        print(f"   Sharpe Ratio: {overall_best['sharpe_ratio']:.4f}")
        print(f"   Total PnL: {overall_best['total_pnl']:.2%}")
        print(f"   Win Rate: {overall_best['win_rate']:.2%}")
        print(f"   Total Trades: {overall_best['total_trades']}")
        
        print(f"\n📈 BEST PARAMETERS BY RISK-REWARD RATIO:")
        print("-" * 60)
        for rr in sorted(best_by_rr.keys()):
            data = best_by_rr[rr]
            print(f"RR {rr:3.1f}: Short={int(data['short_window']):2d}, Long={int(data['long_window']):2d} | "
                  f"Score={data['score']:8.2f} | PnL={data['total_pnl']:6.2%} | "
                  f"Sharpe={data['sharpe_ratio']:.4f} | Trades={data['total_trades']}")
        
        # Most robust parameters
        if robustness['robust_parameters']:
            print(f"\n🛡️  MOST ROBUST PARAMETERS (Consistent across RR ratios):")
            print("-" * 60)
            for i, (param_key, data) in enumerate(robustness['robust_parameters'][:5]):
                print(f"{i+1}. Short={int(data['short_window']):2d}, Long={int(data['long_window']):2d} | "
                      f"Appears in {int(data['appearances'])} RR ratios | "
                      f"Avg Score: {data['avg_score']:.2f}")
        
        # Parameter range analysis
        print(f"\n📊 PARAMETER RANGE ANALYSIS:")
        print("-" * 60)
        
        short_windows = [data['short_windows'] for data in self.results.values()]
        long_windows = [data['long_windows'] for data in self.results.values()]
        
        # Flatten lists
        all_short = [w for sublist in short_windows for w in sublist]
        all_long = [w for sublist in long_windows for w in sublist]
        
        print(f"Short Window Range: {min(all_short)} - {max(all_short)}")
        print(f"Long Window Range: {min(all_long)} - {max(all_long)}")
        
        # Find most common optimal ranges
        optimal_short = [data['short_windows'][np.argmax(data[metric])] for data in self.results.values()]
        optimal_long = [data['long_windows'][np.argmax(data[metric])] for data in self.results.values()]
        
        print(f"Most Common Optimal Short Windows: {sorted(set(optimal_short))}")
        print(f"Most Common Optimal Long Windows: {sorted(set(optimal_long))}")
        
        return {
            'overall_best': overall_best,
            'best_by_rr': best_by_rr,
            'robustness_analysis': robustness,
            'parameter_ranges': {
                'short_min': min(all_short),
                'short_max': max(all_short),
                'long_min': min(all_long),
                'long_max': max(all_long)
            },
            'common_optimal': {
                'short_windows': sorted(set(optimal_short)),
                'long_windows': sorted(set(optimal_long))
            }
        }
    
    def create_parameter_selection_guide(self, metric: str = 'composite_score') -> None:
        """
        Create a comprehensive parameter selection guide with visualizations.
        
        Args:
            metric: Metric to analyze
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return
        
        print(f"\n📋 PARAMETER SELECTION GUIDE")
        print("=" * 80)
        
        # Get recommendations
        recommendations = self.create_parameter_recommendations(metric)
        
        print(f"\n🎯 HOW TO CHOOSE THE BEST PARAMETERS:")
        print("-" * 80)
        print("1. 🏆 OVERALL BEST: Use the highest scoring combination across all risk-reward ratios")
        print("   - Best for: Maximum performance regardless of risk tolerance")
        print("   - Risk: May not be optimal for your specific risk preference")
        
        print("\n2. 🛡️  ROBUST PARAMETERS: Use parameters that perform well across multiple RR ratios")
        print("   - Best for: Consistent performance, less sensitive to risk-reward changes")
        print("   - Risk: May not achieve maximum performance in any single scenario")
        
        print("\n3. 📊 RISK-SPECIFIC: Choose parameters optimized for your preferred risk-reward ratio")
        print("   - Best for: Aligned with your risk tolerance and trading style")
        print("   - Risk: May not perform well if market conditions change")
        
        print("\n4. 🔄 ADAPTIVE: Use different parameters for different market conditions")
        print("   - Best for: Dynamic trading strategies")
        print("   - Risk: Requires more monitoring and adjustment")
        
        print(f"\n💡 RECOMMENDATIONS BASED ON YOUR DATA:")
        print("-" * 80)
        
        overall = recommendations['overall_best']
        print(f"• For MAXIMUM PERFORMANCE: Short={int(overall['short_window'])}, Long={int(overall['long_window'])}, RR={overall['risk_reward_ratio']}")
        print(f"  Expected: {overall['score']:.0f} composite score, {overall['total_pnl']:.1f}% PnL, {overall['sharpe_ratio']:.3f} Sharpe, {overall['total_trades']} trades")
        
        if recommendations['robustness_analysis']['robust_parameters']:
            robust = recommendations['robustness_analysis']['robust_parameters'][0][1]
            # Find the most common risk-reward ratio for this parameter combination
            robust_rr_counts = {}
            for rr in robust['risk_reward_ratios']:
                robust_rr_counts[rr] = robust_rr_counts.get(rr, 0) + 1
            most_common_rr = max(robust_rr_counts, key=robust_rr_counts.get)
            
            print(f"• For CONSISTENCY: Short={int(robust['short_window'])}, Long={int(robust['long_window'])}, RR={most_common_rr}")
            print(f"  Appears in {int(robust['appearances'])} risk-reward ratios, avg score: {robust['avg_score']:.0f}")
        
        common_short = recommendations['common_optimal']['short_windows']
        common_long = recommendations['common_optimal']['long_windows']
        print(f"• For BALANCED APPROACH: Short in {common_short}, Long in {common_long}")
        print(f"  These ranges appear most frequently in optimal solutions")
        
        print(f"\n⚠️  IMPORTANT CONSIDERATIONS:")
        print("-" * 80)
        print("• Backtest results are historical and may not predict future performance")
        print("• Consider transaction costs and slippage in live trading")
        print("• Monitor performance and be ready to adjust parameters")
        print("• Test parameters on out-of-sample data before live trading")
        print("• Consider market regime changes that might affect optimal parameters")
    
    def create_individual_3d_gaussian_bell_curves(self, metric: str = 'composite_score', 
                                                percentile_threshold: float = None) -> None:
        """
        Create individual 3D Gaussian bell curves for each risk-reward ratio (one plot per ratio).
        These show the probability density distribution of optimal parameters.
        
        Args:
            metric: Metric to analyze
            percentile_threshold: Percentile threshold for defining "optimal" regions
        """
        if not self.results:
            print("❌ No results available. Run run_optimization_grid() first.")
            return
        
        # Analyze optimal regions first
        pct = self.percentile_threshold if percentile_threshold is None else percentile_threshold
        optimal_regions = self.analyze_optimal_regions(metric, pct)
        
        for rr, data in self.results.items():
            x = np.array(data['short_windows'])
            y = np.array(data['long_windows'])
            z = np.array(data[metric])
            
            if len(x) < 3:
                print(f"⚠️  Not enough data points for RR={rr} (need at least 3, have {len(x)})")
                continue
            
            # Get optimal region info
            if rr in optimal_regions and 'gaussian_mean' in optimal_regions[rr]:
                opt_info = optimal_regions[rr]
                mean = opt_info['gaussian_mean']
                cov = opt_info['gaussian_cov']
                
                # Add regularization to ensure positive definite covariance
                regularization = 0.1 * np.eye(cov.shape[0])
                cov_regularized = cov + regularization
                
                # Create a grid for the Gaussian surface
                short_range = np.linspace(x.min(), x.max(), 50)
                long_range = np.linspace(y.min(), y.max(), 50)
                X_grid, Y_grid = np.meshgrid(short_range, long_range)
                
                try:
                    # Calculate 2D Gaussian probability density
                    pos = np.dstack((X_grid, Y_grid))
                    rv = stats.multivariate_normal(mean, cov_regularized, allow_singular=True)
                    Z_gaussian = rv.pdf(pos)
                except Exception as e:
                    print(f"⚠️  Error creating Gaussian for RR={rr}: {e}")
                    # Fallback: create a simple Gaussian centered at the mean
                    std_short = (x.max() - x.min()) / 4
                    std_long = (y.max() - y.min()) / 4
                    cov_fallback = np.array([[std_short**2, 0], [0, std_long**2]])
                    rv = stats.multivariate_normal(mean, cov_fallback)
                    Z_gaussian = rv.pdf(pos)
                
                # Normalize to 0-1 range for better visualization
                Z_gaussian = Z_gaussian / Z_gaussian.max()
                
                # Create 3D Gaussian bell curve surface
                gaussian_surface = go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_gaussian,
                    colorscale='Viridis',
                    opacity=0.8,
                    name=f'Gaussian PDF',
                    colorbar=dict(title='Probability Density')
                )
                
                # Add original data points (projected onto z=0 plane)
                scatter = go.Scatter3d(
                    x=x, y=y, z=np.zeros_like(z),
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=z,
                        colorscale='Viridis',
                        opacity=0.9,
                        line=dict(width=1, color='white')
                    ),
                    text=[f"Short: {x[j]}, Long: {y[j]}, Score: {z[j]:.3f}" for j in range(len(x))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Data Points'
                )
                
                # Highlight optimal points (projected onto the Gaussian)
                threshold = opt_info['threshold']
                optimal_mask = z >= threshold
                optimal_x = x[optimal_mask]
                optimal_y = y[optimal_mask]
                optimal_z = z[optimal_mask]
                
                traces = [gaussian_surface, scatter]
                
                if len(optimal_x) > 0:
                    # Project optimal points onto the Gaussian surface
                    optimal_pos = np.column_stack([optimal_x, optimal_y])
                    optimal_gaussian_z = rv.pdf(optimal_pos)
                    if np.isscalar(optimal_gaussian_z):
                        optimal_gaussian_z = optimal_gaussian_z / Z_gaussian.max()
                    else:
                        optimal_gaussian_z = optimal_gaussian_z / Z_gaussian.max()
                    
                    optimal_scatter = go.Scatter3d(
                        x=optimal_x, y=optimal_y, z=optimal_gaussian_z,
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='red',
                            opacity=1.0
                        ),
                        text=[f"Optimal: Short={optimal_x[j]}, Long={optimal_y[j]}, Score={optimal_z[j]:.3f}" for j in range(len(optimal_x))],
                        hovertemplate='%{text}<extra></extra>',
                        name=f'Optimal Points'
                    )
                    traces.append(optimal_scatter)
                
                # Highlight best point
                best_idx = np.argmax(z)
                best_x, best_y, best_z = x[best_idx], y[best_idx], z[best_idx]
                best_pos = np.array([[best_x, best_y]])
                best_gaussian_z = rv.pdf(best_pos)
                if np.isscalar(best_gaussian_z):
                    best_gaussian_z = best_gaussian_z / Z_gaussian.max()
                else:
                    best_gaussian_z = best_gaussian_z[0] / Z_gaussian.max()
                
                best_scatter = go.Scatter3d(
                    x=[best_x], y=[best_y], z=[best_gaussian_z],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='diamond',
                        opacity=1.0
                    ),
                    text=[f"BEST: Short={best_x}, Long={best_y}, Score={best_z:.3f}"],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Best Point'
                )
                traces.append(best_scatter)
                
                # Add confidence ellipse as wireframe
                if 'confidence_ellipse' in opt_info:
                    ellipse_info = opt_info['confidence_ellipse']
                    center = ellipse_info['center']
                    width = ellipse_info['width']
                    height = ellipse_info['height']
                    angle = ellipse_info['angle']
                    
                    # Create ellipse points
                    t = np.linspace(0, 2*np.pi, 100)
                    ellipse_x = center[0] + (width/2) * np.cos(t) * np.cos(np.radians(angle)) - (height/2) * np.sin(t) * np.sin(np.radians(angle))
                    ellipse_y = center[1] + (width/2) * np.cos(t) * np.sin(np.radians(angle)) + (height/2) * np.sin(t) * np.cos(np.radians(angle))
                    ellipse_z = np.zeros_like(ellipse_x)  # Project onto z=0
                    
                    ellipse_wireframe = go.Scatter3d(
                        x=ellipse_x, y=ellipse_y, z=ellipse_z,
                        mode='lines',
                        line=dict(color='red', width=3),
                        name=f'95% Confidence Region'
                    )
                    traces.append(ellipse_wireframe)
                
                # Create individual figure
                fig = go.Figure(data=traces)
                
                # Calculate stability metrics for the best point
                best_x, best_y = x[best_idx], y[best_idx]
                best_pos = np.array([[best_x, best_y]])
                # Also pull win rate and number of trades for the best point
                best_win_rate = float(data['win_rate'][best_idx]) if 'win_rate' in data else np.nan
                best_total_trades = int(data['total_trades'][best_idx]) if 'total_trades' in data else 0
                best_total_pnl = float(data['total_pnl'][best_idx]) if 'total_pnl' in data else np.nan
                
                # Distance from mean (in standard deviations)
                mean = opt_info['gaussian_mean']
                cov = opt_info['gaussian_cov']
                diff = best_pos - mean
                mahalanobis_distance = np.sqrt(diff @ np.linalg.inv(cov) @ diff.T)[0, 0]
                
                # Standard deviation of the distribution
                eigenvals = np.linalg.eigvals(cov)
                avg_std = np.sqrt(np.mean(eigenvals))
                
                # Stability score (lower distance + higher std = more stable)
                stability_score = avg_std / (1 + mahalanobis_distance)
                
                # Update layout with stability + performance metrics
                title = (
                    f'MA Optimization - RR {rr} - 3D Gaussian Bell Curve - {metric.replace("_", " ").title()}<br>'
                    f'Best: ({best_x}, {best_y}) | Dist: {mahalanobis_distance:.2f}σ | '
                    f'AvgStd: {avg_std:.2f} | Stability: {stability_score:.3f} | '
                    f'WinRate: {best_win_rate:.1%} | PnL: {best_total_pnl:.1%} | Trades: {best_total_trades}'
                )
                
                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title="Short Window",
                        yaxis_title="Long Window",
                        zaxis_title="Probability Density"
                    ),
                    width=1100,
                    height=800,
                    title_font_size=18,
                    margin=dict(l=20, r=20, t=90, b=20)
                )
                
                # Save as HTML file
                filename = os.path.join(self.output_dir, f"ma_individual_gaussian_bell_rr_{rr}_{metric.replace('_', '_')}.html")
                plot(fig, filename=filename, auto_open=self.auto_open)
                print(f"📁 Individual Gaussian bell curve for RR={rr} saved as: {filename}")
                
            else:
                # Fallback: show data points without Gaussian
                scatter = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=z,
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=[f"Short: {x[j]}, Long: {y[j]}, Score: {z[j]:.3f}" for j in range(len(x))],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Data Points (No Gaussian)'
                )
                
                fig = go.Figure(data=[scatter])
                fig.update_layout(
                    title=f'MA Optimization - Risk-Reward Ratio {rr} - Data Points Only - {metric.replace("_", " ").title()}',
                    scene=dict(
                        xaxis_title="Short Window",
                        yaxis_title="Long Window",
                        zaxis_title=metric.replace('_', ' ').title()
                    ),
                    width=800,
                    height=600
                )
                
                filename = os.path.join(self.output_dir, f"ma_individual_data_points_rr_{rr}_{metric.replace('_', '_')}.html")
                plot(fig, filename=filename, auto_open=self.auto_open)
                print(f"📁 Individual data points for RR={rr} saved as: {filename}")
    
    def print_optimization_summary(self) -> None:
        """
        Print a summary of the optimization results.
        """
        if not self.results:
            print("❌ No results available.")
            return
        
        print("\n" + "="*80)
        print("MOVING AVERAGE OPTIMIZATION SUMMARY")
        print("="*80)
        
        for rr, data in self.results.items():
            if len(data['composite_score']) == 0:
                continue
                
            scores = np.array(data['composite_score'])
            valid_scores = scores[scores > -999]
            
            if len(valid_scores) == 0:
                print(f"\nRisk-Reward Ratio {rr}: No valid results")
                continue
            
            best_idx = np.argmax(valid_scores)
            best_short = data['short_windows'][best_idx]
            best_long = data['long_windows'][best_idx]
            best_score = valid_scores[best_idx]
            
            print(f"\nRisk-Reward Ratio {rr}:")
            print(f"  Best Parameters: short_window={best_short}, long_window={best_long}")
            print(f"  Best Composite Score: {best_score:.4f}")
            print(f"  Best Sharpe Ratio: {data['sharpe_ratio'][best_idx]:.4f}")
            print(f"  Best Total PnL: {data['total_pnl'][best_idx]:.2%}")
            print(f"  Best Win Rate: {data['win_rate'][best_idx]:.2%}")
            print(f"  Total Trades: {data['total_trades'][best_idx]}")
            print(f"  Total Valid Combinations: {len(valid_scores)}")


def main():
    """
    Main function to demonstrate 3D optimization visualization.
    """
    print("3D MOVING AVERAGE OPTIMIZATION VISUALIZER")
    print("="*60)
    
    # Configuration
    symbol = "ETHUSDT"
    end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d %H:%M:%S')
    interval = "15m"
    trading_fee = 0.0  # 0% fee for this example
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    
    try:
        # Fetch data
        binance_fetcher = BinanceDataFetcher(api_key=cfg.BINANCE_API_KEY, api_secret=cfg.BINANCE_SECRET_KEY)
        data = binance_fetcher.fetch_historical_data(symbol, start_date, end_date, interval=interval)
        
        if data.empty:
            print(f"❌ No data fetched for {symbol}")
            return
        
        print(f"✅ Successfully fetched {len(data)} data points")
        
        # Create visualizer
        visualizer = MAOptimization3DVisualizer(data, trading_fee)
        
        # Define parameter ranges
        short_window_range = [5, 10, 15, 20, 25, 30]
        long_window_range = [20, 30, 40, 50, 60, 70]
        risk_reward_ratios = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        
        # Run optimization grid
        results = visualizer.run_optimization_grid(
            short_window_range, long_window_range, risk_reward_ratios
        )
        
        # Print summary
        visualizer.print_optimization_summary()
        
        # Create summary plot (all risk-reward ratios in one plot)
        print("\n🎨 Creating summary plot (all risk-reward ratios)...")
        visualizer.create_summary_plot(metric='composite_score')
        
        # Create 3D plots (grid view)
        print("\n🎨 Creating 3D plots (grid view)...")
        visualizer.create_3d_plots(metric='composite_score')
        
        # Create individual 3D plots for each risk-reward ratio
        print("\n🎨 Creating individual 3D plots...")
        visualizer.create_individual_3d_plots(metric='composite_score')
        
        # Create 2D heatmaps
        print("\n🎨 Creating 2D heatmaps...")
        visualizer.create_2d_heatmaps(metric='composite_score')
        
        # Create distribution contour plots
        print("\n🎨 Creating distribution contour plots...")
        visualizer.create_distribution_contour_plots(metric='composite_score')
        
        # Create optimal regions analysis
        print("\n🎨 Creating optimal regions analysis...")
        visualizer.create_optimal_regions_plot(metric='composite_score', percentile_threshold=DEFAULT_OPTIMAL_PERCENTILE)
        
        # Create 3D Gaussian surface plots
        print("\n🎨 Creating 3D Gaussian surface plots...")
        visualizer.create_3d_gaussian_surface_plots(metric='composite_score')
        
        # Create combined 3D Gaussian plot
        print("\n🎨 Creating combined 3D Gaussian landscape...")
        visualizer.create_combined_3d_plot(metric='composite_score')
        
        # Create 3D Gaussian bell curves (true bell shapes)
        print("\n🎨 Creating 3D Gaussian bell curves...")
        visualizer.create_3d_gaussian_bell_curves(metric='composite_score', percentile_threshold=DEFAULT_OPTIMAL_PERCENTILE)
        
        # Create individual 3D Gaussian bell curves
        print("\n🎨 Creating individual 3D Gaussian bell curves...")
        visualizer.create_individual_3d_gaussian_bell_curves(metric='composite_score', percentile_threshold=DEFAULT_OPTIMAL_PERCENTILE)
        
        # Create parameter selection guide
        print("\n📋 Creating parameter selection guide...")
        visualizer.create_parameter_selection_guide(metric='composite_score')
        
        # Create additional plots for other metrics
        print("\n🎨 Creating Sharpe ratio summary plot...")
        visualizer.create_summary_plot(metric='sharpe_ratio')
        
        print("\n🎨 Creating Total PnL summary plot...")
        visualizer.create_summary_plot(metric='total_pnl')
        
        print("\n✅ Visualization complete!")
        
    except Exception as e:
        logger.error(f"Error in 3D visualization: {e}")
        print(f"❌ Error: {e}")


def demo_with_sample_data():
    """
    Demo function using sample data (no API credentials required).
    """
    print("DEMO MODE: Using sample data for 3D visualization")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range(
        start='2025-01-01 09:15:00',
        end='2025-01-31 15:30:00',
        freq='15min'
    )
    
    # Filter for weekdays and market hours
    market_dates = []
    for date in dates:
        if date.weekday() < 5:  # Monday to Friday
            if '09:15' <= date.strftime('%H:%M') <= '15:30':
                market_dates.append(date)
    
    # Generate sample OHLCV data
    np.random.seed(42)
    n_points = len(market_dates)
    base_price = 500.0
    prices = [base_price]
    
    for i in range(1, n_points):
        change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))
    
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(100000, 1000000) for _ in range(n_points)]
    }, index=market_dates)
    
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    print(f"📊 Generated {len(data)} sample data points")
    
    # Create visualizer
    visualizer = MAOptimization3DVisualizer(data, trading_fee=0.0)
    
    # Define parameter ranges (smaller for demo)
    short_window_range = [5, 10, 15, 20]
    long_window_range = [20, 30, 40, 50]
    risk_reward_ratios = [1.5, 2.0, 2.5, 3.0]
    
    # Run optimization grid
    results = visualizer.run_optimization_grid(
        short_window_range, long_window_range, risk_reward_ratios
    )
    
    # Print summary
    visualizer.print_optimization_summary()
    
    # Create visualizations
    print("\n🎨 Creating summary plot (all risk-reward ratios)...")
    visualizer.create_summary_plot(metric='composite_score')
    
    print("\n🎨 Creating 3D plots (grid view)...")
    visualizer.create_3d_plots(metric='composite_score')
    
    print("\n🎨 Creating individual 3D plots...")
    visualizer.create_individual_3d_plots(metric='composite_score')
    
    print("\n🎨 Creating 2D heatmaps...")
    visualizer.create_2d_heatmaps(metric='composite_score')
    
    # Create distribution contour plots
    print("\n🎨 Creating distribution contour plots...")
    visualizer.create_distribution_contour_plots(metric='composite_score')
    
    # Create optimal regions analysis
    print("\n🎨 Creating optimal regions analysis...")
    visualizer.create_optimal_regions_plot(metric='composite_score', percentile_threshold=80.0)
    
    # Create 3D Gaussian surface plots
    print("\n🎨 Creating 3D Gaussian surface plots...")
    visualizer.create_3d_gaussian_surface_plots(metric='composite_score')
    
    # Create combined 3D Gaussian plot
    print("\n🎨 Creating combined 3D Gaussian landscape...")
    visualizer.create_combined_3d_plot(metric='composite_score')
    
    # Create 3D Gaussian bell curves (true bell shapes)
    print("\n🎨 Creating 3D Gaussian bell curves...")
    visualizer.create_3d_gaussian_bell_curves(metric='composite_score', percentile_threshold=80.0)
    
    # Create individual 3D Gaussian bell curves
    print("\n🎨 Creating individual 3D Gaussian bell curves...")
    visualizer.create_individual_3d_gaussian_bell_curves(metric='composite_score', percentile_threshold=80.0)
    
    # Create parameter selection guide
    print("\n📋 Creating parameter selection guide...")
    visualizer.create_parameter_selection_guide(metric='composite_score')
    
    print("\n✅ Demo visualization complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_with_sample_data()
    else:
        main()
