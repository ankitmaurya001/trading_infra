#!/usr/bin/env python3
"""
Run Parameter Validation - Check if current parameters are still optimal

This script validates that your current strategy parameters are still optimal
by running optimization on recent data and comparing results.

Usage:
    python run_parameter_validation.py \
        --symbol SILVERMIC26FEBFUT \
        --exchange MCX \
        --interval 15m \
        --short 4 \
        --long 58 \
        --rr 6.0

Or use config file:
    python run_parameter_validation.py --config params.json
"""

import argparse
import json
import os
from datetime import datetime
from parameter_validator import ParameterValidator, validate_ma_parameters


def load_params_from_config(config_path: str) -> dict:
    """Load parameters from JSON config file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Validate current strategy parameters"
    )
    
    # Parameter inputs
    parser.add_argument('--symbol', type=str, required=True,
                       help='Trading symbol (e.g., SILVERMIC26FEBFUT)')
    parser.add_argument('--exchange', type=str, default='MCX',
                       help='Exchange (NSE, BSE, MCX)')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Data interval (15m, 1h, 1d)')
    
    # Current parameters
    parser.add_argument('--short', type=int,
                       help='Current short window')
    parser.add_argument('--long', type=int,
                       help='Current long window')
    parser.add_argument('--rr', type=float,
                       help='Current risk-reward ratio')
    
    # Config file (alternative to individual params)
    parser.add_argument('--config', type=str,
                       help='JSON config file with parameters')
    
    # Validation settings
    parser.add_argument('--validation-frequency-days', type=int, default=7,
                       help='How often to validate (default: 7 = weekly)')
    parser.add_argument('--data-window-days', type=int, default=30,
                       help='Recent data window in days (default: 30)')
    
    # Distance thresholds
    parser.add_argument('--distance-threshold-monitor', type=float, default=3.0,
                       help='Distance threshold for monitoring (default: 3.0)')
    parser.add_argument('--distance-threshold-warning', type=float, default=7.0,
                       help='Distance threshold for warning (default: 7.0)')
    parser.add_argument('--distance-threshold-critical', type=float, default=12.0,
                       help='Distance threshold for critical (default: 12.0)')
    
    # Performance gap threshold
    parser.add_argument('--performance-gap-threshold', type=float, default=0.05,
                       help='Performance gap threshold (default: 0.05 = 5%%)')
    
    args = parser.parse_args()
    
    # Load parameters
    if args.config:
        config = load_params_from_config(args.config)
        current_params = {
            'short_window': config.get('short_window'),
            'long_window': config.get('long_window'),
            'risk_reward_ratio': config.get('risk_reward_ratio')
        }
    else:
        if not all([args.short, args.long, args.rr]):
            parser.error("Must provide --short, --long, --rr OR --config")
        current_params = {
            'short_window': args.short,
            'long_window': args.long,
            'risk_reward_ratio': args.rr
        }
    
    print("=" * 80)
    print("🔍 PARAMETER VALIDATION")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Interval: {args.interval}")
    print(f"Current Parameters:")
    print(f"  Short Window: {current_params['short_window']}")
    print(f"  Long Window: {current_params['long_window']}")
    print(f"  Risk-Reward Ratio: {current_params['risk_reward_ratio']}")
    print(f"\nValidation Settings:")
    print(f"  Frequency: Every {args.validation_frequency_days} days")
    print(f"  Data Window: {args.data_window_days} days")
    print("=" * 80)
    
    # Create validator
    validator = ParameterValidator(
        validation_frequency_days=args.validation_frequency_days,
        data_window_days=args.data_window_days,
        distance_threshold_monitor=args.distance_threshold_monitor,
        distance_threshold_warning=args.distance_threshold_warning,
        distance_threshold_critical=args.distance_threshold_critical,
        performance_gap_threshold=args.performance_gap_threshold,
        exchange=args.exchange
    )
    
    # Run validation
    try:
        result = validator.validate_parameters(
            current_params=current_params,
            symbol=args.symbol,
            interval=args.interval,
            exchange=args.exchange
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("📋 VALIDATION RESULTS")
        print("=" * 80)
        print(result.alert_message)
        print("\n" + "-" * 80)
        print("📊 Performance Comparison:")
        print(f"  Current Parameters Performance:")
        print(f"    Total PnL: {result.current_params_performance.get('total_pnl', 0):.2%}")
        print(f"    Sharpe Ratio: {result.current_params_performance.get('sharpe_ratio', 0):.3f}")
        print(f"    Win Rate: {result.current_params_performance.get('win_rate', 0):.2%}")
        print(f"    Max Drawdown: {result.current_params_performance.get('max_drawdown', 0):.2%}")
        print(f"\n  New Optimal Parameters Performance:")
        print(f"    Total PnL: {result.new_optimal_performance.get('total_pnl', 0):.2%}")
        print(f"    Sharpe Ratio: {result.new_optimal_performance.get('sharpe_ratio', 0):.3f}")
        print(f"    Win Rate: {result.new_optimal_performance.get('win_rate', 0):.2%}")
        print(f"    Max Drawdown: {result.new_optimal_performance.get('max_drawdown', 0):.2%}")
        print(f"\n  Performance Gap: {result.performance_gap:.2%}")
        print(f"  Parameter Distance: {result.parameter_distance:.2f}")
        print("=" * 80)
        
        # Save result
        output_dir = "validation_results"
        saved_path = validator.save_validation_result(result, output_dir)
        print(f"\n💾 Results saved to: {saved_path}")
        
        # Exit code based on result
        if result.should_reoptimize:
            print("\n⚠️  RECOMMENDATION: Re-optimize parameters!")
            exit(1)
        else:
            print("\n✅ Parameters still optimal - no action needed")
            exit(0)
            
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

