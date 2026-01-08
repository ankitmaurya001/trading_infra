#!/usr/bin/env python3
"""
Parameter Validation Service - Continuous Parameter Drift Detection

Runs parameter validation in an infinite loop with configurable frequency.
Saves results to logs that can be picked up by the trading dashboard.

Usage:
    python parameter_validation_service.py --config validation_config.json

Or integrate with trading engine:
    The trading engine can optionally start this service alongside trading.
"""

import argparse
import json
import os
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import threading

from parameter_validator import ParameterValidator, ValidationResult
import config as cfg


class ParameterValidationService:
    """
    Service that continuously validates strategy parameters.
    Runs in background and saves results to logs.
    """
    
    def __init__(self, config_file: str = "parameter_validation_config.json"):
        """
        Initialize the validation service.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Service state
        self.is_running = False
        self.last_validation_time = None
        self.validation_history = []
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize validator
        self.validator = ParameterValidator(
            validation_frequency_days=self.config.get('validation_frequency_days', 7),
            data_window_days=self.config.get('data_window_days', 30),
            distance_threshold_monitor=self.config.get('distance_threshold_monitor', 3.0),
            distance_threshold_warning=self.config.get('distance_threshold_warning', 7.0),
            distance_threshold_critical=self.config.get('distance_threshold_critical', 12.0),
            performance_gap_threshold=self.config.get('performance_gap_threshold', 0.05),
            trading_fee=self.config.get('trading_fee', 0.0),
            exchange=self.config.get('exchange', 'MCX')
        )
        
        # Log folder for saving results
        self.log_folder = self.config.get('log_folder', 'logs')
        os.makedirs(self.log_folder, exist_ok=True)
        
        self.logger.info("✅ Parameter Validation Service initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_file):
            print(f"Creating default config file: {self.config_file}")
            default_config = {
                "symbols": [
                    {
                        "symbol": "SILVERMIC26FEBFUT",
                        "exchange": "MCX",
                        "interval": "15m",
                        "current_params": {
                            "short_window": 4,
                            "long_window": 58,
                            "risk_reward_ratio": 6.0
                        }
                    }
                ],
                "validation_frequency_days": 7,
                "data_window_days": 30,
                "distance_threshold_monitor": 3.0,
                "distance_threshold_warning": 7.0,
                "distance_threshold_critical": 12.0,
                "performance_gap_threshold": 0.05,
                "trading_fee": 0.0,
                "exchange": "MCX",
                "log_folder": "logs",
                "check_interval_hours": 24,  # How often to check if validation is due
                "log_level": "INFO"
            }
            self._save_config(default_config)
            return default_config
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from: {self.config_file}")
            return config
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            sys.exit(1)
    
    def _save_config(self, config: Dict):
        """Save configuration to JSON file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def _setup_logging(self):
        """Setup logging system."""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.stop_service()
        sys.exit(0)
    
    def _should_run_validation(self, symbol_config: Dict) -> bool:
        """
        Check if validation should run for a symbol.
        
        Args:
            symbol_config: Symbol configuration dict
            
        Returns:
            True if validation should run
        """
        symbol = symbol_config['symbol']
        
        # Check last validation time for this symbol
        last_validation_file = os.path.join(
            self.log_folder,
            f"validation_status_{symbol}.json"
        )
        
        if os.path.exists(last_validation_file):
            try:
                with open(last_validation_file, 'r') as f:
                    status = json.load(f)
                    last_time_str = status.get('last_validation_time')
                    if last_time_str:
                        last_time = datetime.fromisoformat(last_time_str)
                        days_since = (datetime.now() - last_time).days
                        return days_since >= self.config.get('validation_frequency_days', 7)
            except Exception as e:
                self.logger.warning(f"Error reading validation status for {symbol}: {e}")
        
        # No previous validation, should run
        return True
    
    def _run_validation_for_symbol(self, symbol_config: Dict) -> Optional[ValidationResult]:
        """
        Run validation for a single symbol.
        
        Args:
            symbol_config: Symbol configuration dict
            
        Returns:
            ValidationResult or None if failed
        """
        symbol = symbol_config['symbol']
        exchange = symbol_config.get('exchange', self.config.get('exchange', 'MCX'))
        interval = symbol_config.get('interval', '15m')
        current_params = symbol_config.get('current_params', {})
        
        self.logger.info("=" * 80)
        self.logger.info(f"🔍 Running validation for {symbol}")
        self.logger.info("=" * 80)
        
        try:
            result = self.validator.validate_parameters(
                current_params=current_params,
                symbol=symbol,
                interval=interval,
                exchange=exchange
            )
            
            # Save result
            self._save_validation_result(symbol, result)
            
            # Update status
            self._update_validation_status(symbol, result)
            
            self.logger.info(f"✅ Validation completed for {symbol}")
            self.logger.info(f"   Status: {result.alert_level}")
            self.logger.info(f"   Should re-optimize: {result.should_reoptimize}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_validation_result(self, symbol: str, result: ValidationResult):
        """Save validation result to file."""
        try:
            # Save detailed result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(
                self.log_folder,
                f"validation_{symbol}_{timestamp}.json"
            )
            
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"💾 Validation result saved: {result_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving validation result: {e}")
    
    def _update_validation_status(self, symbol: str, result: ValidationResult):
        """Update validation status file for dashboard."""
        try:
            status_file = os.path.join(
                self.log_folder,
                f"validation_status_{symbol}.json"
            )
            
            # Load existing status if it exists
            status = {}
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = json.load(f)
            
            # Update status
            status.update({
                'symbol': symbol,
                'last_validation_time': datetime.now().isoformat(),
                'last_validation_date': result.validation_date,
                'current_params': result.current_params,
                'new_optimal_params': result.new_optimal_params,
                'parameter_distance': result.parameter_distance,
                'performance_gap': result.performance_gap,
                'should_reoptimize': result.should_reoptimize,
                'alert_level': result.alert_level,
                'alert_message': result.alert_message,
                'current_params_performance': result.current_params_performance,
                'new_optimal_performance': result.new_optimal_performance,
                'validation_data_period': result.validation_data_period,
                'validation_frequency_days': self.config.get('validation_frequency_days', 7),
                'next_validation_due': (
                    datetime.now() + timedelta(days=self.config.get('validation_frequency_days', 7))
                ).isoformat()
            })
            
            # Add to history (keep last 10)
            if 'validation_history' not in status:
                status['validation_history'] = []
            
            history_entry = {
                'validation_date': result.validation_date,
                'alert_level': result.alert_level,
                'parameter_distance': result.parameter_distance,
                'performance_gap': result.performance_gap,
                'should_reoptimize': result.should_reoptimize
            }
            
            status['validation_history'].append(history_entry)
            # Keep only last 10
            status['validation_history'] = status['validation_history'][-10:]
            
            # Save status
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
            
            self.logger.info(f"💾 Validation status updated: {status_file}")
            
        except Exception as e:
            self.logger.error(f"Error updating validation status: {e}")
    
    def _validation_loop(self):
        """Main validation loop that runs continuously."""
        self.logger.info("🚀 Starting parameter validation service...")
        self.logger.info(f"📊 Monitoring {len(self.config.get('symbols', []))} symbol(s)")
        self.logger.info(f"⏰ Validation frequency: Every {self.config.get('validation_frequency_days', 7)} days")
        self.logger.info(f"🔄 Check interval: Every {self.config.get('check_interval_hours', 24)} hours")
        self.logger.info("=" * 80)
        
        check_interval_seconds = self.config.get('check_interval_hours', 24) * 3600
        
        while self.is_running:
            try:
                symbols = self.config.get('symbols', [])
                
                if not symbols:
                    self.logger.warning("⚠️  No symbols configured. Waiting...")
                    time.sleep(check_interval_seconds)
                    continue
                
                # Check each symbol
                for symbol_config in symbols:
                    if not self.is_running:
                        break
                    
                    symbol = symbol_config['symbol']
                    
                    if self._should_run_validation(symbol_config):
                        self.logger.info(f"⏰ Validation due for {symbol}")
                        result = self._run_validation_for_symbol(symbol_config)
                        
                        if result:
                            self.last_validation_time = datetime.now()
                    else:
                        # Check when next validation is due
                        status_file = os.path.join(
                            self.log_folder,
                            f"validation_status_{symbol}.json"
                        )
                        if os.path.exists(status_file):
                            with open(status_file, 'r') as f:
                                status = json.load(f)
                                next_due = status.get('next_validation_due')
                                if next_due:
                                    next_due_dt = datetime.fromisoformat(next_due)
                                    hours_until = (next_due_dt - datetime.now()).total_seconds() / 3600
                                    self.logger.debug(f"⏳ {symbol}: Next validation in {hours_until:.1f} hours")
                
                # Wait before next check
                self.logger.debug(f"⏳ Waiting {check_interval_seconds/3600:.1f} hours until next check...")
                time.sleep(check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"❌ Error in validation loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)  # Wait 1 minute before retrying
    
    def start_service(self):
        """Start the validation service."""
        if self.is_running:
            self.logger.warning("Service is already running")
            return
        
        self.is_running = True
        
        # Start validation loop in a separate thread
        validation_thread = threading.Thread(
            target=self._validation_loop,
            daemon=True
        )
        validation_thread.start()
        
        self.logger.info("✅ Parameter validation service started")
    
    def stop_service(self):
        """Stop the validation service."""
        self.is_running = False
        self.logger.info("🛑 Parameter validation service stopped")
    
    def run(self):
        """Main run method - starts service and keeps process alive."""
        try:
            self.start_service()
            self.logger.info("Service running. Press Ctrl+C to stop.")
            
            # Keep main thread alive
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt. Shutting down...")
            self.stop_service()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.stop_service()
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parameter Validation Service - Continuous parameter drift detection"
    )
    parser.add_argument(
        "--config",
        default="parameter_validation_config.json",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Create and run service
    service = ParameterValidationService(config_file=args.config)
    service.run()


if __name__ == "__main__":
    main()

