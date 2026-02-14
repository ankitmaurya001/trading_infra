#!/usr/bin/env python3
"""
Test script to verify the modular structure
Tests the basic functionality without requiring external dependencies
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        print("✅ pandas and numpy imported successfully")
    except ImportError as e:
        print(f"⚠️  pandas/numpy not available: {e}")
        return False
    
    try:
        # Test our custom modules
        from strategy_manager import StrategyManager
        print("✅ StrategyManager imported successfully")
    except ImportError as e:
        print(f"❌ StrategyManager import failed: {e}")
        return False
    
    try:
        from trading_engine import TradingEngine
        print("✅ TradingEngine imported successfully")
    except ImportError as e:
        print(f"❌ TradingEngine import failed: {e}")
        return False
    
    try:
        from optimization_runner import run_quick_optimization
        print("✅ OptimizationRunner imported successfully")
    except ImportError as e:
        print(f"❌ OptimizationRunner import failed: {e}")
        return False
    
    return True

def test_strategy_manager():
    """Test StrategyManager basic functionality"""
    print("\n🧪 Testing StrategyManager...")
    
    try:
        from strategy_manager import StrategyManager
        
        # Initialize
        manager = StrategyManager()
        print("✅ StrategyManager initialized")
        
        # Test parameter setting
        manager.set_manual_parameters(
            ma_params={'short_window': 20, 'long_window': 50, 'risk_reward_ratio': 2.0, 'trading_fee': 0.001}
        )
        print("✅ Manual parameters set")
        
        # Test parameter retrieval
        params = manager.get_all_parameters()
        print(f"✅ Parameters retrieved: {list(params.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ StrategyManager test failed: {e}")
        return False

def test_trading_engine():
    """Test TradingEngine basic functionality"""
    print("\n🧪 Testing TradingEngine...")
    
    try:
        from trading_engine import TradingEngine
        
        # Initialize
        engine = TradingEngine(initial_balance=10000)
        print("✅ TradingEngine initialized")
        
        # Test status
        status = engine.get_current_status()
        print(f"✅ Status retrieved: Balance=${status['current_balance']:,.2f}")
        
        # Test metrics
        metrics = engine.calculate_performance_metrics()
        print(f"✅ Metrics calculated: {metrics['total_trades']} trades")
        
        return True
        
    except Exception as e:
        print(f"❌ TradingEngine test failed: {e}")
        return False

def test_optimization_runner():
    """Test OptimizationRunner basic functionality"""
    print("\n🧪 Testing OptimizationRunner...")
    
    try:
        from optimization_runner import get_optimization_parameters
        
        # Test parameter extraction
        sample_results = {
            'ma': {
                'parameters': {'short_window': 20, 'long_window': 50, 'risk_reward_ratio': 2.0, 'trading_fee': 0.001},
                'metrics': {'sharpe_ratio': 1.5, 'total_pnl': 0.15}
            }
        }
        
        params = get_optimization_parameters(sample_results, 'ma')
        print(f"✅ Parameter extraction: {list(params.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ OptimizationRunner test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        'strategy_manager.py',
        'trading_engine.py', 
        'optimization_runner.py',
        'simplified_live_trading.py',
        'example_modular_usage.py',
        'MODULAR_ARCHITECTURE_README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 MODULAR STRUCTURE TEST")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("StrategyManager", test_strategy_manager),
        ("TradingEngine", test_trading_engine),
        ("OptimizationRunner", test_optimization_runner)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Modular structure is working correctly.")
        print("\n📚 Next steps:")
        print("   1. Install dependencies: pip install yfinance pandas numpy streamlit plotly")
        print("   2. Run example: python example_modular_usage.py")
        print("   3. Start trading: streamlit run simplified_live_trading.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
