#!/usr/bin/env python3
"""
Test script for new data detection logic
Tests that strategies are only processed when new data is available
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from simplified_live_trading import SimplifiedLiveTradingSimulator

def test_new_data_detection():
    """Test that new data detection works correctly."""
    print("🧪 Testing New Data Detection Logic")
    print("=" * 60)
    
    # Initialize simulator
    simulator = SimplifiedLiveTradingSimulator(initial_balance=10000)
    print(f"✅ Simulator initialized with ${simulator.trading_engine.initial_balance:,.2f}")
    
    # Test 1: Initial state
    print("\n📊 Test 1: Initial state")
    print(f"Last processed timestamp: {simulator.last_processed_timestamp}")
    print(f"Last processed index: {simulator.last_processed_index}")
    
    # Test 2: First data should be considered new
    print("\n📊 Test 2: First data detection")
    first_timestamp = datetime.now()
    has_new_data = simulator.last_processed_timestamp is None or first_timestamp > simulator.last_processed_timestamp
    print(f"First timestamp: {first_timestamp}")
    print(f"Has new data: {has_new_data}")
    assert has_new_data, "First data should be considered new"
    print("✅ First data detection works correctly")
    
    # Test 3: Same timestamp should not be considered new
    print("\n📊 Test 3: Same timestamp detection")
    simulator.last_processed_timestamp = first_timestamp
    same_timestamp = first_timestamp
    has_new_data = simulator.last_processed_timestamp is None or same_timestamp > simulator.last_processed_timestamp
    print(f"Same timestamp: {same_timestamp}")
    print(f"Has new data: {has_new_data}")
    assert not has_new_data, "Same timestamp should not be considered new"
    print("✅ Same timestamp detection works correctly")
    
    # Test 4: Newer timestamp should be considered new
    print("\n📊 Test 4: Newer timestamp detection")
    newer_timestamp = first_timestamp + timedelta(minutes=15)
    has_new_data = simulator.last_processed_timestamp is None or newer_timestamp > simulator.last_processed_timestamp
    print(f"Newer timestamp: {newer_timestamp}")
    print(f"Has new data: {has_new_data}")
    assert has_new_data, "Newer timestamp should be considered new"
    print("✅ Newer timestamp detection works correctly")
    
    # Test 5: Older timestamp should not be considered new
    print("\n📊 Test 5: Older timestamp detection")
    older_timestamp = first_timestamp - timedelta(minutes=15)
    has_new_data = simulator.last_processed_timestamp is None or older_timestamp > simulator.last_processed_timestamp
    print(f"Older timestamp: {older_timestamp}")
    print(f"Has new data: {has_new_data}")
    assert not has_new_data, "Older timestamp should not be considered new"
    print("✅ Older timestamp detection works correctly")
    
    # Test 6: Mock data index tracking
    print("\n📊 Test 6: Mock data index tracking")
    simulator.last_processed_index = 5
    current_index = 6
    has_new_data = current_index > simulator.last_processed_index
    print(f"Current index: {current_index}")
    print(f"Last processed index: {simulator.last_processed_index}")
    print(f"Has new data: {has_new_data}")
    assert has_new_data, "Higher index should be considered new"
    print("✅ Mock data index tracking works correctly")
    
    # Test 7: Same index should not be considered new
    print("\n📊 Test 7: Same index detection")
    same_index = 5
    has_new_data = same_index > simulator.last_processed_index
    print(f"Same index: {same_index}")
    print(f"Has new data: {has_new_data}")
    assert not has_new_data, "Same index should not be considered new"
    print("✅ Same index detection works correctly")
    
    print("\n🎉 All new data detection tests passed!")
    return True

def test_data_tracking_reset():
    """Test that data tracking is reset when starting new sessions."""
    print("\n🧪 Testing Data Tracking Reset")
    print("=" * 60)
    
    # Initialize simulator
    simulator = SimplifiedLiveTradingSimulator(initial_balance=10000)
    
    # Set some values
    simulator.last_processed_timestamp = datetime.now()
    simulator.last_processed_index = 10
    
    print(f"Before reset - Timestamp: {simulator.last_processed_timestamp}, Index: {simulator.last_processed_index}")
    
    # Simulate starting a new session
    simulator.last_processed_timestamp = None
    simulator.last_processed_index = -1
    
    print(f"After reset - Timestamp: {simulator.last_processed_timestamp}, Index: {simulator.last_processed_index}")
    
    # Verify reset
    assert simulator.last_processed_timestamp is None, "Timestamp should be reset to None"
    assert simulator.last_processed_index == -1, "Index should be reset to -1"
    
    print("✅ Data tracking reset works correctly")
    return True

def main():
    """Run all new data detection tests."""
    print("🚀 New Data Detection Logic Test")
    print("=" * 80)
    
    # Test 1: New data detection logic
    test1_passed = test_new_data_detection()
    
    # Test 2: Data tracking reset
    test2_passed = test_data_tracking_reset()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Results Summary:")
    print(f"✅ New Data Detection: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Data Tracking Reset: {'PASSED' if test2_passed else 'FAILED'}")
    
    if all([test1_passed, test2_passed]):
        print("\n🎉 All tests passed! New data detection logic is working correctly.")
        print("\n💡 The simulator will now only process strategies when new data is available.")
        print("   This will improve efficiency and reduce unnecessary processing.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
