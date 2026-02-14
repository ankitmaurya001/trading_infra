#!/usr/bin/env python3
"""
Launcher script for the Live Trading Simulator
Run this script to start the live trading simulator UI
"""

import subprocess
import sys
import os

def main():
    """Launch the live trading simulator"""
    print("🚀 Starting Live Trading Simulator...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("live_trading_simulator.py"):
        print("Error: live_trading_simulator.py not found in current directory")
        print("Please run this script from the trading_infra directory")
        sys.exit(1)
    
    # Check if required dependencies are installed
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import yfinance
        print("✅ All required dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Launch the Streamlit app
    print("🌐 Launching Streamlit app...")
    print("The app will open in your default web browser")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the simulator")
    print("=" * 50)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "live_trading_simulator.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Live Trading Simulator stopped by user")
    except Exception as e:
        print(f"❌ Error launching simulator: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
