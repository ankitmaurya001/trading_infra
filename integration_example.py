#!/usr/bin/env python3
"""
Integration Example: Using Validated Strategies with Live Trading

This example shows how to:
1. Run comprehensive validation to find the best strategy
2. Use the validated strategy with the live trading system
3. Monitor performance and make adjustments
"""

import json
import os
from datetime import datetime, timedelta
from comprehensive_strategy_validation import ComprehensiveStrategyValidator
from simplified_binance_live_trading import SimplifiedLiveTradingSimulator
import streamlit as st


def run_validation_and_get_best_strategy(symbol: str = "ETHUSDT", 
                                       days_back: int = 30,
                                       train_ratio: float = 0.7):
    """
    Run comprehensive validation and return the best strategy setup.
    
    Args:
        symbol: Trading symbol
        days_back: Days of historical data to use
        train_ratio: Ratio of data for training
        
    Returns:
        Dictionary with best strategy setup
    """
    print(f"🔍 Running comprehensive validation for {symbol}...")
    
    # Initialize validator
    validator = ComprehensiveStrategyValidator(
        initial_balance=10000,
        max_leverage=10.0,
        max_loss_percent=2.0,
        trading_fee=0.001
    )
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # Step 1: Fetch and split data
        train_data, test_data = validator.fetch_and_split_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="15m",
            train_ratio=train_ratio
        )
        
        # Step 2: Optimize strategies
        optimization_results = validator.optimize_strategies_on_train_data(
            strategies_to_optimize=['ma', 'rsi', 'donchian']
        )
        
        # Step 3: Validate on test data
        validation_results = validator.validate_strategies_on_test_data(
            strategies_to_validate=['ma', 'rsi', 'donchian'],
            mock_trading_delay=0.01
        )
        
        # Step 4: Get recommendations
        recommendations = validator.compare_strategies()
        
        # Step 5: Get live trading setup
        live_setup = validator.get_live_trading_setup()
        
        # Save results
        validator.save_results("validation_results")
        
        print(f"✅ Validation completed! Best strategy: {live_setup['strategy_name']}")
        return live_setup
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None


def setup_live_trading_with_validated_strategy(live_setup: dict):
    """
    Setup live trading with the validated strategy.
    
    Args:
        live_setup: Dictionary with strategy setup from validation
        
    Returns:
        Configured trading simulator
    """
    print(f"🚀 Setting up live trading with {live_setup['strategy_name']}...")
    
    # Initialize simulator
    simulator = SimplifiedLiveTradingSimulator(
        initial_balance=live_setup['initial_balance'],
        max_leverage=live_setup['max_leverage'],
        max_loss_percent=live_setup['max_loss_percent']
    )
    
    # Setup strategy with validated parameters
    strategy_key = live_setup['strategy_key']
    parameters = live_setup['parameters']
    
    # Convert strategy parameters to the format expected by the simulator
    if strategy_key == 'ma':
        ma_params = parameters
        rsi_params = None
        donchian_params = None
    elif strategy_key == 'rsi':
        ma_params = None
        rsi_params = parameters
        donchian_params = None
    elif strategy_key == 'donchian':
        ma_params = None
        rsi_params = None
        donchian_params = parameters
    else:
        raise ValueError(f"Unknown strategy key: {strategy_key}")
    
    # Set manual parameters
    success = simulator.set_manual_parameters(
        ma_params=ma_params,
        rsi_params=rsi_params,
        donchian_params=donchian_params,
        enabled_strategies=[strategy_key]
    )
    
    if success:
        print(f"✅ Live trading setup completed!")
        print(f"📊 Strategy: {live_setup['strategy_name']}")
        print(f"⚙️  Parameters: {parameters}")
        print(f"📈 Expected Performance:")
        print(f"  - Sharpe Ratio: {live_setup['performance_metrics'].get('sharpe_ratio', 'N/A'):.3f}")
        print(f"  - Win Rate: {live_setup['performance_metrics'].get('win_rate', 'N/A'):.2%}")
        print(f"  - Max Drawdown: {live_setup['performance_metrics'].get('max_drawdown', 'N/A'):.2%}")
        return simulator
    else:
        print(f"❌ Failed to setup live trading")
        return None


def create_streamlit_app():
    """
    Create a Streamlit app that integrates validation with live trading.
    """
    st.set_page_config(
        page_title="Validated Strategy Trading", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 Validated Strategy Trading System")
    st.markdown("Comprehensive validation + Live trading integration")
    
    # Sidebar configuration
    st.sidebar.header("📊 Configuration")
    
    symbol = st.sidebar.text_input("Symbol", value="ETHUSDT")
    days_back = st.sidebar.slider("Days of Historical Data", 7, 90, 30)
    train_ratio = st.sidebar.slider("Train/Test Split", 0.5, 0.9, 0.7)
    
    # Validation section
    st.subheader("🔍 Strategy Validation")
    
    if st.button("🚀 Run Comprehensive Validation", type="primary"):
        with st.spinner("Running comprehensive validation..."):
            live_setup = run_validation_and_get_best_strategy(
                symbol=symbol,
                days_back=days_back,
                train_ratio=train_ratio
            )
            
            if live_setup:
                st.session_state.live_setup = live_setup
                st.success("Validation completed! Best strategy found.")
                
                # Display validation results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Strategy", live_setup['strategy_name'])
                with col2:
                    st.metric("Sharpe Ratio", f"{live_setup['performance_metrics'].get('sharpe_ratio', 0):.3f}")
                with col3:
                    st.metric("Win Rate", f"{live_setup['performance_metrics'].get('win_rate', 0):.2%}")
                with col4:
                    st.metric("Max Drawdown", f"{live_setup['performance_metrics'].get('max_drawdown', 0):.2%}")
                
                st.json(live_setup['parameters'])
            else:
                st.error("Validation failed. Check the logs for details.")
    
    # Live trading section
    if 'live_setup' in st.session_state:
        st.subheader("🚀 Live Trading Setup")
        
        live_setup = st.session_state.live_setup
        
        if st.button("⚙️ Setup Live Trading", type="primary"):
            with st.spinner("Setting up live trading..."):
                simulator = setup_live_trading_with_validated_strategy(live_setup)
                
                if simulator:
                    st.session_state.simulator = simulator
                    st.success("Live trading setup completed!")
                else:
                    st.error("Failed to setup live trading")
        
        # Live trading controls
        if 'simulator' in st.session_state:
            st.subheader("🎮 Live Trading Controls")
            
            simulator = st.session_state.simulator
            status = simulator.get_current_status()
            
            # Display current status
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Status", "Running" if status['is_running'] else "Stopped")
            with col2:
                st.metric("Balance", f"${status['current_balance']:,.2f}")
            with col3:
                st.metric("Total PnL", f"${status['total_pnl']:,.2f}")
            with col4:
                st.metric("Total Trades", status['total_trades'])
            
            # Trading controls
            col1, col2 = st.columns(2)
            with col1:
                if not status['is_running']:
                    if st.button("▶️ Start Trading", type="primary"):
                        success = simulator.start_live_trading(
                            symbol=symbol,
                            interval="15m",
                            polling_frequency=60,
                            mock_mode=True,  # Start with mock mode for safety
                            mock_days_back=7,
                            mock_delay=0.1
                        )
                        if success:
                            st.rerun()
                        else:
                            st.error("Failed to start trading")
            with col2:
                if status['is_running']:
                    if st.button("⏹️ Stop Trading", type="secondary"):
                        simulator.stop_live_trading()
                        st.rerun()
            
            # Performance metrics
            if not simulator.get_trade_history_df().empty:
                st.subheader("📊 Performance Metrics")
                metrics = simulator.calculate_performance_metrics()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
                with col2:
                    st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
                with col3:
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
                with col4:
                    st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
    
    # Results section
    st.subheader("📁 Validation Results")
    
    if os.path.exists("validation_results"):
        result_files = os.listdir("validation_results")
        if result_files:
            st.write("Available result files:")
            for file in result_files:
                if file.endswith('.json'):
                    st.write(f"📄 {file}")
                elif file.endswith('.csv'):
                    st.write(f"📊 {file}")
        else:
            st.info("No validation results available. Run validation first.")
    else:
        st.info("No validation results directory found. Run validation first.")


def main():
    """
    Main function - can be used for both programmatic and Streamlit usage.
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        # Run Streamlit app
        create_streamlit_app()
    else:
        # Run programmatic example
        print("🎯 VALIDATION + LIVE TRADING INTEGRATION EXAMPLE")
        print("="*60)
        
        # Step 1: Run validation
        print("\n🔍 Step 1: Running comprehensive validation...")
        live_setup = run_validation_and_get_best_strategy(
            symbol="ETHUSDT",
            days_back=30,
            train_ratio=0.7
        )
        
        if live_setup:
            print(f"\n✅ Validation completed!")
            print(f"🏆 Best Strategy: {live_setup['strategy_name']}")
            print(f"⚙️  Parameters: {live_setup['parameters']}")
            
            # Step 2: Setup live trading
            print(f"\n🚀 Step 2: Setting up live trading...")
            simulator = setup_live_trading_with_validated_strategy(live_setup)
            
            if simulator:
                print(f"\n✅ Live trading setup completed!")
                print(f"🎯 Ready to start trading with validated strategy!")
                print(f"\n💡 Next steps:")
                print(f"1. Use simulator.start_live_trading() to begin")
                print(f"2. Monitor performance with simulator.get_current_status()")
                print(f"3. Adjust parameters if needed")
            else:
                print(f"\n❌ Failed to setup live trading")
        else:
            print(f"\n❌ Validation failed")


if __name__ == "__main__":
    main()
