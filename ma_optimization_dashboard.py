import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime, timedelta
import logging
import warnings

# Suppress WebSocket and other harmless warnings
logging.getLogger("tornado.websocket").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

from ma_3d_optimization_visualizer import MAOptimization3DVisualizer
import config as cfg
from data_fetcher import BinanceDataFetcher


def read_html_file(path: str) -> str:
    if not os.path.exists(path):
        return f"<div style='color:red'>File not found: {path}</div>"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    st.set_page_config(
        page_title="MA Optimization Dashboard", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Moving Average Optimization Dashboard")
    
    # Initialize session state
    if 'plots_generated' not in st.session_state:
        st.session_state.plots_generated = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'current_visualizer' not in st.session_state:
        st.session_state.current_visualizer = None
    if 'run_count' not in st.session_state:
        st.session_state.run_count = 0
    
    # Add info about WebSocket errors
    with st.expander("ℹ️ About WebSocket Errors", expanded=False):
        st.info("""
        **WebSocket errors are harmless!** 
        
        The `WebSocketClosedError` messages you see in the terminal are normal and don't affect functionality. 
        They occur when:
        - You click on interactive plots
        - The browser refreshes or loses connection
        - Streamlit tries to update the display
        
        These errors are now suppressed in the dashboard. The plots will work perfectly fine!
        """)

    with st.sidebar:
        st.header("Data & Parameters")
        mode = st.radio("Mode", ["Live (Binance)", "Demo"], index=1)
        symbol = st.text_input("Symbol", value="ETHUSDT")
        interval = st.selectbox("Interval", ["5m", "15m", "1h"], index=1)
        days = st.slider("Lookback Days", 3, 60, 20)
        trading_fee = st.number_input("Trading Fee (decimal)", value=0.0, step=0.001, format="%.4f")
        metric = st.selectbox("Metric", ["composite_score", "sharpe_ratio", "total_pnl"], index=0)
        percentile = st.slider("Optimal Percentile", 60, 95, 80)
        short_range = st.multiselect("Short Windows", [5, 10, 15, 20, 25, 30], default=[5, 10, 15, 20, 25, 30])
        long_range = st.multiselect("Long Windows", [20, 30, 40, 50, 60, 70], default=[20, 30, 40, 50, 60, 70])
        rr_ratios = st.multiselect("Risk-Reward Ratios", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], default=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

        run_btn = st.button("Run Optimization")
        
        # Add clear cache button
        if st.session_state.plots_generated:
            if st.button("🔄 Clear Cache & Regenerate"):
                st.session_state.plots_generated = False
                st.session_state.current_data = None
                st.session_state.current_visualizer = None
                st.success("Cache cleared! Click 'Run Optimization' to regenerate plots.")

    # Debug info (remove this later)
    with st.expander("🔍 Debug Info", expanded=False):
        st.write(f"Plots generated: {st.session_state.plots_generated}")
        st.write(f"Run button clicked: {run_btn}")
        st.write(f"Current data exists: {st.session_state.current_data is not None}")
        st.write(f"Run count: {st.session_state.run_count}")
    
    if not run_btn and not st.session_state.plots_generated:
        st.info("Configure parameters in the sidebar and click 'Run Optimization'.")
        return

    # Only regenerate if run button was clicked or no plots exist
    if run_btn or not st.session_state.plots_generated:
        # Fetch or create data
        if mode.startswith("Live"):
            end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
            try:
                fetcher = BinanceDataFetcher(api_key=cfg.BINANCE_API_KEY, api_secret=cfg.BINANCE_SECRET_KEY)
                data = fetcher.fetch_historical_data(symbol, start_date, end_date, interval=interval)
                if data.empty:
                    st.error("No data returned. Try a different symbol/interval or switch to Demo.")
                    return
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return
        else:
            # Demo synthetic data from the visualizer's demo function logic
            import numpy as np
            idx = pd.date_range(end=datetime.now(), periods=500, freq="15min")
            prices = 2000 + np.cumsum(np.random.normal(0, 5, size=len(idx)))
            data = pd.DataFrame({
                'datetime': idx,
                'open': prices + np.random.normal(0, 1, len(idx)),
                'high': prices + np.abs(np.random.normal(0, 2, len(idx))),
                'low': prices - np.abs(np.random.normal(0, 2, len(idx))),
                'close': prices,
                'volume': np.random.randint(100, 1000, len(idx))
            })
            data.set_index('datetime', inplace=True)

        # Run optimizer + visualizer (suppress auto-open)
        visualizer = MAOptimization3DVisualizer(data, trading_fee=trading_fee, auto_open=False)
        visualizer.run_optimization_grid(short_range, long_range, rr_ratios)

        # Generate plots (files saved to CWD) with error handling
        st.success("Optimization complete. Generating plots...")
        
        plot_functions = [
            ("Summary Plot", lambda: visualizer.create_summary_plot(metric=metric)),
            ("3D Grid Plots", lambda: visualizer.create_3d_plots(metric=metric)),
            ("Individual 3D Plots", lambda: visualizer.create_individual_3d_plots(metric=metric)),
            ("2D Heatmaps", lambda: visualizer.create_2d_heatmaps(metric=metric)),
            ("Distribution Contours", lambda: visualizer.create_distribution_contour_plots(metric=metric)),
            ("Optimal Regions", lambda: visualizer.create_optimal_regions_plot(metric=metric, percentile_threshold=percentile)),
            ("3D Gaussian Surfaces", lambda: visualizer.create_3d_gaussian_surface_plots(metric=metric)),
            ("Combined 3D Plot", lambda: visualizer.create_combined_3d_plot(metric=metric)),
            ("3D Bell Curves", lambda: visualizer.create_3d_gaussian_bell_curves(metric=metric, percentile_threshold=percentile)),
            ("Individual Bell Curves", lambda: visualizer.create_individual_3d_gaussian_bell_curves(metric=metric, percentile_threshold=percentile)),
            ("Parameter Guide", lambda: visualizer.create_parameter_selection_guide(metric=metric))
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (plot_name, plot_func) in enumerate(plot_functions):
            try:
                status_text.text(f"Generating {plot_name}...")
                plot_func()
                progress_bar.progress((i + 1) / len(plot_functions))
            except Exception as e:
                st.warning(f"⚠️ Error generating {plot_name}: {str(e)}")
                continue
        
        status_text.text("✅ All plots generated successfully!")
        progress_bar.empty()
        status_text.empty()
        
        # Store in session state
        st.session_state.plots_generated = True
        st.session_state.current_data = data
        st.session_state.current_visualizer = visualizer
        st.session_state.run_count += 1
    else:
        st.success("✅ Plots already generated! Click on tabs below to view them.")

    # Build tabs
    tabs = st.tabs([
        "Summary",
        "3D Grid",
        "3D Individual",
        "2D Heatmaps",
        "Contours",
        "Optimal Regions",
        "Gaussian Surfaces",
        "Combined 3D",
        "Bell Curves",
        "Bell Curves (Individual)",
        "Recommendations"
    ])

    # Embed saved HTMLs
    with tabs[0]:
        html = read_html_file(f"ma_summary_plot_{metric.replace('_', '_')}.html")
        components.html(html, height=820, scrolling=True)

    with tabs[1]:
        html = read_html_file(f"ma_3d_plots_{metric.replace('_', '_')}.html")
        components.html(html, height=900, scrolling=True)

    with tabs[2]:
        st.info("Scroll the tabs below to view per-RR plots.")
        for rr in rr_ratios:
            st.markdown(f"### RR = {rr}")
            html = read_html_file(f"ma_individual_3d_plot_rr_{rr}_{metric.replace('_', '_')}.html")
            components.html(html, height=650, scrolling=True)

    with tabs[3]:
        html = read_html_file(f"ma_2d_heatmaps_{metric.replace('_', '_')}.html")
        components.html(html, height=900, scrolling=True)

    with tabs[4]:
        html = read_html_file(f"ma_distribution_contours_{metric.replace('_', '_')}.html")
        components.html(html, height=900, scrolling=True)

    with tabs[5]:
        html = read_html_file(f"ma_optimal_regions_{metric.replace('_', '_')}.html")
        components.html(html, height=900, scrolling=True)

    with tabs[6]:
        html = read_html_file(f"ma_3d_gaussian_surfaces_{metric.replace('_', '_')}.html")
        components.html(html, height=900, scrolling=True)

    with tabs[7]:
        html = read_html_file(f"ma_combined_3d_gaussian_{metric.replace('_', '_')}.html")
        components.html(html, height=900, scrolling=True)

    with tabs[8]:
        html = read_html_file(f"ma_3d_gaussian_bell_curves_{metric.replace('_', '_')}.html")
        components.html(html, height=900, scrolling=True)

    with tabs[9]:
        for rr in rr_ratios:
            st.markdown(f"### RR = {rr}")
            html = read_html_file(f"ma_individual_gaussian_bell_rr_{rr}_{metric.replace('_', '_')}.html")
            components.html(html, height=650, scrolling=True)

    with tabs[10]:
        st.markdown("#### Console Output (Recommendations)")
        st.write("See terminal/log output for the printed parameter recommendations and robustness analysis.")


if __name__ == "__main__":
    main()
