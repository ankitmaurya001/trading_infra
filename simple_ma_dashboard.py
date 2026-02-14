#!/usr/bin/env python3
"""
Simple MA Optimization Dashboard
Generates all plots and creates a single HTML dashboard to view them.
"""

import os
import sys
import subprocess

def main():
    """Generate plots and create dashboard"""
    print("🚀 SIMPLE MA OPTIMIZATION DASHBOARD")
    print("=" * 50)
    
    # Step 1: Generate all plots
    print("\n📊 Step 1: Generating all plots...")
    try:
        from generate_plots import main as generate_main
        generate_main()
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        return
    
    # Step 2: Create dashboard
    print("\n🎨 Step 2: Creating dashboard...")
    try:
        from create_dashboard import create_dashboard
        create_dashboard()
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        return
    
    # Step 3: Open dashboard
    print("\n🌐 Step 3: Opening dashboard...")
    dashboard_path = os.path.join("ma_optimization_plots", "dashboard.html")
    
    if os.path.exists(dashboard_path):
        print(f"✅ Dashboard ready!")
        print(f"📁 Location: {os.path.abspath(dashboard_path)}")
        print(f"🌐 Open this file in your browser to view all plots")
        
        # Try to open in browser
        try:
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', dashboard_path])
            elif sys.platform.startswith('win'):  # Windows
                subprocess.run(['start', dashboard_path], shell=True)
            elif sys.platform.startswith('linux'):  # Linux
                subprocess.run(['xdg-open', dashboard_path])
            print("🌐 Dashboard opened in your default browser!")
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print(f"   Please manually open: {os.path.abspath(dashboard_path)}")
    else:
        print(f"❌ Dashboard not found at {dashboard_path}")

if __name__ == "__main__":
    main()
