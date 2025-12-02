#!/usr/bin/env python3
"""
Create a simple HTML dashboard to view all generated plots
"""

import os
import glob

def create_dashboard(plot_dir="ma_optimization_plots"):
    """Create an HTML dashboard with all generated plots
    
    Args:
        plot_dir (str): Directory containing the plot HTML files (default: "ma_optimization_plots")
    """
    
    # Find all HTML files in the plots directory
    if not os.path.exists(plot_dir):
        print(f"❌ Directory '{plot_dir}' not found. Run generate_plots.py first.")
        return
    
    html_files = glob.glob(os.path.join(plot_dir, "*.html"))
    html_files = [f for f in html_files if not f.endswith("dashboard.html")]
    
    if not html_files:
        print(f"❌ No HTML files found in '{plot_dir}' directory.")
        return
    
    print(f"📊 Found {len(html_files)} plot files")
    
    # Create dashboard HTML
    dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MA Optimization Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .plot-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .plot-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        .plot-title {{
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .plot-iframe {{
            width: 100%;
            height: 500px;
            border: none;
            border-radius: 5px;
        }}
        .plot-link {{
            display: inline-block;
            background: #667eea;
            color: white;
            text-decoration: none;
            padding: 10px 20px;
            border-radius: 5px;
            margin-top: 10px;
            transition: background 0.3s ease;
        }}
        .plot-link:hover {{
            background: #5a6fd8;
        }}
        .stats {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .stats h3 {{
            color: #333;
            margin-top: 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat-item {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 MA Optimization Dashboard</h1>
        <p>Interactive 3D Visualizations for Moving Average Crossover Strategy</p>
    </div>
    
    <div class="stats">
        <h3>📈 Dashboard Statistics</h3>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-number">{len(html_files)}</div>
                <div class="stat-label">Total Plots</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">3D</div>
                <div class="stat-label">Visualizations</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">Interactive</div>
                <div class="stat-label">Plotly Charts</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">Real-time</div>
                <div class="stat-label">Analysis</div>
            </div>
        </div>
    </div>
    
    <div class="plots-grid">
"""

    # Add each plot to the dashboard
    for i, html_file in enumerate(sorted(html_files)):
        filename = os.path.basename(html_file)
        plot_name = filename.replace('.html', '').replace('_', ' ').title()
        
        # Create a more readable name
        plot_name = plot_name.replace('Ma ', 'MA ').replace('Rr', 'RR').replace('3d', '3D')
        
        dashboard_html += f"""
        <div class="plot-card">
            <div class="plot-title">{plot_name}</div>
            <iframe src="{filename}" class="plot-iframe"></iframe>
            <a href="{filename}" target="_blank" class="plot-link">🔗 Open in New Tab</a>
        </div>
"""

    dashboard_html += """
    </div>
    
    <div class="footer">
        <p>Generated by MA Optimization Visualizer | Interactive 3D Trading Strategy Analysis</p>
    </div>
</body>
</html>
"""

    # Save dashboard
    dashboard_path = os.path.join(plot_dir, "dashboard.html")
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    
    print(f"✅ Dashboard created: {dashboard_path}")
    print(f"🌐 Open {dashboard_path} in your browser to view all plots!")

if __name__ == "__main__":
    create_dashboard()
