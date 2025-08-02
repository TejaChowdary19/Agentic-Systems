# simplified_visualizer.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

def create_visualizations(results_path):
    """Create visualizations from evaluation results."""
    print(f"Creating visualizations from: {results_path}")
    
    # Load the results
    with open(results_path, 'r') as f:
        evaluation_data = json.load(f)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(results_path), "visualizations")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract data
    results = evaluation_data.get("results", [])
    summary = evaluation_data.get("summary", {})
    
    if not results:
        print("No results data found in the file")
        return None
    
    # Organize data by dataset size
    sizes = []
    runtimes = []
    success_rates = []
    
    for size, data in summary.get("by_size", {}).items():
        sizes.append(int(size))
        runtimes.append(data.get("average_runtime", 0))
        
        tests = data.get("tests", 0)
        successful = data.get("successful", 0)
        success_rates.append((successful / tests * 100) if tests > 0 else 0)
    
    # Sort by size
    size_indices = np.argsort(sizes)
    sizes = [sizes[i] for i in size_indices]
    runtimes = [runtimes[i] for i in size_indices]
    success_rates = [success_rates[i] for i in size_indices]
    
    # Create visualization 1: Runtime by dataset size
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, runtimes, 'o-', linewidth=2, markersize=10)
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Runtime vs Dataset Size')
    plt.grid(True)
    plt.tight_layout()
    
    runtime_plot_path = os.path.join(output_dir, "runtime_by_size.png")
    plt.savefig(runtime_plot_path)
    plt.close()
    print(f"Created plot: {runtime_plot_path}")
    
    # Create visualization 2: Success rate by dataset size
    plt.figure(figsize=(10, 6))
    plt.bar(sizes, success_rates, color='green', alpha=0.7)
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate vs Dataset Size')
    plt.ylim(0, 105)  # Limit y-axis to 0-105%
    for i, rate in enumerate(success_rates):
        plt.text(sizes[i], rate + 2, f"{rate:.0f}%", ha='center')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    success_plot_path = os.path.join(output_dir, "success_rate_by_size.png")
    plt.savefig(success_plot_path)
    plt.close()
    print(f"Created plot: {success_plot_path}")
    
    # Analyze metrics if available
    metrics_data = {}
    step_runtimes = {}
    
    for result in results:
        if result.get("status") == "success" and "metrics_summary" in result:
            size = result.get("dataset_size")
            if size not in metrics_data:
                metrics_data[size] = []
            
            metrics_data[size].append(result["metrics_summary"])
            
            # Extract step runtimes
            if "efficiency" in result["metrics_summary"]:
                efficiency = result["metrics_summary"]["efficiency"]
                for key, value in efficiency.items():
                    if key.startswith("avg_") and key.endswith("_runtime") and key != "avg_runtime":
                        step = key.replace("avg_", "").replace("_runtime", "")
                        if step not in step_runtimes:
                            step_runtimes[step] = []
                        step_runtimes[step].append(value)
    
    # Create visualization 3: Step runtimes
    if step_runtimes:
        plt.figure(figsize=(12, 6))
        
        steps = list(step_runtimes.keys())
        avg_runtimes = [np.mean(step_runtimes[step]) for step in steps]
        
        # Sort by runtime
        sort_indices = np.argsort(avg_runtimes)
        steps = [steps[i] for i in sort_indices]
        avg_runtimes = [avg_runtimes[i] for i in sort_indices]
        
        plt.barh(steps, avg_runtimes, color='skyblue')
        plt.xlabel('Average Runtime (seconds)')
        plt.title('Runtime by Processing Step')
        for i, runtime in enumerate(avg_runtimes):
            plt.text(runtime + 0.05, i, f"{runtime:.2f}s", va='center')
        plt.grid(True, axis='x')
        plt.tight_layout()
        
        steps_plot_path = os.path.join(output_dir, "step_runtimes.png")
        plt.savefig(steps_plot_path)
        plt.close()
        print(f"Created plot: {steps_plot_path}")
    
    # Extract insight counts if available
    insight_counts = {}
    
    for result in results:
        if result.get("status") == "success" and "metrics_summary" in result:
            size = result.get("dataset_size")
            if "system" in result["metrics_summary"]:
                system = result["metrics_summary"]["system"]
                if "avg_insight_count" in system:
                    if size not in insight_counts:
                        insight_counts[size] = []
                    insight_counts[size].append(system["avg_insight_count"])
    
    # Create visualization 4: Insight counts by dataset size
    if insight_counts:
        plt.figure(figsize=(10, 6))
        
        sizes_with_insights = sorted(insight_counts.keys())
        avg_insights = [np.mean(insight_counts[size]) for size in sizes_with_insights]
        
        plt.plot(sizes_with_insights, avg_insights, 'o-', linewidth=2, markersize=10, color='orange')
        plt.xlabel('Dataset Size (rows)')
        plt.ylabel('Average Number of Insights')
        plt.title('Insight Generation vs Dataset Size')
        plt.grid(True)
        plt.tight_layout()
        
        insights_plot_path = os.path.join(output_dir, "insights_by_size.png")
        plt.savefig(insights_plot_path)
        plt.close()
        print(f"Created plot: {insights_plot_path}")
    
    # Create summary dashboard
    plt.figure(figsize=(12, 10))
    
    # Runtime plot
    plt.subplot(2, 2, 1)
    plt.plot(sizes, runtimes, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Dataset Size')
    plt.grid(True)
    
    # Success rate plot
    plt.subplot(2, 2, 2)
    plt.bar(sizes, success_rates, color='green', alpha=0.7)
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate vs Dataset Size')
    plt.ylim(0, 105)
    
    # Step runtimes
    if step_runtimes:
        plt.subplot(2, 2, 3)
        plt.barh(steps[:5] if len(steps) > 5 else steps, 
                avg_runtimes[:5] if len(avg_runtimes) > 5 else avg_runtimes, 
                color='skyblue')
        plt.xlabel('Runtime (seconds)')
        plt.title('Top 5 Processing Steps')
        plt.grid(True, axis='x')
    
    # Insight counts
    if insight_counts:
        plt.subplot(2, 2, 4)
        plt.plot(sizes_with_insights, avg_insights, 'o-', linewidth=2, markersize=8, color='orange')
        plt.xlabel('Dataset Size (rows)')
        plt.ylabel('Insights Count')
        plt.title('Insights vs Dataset Size')
        plt.grid(True)
    
    plt.tight_layout()
    
    dashboard_path = os.path.join(output_dir, "performance_dashboard.png")
    plt.savefig(dashboard_path)
    plt.close()
    print(f"Created dashboard: {dashboard_path}")
    
    # Create HTML report
    html_path = create_html_report(evaluation_data, output_dir)
    
    return output_dir

def create_html_report(evaluation_data, visualization_dir):
    """Create a simple HTML report."""
    results = evaluation_data.get("results", [])
    summary = evaluation_data.get("summary", {})
    
    # Get visualization files
    vis_files = [f for f in os.listdir(visualization_dir) if f.endswith('.png')]
    
    # Create HTML content
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis System Performance Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            .section { margin-bottom: 30px; }
            .metric { margin-bottom: 10px; }
            .metric-name { font-weight: bold; width: 200px; display: inline-block; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .visualization { margin: 20px 0; text-align: center; }
            .visualization img { max-width: 100%; border: 1px solid #ddd; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        <h1>Data Analysis System Performance Report</h1>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <div class="section">
            <h2>Test Summary</h2>
            <div class="metric">
                <span class="metric-name">Total Tests:</span> """ + str(summary.get("total_tests", 0)) + """
            </div>
            <div class="metric">
                <span class="metric-name">Successful Tests:</span> """ + str(summary.get("successful_tests", 0)) + """
            </div>
            <div class="metric">
                <span class="metric-name">Failed Tests:</span> """ + str(summary.get("failed_tests", 0)) + """
            </div>
            <div class="metric">
                <span class="metric-name">Average Runtime:</span> """ + f"{summary.get('average_runtime', 0):.2f} seconds" + """
            </div>
        </div>
        
        <div class="section">
            <h2>Performance by Dataset Size</h2>
            <table>
                <tr>
                    <th>Dataset Size</th>
                    <th>Tests</th>
                    <th>Success Rate</th>
                    <th>Average Runtime</th>
                </tr>
    """
    
    # Add performance by size
    for size, data in summary.get("by_size", {}).items():
        tests = data.get("tests", 0)
        successful = data.get("successful", 0)
        success_rate = (successful / tests * 100) if tests > 0 else 0
        
        html += f"""
                <tr>
                    <td>{size} rows</td>
                    <td>{tests}</td>
                    <td>{success_rate:.0f}%</td>
                    <td>{data.get('average_runtime', 0):.2f} seconds</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
    """
    
    # Add visualizations
    for vis_file in vis_files:
        # Create a nicer title
        title = vis_file.replace('.png', '').replace('_', ' ').title()
        
        html += f"""
            <div class="visualization">
                <h3>{title}</h3>
                <img src="{vis_file}" alt="{title}">
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_path = os.path.join(visualization_dir, "performance_report.html")
    with open(html_path, 'w') as f:
        f.write(html)
    
    print(f"Created HTML report: {html_path}")
    return html_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        create_visualizations(results_path)
    else:
        print("Usage: python simplified_visualizer.py <results_file.json>")