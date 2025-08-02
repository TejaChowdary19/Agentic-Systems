# run_evaluation.py

import os
import sys
import time
from datetime import datetime

# Add utility folder to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the metrics testing module
try:
    from metrics_test import run_metrics_test, generate_metrics_report
    
    # Try to import the visualization module, but continue even if it fails
    try:
        from metrics_visualizer import visualize_metrics_report, create_html_report
        visualization_available = True
    except ImportError:
        print("Warning: Visualization module could not be imported. Only metrics collection will be available.")
        visualization_available = False
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure all required modules are available.")
    sys.exit(1)

def run_evaluation(dataset_sizes=None, noise_levels=None, repetitions=2):
    """Run a full evaluation of the data analysis system."""
    print("\n====== DATA ANALYSIS SYSTEM EVALUATION ======\n")
    
    # Use default values if not provided
    if dataset_sizes is None:
        dataset_sizes = [100, 500, 1000]
    
    if noise_levels is None:
        noise_levels = [0.1, 0.3, 0.5]
    
    start_time = time.time()
    
    print(f"Starting evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Noise levels: {noise_levels}")
    print(f"Repetitions: {repetitions}")
    print("\n")
    
    # Run the metrics tests
    results_path = run_metrics_test(dataset_sizes, noise_levels, repetitions)
    
    # Generate metrics report
    report_path = generate_metrics_report(results_path)
    
    # Create visualizations if available
    if visualization_available:
        try:
            vis_dir = visualize_metrics_report(report_path)
            html_path = create_html_report(report_path, vis_dir)
            visualization_output = f"HTML Report: {html_path}"
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
            visualization_output = "Visualizations could not be created."
    else:
        visualization_output = "Visualization module not available."
    
    end_time = time.time()
    
    print("\n====== EVALUATION COMPLETE ======")
    print(f"Total evaluation time: {end_time - start_time:.2f} seconds")
    print(f"Results available at: {os.path.abspath(os.path.dirname(report_path))}")
    print(visualization_output)
    
    return report_path

if __name__ == "__main__":
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        try:
            # Format should be: dataset_sizes noise_levels repetitions
            # Example: "100,500,1000" "0.1,0.3,0.5" 2
            dataset_sizes = [int(s) for s in sys.argv[1].split(',')]
            noise_levels = [float(n) for n in sys.argv[2].split(',')] if len(sys.argv) > 2 else None
            repetitions = int(sys.argv[3]) if len(sys.argv) > 3 else 2
            
            run_evaluation(dataset_sizes, noise_levels, repetitions)
        except Exception as e:
            print(f"Error parsing arguments: {str(e)}")
            print("Usage: python run_evaluation.py [dataset_sizes] [noise_levels] [repetitions]")
            print("Example: python run_evaluation.py 100,500,1000 0.1,0.3,0.5 2")
            sys.exit(1)
    else:
        # Use default values
        run_evaluation()