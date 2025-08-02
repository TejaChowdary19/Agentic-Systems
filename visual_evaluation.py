# visual_evaluation.py

import os
import sys
import time
from datetime import datetime
import json
import pandas as pd
import numpy as np
from data_analysis_system.simplified_main import run_data_analysis
from simplified_visualizer import create_visualizations

def create_test_dataset(size=100, noise_level=0.2, file_prefix="test"):
    """Create a test dataset with controllable characteristics."""
    # Create a directory for test data
    if not os.path.exists("test_data"):
        os.makedirs("test_data")
    
    # Set random seed for reproducibility
    np.random.seed(int(time.time()) % 1000)
    
    # Create sample dates
    date_range = pd.date_range(start='2023-01-01', periods=size, freq='D')
    
    # Create features with clear relationships
    x = np.linspace(0, 10, size)
    y = 2 * x + 1 + np.random.normal(0, noise_level * 5, size)  # Linear relationship with controllable noise
    z = np.sin(x) + np.random.normal(0, noise_level * 2, size)  # Sinusoidal relationship
    
    # Create categorical variable
    categories = np.random.choice(['A', 'B', 'C'], size)
    
    # Make category A have higher y values
    y[categories == 'A'] += 2
    
    # Add outliers
    outliers_count = int(size * 0.05)  # 5% outliers
    if outliers_count > 0:
        outlier_indices = np.random.choice(size, outliers_count, replace=False)
        y[outlier_indices] = y[outlier_indices] + np.random.choice([-1, 1], outliers_count) * np.random.uniform(10, 20, outliers_count)
    
    # Create the DataFrame
    data = {
        'date': date_range,
        'x': x,
        'y': y,
        'z': z,
        'category': categories
    }
    
    df = pd.DataFrame(data)
    
    # Add missing values
    missing_count = int(size * 0.05 * 3)  # 5% missing values in 3 columns
    if missing_count > 0:
        for _ in range(missing_count):
            row = np.random.randint(0, size)
            col = np.random.choice(['x', 'y', 'z'])  # Only add missing values to numeric columns
            df.loc[row, col] = np.nan
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"test_data/{file_prefix}_{size}_{timestamp}.csv"
    df.to_csv(file_path, index=False)
    
    print(f"Created test dataset: {file_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Added approximately {outliers_count} outliers")
    
    return file_path

def run_visual_evaluation(dataset_sizes=None, repetitions=2):
    """Run an evaluation of the data analysis system with visualizations."""
    print("\n====== DATA ANALYSIS SYSTEM EVALUATION ======\n")
    
    # Use default values if not provided
    if dataset_sizes is None:
        dataset_sizes = [100, 500]  # Use smaller sizes for quicker testing
    
    start_time = time.time()
    
    print(f"Starting evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Repetitions: {repetitions}")
    print("\n")
    
    # Create output directory
    base_output_dir = f"visual_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Initialize results
    evaluation_results = {
        "configuration": {
            "dataset_sizes": dataset_sizes,
            "repetitions": repetitions,
            "timestamp": datetime.now().isoformat()
        },
        "results": []
    }
    
    # Run tests for each dataset size
    for size in dataset_sizes:
        print(f"\n===== Testing with dataset size={size} =====")
        
        for rep in range(repetitions):
            print(f"\nRepetition {rep+1}/{repetitions}")
            
            # Create test dataset
            data_path = create_test_dataset(size=size, file_prefix=f"visual_test")
            
            # Set up arguments
            class Args:
                source_type = "file"
                data_source = data_path
                objective = f"Analyze test dataset (size={size}, rep={rep+1})"
                business_context = "Performance testing"
                target_column = "y"
                output_dir = os.path.join(base_output_dir, f"size_{size}_rep_{rep+1}")
            
            # Set up configuration
            config = {
                "output_dir": Args.output_dir,
                "collect_metrics": True,
                "metrics_dir": os.path.join(base_output_dir, "metrics")
            }
            
            # Ensure output directory exists
            if not os.path.exists(Args.output_dir):
                os.makedirs(Args.output_dir)
            
            # Run the analysis
            print(f"Running analysis on {data_path}...")
            test_start_time = time.time()
            
            try:
                result = run_data_analysis(Args, config)
                test_end_time = time.time()
                
                # Record test results
                test_result = {
                    "dataset_size": size,
                    "repetition": rep + 1,
                    "status": result["status"],
                    "runtime": test_end_time - test_start_time,
                    "output_dir": Args.output_dir,
                    "data_path": data_path
                }
                
                if result["status"] == "success" and "metrics_summary" in result:
                    test_result["metrics_summary"] = result["metrics_summary"]
                    
                    # Print metrics summary
                    print(f"Analysis completed in {test_end_time - test_start_time:.2f} seconds")
                    print(f"Status: {result['status']}")
                    
                    if "efficiency" in result["metrics_summary"]:
                        efficiency = result["metrics_summary"]["efficiency"]
                        if "avg_runtime" in efficiency:
                            print(f"Total processing time: {efficiency['avg_runtime']:.2f} seconds")
                        
                        # Print step runtimes
                        for key, value in efficiency.items():
                            if key.startswith("avg_") and key.endswith("_runtime") and key != "avg_runtime":
                                step = key.replace("avg_", "").replace("_runtime", "")
                                print(f"  {step}: {value:.2f} seconds")
                    
                    if "system" in result["metrics_summary"]:
                        system = result["metrics_summary"]["system"]
                        if "avg_insight_count" in system:
                            print(f"Insights generated: {system['avg_insight_count']:.0f}")
                        
                        if "avg_recommendation_count" in system:
                            print(f"Recommendations: {system['avg_recommendation_count']:.0f}")
                else:
                    print(f"Analysis failed: {result.get('error_message', 'Unknown error')}")
                
                evaluation_results["results"].append(test_result)
                
            except Exception as e:
                test_end_time = time.time()
                print(f"Error during analysis: {str(e)}")
                
                # Record error
                evaluation_results["results"].append({
                    "dataset_size": size,
                    "repetition": rep + 1,
                    "status": "error",
                    "error_message": str(e),
                    "runtime": test_end_time - test_start_time,
                    "output_dir": Args.output_dir,
                    "data_path": data_path
                })
    
    # Calculate summary statistics
    summary = {
        "total_tests": len(evaluation_results["results"]),
        "successful_tests": sum(1 for r in evaluation_results["results"] if r["status"] == "success"),
        "failed_tests": sum(1 for r in evaluation_results["results"] if r["status"] != "success"),
        "average_runtime": sum(r.get("runtime", 0) for r in evaluation_results["results"]) / len(evaluation_results["results"]) if evaluation_results["results"] else 0,
        "by_size": {}
    }
    
    # Calculate statistics by dataset size
    for size in dataset_sizes:
        size_results = [r for r in evaluation_results["results"] if r["dataset_size"] == size]
        if size_results:
            summary["by_size"][str(size)] = {
                "tests": len(size_results),
                "successful": sum(1 for r in size_results if r["status"] == "success"),
                "average_runtime": sum(r.get("runtime", 0) for r in size_results) / len(size_results)
            }
    
    evaluation_results["summary"] = summary
    
    # Save results
    results_path = os.path.join(base_output_dir, f"evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    end_time = time.time()
    
    print("\n====== EVALUATION COMPLETE ======")
    print(f"Total evaluation time: {end_time - start_time:.2f} seconds")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Successful tests: {summary['successful_tests']}")
    print(f"Failed tests: {summary['failed_tests']}")
    print(f"Average runtime: {summary['average_runtime']:.2f} seconds")
    
    print("\nPerformance by dataset size:")
    for size, data in summary["by_size"].items():
        print(f"  Size {size}: {data['average_runtime']:.2f} seconds (Success rate: {data['successful']/data['tests']*100 if data['tests'] > 0 else 0:.0f}%)")
    
    print(f"\nDetailed results saved to: {results_path}")
    
    # Create visualizations
    try:
        print("\nGenerating visualizations...")
        vis_dir = create_visualizations(results_path)
        print(f"Visualizations created in: {vis_dir}")
        print(f"View the HTML report at: {os.path.join(vis_dir, 'performance_report.html')}")
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
    
    return base_output_dir

if __name__ == "__main__":
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        try:
            # Format should be: dataset_sizes repetitions
            # Example: "100,500,1000" 2
            dataset_sizes = [int(s) for s in sys.argv[1].split(',')]
            repetitions = int(sys.argv[2]) if len(sys.argv) > 2 else 2
            
            run_visual_evaluation(dataset_sizes, repetitions)
        except Exception as e:
            print(f"Error parsing arguments: {str(e)}")
            print("Usage: python visual_evaluation.py [dataset_sizes] [repetitions]")
            print("Example: python visual_evaluation.py 100,500,1000 2")
            sys.exit(1)
    else:
        # Use default values
        run_visual_evaluation()