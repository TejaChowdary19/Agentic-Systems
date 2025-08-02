# metrics_test.py

import os
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

# Import the simplified main module
# Note: Update the import to use the updated simplified main
from data_analysis_system.simplified_main import run_data_analysis

def create_test_dataset(size, noise_level=0.2, outliers_pct=0.05, missing_pct=0.05, file_prefix="test"):
    """Create a test dataset with controllable characteristics."""
    # Create a directory for test data
    if not os.path.exists("test_data"):
        os.makedirs("test_data")
    
    # Set random seed for reproducibility
    np.random.seed(int(time.time()))
    
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
    outliers_count = int(size * outliers_pct)
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
    missing_count = int(size * missing_pct * len(df.columns))
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

def run_metrics_test(dataset_sizes, noise_levels, repetitions=3):
    """Run multiple analyses with different dataset characteristics to collect metrics."""
    test_results = []
    
    # Test with different dataset sizes
    for size in dataset_sizes:
        for noise in noise_levels:
            print(f"\n===== Testing with dataset size={size}, noise={noise} =====")
            
            for rep in range(repetitions):
                print(f"\nRepetition {rep+1}/{repetitions}")
                
                # Create test dataset
                data_path = create_test_dataset(size=size, noise_level=noise, file_prefix=f"metrics_test")
                
                # Set up arguments
                class Args:
                    source_type = "file"
                    data_source = data_path
                    objective = f"Analyze test dataset (size={size}, noise={noise}, rep={rep+1})"
                    business_context = "Performance testing"
                    target_column = "y"
                    output_dir = f"metrics_test_outputs/size_{size}_noise_{noise}_rep_{rep+1}"
                
                # Set up configuration
                config = {
                    "output_dir": Args.output_dir,
                    "collect_metrics": True,
                    "metrics_dir": "metrics_test_results"
                }
                
                # Ensure output directory exists
                if not os.path.exists(Args.output_dir):
                    os.makedirs(Args.output_dir)
                
                # Run the analysis
                print(f"Running analysis on {data_path}...")
                start_time = time.time()
                
                try:
                    result = run_data_analysis(Args, config)
                    end_time = time.time()
                    
                    # Record basic test results
                    test_result = {
                        "dataset_size": size,
                        "noise_level": noise,
                        "repetition": rep + 1,
                        "status": result["status"],
                        "runtime": end_time - start_time,
                        "output_dir": Args.output_dir,
                        "data_path": data_path
                    }
                    
                    if result["status"] == "success" and "metrics_summary" in result:
                        test_result["metrics_summary"] = result["metrics_summary"]
                    
                    test_results.append(test_result)
                    
                    print(f"Analysis completed in {end_time - start_time:.2f} seconds")
                    print(f"Status: {result['status']}")
                    
                    # Print metrics summary if available
                    if result["status"] == "success" and "metrics_summary" in result:
                        metrics = result["metrics_summary"]
                        if "efficiency" in metrics and "avg_runtime" in metrics["efficiency"]:
                            print(f"Reported runtime: {metrics['efficiency']['avg_runtime']:.2f} seconds")
                        
                        if "system" in metrics:
                            system = metrics["system"]
                            if "avg_insight_count" in system:
                                print(f"Insights generated: {system['avg_insight_count']:.0f}")
                            
                            if "avg_recommendation_count" in system:
                                print(f"Recommendations: {system['avg_recommendation_count']:.0f}")
                    
                except Exception as e:
                    end_time = time.time()
                    print(f"Error during analysis: {str(e)}")
                    
                    # Record error
                    test_results.append({
                        "dataset_size": size,
                        "noise_level": noise,
                        "repetition": rep + 1,
                        "status": "error",
                        "error_message": str(e),
                        "runtime": end_time - start_time,
                        "output_dir": Args.output_dir,
                        "data_path": data_path
                    })
    
    # Save test results
    results_dir = "metrics_test_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_path = os.path.join(results_dir, f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump({
            "test_configuration": {
                "dataset_sizes": dataset_sizes,
                "noise_levels": noise_levels,
                "repetitions": repetitions,
                "timestamp": datetime.now().isoformat()
            },
            "results": test_results
        }, f, indent=2)
    
    print(f"\nTest results saved to: {results_path}")
    return results_path

def generate_metrics_report(results_path):
    """Generate a report analyzing the metrics collected during testing."""
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    # Load test results
    with open(results_path, 'r') as f:
        test_data = json.load(f)
    
    test_config = test_data["test_configuration"]
    results = test_data["results"]
    
    # Prepare report data
    report = {
        "test_configuration": test_config,
        "summary": {
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r["status"] == "success"),
            "failed_tests": sum(1 for r in results if r["status"] != "success"),
            "average_runtime": sum(r.get("runtime", 0) for r in results) / len(results) if results else 0
        },
        "by_dataset_size": {},
        "by_noise_level": {},
        "performance_metrics": {},
        "insights": []
    }
    
    # Analyze by dataset size
    for size in test_config["dataset_sizes"]:
        size_results = [r for r in results if r["dataset_size"] == size]
        if size_results:
            report["by_dataset_size"][str(size)] = {
                "tests": len(size_results),
                "successful": sum(1 for r in size_results if r["status"] == "success"),
                "average_runtime": sum(r.get("runtime", 0) for r in size_results) / len(size_results)
            }
    
    # Analyze by noise level
    for noise in test_config["noise_levels"]:
        noise_results = [r for r in results if r["noise_level"] == noise]
        if noise_results:
            report["by_noise_level"][str(noise)] = {
                "tests": len(noise_results),
                "successful": sum(1 for r in noise_results if r["status"] == "success"),
                "average_runtime": sum(r.get("runtime", 0) for r in noise_results) / len(noise_results)
            }
    
    # Extract performance metrics
    metrics_keys = set()
    for result in results:
        if result["status"] == "success" and "metrics_summary" in result:
            for category in result["metrics_summary"]:
                for metric in result["metrics_summary"][category]:
                    metrics_keys.add(f"{category}.{metric}")
    
    # Initialize performance metrics
    for key in metrics_keys:
        report["performance_metrics"][key] = {
            "values": [],
            "by_dataset_size": {},
            "by_noise_level": {}
        }
        
        # Initialize by dataset size
        for size in test_config["dataset_sizes"]:
            report["performance_metrics"][key]["by_dataset_size"][str(size)] = []
        
        # Initialize by noise level
        for noise in test_config["noise_levels"]:
            report["performance_metrics"][key]["by_noise_level"][str(noise)] = []
    
    # Populate performance metrics
    for result in results:
        if result["status"] == "success" and "metrics_summary" in result:
            for category in result["metrics_summary"]:
                for metric, value in result["metrics_summary"][category].items():
                    key = f"{category}.{metric}"
                    
                    # Skip non-numeric values
                    if not isinstance(value, (int, float)):
                        continue
                    
                    # Add to overall values
                    report["performance_metrics"][key]["values"].append(value)
                    
                    # Add to dataset size specific values
                    size_key = str(result["dataset_size"])
                    report["performance_metrics"][key]["by_dataset_size"][size_key].append(value)
                    
                    # Add to noise level specific values
                    noise_key = str(result["noise_level"])
                    report["performance_metrics"][key]["by_noise_level"][noise_key].append(value)
    
    # Calculate averages for performance metrics
    for key in report["performance_metrics"]:
        values = report["performance_metrics"][key]["values"]
        if values:
            report["performance_metrics"][key]["average"] = sum(values) / len(values)
        
        # Calculate averages by dataset size
        for size in report["performance_metrics"][key]["by_dataset_size"]:
            size_values = report["performance_metrics"][key]["by_dataset_size"][size]
            if size_values:
                report["performance_metrics"][key]["by_dataset_size"][size] = sum(size_values) / len(size_values)
        
        # Calculate averages by noise level
        for noise in report["performance_metrics"][key]["by_noise_level"]:
            noise_values = report["performance_metrics"][key]["by_noise_level"][noise]
            if noise_values:
                report["performance_metrics"][key]["by_noise_level"][noise] = sum(noise_values) / len(noise_values)
    
    # Generate insights
    
    # Insight 1: Runtime scaling with dataset size
    runtime_by_size = {}
    for size in test_config["dataset_sizes"]:
        size_results = [r for r in results if r["dataset_size"] == size]
        if size_results:
            runtime_by_size[size] = sum(r.get("runtime", 0) for r in size_results) / len(size_results)
    
    if len(runtime_by_size) >= 2:
        sizes = sorted(runtime_by_size.keys())
        smallest_size = sizes[0]
        largest_size = sizes[-1]
        
        if runtime_by_size[smallest_size] > 0:
            scaling_factor = runtime_by_size[largest_size] / runtime_by_size[smallest_size]
            size_ratio = largest_size / smallest_size
            
            report["insights"].append({
                "type": "runtime_scaling",
                "description": f"Runtime scales by a factor of {scaling_factor:.2f}x when dataset size increases by {size_ratio:.2f}x",
                "details": {
                    "smallest_size": smallest_size,
                    "largest_size": largest_size,
                    "smallest_runtime": runtime_by_size[smallest_size],
                    "largest_runtime": runtime_by_size[largest_size]
                }
            })
    
    # Insight 2: Effect of noise on accuracy
    if "accuracy.correlation_significance" in report["performance_metrics"]:
        metric = report["performance_metrics"]["accuracy.correlation_significance"]
        if metric.get("by_noise_level"):
            noise_levels = sorted([float(n) for n in metric["by_noise_level"].keys()])
            if len(noise_levels) >= 2:
                lowest_noise = str(noise_levels[0])
                highest_noise = str(noise_levels[-1])
                
                if lowest_noise in metric["by_noise_level"] and highest_noise in metric["by_noise_level"]:
                    low_accuracy = metric["by_noise_level"][lowest_noise]
                    high_accuracy = metric["by_noise_level"][highest_noise]
                    
                    accuracy_change = ((high_accuracy - low_accuracy) / low_accuracy * 100) if low_accuracy else 0
                    
                    report["insights"].append({
                        "type": "noise_effect",
                        "description": f"Increasing noise from {float(lowest_noise):.2f} to {float(highest_noise):.2f} changes correlation significance by {accuracy_change:.2f}%",
                        "details": {
                            "lowest_noise": float(lowest_noise),
                            "highest_noise": float(highest_noise),
                            "lowest_noise_accuracy": low_accuracy,
                            "highest_noise_accuracy": high_accuracy
                        }
                    })
    
    # Insight 3: Insight generation performance
    if "system.avg_insight_count" in report["performance_metrics"]:
        metric = report["performance_metrics"]["system.avg_insight_count"]
        if metric.get("by_dataset_size"):
            sizes = sorted([int(s) for s in metric["by_dataset_size"].keys()])
            if len(sizes) >= 2:
                smallest_size = str(sizes[0])
                largest_size = str(sizes[-1])
                
                if smallest_size in metric["by_dataset_size"] and largest_size in metric["by_dataset_size"]:
                    small_insights = metric["by_dataset_size"][smallest_size]
                    large_insights = metric["by_dataset_size"][largest_size]
                    
                    insight_ratio = large_insights / small_insights if small_insights else 0
                    size_ratio = int(largest_size) / int(smallest_size)
                    
                    report["insights"].append({
                        "type": "insight_scaling",
                        "description": f"Number of insights scales by a factor of {insight_ratio:.2f}x when dataset size increases by {size_ratio:.2f}x",
                        "details": {
                            "smallest_size": int(smallest_size),
                            "largest_size": int(largest_size),
                            "smallest_size_insights": small_insights,
                            "largest_size_insights": large_insights
                        }
                    })
    
    # Save report
    report_path = os.path.join(os.path.dirname(results_path), f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print report summary
    print("\n===== Metrics Report Summary =====")
    print(f"Total tests: {report['summary']['total_tests']}")
    print(f"Successful tests: {report['summary']['successful_tests']}")
    print(f"Failed tests: {report['summary']['failed_tests']}")
    print(f"Average runtime: {report['summary']['average_runtime']:.2f} seconds")
    
    print("\nPerformance by dataset size:")
    for size, data in report["by_dataset_size"].items():
        print(f"  Size {size}: {data['average_runtime']:.2f} seconds")
    
    print("\nPerformance by noise level:")
    for noise, data in report["by_noise_level"].items():
        print(f"  Noise {noise}: {data['average_runtime']:.2f} seconds")
    
    print("\nKey Insights:")
    for insight in report["insights"]:
        print(f"  {insight['description']}")
    
    print(f"\nDetailed report saved to: {report_path}")
    return report_path

if __name__ == "__main__":
    # Configure test parameters
    dataset_sizes = [100, 500, 1000]  # Vary sizes to test scalability
    noise_levels = [0.1, 0.3, 0.5]    # Vary noise to test robustness
    repetitions = 2                   # Number of repetitions for each configuration
    
    # Run the metrics tests
    results_path = run_metrics_test(dataset_sizes, noise_levels, repetitions)
    
    # Generate and print the metrics report
    generate_metrics_report(results_path)