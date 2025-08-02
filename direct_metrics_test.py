# direct_metrics_test.py
# A direct approach to testing efficiency and reliability metrics

import pandas as pd
import numpy as np
import time
import os
import json
import psutil
from datetime import datetime, timedelta

from data_analysis_system.simplified_main import SimplifiedOrchestrator
from data_analysis_system.tools.data_retrieval import DataRetrievalTool
from data_analysis_system.tools.data_cleaning import DataCleaningTool
from data_analysis_system.tools.statistical_analysis import StatisticalAnalysisTool
from data_analysis_system.tools.insight_generator import InsightGeneratorTool

def create_test_dataset(size=1000, output_dir="test_datasets"):
    """Create a test dataset with known patterns."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create date range
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(size)]
    
    # Create features with clear patterns
    np.random.seed(42)
    x1 = np.random.normal(0, 1, size)
    x2 = np.random.normal(0, 1, size)
    
    # Create a trend
    trend = np.linspace(0, 10, size) + np.random.normal(0, 1, size)
    
    # Create target with clear relationship to features
    y = 2 * x1 + 0.5 * x2 + np.random.normal(0, 1, size)
    
    # Create categorical variable
    categories = np.random.choice(['A', 'B', 'C'], size)
    
    # Make category A have higher y values
    y[categories == 'A'] += 2
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'x1': x1,
        'x2': x2,
        'trend': trend,
        'target': y,
        'category': categories
    })
    
    # Add some outliers
    df.loc[10, 'target'] = 20  # Add a positive outlier
    df.loc[20, 'target'] = -15  # Add a negative outlier
    
    # Save to CSV
    filename = os.path.join(output_dir, f"test_data_{size}.csv")
    df.to_csv(filename, index=False)
    
    return filename

def test_efficiency_metrics(dataset_sizes=[100, 500, 1000], output_dir="efficiency_test_results"):
    """Test efficiency metrics directly."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {
        "dataset_sizes": dataset_sizes,
        "processing_times": [],
        "memory_usages": [],
        "cpu_usages": []
    }
    
    process = psutil.Process(os.getpid())
    
    for size in dataset_sizes:
        print(f"\nTesting efficiency with {size} rows")
        
        # Create test dataset
        filename = create_test_dataset(size)
        print(f"Created test dataset: {filename}")
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Start timing
        start_time = time.time()
        
        # Create orchestrator
        orchestrator = SimplifiedOrchestrator({"output_dir": output_dir})
        
        # Run the analysis
        context = {
            "data_source": {
                "source_type": "file",
                "source_path": filename
            },
            "analysis_objective": "Efficiency testing",
            "target_column": "target"
        }
        
        try:
            # Monitor CPU during execution
            cpu_samples = []
            
            def sample_cpu():
                return process.cpu_percent()
            
            # Run the orchestrator
            result = orchestrator.run(context)
            
            # Sample CPU a few times during execution
            for _ in range(10):
                cpu_samples.append(sample_cpu())
                time.sleep(0.1)
            
            # End timing
            end_time = time.time()
            
            # Get final memory usage
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Calculate metrics
            processing_time = end_time - start_time
            memory_usage = final_memory - initial_memory
            avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
            
            # Store results
            results["processing_times"].append(processing_time)
            results["memory_usages"].append(memory_usage)
            results["cpu_usages"].append(avg_cpu)
            
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Memory usage: {memory_usage:.2f} MB")
            print(f"Avg CPU usage: {avg_cpu:.2f}%")
            
        except Exception as e:
            print(f"Error during execution: {str(e)}")
            results["processing_times"].append(None)
            results["memory_usages"].append(None)
            results["cpu_usages"].append(None)
    
    # Calculate summary metrics
    valid_times = [t for t in results["processing_times"] if t is not None]
    valid_sizes = [s for i, s in enumerate(dataset_sizes) if results["processing_times"][i] is not None]
    
    if valid_times and valid_sizes:
        time_per_row = sum(t/s for t, s in zip(valid_times, valid_sizes)) / len(valid_times)
        results["summary"] = {
            "avg_time_per_100_rows": time_per_row * 100
        }
        
        if len(valid_times) >= 2:
            # Estimate base memory and memory per row
            mem_usages = [m for m in results["memory_usages"] if m is not None]
            
            # Simple linear regression to estimate base memory and per-row usage
            x = np.array(valid_sizes).reshape(-1, 1)
            y = np.array(mem_usages)
            
            # Using numpy's polyfit to estimate slope and intercept
            slope, intercept = np.polyfit(x.flatten(), y, 1)
            
            results["summary"]["base_memory_mb"] = intercept
            results["summary"]["memory_per_row_kb"] = slope * 1024  # Convert to KB
        
        # Average CPU usage
        valid_cpu = [c for c in results["cpu_usages"] if c is not None]
        if valid_cpu:
            results["summary"]["avg_cpu_percent"] = sum(valid_cpu) / len(valid_cpu)
    
    # Save results
    results_file = os.path.join(output_dir, "efficiency_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEfficiency results saved to: {results_file}")
    
    # Print summary
    if "summary" in results:
        summary = results["summary"]
        print("\nEfficiency Summary:")
        print(f"Avg. time per 100 rows: {summary.get('avg_time_per_100_rows', 0):.2f} seconds")
        print(f"Base memory usage: {summary.get('base_memory_mb', 0):.2f} MB")
        print(f"Memory per row: {summary.get('memory_per_row_kb', 0):.2f} KB")
        print(f"Avg. CPU usage: {summary.get('avg_cpu_percent', 0):.2f}%")
    
    return results

def test_reliability_metrics(runs=10, dataset_size=500, output_dir="reliability_test_results"):
    """Test reliability metrics directly."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {
        "total_runs": runs,
        "successful_runs": 0,
        "error_runs": 0,
        "consistency_scores": []
    }
    
    # Create test dataset once for consistency tests
    print(f"\nCreating test dataset for reliability testing ({dataset_size} rows)")
    filename = create_test_dataset(dataset_size)
    print(f"Created test dataset: {filename}")
    
    # Create error test datasets
    error_datasets = []
    
    # 1. Dataset with missing values
    df_missing = pd.read_csv(filename)
    missing_idx = np.random.choice(len(df_missing), size=int(len(df_missing)*0.2), replace=False)
    df_missing.loc[missing_idx, 'target'] = np.nan
    missing_file = os.path.join(os.path.dirname(filename), "missing_values.csv")
    df_missing.to_csv(missing_file, index=False)
    error_datasets.append(("missing_values", missing_file))
    
    # 2. Dataset with invalid format
    invalid_file = os.path.join(os.path.dirname(filename), "invalid_format.csv")
    with open(invalid_file, 'w') as f:
        f.write("This is not a valid CSV file format.")
    error_datasets.append(("invalid_format", invalid_file))
    
    # First, run multiple times on the same dataset to check consistency
    print("\nRunning consistency tests...")
    insights_results = []
    
    for i in range(runs):
        print(f"Consistency run {i+1}/{runs}")
        
        # Create new orchestrator for each run
        orchestrator = SimplifiedOrchestrator({"output_dir": output_dir})
        
        # Run the analysis
        context = {
            "data_source": {
                "source_type": "file",
                "source_path": filename
            },
            "analysis_objective": "Reliability testing",
            "target_column": "target"
        }
        
        try:
            result = orchestrator.run(context)
            
            if result["status"] == "success":
                results["successful_runs"] += 1
                
                # Get insights for consistency comparison
                insights_path = result.get("file_paths", {}).get("insights")
                if insights_path and os.path.exists(insights_path):
                    with open(insights_path, 'r') as f:
                        insights = json.load(f)
                    insights_results.append(insights)
            else:
                results["error_runs"] += 1
                print(f"  Run failed with status: {result['status']}")
        
        except Exception as e:
            results["error_runs"] += 1
            print(f"  Error during execution: {str(e)}")
    
    # Calculate consistency score if we have multiple successful runs
    if len(insights_results) >= 2:
        consistency_scores = []
        
        # Compare each pair of results
        for i in range(len(insights_results)):
            for j in range(i+1, len(insights_results)):
                # Simple similarity calculation based on number of insights
                insights1 = insights_results[i].get("insights", {})
                insights2 = insights_results[j].get("insights", {})
                
                similarity = calculate_insight_similarity(insights1, insights2)
                consistency_scores.append(similarity)
        
        if consistency_scores:
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
            results["consistency_scores"] = consistency_scores
            results["avg_consistency"] = avg_consistency
            print(f"Average consistency score: {avg_consistency:.2f}")
    
    # Next, test error handling
    print("\nTesting error handling...")
    results["error_handling"] = {}
    
    for error_type, error_file in error_datasets:
        print(f"Testing with error type: {error_type}")
        
        # Create new orchestrator
        orchestrator = SimplifiedOrchestrator({"output_dir": output_dir})
        
        # Run the analysis
        context = {
            "data_source": {
                "source_type": "file",
                "source_path": error_file
            },
            "analysis_objective": f"Error handling test: {error_type}",
            "target_column": "target"
        }
        
        try:
            result = orchestrator.run(context)
            
            if result["status"] == "success":
                results["error_handling"][error_type] = "recovered"
                print(f"  Successfully recovered from {error_type} error")
            else:
                results["error_handling"][error_type] = "failed"
                print(f"  Failed to recover from {error_type} error")
        
        except Exception as e:
            results["error_handling"][error_type] = "exception"
            print(f"  Exception with {error_type} error: {str(e)}")
    
    # Calculate error rate
    results["error_rate"] = results["error_runs"] / results["total_runs"] if results["total_runs"] > 0 else 0
    
    # Calculate recovery rate
    recovery_count = sum(1 for status in results["error_handling"].values() if status == "recovered")
    results["recovery_rate"] = recovery_count / len(results["error_handling"]) if results["error_handling"] else 0
    
    # Save results
    results_file = os.path.join(output_dir, "reliability_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReliability results saved to: {results_file}")
    
    # Print summary
    print("\nReliability Summary:")
    print(f"Error rate: {results['error_rate']*100:.2f}%")
    print(f"Recovery rate: {results['recovery_rate']*100:.2f}%")
    print(f"Consistency: {results.get('avg_consistency', 0)*100:.2f}%")
    
    return results

def calculate_insight_similarity(insights1, insights2):
    """Calculate a simple similarity score between two sets of insights."""
    if not insights1 or not insights2:
        return 0
    
    # Get common insight types
    common_types = set(insights1.keys()) & set(insights2.keys())
    if not common_types:
        return 0
    
    similarity_scores = []
    
    for insight_type in common_types:
        items1 = insights1.get(insight_type, [])
        items2 = insights2.get(insight_type, [])
        
        if not items1 or not items2:
            continue
        
        # Count number of "similar" insights
        similar_count = 0
        
        # Different comparison logic based on insight type
        if insight_type == "correlations":
            # For correlations, check if the same pairs of columns are correlated
            pairs1 = [(item.get("column1"), item.get("column2")) for item in items1]
            pairs2 = [(item.get("column1"), item.get("column2")) for item in items2]
            
            # Count pairs that appear in both results (in either order)
            for pair in pairs1:
                if pair in pairs2 or (pair[1], pair[0]) in pairs2:
                    similar_count += 1
        
        elif insight_type == "trends":
            # For trends, check if the same columns have the same direction
            trends1 = [(item.get("column"), item.get("direction")) for item in items1]
            trends2 = [(item.get("column"), item.get("direction")) for item in items2]
            
            similar_count = len(set(trends1) & set(trends2))
        
        elif insight_type == "outliers":
            # For outliers, check if the same columns have outliers
            cols1 = [item.get("column") for item in items1]
            cols2 = [item.get("column") for item in items2]
            
            similar_count = len(set(cols1) & set(cols2))
        
        elif insight_type == "segments":
            # For segments, check if the same segment columns and value columns are identified
            segs1 = [(item.get("segment_column"), item.get("value_column")) for item in items1]
            segs2 = [(item.get("segment_column"), item.get("value_column")) for item in items2]
            
            similar_count = len(set(segs1) & set(segs2))
        
        # Calculate similarity for this insight type
        max_items = max(len(items1), len(items2))
        type_similarity = similar_count / max_items if max_items > 0 else 0
        similarity_scores.append(type_similarity)
    
    # Return average similarity across all insight types
    return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

def main():
    """Run both efficiency and reliability tests."""
    print("Starting performance evaluation...")
    
    # Test efficiency with smaller datasets for quicker results
    dataset_sizes = [100, 500, 1000]
    efficiency_results = test_efficiency_metrics(dataset_sizes)
    
    # Test reliability with fewer runs for quicker results
    reliability_results = test_reliability_metrics(runs=5)
    
    # Combine results for final report
    combined_results = {
        "efficiency": efficiency_results,
        "reliability": reliability_results,
        "summary": {
            "avg_processing_time_per_100rows": efficiency_results.get("summary", {}).get("avg_time_per_100_rows", 0),
            "base_memory_usage_mb": efficiency_results.get("summary", {}).get("base_memory_mb", 0),
            "memory_per_row_kb": efficiency_results.get("summary", {}).get("memory_per_row_kb", 0),
            "avg_cpu_utilization": efficiency_results.get("summary", {}).get("avg_cpu_percent", 0),
            "error_rate": reliability_results.get("error_rate", 0),
            "recovery_rate": reliability_results.get("recovery_rate", 0),
            "consistency": reliability_results.get("avg_consistency", 0)
        }
    }
    
    # Save combined results
    with open("performance_metrics_summary.json", 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print("\nPerformance evaluation complete!")
    print("Summary metrics:")
    summary = combined_results["summary"]
    print(f"Avg. Processing Time: {summary['avg_processing_time_per_100rows']:.2f} seconds per 100 rows")
    print(f"Base Memory Usage: {summary['base_memory_usage_mb']:.2f} MB")
    print(f"Memory per Row: {summary['memory_per_row_kb']:.2f} KB")
    print(f"Avg. CPU Utilization: {summary['avg_cpu_utilization']:.2f}%")
    print(f"Error Rate: {summary['error_rate']*100:.2f}%")
    print(f"Recovery Rate: {summary['recovery_rate']*100:.2f}%")
    print(f"Consistency: {summary['consistency']*100:.2f}%")
    print("\nDetailed results saved to performance_metrics_summary.json")

if __name__ == "__main__":
    main()