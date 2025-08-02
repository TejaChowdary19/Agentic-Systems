# reliability_test.py

import pandas as pd
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from data_analysis_system.simplified_main import run_data_analysis

def create_test_dataset():
    """Create a fixed dataset for reliability testing."""
    np.random.seed(42)  # For reproducibility
    
    # Create a dataset with 1000 rows and clear patterns
    n = 1000
    
    # Independent variables with clear relationships
    x1 = np.linspace(0, 10, n)
    x2 = np.linspace(5, 15, n)
    
    # Target variable with defined relationship: y = 2*x1 - 0.5*x2 + noise
    y = 2 * x1 - 0.5 * x2 + np.random.normal(0, 1, n)
    
    # Categorical variable with clear segment differences
    categories = np.random.choice(['A', 'B', 'C'], n, p=[0.5, 0.3, 0.2])
    
    # Make category A have higher y values
    y[categories == 'A'] += 3
    
    # Create time variable for trend analysis
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    # Create the DataFrame
    data = {
        'date': dates,
        'x1': x1,
        'x2': x2, 
        'y': y,
        'category': categories
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('reliability_data', exist_ok=True)
    test_file = 'reliability_data/reliability_dataset.csv'
    df.to_csv(test_file, index=False)
    
    print(f"Created reliability test dataset: {test_file}")
    print(f"Data shape: {df.shape}")
    
    return test_file

def extract_insights(result):
    """Extract key insights from analysis result for comparison."""
    insights = []
    
    if "key_findings" in result:
        for finding in result["key_findings"]:
            if "finding" in finding:
                insights.append(finding["finding"])
    
    return sorted(insights)

def calculate_similarity(insights1, insights2):
    """Calculate similarity between two sets of insights."""
    if not insights1 or not insights2:
        return 0.0
    
    # Count matching insights
    matches = 0
    for insight in insights1:
        if insight in insights2:
            matches += 1
    
    # Calculate Jaccard similarity
    similarity = matches / (len(insights1) + len(insights2) - matches)
    return similarity

def run_reliability_test(iterations=10):
    """Run the analysis multiple times and evaluate consistency."""
    print("\n" + "="*50)
    print("RELIABILITY TESTING")
    print("="*50)
    
    # Create test dataset
    data_path = create_test_dataset()
    
    # Store results from each iteration
    all_results = []
    execution_times = []
    success_rates = []
    all_insights = []
    
    print(f"\nRunning {iterations} iterations...")
    
    # Run multiple iterations
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...")
        
        # Set up arguments
        class Args:
            source_type = "file"
            data_source = data_path
            objective = "Reliability testing"
            business_context = "Evaluating system reliability"
            target_column = "y"
            output_dir = f"reliability_results/iteration_{i+1}"
        
        args = Args()
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run the analysis and time it
        start_time = time.time()
        result = run_data_analysis(args)
        execution_time = time.time() - start_time
        
        # Store execution time
        execution_times.append(execution_time)
        
        # Store success status
        success_rates.append(1 if result["status"] == "success" else 0)
        
        # Extract insights
        insights = extract_insights(result)
        all_insights.append(insights)
        
        # Store result
        all_results.append(result)
        
        print(f"  Status: {result['status']}")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Insights found: {len(insights)}")
    
    # Calculate metrics
    avg_execution_time = sum(execution_times) / len(execution_times)
    std_execution_time = np.std(execution_times)
    success_rate = sum(success_rates) / len(success_rates) * 100
    
    # Calculate insight consistency
    similarity_scores = []
    for i in range(len(all_insights)):
        for j in range(i+1, len(all_insights)):
            similarity = calculate_similarity(all_insights[i], all_insights[j])
            similarity_scores.append(similarity)
    
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    # Calculate coefficient of variation for execution times
    cv_execution_time = (std_execution_time / avg_execution_time) * 100 if avg_execution_time > 0 else 0
    
    # Print results
    print("\nReliability Test Results:")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Execution Time: {avg_execution_time:.2f} seconds")
    print(f"Execution Time Variability (CV): {cv_execution_time:.2f}%")
    print(f"Insight Consistency: {avg_similarity:.2f} (0-1 scale)")
    
    # Create results directory
    os.makedirs('reliability_results', exist_ok=True)
    
    # Save metrics
    metrics = {
        "iterations": iterations,
        "success_rate": success_rate,
        "avg_execution_time": avg_execution_time,
        "std_execution_time": std_execution_time,
        "cv_execution_time": cv_execution_time,
        "avg_similarity": avg_similarity,
        "execution_times": execution_times,
        "success_rates": success_rates,
        "insight_counts": [len(insights) for insights in all_insights]
    }
    
    with open('reliability_results/reliability_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate charts
    generate_reliability_charts(metrics)
    
    return metrics

def generate_reliability_charts(metrics):
    """Generate charts visualizing reliability metrics."""
    # Create directory for charts
    os.makedirs('reliability_results', exist_ok=True)
    
    # Create execution time variation chart
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics["execution_times"])+1), metrics["execution_times"], marker='o')
    plt.axhline(y=metrics["avg_execution_time"], color='r', linestyle='--', label=f'Average: {metrics["avg_execution_time"]:.2f}s')
    plt.title('Execution Time Across Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('reliability_results/execution_time_variation.png')
    plt.close()
    
    # Create insight count variation chart
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics["insight_counts"])+1), metrics["insight_counts"], marker='o')
    avg_insights = sum(metrics["insight_counts"]) / len(metrics["insight_counts"])
    plt.axhline(y=avg_insights, color='r', linestyle='--', label=f'Average: {avg_insights:.1f}')
    plt.title('Number of Insights Across Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Insights')
    plt.legend()
    plt.grid(True)
    plt.savefig('reliability_results/insight_count_variation.png')
    plt.close()
    
    print(f"\nReliability charts saved to reliability_results/")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    run_reliability_test(iterations=10)