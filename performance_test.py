# performance_test.py

import pandas as pd
import numpy as np
import os
import json
import time
import psutil
import matplotlib.pyplot as plt
from data_analysis_system.simplified_main import run_data_analysis

def create_dataset(size, complexity='medium'):
    """
    Create a synthetic dataset of specified size and complexity.
    
    Parameters:
    - size: Number of rows
    - complexity: 'low', 'medium', or 'high' - affects number of columns and relationships
    """
    np.random.seed(42)  # For reproducibility
    
    # Determine number of columns based on complexity
    if complexity == 'low':
        num_numeric = 3
        num_categorical = 1
    elif complexity == 'medium':
        num_numeric = 5
        num_categorical = 2
    else:  # high
        num_numeric = 10
        num_categorical = 3
    
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
    
    # Create dataframe
    df = pd.DataFrame()
    df['date'] = dates
    
    # Add numeric columns with relationships
    for i in range(num_numeric):
        # Base column with linear trend
        if i == 0:
            df[f'numeric_{i}'] = np.linspace(0, 100, size) + np.random.normal(0, 10, size)
        # Columns with relationships to first column
        else:
            coef = np.random.uniform(-2, 2)
            df[f'numeric_{i}'] = coef * df['numeric_0'] + np.random.normal(0, 20, size)
    
    # Add categorical columns
    categories = ['A', 'B', 'C', 'D', 'E']
    for i in range(num_categorical):
        df[f'category_{i}'] = np.random.choice(categories, size)
        
        # Make categories have different distributions for target variable
        for cat in categories:
            mask = df[f'category_{i}'] == cat
            if mask.any():
                df.loc[mask, 'numeric_0'] += np.random.uniform(-20, 20)
    
    # Add target variable with known relationship
    target_coefs = np.random.uniform(-1, 1, num_numeric)
    df['target'] = sum(target_coefs[i] * df[f'numeric_{i}'] for i in range(num_numeric))
    df['target'] += np.random.normal(0, 10, size)
    
    # Create directory if it doesn't exist
    os.makedirs('performance_data', exist_ok=True)
    
    # Save to CSV
    filename = f'performance_data/dataset_{size}_{complexity}.csv'
    df.to_csv(filename, index=False)
    
    print(f"Created dataset with {size} rows and {len(df.columns)} columns: {filename}")
    return filename

def measure_performance(data_path, target_col='target'):
    """Measure performance metrics while running analysis."""
    # Set up arguments
    class Args:
        source_type = "file"
        data_source = data_path
        objective = "Performance testing"
        business_context = "Evaluating system performance"
        target_column = target_col
        output_dir = "performance_results"
    
    args = Args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Start timing
    start_time = time.time()
    
    # Run the analysis
    result = run_data_analysis(args)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Get peak memory usage
    peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
    memory_used = peak_memory - initial_memory
    
    # Calculate dataset size
    data_size_bytes = os.path.getsize(data_path)
    data_size_mb = data_size_bytes / (1024 * 1024)  # MB
    
    # Get dataset dimensions
    df = pd.read_csv(data_path)
    rows, columns = df.shape
    
    # Return performance metrics
    metrics = {
        "status": result["status"],
        "execution_time": execution_time,
        "memory_used": memory_used,
        "peak_memory": peak_memory,
        "data_size_mb": data_size_mb,
        "data_rows": rows,
        "data_columns": columns,
        "processing_rate": rows / execution_time if execution_time > 0 else 0  # rows per second
    }
    
    return metrics

def run_performance_tests():
    """Run a series of performance tests with increasing dataset sizes."""
    print("\n" + "="*50)
    print("PERFORMANCE TESTING")
    print("="*50)
    
    # Define dataset sizes to test
    sizes = [100, 1000, 10000, 50000]
    complexity = 'medium'
    
    # Store results
    results = []
    
    for size in sizes:
        print(f"\nTesting with dataset size: {size} rows")
        
        # Create dataset
        data_path = create_dataset(size, complexity)
        
        # Measure performance
        print(f"Running analysis...")
        metrics = measure_performance(data_path)
        
        # Add size to metrics
        metrics["size"] = size
        results.append(metrics)
        
        # Print results
        print(f"Status: {metrics['status']}")
        print(f"Execution time: {metrics['execution_time']:.2f} seconds")
        print(f"Memory used: {metrics['memory_used']:.2f} MB")
        print(f"Processing rate: {metrics['processing_rate']:.2f} rows/second")
    
    # Save results
    os.makedirs('performance_results', exist_ok=True)
    with open('performance_results/performance_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate charts
    generate_performance_charts(results)
    
    return results

def generate_performance_charts(results):
    """Generate charts visualizing performance metrics."""
    # Extract data for plotting
    sizes = [r["size"] for r in results]
    times = [r["execution_time"] for r in results]
    memory = [r["memory_used"] for r in results]
    rates = [r["processing_rate"] for r in results]
    
    # Create directory for charts
    os.makedirs('performance_results', exist_ok=True)
    
    # Create execution time chart
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, marker='o', linestyle='-', linewidth=2)
    plt.title('Execution Time vs Dataset Size')
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig('performance_results/execution_time.png')
    plt.close()
    
    # Create memory usage chart
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, memory, marker='o', linestyle='-', linewidth=2)
    plt.title('Memory Usage vs Dataset Size')
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Memory Usage (MB)')
    plt.grid(True)
    plt.savefig('performance_results/memory_usage.png')
    plt.close()
    
    # Create processing rate chart
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, rates, marker='o', linestyle='-', linewidth=2)
    plt.title('Processing Rate vs Dataset Size')
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Processing Rate (rows/second)')
    plt.grid(True)
    plt.savefig('performance_results/processing_rate.png')
    plt.close()
    
    print(f"\nPerformance charts saved to performance_results/")

if __name__ == "__main__":
    run_performance_tests()