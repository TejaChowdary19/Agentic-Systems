# test_clean_data.py

import pandas as pd
import numpy as np
import os
import json
import time
from data_analysis_system.simplified_main import run_data_analysis

def create_clean_dataset():
    """Create a well-structured dataset with clear patterns."""
    np.random.seed(42)
    
    # Create a dataset with 100 rows and clear relationships
    n = 100
    
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
    test_file = 'test_data/clean_data.csv'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    df.to_csv(test_file, index=False)
    
    print(f"Created clean test dataset: {test_file}")
    print(f"Data shape: {df.shape}")
    print("Data preview:")
    print(df.head())
    
    return test_file

def run_test():
    """Run the test on clean data."""
    print("\n" + "="*50)
    print("TEST CASE 1: BASIC DATA ANALYSIS WITH CLEAN DATA")
    print("="*50)
    
    start_time = time.time()
    
    # Create test dataset
    data_path = create_clean_dataset()
    
    # Set up arguments
    class Args:
        source_type = "file"
        data_source = data_path
        objective = "Analyze relationships between variables and identify key patterns"
        business_context = "Research and development analysis"
        target_column = "y"
        output_dir = "test_results/clean_data"
    
    args = Args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nRunning analysis on clean data...")
    
    # Run the analysis
    result = run_data_analysis(args)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Evaluate results
    if result["status"] == "success":
        print(f"\n✅ Test passed: Analysis completed successfully")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        # Check for key findings
        key_findings = []
        if "key_findings" in result:
            key_findings = result["key_findings"]
            print(f"\nKey findings detected: {len(key_findings)}")
            for i, finding in enumerate(key_findings):
                print(f"  {i+1}. {finding.get('finding', '')}")
        
        # Check for expected insights
        expected_insights = [
            "correlation between x1 and y",
            "correlation between x2 and y",
            "segment 'A' has higher y"
        ]
        
        found_insights = [False, False, False]
        
        for finding in key_findings:
            finding_text = finding.get('finding', '').lower()
            
            if "correlation" in finding_text and "x1" in finding_text and "y" in finding_text:
                found_insights[0] = True
            
            if "correlation" in finding_text and "x2" in finding_text and "y" in finding_text:
                found_insights[1] = True
                
            if "segment" in finding_text and "a" in finding_text and "higher y" in finding_text:
                found_insights[2] = True
        
        # Report on expected insights
        print("\nChecking for expected insights:")
        for i, (expected, found) in enumerate(zip(expected_insights, found_insights)):
            status = "✅ Found" if found else "❌ Not found"
            print(f"  {status}: {expected_insights[i]}")
        
        # Calculate success rate
        success_rate = sum(found_insights) / len(found_insights) * 100
        print(f"\nSuccess rate: {success_rate:.1f}%")
        
        # Save test results
        test_results = {
            "test_case": "Clean Data Analysis",
            "status": "passed",
            "execution_time": execution_time,
            "expected_insights": expected_insights,
            "found_insights": found_insights,
            "success_rate": success_rate,
            "key_findings": key_findings,
            "full_result": result
        }
        
        with open(os.path.join(args.output_dir, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        return test_results
    else:
        print(f"\n❌ Test failed: {result.get('error_message', 'Unknown error')}")
        
        # Save test results
        test_results = {
            "test_case": "Clean Data Analysis",
            "status": "failed",
            "execution_time": execution_time,
            "error": result.get('error_message', 'Unknown error'),
            "full_result": result
        }
        
        with open(os.path.join(args.output_dir, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        return test_results

if __name__ == "__main__":
    run_test()