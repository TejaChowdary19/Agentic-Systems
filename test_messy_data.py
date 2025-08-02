# test_messy_data.py

import pandas as pd
import numpy as np
import os
import json
import time
from data_analysis_system.simplified_main import run_data_analysis

def create_messy_dataset():
    """Create a dataset with various data quality issues."""
    np.random.seed(42)
    
    # Create a dataset with 100 rows and clear relationships but with data issues
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
    
    # Add data quality issues:
    
    # 1. Missing values (10% of data)
    for col in ['x1', 'x2', 'y']:
        missing_indices = np.random.choice(n, size=int(n*0.1), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    # 2. Outliers (5% of data)
    for col in ['x1', 'x2', 'y']:
        outlier_indices = np.random.choice(n, size=int(n*0.05), replace=False)
        df.loc[outlier_indices, col] = df[col].mean() + 5 * df[col].std()
    
    # 3. Duplicate rows (5% of data)
    duplicate_indices = np.random.choice(n, size=int(n*0.05), replace=False)
    for idx in duplicate_indices:
        df = pd.concat([df, df.iloc[[idx]]], ignore_index=True)
    
    # Save to CSV
    test_file = 'test_data/messy_data.csv'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    df.to_csv(test_file, index=False)
    
    print(f"Created messy test dataset: {test_file}")
    print(f"Data shape: {df.shape}")
    print("Data preview:")
    print(df.head())
    print("\nData quality issues:")
    print(f"- Missing values: {df.isna().sum().sum()}")
    print(f"- Duplicate rows: {df.duplicated().sum()}")
    
    return test_file

def run_test():
    """Run the test on messy data."""
    print("\n" + "="*50)
    print("TEST CASE 2: MESSY DATA HANDLING")
    print("="*50)
    
    start_time = time.time()
    
    # Create test dataset
    data_path = create_messy_dataset()
    
    # Set up arguments
    class Args:
        source_type = "file"
        data_source = data_path
        objective = "Analyze relationships between variables and identify key patterns"
        business_context = "Research and development analysis"
        target_column = "y"
        output_dir = "test_results/messy_data"
    
    args = Args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nRunning analysis on messy data...")
    
    # Run the analysis
    result = run_data_analysis(args)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Evaluate results
    if result["status"] == "success":
        print(f"\n✅ Test passed: Analysis completed successfully despite data quality issues")
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
        
        # Check if data cleaning was performed
        cleaned_data_path = result.get("file_paths", {}).get("cleaned_data", "")
        if cleaned_data_path and os.path.exists(cleaned_data_path):
            original_df = pd.read_csv(data_path)
            cleaned_df = pd.read_csv(cleaned_data_path)
            
            # Check data cleaning results
            print("\nData cleaning results:")
            print(f"  Original data shape: {original_df.shape}")
            print(f"  Cleaned data shape: {cleaned_df.shape}")
            print(f"  Original missing values: {original_df.isna().sum().sum()}")
            print(f"  Cleaned missing values: {cleaned_df.isna().sum().sum()}")
            print(f"  Original duplicate rows: {original_df.duplicated().sum()}")
            print(f"  Cleaned duplicate rows: {cleaned_df.duplicated().sum()}")
            
            # Evaluate cleaning quality
            cleaning_success = []
            
            # Check if missing values were handled
            missing_reduced = cleaned_df.isna().sum().sum() < original_df.isna().sum().sum()
            cleaning_success.append(missing_reduced)
            print(f"  Missing values handled: {'✅' if missing_reduced else '❌'}")
            
            # Check if duplicates were removed
            duplicates_reduced = cleaned_df.duplicated().sum() < original_df.duplicated().sum()
            cleaning_success.append(duplicates_reduced)
            print(f"  Duplicates removed: {'✅' if duplicates_reduced else '❌'}")
            
            # Calculate cleaning success rate
            cleaning_rate = sum(cleaning_success) / len(cleaning_success) * 100
            print(f"\nCleaning success rate: {cleaning_rate:.1f}%")
        else:
            print("\n❌ No cleaned data file found")
            cleaning_rate = 0
        
        # Save test results
        test_results = {
            "test_case": "Messy Data Handling",
            "status": "passed",
            "execution_time": execution_time,
            "expected_insights": expected_insights,
            "found_insights": found_insights,
            "insight_success_rate": success_rate,
            "cleaning_success_rate": cleaning_rate,
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
            "test_case": "Messy Data Handling",
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