# test_small_data.py

import pandas as pd
import numpy as np
import os
import json
import time
from data_analysis_system.simplified_main import run_data_analysis

def create_small_dataset():
    """Create a very small dataset (10 rows)."""
    np.random.seed(42)
    
    # Create a small dataset with only 10 rows
    n = 10
    
    # Independent variables
    x1 = np.linspace(0, 10, n)
    x2 = np.linspace(5, 15, n)
    
    # Target variable with defined relationship: y = 2*x1 - 0.5*x2 + noise
    y = 2 * x1 - 0.5 * x2 + np.random.normal(0, 1, n)
    
    # Categorical variable
    categories = np.array(['A', 'B', 'A', 'B', 'A', 'A', 'B', 'A', 'B', 'A'])
    
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
    test_file = 'test_data/small_data.csv'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    df.to_csv(test_file, index=False)
    
    print(f"Created small test dataset: {test_file}")
    print(f"Data shape: {df.shape}")
    print("Data preview:")
    print(df)
    
    return test_file

def run_test():
    """Run the test on a small dataset."""
    print("\n" + "="*50)
    print("TEST CASE 3: SMALL DATASET ANALYSIS")
    print("="*50)
    
    start_time = time.time()
    
    # Create test dataset
    data_path = create_small_dataset()
    
    # Set up arguments
    class Args:
        source_type = "file"
        data_source = data_path
        objective = "Analyze relationships between variables despite limited data"
        business_context = "Pilot study with limited samples"
        target_column = "y"
        output_dir = "test_results/small_data"
    
    args = Args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nRunning analysis on small dataset...")
    
    # Run the analysis
    result = run_data_analysis(args)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Evaluate results
    if result["status"] == "success":
        print(f"\n✅ Test passed: Analysis completed on small dataset")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        # Check for key findings
        key_findings = []
        if "key_findings" in result:
            key_findings = result["key_findings"]
            print(f"\nKey findings detected: {len(key_findings)}")
            for i, finding in enumerate(key_findings):
                print(f"  {i+1}. {finding.get('finding', '')}")
        
        # Calculate statistically valid findings
        valid_findings = []
        questionable_findings = []
        
        for finding in key_findings:
            finding_text = finding.get('finding', '').lower()
            
            # In small datasets, correlations are often not statistically significant
            if "correlation" in finding_text and "strong" in finding_text:
                questionable_findings.append(finding.get('finding', ''))
            elif "trend" in finding_text:
                # Trends with only 10 points are questionable
                questionable_findings.append(finding.get('finding', ''))
            else:
                valid_findings.append(finding.get('finding', ''))
        
        # Report on finding validity
        print("\nStatistically valid findings:")
        for i, finding in enumerate(valid_findings):
            print(f"  {i+1}. {finding}")
            
        print("\nQuestionable findings (limited statistical power):")
        for i, finding in enumerate(questionable_findings):
            print(f"  {i+1}. {finding}")
        
        # Check if the system acknowledges statistical limitations
        limitations_acknowledged = False
        if "summary" in result:
            summary_text = result["summary"].lower()
            statistical_terms = ["small sample", "limited data", "few observations", 
                               "statistical power", "significance", "limited statistical"]
            
            for term in statistical_terms:
                if term in summary_text:
                    limitations_acknowledged = True
                    break
        
        print(f"\nAcknowledgment of statistical limitations: {'✅' if limitations_acknowledged else '❌'}")
        
        # Save test results
        test_results = {
            "test_case": "Small Dataset Analysis",
            "status": "passed",
            "execution_time": execution_time,
            "valid_findings": valid_findings,
            "questionable_findings": questionable_findings,
            "limitations_acknowledged": limitations_acknowledged,
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
            "test_case": "Small Dataset Analysis",
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