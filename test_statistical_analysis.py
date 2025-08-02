# test_statistical_analysis.py

import json
import os
import pandas as pd
import numpy as np
from data_analysis_system.tools.statistical_analysis import StatisticalAnalysisTool

def main():
    # Create a test dataset with some relationships
    np.random.seed(42)  # For reproducibility
    
    # Create features with relationships
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(0, 1, 100)
    
    # y has strong relationship with x1, weak with x2
    y = 2 * x1 + 0.5 * x2 + np.random.normal(0, 1, 100)
    
    # Create categorical variable
    categories = np.random.choice(['A', 'B', 'C'], 100)
    
    test_data = {
        'x1': x1,
        'x2': x2,
        'y': y,
        'category': categories
    }
    
    test_df = pd.DataFrame(test_data)
    test_file = 'test_data_for_analysis.csv'
    test_df.to_csv(test_file, index=False)
    
    print(f"Created test file: {test_file}")
    print(f"Test data shape: {test_df.shape}")
    print("Test data preview:")
    print(test_df.head())
    print("\nTest data summary:")
    print(test_df.describe())
    print("\n")
    
    # Initialize the Statistical Analysis Tool
    stat_analysis_tool = StatisticalAnalysisTool()
    
    # Test statistical analysis
    print("Testing statistical analysis...")
    result = stat_analysis_tool._run(
        data_path=test_file,
        analysis_types=["descriptive", "correlation", "regression"],
        target_column="y"
    )
    
    # Parse and print the result
    result_dict = json.loads(result)
    print(f"Status: {result_dict['status']}")
    print(f"Analysis results path: {result_dict['analysis_results_path']}")
    
    # Print descriptive statistics
    if "descriptive" in result_dict["analyses"]:
        print("\nDescriptive Statistics:")
        for col, stats in result_dict["analyses"]["descriptive"].items():
            print(f"\n{col}:")
            for stat, value in stats.items():
                print(f"  {stat}: {value}")
    
    # Print correlation results
    if "correlation" in result_dict["analyses"]:
        print("\nHighest Correlations:")
        for pair, corr in result_dict["analyses"]["correlation"]["highest_correlations"].items():
            print(f"  {pair}: {corr}")
    
    # Print regression results
    if "regression" in result_dict["analyses"]:
        print("\nRegression Results (target: y):")
        for feature, stats in result_dict["analyses"]["regression"].items():
            print(f"\n{feature}:")
            print(f"  Coefficient: {stats['slope']}")
            print(f"  Intercept: {stats['intercept']}")
            print(f"  R-squared: {stats['r_squared']}")
            print(f"  p-value: {stats['p_value']}")
    
    # Clean up
    os.remove(test_file)
    if os.path.exists(result_dict["analysis_results_path"]):
        os.remove(result_dict["analysis_results_path"])
    print(f"\nRemoved test files: {test_file}, {result_dict['analysis_results_path']}")

if __name__ == "__main__":
    main()