# test_data_cleaning.py

import json
import os
import pandas as pd
import numpy as np
from data_analysis_system.tools.data_cleaning import DataCleaningTool

def main():
    # Create a test CSV file with some issues
    test_data = {
        'id': [1, 2, 3, 4, 5, 5],  # Duplicate row
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Eve'],
        'age': [25, 30, np.nan, 40, 45, 45],  # Missing value
        'score': [85.5, 90.0, 88.5, 200.0, 87.0, 87.0]  # Outlier (200)
    }
    
    test_df = pd.DataFrame(test_data)
    test_file = 'test_data_with_issues.csv'
    test_df.to_csv(test_file, index=False)
    
    print(f"Created test file: {test_file}")
    print(f"Test data shape: {test_df.shape}")
    print("Test data preview:")
    print(test_df)
    print("\n")
    
    # Initialize the Data Cleaning Tool
    data_cleaning_tool = DataCleaningTool()
    
    # Test data cleaning
    print("Testing data cleaning...")
    result = data_cleaning_tool._run(
        data_path=test_file,
        operations=["handle_missing", "remove_duplicates", "remove_outliers", "normalize"]
    )
    
    # Parse and print the result
    result_dict = json.loads(result)
    print(f"Status: {result_dict['status']}")
    print("Operations performed:")
    for op in result_dict['operations_performed']:
        print(f"- {op}")
    print(f"Cleaned data shape: {result_dict['cleaned_data_shape']}")
    print(f"Cleaned data path: {result_dict['cleaned_data_path']}")
    print("Sample cleaned data:")
    for record in result_dict['sample']:
        print(record)
    
    # Load and display the cleaned data for verification
    cleaned_file = result_dict['cleaned_data_path']
    if os.path.exists(cleaned_file):
        cleaned_df = pd.read_csv(cleaned_file)
        print("\nVerification - Cleaned data:")
        print(cleaned_df)
    
    # Clean up
    os.remove(test_file)
    if os.path.exists(cleaned_file):
        os.remove(cleaned_file)
    print(f"\nRemoved test files: {test_file}, {cleaned_file}")

if __name__ == "__main__":
    main()