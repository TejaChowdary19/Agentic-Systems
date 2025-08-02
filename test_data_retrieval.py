# test_data_retrieval.py

import json
import os
import pandas as pd
from data_analysis_system.tools.data_retrieval import DataRetrievalTool

def main():
    # Create a test CSV file
    test_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'score': [85.5, 90.0, 88.5, 92.5, 87.0]
    }
    
    test_df = pd.DataFrame(test_data)
    test_file = 'test_data.csv'
    test_df.to_csv(test_file, index=False)
    
    print(f"Created test file: {test_file}")
    print(f"Test data shape: {test_df.shape}")
    print("Test data preview:")
    print(test_df.head())
    print("\n")
    
    # Initialize the Data Retrieval Tool
    data_retrieval_tool = DataRetrievalTool()
    
    # Test file retrieval
    print("Testing file retrieval...")
    result = data_retrieval_tool._run(
        source_type="file",
        source_path=test_file
    )
    
    # Parse and print the result
    result_dict = json.loads(result)
    print(f"Status: {result_dict['status']}")
    print(f"Data shape: {result_dict['data_shape']}")
    print(f"Columns: {result_dict['columns']}")
    print("Sample data:")
    for record in result_dict['sample']:
        print(record)
    
    # Clean up
    os.remove(test_file)
    print(f"\nRemoved test file: {test_file}")

if __name__ == "__main__":
    main()