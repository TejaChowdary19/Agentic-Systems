# test_run_function.py
import pandas as pd
import numpy as np
import os
from data_analysis_system.simplified_main import run_data_analysis

print("Successfully imported run_data_analysis function")

# Create a sample dataset
def create_test_file():
    """Create a simple test dataset."""
    np.random.seed(42)
    
    # Create sample data
    data = {
        'x': np.random.rand(20),
        'y': np.random.rand(20),
        'z': np.random.rand(20)
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    test_file = 'test_run_data.csv'
    df.to_csv(test_file, index=False)
    
    print(f"Created test file: {test_file}")
    print(f"Data shape: {df.shape}")
    
    return test_file

# Create the test file
data_file = create_test_file()

# Define a simple Args class
class Args:
    source_type = "file"
    data_source = data_file  # Use the file we just created
    objective = "Test"
    business_context = "Test"
    target_column = "y"
    output_dir = "test_run_outputs"

# Ensure output directory exists
if not os.path.exists(Args.output_dir):
    os.makedirs(Args.output_dir)

# Try running the function
try:
    result = run_data_analysis(Args)
    print(f"Function ran with status: {result['status']}")
    
    if result["status"] == "success":
        print("\nAnalysis completed successfully!")
        if "summary" in result:
            print("\nSummary:")
            print(result["summary"])
    else:
        print(f"\nAnalysis failed: {result.get('error_message', 'Unknown error')}")
        
except Exception as e:
    print(f"Error running function: {e}")

# Clean up
if os.path.exists(data_file):
    os.remove(data_file)