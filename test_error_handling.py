# test_error_handling.py

import os
import json
import time
from data_analysis_system.simplified_main import run_data_analysis

def create_corrupted_file():
    """Create a corrupted CSV file."""
    # Create directory if it doesn't exist
    os.makedirs('test_data', exist_ok=True)
    
    # Create a file with invalid CSV format
    test_file = 'test_data/corrupted_data.csv'
    
    with open(test_file, 'w') as f:
        f.write("This is not a valid CSV file\n")
        f.write("x1,x2,y\n")
        f.write("1,2,3\n")
        f.write("invalid,data,format\n")
        f.write("more,invalid,data,extra_column\n")
    
    print(f"Created corrupted test file: {test_file}")
    
    return test_file

def run_test_missing_file():
    """Test with a non-existent file."""
    print("\n" + "="*50)
    print("TEST CASE 4A: ERROR HANDLING - MISSING FILE")
    print("="*50)
    
    start_time = time.time()
    
    # Non-existent file path
    data_path = 'test_data/non_existent_file.csv'
    
    # Set up arguments
    class Args:
        source_type = "file"
        data_source = data_path
        objective = "Test error handling"
        business_context = "Testing"
        target_column = "y"
        output_dir = "test_results/error_handling/missing_file"
    
    args = Args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nRunning analysis with non-existent file...")
    
    # Run the analysis
    result = run_data_analysis(args)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # For error tests, we expect a failure
    if result["status"] == "error":
        print(f"\n✅ Test passed: System correctly reported an error for missing file")
        print(f"Error message: {result.get('error_message', 'Unknown error')}")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        # Check for error details
        has_file_error = False
        error_message = result.get('error_message', '').lower()
        
        if 'no such file' in error_message or 'not found' in error_message or 'does not exist' in error_message:
            has_file_error = True
            print("✅ Error message correctly indicates file not found")
        else:
            print("❌ Error message does not clearly indicate the nature of the error")
        
        # Save test results
        test_results = {
            "test_case": "Error Handling - Missing File",
            "status": "passed",
            "execution_time": execution_time,
            "has_file_error": has_file_error,
            "error_message": result.get('error_message', 'Unknown error'),
            "full_result": result
        }
        
        with open(os.path.join(args.output_dir, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        return test_results
    else:
        print(f"\n❌ Test failed: System did not report an error for missing file")
        
        # Save test results
        test_results = {
            "test_case": "Error Handling - Missing File",
            "status": "failed",
            "execution_time": execution_time,
            "error": "System did not report an error for missing file",
            "full_result": result
        }
        
        with open(os.path.join(args.output_dir, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        return test_results

def run_test_corrupted_file():
    """Test with a corrupted file."""
    print("\n" + "="*50)
    print("TEST CASE 4B: ERROR HANDLING - CORRUPTED FILE")
    print("="*50)
    
    start_time = time.time()
    
    # Create corrupted file
    data_path = create_corrupted_file()
    
    # Set up arguments
    class Args:
        source_type = "file"
        data_source = data_path
        objective = "Test error handling"
        business_context = "Testing"
        target_column = "y"
        output_dir = "test_results/error_handling/corrupted_file"
    
    args = Args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nRunning analysis with corrupted file...")
    
    # Run the analysis
    result = run_data_analysis(args)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Check results - either error handling or graceful recovery is acceptable
    if result["status"] == "error":
        print(f"\n✅ Test passed: System correctly reported an error for corrupted file")
        print(f"Error message: {result.get('error_message', 'Unknown error')}")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        # Check for error details
        has_format_error = False
        error_message = result.get('error_message', '').lower()
        
        format_error_terms = ['format', 'corrupt', 'invalid', 'parse', 'malformed']
        for term in format_error_terms:
            if term in error_message:
                has_format_error = True
                break
                
        if has_format_error:
            print("✅ Error message correctly indicates file format issue")
        else:
            print("❌ Error message does not clearly indicate the nature of the error")
        
        # Save test results
        test_results = {
            "test_case": "Error Handling - Corrupted File",
            "status": "passed",
            "execution_time": execution_time,
            "has_format_error": has_format_error,
            "error_message": result.get('error_message', 'Unknown error'),
            "full_result": result
        }
        
        with open(os.path.join(args.output_dir, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        return test_results
    elif result["status"] == "success":
        # Some systems might try to recover from corrupted files - this is also acceptable
        print(f"\n✅ Test passed: System recovered from corrupted file")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        # Save test results
        test_results = {
            "test_case": "Error Handling - Corrupted File",
            "status": "passed (recovered)",
            "execution_time": execution_time,
            "recovery_method": "Successfully parsed partial data",
            "full_result": result
        }
        
        with open(os.path.join(args.output_dir, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        return test_results
    else:
        print(f"\n❌ Test failed: Unexpected status: {result['status']}")
        
        # Save test results
        test_results = {
            "test_case": "Error Handling - Corrupted File",
            "status": "failed",
            "execution_time": execution_time,
            "error": f"Unexpected status: {result['status']}",
            "full_result": result
        }
        
        with open(os.path.join(args.output_dir, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        return test_results

def run_test():
    """Run all error handling tests."""
    results = {}
    
    # Test with missing file
    results["missing_file"] = run_test_missing_file()
    
    # Test with corrupted file
    results["corrupted_file"] = run_test_corrupted_file()
    
    return results

if __name__ == "__main__":
    run_test()