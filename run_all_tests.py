# run_all_tests.py

import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from test_clean_data import run_test as test_clean_data
from test_messy_data import run_test as test_messy_data
from test_small_data import run_test as test_small_data
from test_error_handling import run_test as test_error_handling

def run_all_tests():
    """Run all test cases and compile results."""
    print("\n" + "="*50)
    print("RUNNING ALL FUNCTIONAL TESTS")
    print("="*50)
    
    # Create results directory
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Track overall start time
    overall_start_time = time.time()
    
    # Run all tests
    test_results = {}
    
    print("\nRunning Test 1: Clean Data Analysis...")
    test_results["clean_data"] = test_clean_data()
    
    print("\nRunning Test 2: Messy Data Handling...")
    test_results["messy_data"] = test_messy_data()
    
    print("\nRunning Test 3: Small Dataset Analysis...")
    test_results["small_data"] = test_small_data()
    
    print("\nRunning Test 4: Error Handling Tests...")
    error_results = test_error_handling()
    test_results["error_missing_file"] = error_results["missing_file"]
    test_results["error_corrupted_file"] = error_results["corrupted_file"]
    
    # Calculate overall execution time
    overall_execution_time = time.time() - overall_start_time
    
    # Compile summary of results
    summary = {
        "total_tests": len(test_results),
        "tests_passed": sum(1 for result in test_results.values() if result.get("status") == "passed" or result.get("status") == "passed (recovered)"),
        "tests_failed": sum(1 for result in test_results.values() if result.get("status") == "failed"),
        "overall_execution_time": overall_execution_time,
        "test_execution_times": {key: result.get("execution_time", 0) for key, result in test_results.items()},
        "detailed_results": test_results
    }
    
    # Calculate pass rate
    summary["pass_rate"] = (summary["tests_passed"] / summary["total_tests"]) * 100
    
    # Save summary to file
    with open(os.path.join(results_dir, "test_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate report
    generate_report(summary, results_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total tests: {summary['total_tests']}")
    print(f"Tests passed: {summary['tests_passed']}")
    print(f"Tests failed: {summary['tests_failed']}")
    print(f"Pass rate: {summary['pass_rate']:.1f}%")
    print(f"Overall execution time: {summary['overall_execution_time']:.2f} seconds")
    print("\nDetailed execution times:")
    for test, time_taken in summary['test_execution_times'].items():
        print(f"  {test}: {time_taken:.2f} seconds")
    
    print(f"\nDetailed report saved to: {os.path.join(results_dir, 'test_report.html')}")
    
    return summary

def generate_report(summary, results_dir):
    """Generate an HTML report of test results."""
    # Create data for charts
    test_names = list(summary["test_execution_times"].keys())
    execution_times = list(summary["test_execution_times"].values())
    
    # Create execution time chart
    plt.figure(figsize=(10, 6))
    plt.bar(test_names, execution_times)
    plt.title('Test Execution Times')
    plt.xlabel('Test Case')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'execution_times.png'))
    plt.close()
    
    # Create pass/fail chart
    plt.figure(figsize=(8, 8))
    plt.pie([summary["tests_passed"], summary["tests_failed"]], 
            labels=['Passed', 'Failed'], 
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'])
    plt.title('Test Results')
    plt.savefig(os.path.join(results_dir, 'pass_fail.png'))
    plt.close()
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis System Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .test-details {{ margin-bottom: 30px; }}
            .passed {{ color: #28a745; }}
            .failed {{ color: #dc3545; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .charts {{ display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0; }}
            .chart {{ margin: 10px; text-align: center; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Data Analysis System Functional Test Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total Tests:</strong> {summary['total_tests']}</p>
            <p><strong>Tests Passed:</strong> {summary['tests_passed']}</p>
            <p><strong>Tests Failed:</strong> {summary['tests_failed']}</p>
            <p><strong>Pass Rate:</strong> {summary['pass_rate']:.1f}%</p>
            <p><strong>Overall Execution Time:</strong> {summary['overall_execution_time']:.2f} seconds</p>
        </div>
        
        <div class="charts">
            <div class="chart">
                <h3>Test Execution Times</h3>
                <img src="execution_times.png" alt="Execution Times Chart">
            </div>
            <div class="chart">
                <h3>Pass/Fail Results</h3>
                <img src="pass_fail.png" alt="Pass/Fail Chart">
            </div>
        </div>
        
        <div class="test-details">
            <h2>Test Details</h2>
            <table>
                <tr>
                    <th>Test Case</th>
                    <th>Status</th>
                    <th>Execution Time (s)</th>
                    <th>Notes</th>
                </tr>
    """
    
    # Add rows for each test
    for test_name, result in summary["detailed_results"].items():
        status = result.get("status", "unknown")
        status_class = "passed" if status == "passed" or status == "passed (recovered)" else "failed"
        execution_time = result.get("execution_time", 0)
        
        # Determine notes based on test type
        notes = ""
        if test_name == "clean_data":
            success_rate = result.get("success_rate", 0)
            notes = f"Found {success_rate:.1f}% of expected insights"
        elif test_name == "messy_data":
            insight_rate = result.get("insight_success_rate", 0)
            cleaning_rate = result.get("cleaning_success_rate", 0)
            notes = f"Insight rate: {insight_rate:.1f}%, Cleaning rate: {cleaning_rate:.1f}%"
        elif test_name == "small_data":
            valid_count = len(result.get("valid_findings", []))
            questionable_count = len(result.get("questionable_findings", []))
            limitations = "Yes" if result.get("limitations_acknowledged", False) else "No"
            notes = f"Valid findings: {valid_count}, Questionable: {questionable_count}, Acknowledged limitations: {limitations}"
        elif "error" in test_name:
            if "error_message" in result:
                notes = f"Error message: {result['error_message'][:50]}..."
            else:
                notes = "Error handling test"
        
        html_content += f"""
                <tr>
                    <td>{test_name}</td>
                    <td class="{status_class}">{status}</td>
                    <td>{execution_time:.2f}</td>
                    <td>{notes}</td>
                </tr>
        """
    
    # Complete the HTML
    html_content += """
            </table>
        </div>
        
        <div class="recommendations">
            <h2>Recommendations</h2>
            <ul>
    """
    
    # Add recommendations based on test results
    recommendations = []
    
    # Check for clean data test
    clean_data_result = summary["detailed_results"].get("clean_data", {})
    if clean_data_result.get("status") == "passed":
        success_rate = clean_data_result.get("success_rate", 0)
        if success_rate < 100:
            recommendations.append("Improve pattern detection algorithms to identify more expected relationships in clean data")
    
    # Check for messy data test
    messy_data_result = summary["detailed_results"].get("messy_data", {})
    if messy_data_result.get("status") == "passed":
        cleaning_rate = messy_data_result.get("cleaning_success_rate", 0)
        if cleaning_rate < 100:
            recommendations.append("Enhance data cleaning capabilities to better handle missing values, outliers, and duplicates")
    
    # Check for small data test
    small_data_result = summary["detailed_results"].get("small_data", {})
    if small_data_result.get("status") == "passed":
        if not small_data_result.get("limitations_acknowledged", False):
            recommendations.append("Add warnings or notes about statistical limitations when analyzing small datasets")
    
    # Add general recommendations
    recommendations.append("Implement more sophisticated statistical tests to improve insight quality")
    recommendations.append("Add more comprehensive error messages with specific recovery suggestions")
    recommendations.append("Consider implementing parallel processing for better performance with large datasets")
    
    # Add recommendations to HTML
    for recommendation in recommendations:
        html_content += f"<li>{recommendation}</li>\n"
    
    # Complete the HTML
    html_content += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(os.path.join(results_dir, "test_report.html"), 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    run_all_tests()