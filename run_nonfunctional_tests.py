# run_nonfunctional_tests.py

import os
import json
import time
import matplotlib.pyplot as plt
from performance_test import run_performance_tests
from reliability_test import run_reliability_test
from usability_test import run_usability_test

def run_all_nonfunctional_tests():
    """Run all non-functional tests and compile results."""
    print("\n" + "="*50)
    print("RUNNING ALL NON-FUNCTIONAL TESTS")
    print("="*50)
    
    # Create results directory
    results_dir = "nonfunctional_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Track overall start time
    overall_start_time = time.time()
    
    # Run all tests
    print("\nRunning Performance Tests...")
    performance_results = run_performance_tests()
    
    print("\nRunning Reliability Tests...")
    reliability_results = run_reliability_test(iterations=5)  # Reduced iterations for quicker testing
    
    print("\nRunning Usability Tests...")
    usability_results = run_usability_test()
    
    # Calculate overall execution time
    overall_execution_time = time.time() - overall_start_time
    
    # Compile summary of results
    summary = {
        "overall_execution_time": overall_execution_time,
        "performance": {
            "largest_dataset_size": max([r["size"] for r in performance_results]),
            "max_execution_time": max([r["execution_time"] for r in performance_results]),
            "max_memory_used": max([r["memory_used"] for r in performance_results]),
            "max_processing_rate": max([r["processing_rate"] for r in performance_results])
        },
        "reliability": {
            "success_rate": reliability_results["success_rate"],
            "avg_execution_time": reliability_results["avg_execution_time"],
            "execution_time_variability": reliability_results["cv_execution_time"],
            "insight_consistency": reliability_results["avg_similarity"]
        },
        "usability": {
            "overall_usability_score": usability_results["metrics"]["overall_usability_score"] if usability_results else 0,
            "insight_clarity": usability_results["metrics"]["insight_clarity"] if usability_results else 0,
            "insight_actionability": usability_results["metrics"]["insight_actionability"] if usability_results else 0,
            "recommendation_clarity": usability_results["metrics"]["recommendation_clarity"] if usability_results else 0,
            "recommendation_actionability": usability_results["metrics"]["recommendation_actionability"] if usability_results else 0
        }
    }
    
    # Save summary to file
    with open(os.path.join(results_dir, "nonfunctional_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate report
    generate_report(summary, results_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("NON-FUNCTIONAL TEST SUMMARY")
    print("="*50)
    print(f"Overall Execution Time: {summary['overall_execution_time']:.2f} seconds")
    
    print("\nPerformance Summary:")
    print(f"  Largest Dataset: {summary['performance']['largest_dataset_size']} rows")
    print(f"  Max Execution Time: {summary['performance']['max_execution_time']:.2f} seconds")
    print(f"  Max Memory Used: {summary['performance']['max_memory_used']:.2f} MB")
    print(f"  Max Processing Rate: {summary['performance']['max_processing_rate']:.2f} rows/second")
    
    print("\nReliability Summary:")
    print(f"  Success Rate: {summary['reliability']['success_rate']:.1f}%")
    print(f"  Avg Execution Time: {summary['reliability']['avg_execution_time']:.2f} seconds")
    print(f"  Execution Time Variability: {summary['reliability']['execution_time_variability']:.2f}%")
    print(f"  Insight Consistency: {summary['reliability']['insight_consistency']:.2f}")
    
    print("\nUsability Summary:")
    print(f"  Overall Usability Score: {summary['usability']['overall_usability_score']:.1f}/100")
    print(f"  Insight Clarity: {summary['usability']['insight_clarity']:.2f}")
    print(f"  Insight Actionability: {summary['usability']['insight_actionability']:.2f}")
    print(f"  Recommendation Clarity: {summary['usability']['recommendation_clarity']:.2f}")
    print(f"  Recommendation Actionability: {summary['usability']['recommendation_actionability']:.2f}")
    
    print(f"\nDetailed report saved to: {os.path.join(results_dir, 'nonfunctional_report.html')}")
    
    return summary

def generate_report(summary, results_dir):
    """Generate an HTML report of non-functional test results."""
    # Create data for radar chart
    categories = ['Performance', 'Reliability', 'Usability', 'Scalability', 'Efficiency']
    
    # Calculate scores (0-100)
    performance_score = min(100, 100 - (summary["performance"]["max_execution_time"] / 10) * 100)
    reliability_score = summary["reliability"]["success_rate"]
    usability_score = summary["usability"]["overall_usability_score"]
    
    # Calculate scalability score based on processing rate
    max_rate = summary["performance"]["max_processing_rate"]
    scalability_score = min(100, (max_rate / 1000) * 100)
    
    # Calculate efficiency score based on memory usage
    memory_efficiency = min(100, 100 - (summary["performance"]["max_memory_used"] / 100) * 100)
    
    scores = [performance_score, reliability_score, usability_score, scalability_score, memory_efficiency]
    
    # Create radar chart
    plt.figure(figsize=(8, 8))
    
    # Number of variables
    N = len(categories)
    
    # We are going to plot the first line of the data frame
    # But we need to repeat the first value to close the circular graph
    values = scores + [scores[0]]
    
    # What will be the angle of each axis in the plot (divide the plot / number of variables)
    angles = [n / N * 2 * 3.14159 for n in range(N)]
    angles += [angles[0]]
    
    # Draw one axis per variable + add labels
    plt.polar(angles, values, marker='o')
    plt.fill(angles, values, alpha=0.1)
    
    plt.xticks(angles[:-1], categories)
    plt.yticks([20, 40, 60, 80, 100], ['20', '40', '60', '80', '100'], color='grey', size=8)
    plt.title('System Quality Metrics', size=15, y=1.1)
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'quality_radar.png'))
    plt.close()
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis System Non-Functional Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .test-details {{ margin-bottom: 30px; }}
            .passed {{ color: #28a745; }}
            .warning {{ color: #fd7e14; }}
            .failed {{ color: #dc3545; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .charts {{ display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0; }}
            .chart {{ margin: 10px; text-align: center; }}
            img {{ max-width: 100%; height: auto; }}
            .metric {{ margin-bottom: 10px; }}
            .metric-name {{ font-weight: bold; }}
            progress {{ width: 100%; height: 20px; }}
            .score-high {{ color: #28a745; }}
            .score-medium {{ color: #fd7e14; }}
            .score-low {{ color: #dc3545; }}
        </style>
    </head>
    <body>
        <h1>Data Analysis System Non-Functional Test Report</h1>
        
        <div class="summary">
            <h2>Overall Quality Assessment</h2>
            <div class="charts">
                <div class="chart">
                    <img src="quality_radar.png" alt="System Quality Metrics">
                </div>
            </div>
        </div>
        
        <div class="test-details">
            <h2>Performance Test Results</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Rating</th>
                </tr>
                <tr>
                    <td>Largest Dataset Processed</td>
                    <td>{summary['performance']['largest_dataset_size']} rows</td>
                    <td class="{get_rating(summary['performance']['largest_dataset_size'], 1000, 10000, 50000)}">{get_rating_text(summary['performance']['largest_dataset_size'], 1000, 10000, 50000)}</td>
                </tr>
                <tr>
                    <td>Maximum Execution Time</td>
                    <td>{summary['performance']['max_execution_time']:.2f} seconds</td>
                    <td class="{get_rating(summary['performance']['max_execution_time'], 10, 5, 1, reverse=True)}">{get_rating_text(summary['performance']['max_execution_time'], 10, 5, 1, reverse=True)}</td>
                </tr>
                <tr>
                    <td>Maximum Memory Usage</td>
                    <td>{summary['performance']['max_memory_used']:.2f} MB</td>
                    <td class="{get_rating(summary['performance']['max_memory_used'], 500, 200, 100, reverse=True)}">{get_rating_text(summary['performance']['max_memory_used'], 500, 200, 100, reverse=True)}</td>
                </tr>
                <tr>
                    <td>Maximum Processing Rate</td>
                    <td>{summary['performance']['max_processing_rate']:.2f} rows/second</td>
                    <td class="{get_rating(summary['performance']['max_processing_rate'], 100, 1000, 5000)}">{get_rating_text(summary['performance']['max_processing_rate'], 100, 1000, 5000)}</td>
                </tr>
            </table>
            
            <p>Performance tests measured how efficiently the system processes datasets of various sizes.</p>
            <p>View detailed performance results in the <a href="../performance_results/performance_metrics.json">performance metrics file</a>.</p>
        </div>
        
        <div class="test-details">
            <h2>Reliability Test Results</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Rating</th>
                </tr>
                <tr>
                    <td>Success Rate</td>
                    <td>{summary['reliability']['success_rate']:.1f}%</td>
                    <td class="{get_rating(summary['reliability']['success_rate'], 90, 95, 99)}">{get_rating_text(summary['reliability']['success_rate'], 90, 95, 99)}</td>
                </tr>
                <tr>
                    <td>Average Execution Time</td>
                    <td>{summary['reliability']['avg_execution_time']:.2f} seconds</td>
                    <td class="{get_rating(summary['reliability']['avg_execution_time'], 10, 5, 1, reverse=True)}">{get_rating_text(summary['reliability']['avg_execution_time'], 10, 5, 1, reverse=True)}</td>
                </tr>
                <tr>
                    <td>Execution Time Variability</td>
                    <td>{summary['reliability']['execution_time_variability']:.2f}%</td>
                    <td class="{get_rating(summary['reliability']['execution_time_variability'], 20, 10, 5, reverse=True)}">{get_rating_text(summary['reliability']['execution_time_variability'], 20, 10, 5, reverse=True)}</td>
                </tr>
                <tr>
                    <td>Insight Consistency</td>
                    <td>{summary['reliability']['insight_consistency']:.2f}</td>
                    <td class="{get_rating(summary['reliability']['insight_consistency'], 0.7, 0.85, 0.95)}">{get_rating_text(summary['reliability']['insight_consistency'], 0.7, 0.85, 0.95)}</td>
                </tr>
            </table>
            
            <p>Reliability tests measured how consistently the system performs across multiple executions.</p>
            <p>View detailed reliability results in the <a href="../reliability_results/reliability_metrics.json">reliability metrics file</a>.</p>
        </div>
        
        <div class="test-details">
            <h2>Usability Test Results</h2>
            
            <div class="metric">
                <div class="metric-name">Overall Usability Score: 
                    <span class="{get_score_class(summary['usability']['overall_usability_score']/100)}">
                        {summary['usability']['overall_usability_score']:.1f}/100
                    </span>
                </div>
                <progress value="{summary['usability']['overall_usability_score']}" max="100"></progress>
            </div>
            
            <div class="metric">
                <div class="metric-name">Insight Clarity: 
                    <span class="{get_score_class(summary['usability']['insight_clarity'])}">
                        {summary['usability']['insight_clarity']:.2f}
                    </span>
                </div>
                <progress value="{summary['usability']['insight_clarity']*100}" max="100"></progress>
            </div>
            
            <div class="metric">
                <div class="metric-name">Insight Actionability: 
                    <span class="{get_score_class(summary['usability']['insight_actionability'])}">
                        {summary['usability']['insight_actionability']:.2f}
                    </span>
                </div>
                <progress value="{summary['usability']['insight_actionability']*100}" max="100"></progress>
            </div>
            
            <div class="metric">
                <div class="metric-name">Recommendation Clarity: 
                    <span class="{get_score_class(summary['usability']['recommendation_clarity'])}">
                        {summary['usability']['recommendation_clarity']:.2f}
                    </span>
                </div>
                <progress value="{summary['usability']['recommendation_clarity']*100}" max="100"></progress>
            </div>
            
            <div class="metric">
                <div class="metric-name">Recommendation Actionability: 
                    <span class="{get_score_class(summary['usability']['recommendation_actionability'])}">
                        {summary['usability']['recommendation_actionability']:.2f}
                    </span>
                </div>
                <progress value="{summary['usability']['recommendation_actionability']*100}" max="100"></progress>
            </div>
            
            <p>Usability tests evaluated how clear and actionable the system's insights and recommendations are.</p>
            <p>View detailed usability results in the <a href="../usability_results/usability_report.html">usability report</a>.</p>
        </div>
        
        <div class="test-details">
            <h2>Recommendations for Improvement</h2>
            <ul>
    """
    
    # Add recommendations based on test results
    if summary['performance']['max_execution_time'] > 5:
        html_content += "<li>Optimize processing algorithms to improve execution time for large datasets</li>"
    
    if summary['performance']['max_memory_used'] > 200:
        html_content += "<li>Reduce memory usage by implementing more efficient data structures or streaming processing</li>"
    
    if summary['reliability']['execution_time_variability'] > 10:
        html_content += "<li>Improve execution time consistency by optimizing resource usage and algorithm stability</li>"
    
    if summary['reliability']['insight_consistency'] < 0.85:
        html_content += "<li>Enhance insight generation consistency to ensure similar results across runs</li>"
    
    if summary['usability']['insight_clarity'] < 0.7:
        html_content += "<li>Improve insight clarity by ensuring all insights include metrics, directions, and magnitudes</li>"
    
    if summary['usability']['insight_actionability'] < 0.7:
        html_content += "<li>Enhance insight actionability by including more specific recommendations and causal relationships</li>"
    
    # Add general recommendations
    html_content += """
                <li>Implement parallel processing for better performance with large datasets</li>
                <li>Add more advanced statistical methods to improve insight quality</li>
                <li>Enhance visualization capabilities to better communicate insights</li>
                <li>Implement user feedback collection to continuously improve usability</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(os.path.join(results_dir, "nonfunctional_report.html"), 'w') as f:
        f.write(html_content)

def get_rating(value, threshold_fair, threshold_good, threshold_excellent, reverse=False):
    """Get rating class based on thresholds."""
    if reverse:
        if value <= threshold_excellent:
            return "passed"
        elif value <= threshold_good:
            return "passed"
        elif value <= threshold_fair:
            return "warning"
        else:
            return "failed"
    else:
        if value >= threshold_excellent:
            return "passed"
        elif value >= threshold_good:
            return "passed"
        elif value >= threshold_fair:
            return "warning"
        else:
            return "failed"

def get_rating_text(value, threshold_fair, threshold_good, threshold_excellent, reverse=False):
    """Get rating text based on thresholds."""
    if reverse:
        if value <= threshold_excellent:
            return "Excellent"
        elif value <= threshold_good:
            return "Good"
        elif value <= threshold_fair:
            return "Fair"
        else:
            return "Poor"
    else:
        if value >= threshold_excellent:
            return "Excellent"
        elif value >= threshold_good:
            return "Good"
        elif value >= threshold_fair:
            return "Fair"
        else:
            return "Poor"

def get_score_class(score):
    """Get CSS class based on score."""
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-medium"
    else:
        return "score-low"

if __name__ == "__main__":
    run_all_nonfunctional_tests()