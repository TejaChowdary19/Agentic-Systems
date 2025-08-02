# test_insight_generator.py

import json
import os
import pandas as pd
import numpy as np
from data_analysis_system.tools.statistical_analysis import StatisticalAnalysisTool
from data_analysis_system.tools.insight_generator import InsightGeneratorTool

def main():
    # Create a test dataset with clear patterns and relationships
    np.random.seed(42)  # For reproducibility
    
    # Create sample dates (for trend analysis)
    date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create features with clear relationships
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(0, 1, 100)
    
    # Create a variable with an upward trend
    trend = np.linspace(0, 2, 100) + np.random.normal(0, 0.5, 100)
    
    # y has strong relationship with x1, weak with x2
    y = 2 * x1 + 0.5 * x2 + np.random.normal(0, 1, 100)
    
    # Add some outliers
    y[10] = 10  # Strong positive outlier
    y[20] = -10  # Strong negative outlier
    
    # Create categorical variable
    categories = np.random.choice(['A', 'B', 'C'], 100)
    
    # Make category A have higher y values
    y[categories == 'A'] += 2
    
    test_data = {
        'date': date_range,
        'x1': x1,
        'x2': x2,
        'trend': trend,
        'y': y,
        'category': categories
    }
    
    test_df = pd.DataFrame(test_data)
    test_file = 'test_data_for_insights.csv'
    test_df.to_csv(test_file, index=False)
    
    print(f"Created test file: {test_file}")
    print(f"Test data shape: {test_df.shape}")
    print("Test data preview:")
    print(test_df.head())
    print("\n")
    
    # First run statistical analysis to generate analysis results
    print("Running statistical analysis...")
    stat_analysis_tool = StatisticalAnalysisTool()
    analysis_result = stat_analysis_tool._run(
        data_path=test_file,
        analysis_types=["descriptive", "correlation", "regression"],
        target_column="y"
    )
    
    analysis_result_dict = json.loads(analysis_result)
    analysis_results_path = analysis_result_dict["analysis_results_path"]
    print(f"Statistical analysis complete. Results saved to: {analysis_results_path}")
    print("\n")
    
    # Now run the insight generator
    print("Testing insight generator...")
    insight_generator_tool = InsightGeneratorTool()
    result = insight_generator_tool._run(
        data_path=test_file,
        insight_types=["trends", "outliers", "correlations", "predictions", "segments"],
        analysis_results_path=analysis_results_path,
        target_column="y",
        business_context="Sales and marketing performance analysis"
    )
    
    # Parse and print the insights
    insights_dict = json.loads(result)
    print(f"Status: {insights_dict['status']}")
    print(f"Insights path: {insights_dict['insights_path']}")
    
    # Print key insights by type
    if "insights" in insights_dict:
        insights_by_type = insights_dict["insights"]
        
        # Print trend insights
        if "trends" in insights_by_type and insights_by_type["trends"]:
            print("\nTrend Insights:")
            for trend in insights_by_type["trends"]:
                print(f"- {trend['insight']}")
        
        # Print outlier insights
        if "outliers" in insights_by_type and insights_by_type["outliers"]:
            print("\nOutlier Insights:")
            for outlier in insights_by_type["outliers"]:
                print(f"- {outlier['insight']}")
        
        # Print correlation insights
        if "correlations" in insights_by_type and insights_by_type["correlations"]:
            print("\nCorrelation Insights:")
            for corr in insights_by_type["correlations"]:
                print(f"- {corr['insight']}")
        
        # Print prediction insights
        if "predictions" in insights_by_type and insights_by_type["predictions"] and "insights" in insights_by_type["predictions"]:
            print("\nPrediction Insights:")
            for pred in insights_by_type["predictions"]["insights"]:
                print(f"- {pred['insight']}")
                if "evidence" in pred:
                    print(f"  Evidence: {pred['evidence']}")
        
        # Print segment insights
        if "segments" in insights_by_type and insights_by_type["segments"]:
            print("\nSegment Insights:")
            for segment in insights_by_type["segments"]:
                print(f"- {segment['insight']}")
    
    # Print recommended next steps
    if "recommended_next_steps" in insights_dict:
        print("\nRecommended Next Steps:")
        for step in insights_dict["recommended_next_steps"]:
            print(f"- [{step['priority']}] {step['recommendation']}")
    
    # Clean up
    os.remove(test_file)
    if os.path.exists(analysis_results_path):
        os.remove(analysis_results_path)
    if os.path.exists(insights_dict["insights_path"]):
        os.remove(insights_dict["insights_path"])
    print(f"\nRemoved test files: {test_file}, {analysis_results_path}, {insights_dict['insights_path']}")

if __name__ == "__main__":
    main()