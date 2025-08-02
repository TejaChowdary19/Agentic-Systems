# standalone_test.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StandaloneTest")

def create_sample_dataset():
    """Create a simple test dataset."""
    # Create a sample dataframe with some patterns
    np.random.seed(42)
    
    # Create sample dates
    date_range = pd.date_range(start='2023-01-01', periods=50, freq='D')
    
    # Create a variable with trend
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 1, 50)
    
    # Create categorical variable
    categories = np.random.choice(['A', 'B', 'C'], 50)
    
    # Make category A have higher y values
    y[categories == 'A'] += 3
    
    # Add outlier
    y[10] = 30
    
    # Create dataframe
    data = {
        'date': date_range,
        'x': x,
        'y': y,
        'category': categories
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    test_file = 'standalone_test_data.csv'
    df.to_csv(test_file, index=False)
    
    logger.info(f"Created test file: {test_file}")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Data preview:\n{df.head()}")
    
    return test_file

def analyze_data(data_path, output_dir='standalone_outputs'):
    """Perform basic data analysis."""
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"Analyzing data from: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Basic statistics
    stats_result = df.describe()
    logger.info(f"Basic statistics:\n{stats_result}")
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, "statistics.csv")
    stats_result.to_csv(stats_file)
    
    # Create correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    logger.info(f"Correlation matrix:\n{corr_matrix}")
    
    # Save correlation matrix
    corr_file = os.path.join(output_dir, "correlations.csv")
    corr_matrix.to_csv(corr_file)
    
    # Create a visualization
    plt.figure(figsize=(10, 6))
    if 'x' in df.columns and 'y' in df.columns:
        plt.scatter(df['x'], df['y'])
        plt.title('Scatter plot of y vs x')
        plt.xlabel('x')
        plt.ylabel('y')
        scatter_file = os.path.join(output_dir, "scatter_plot.png")
        plt.savefig(scatter_file)
        plt.close()
    
    # Create heatmap of correlations
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    heatmap_file = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_file)
    plt.close()
    
    # Generate insights
    insights = []
    
    # Look for strong correlations
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.5:
                corr_value = corr_matrix.loc[col1, col2]
                direction = "positive" if corr_value > 0 else "negative"
                strength = "strong" if abs(corr_value) > 0.7 else "moderate"
                insights.append({
                    "type": "correlation",
                    "finding": f"{strength.capitalize()} {direction} correlation ({round(corr_value, 2)}) between {col1} and {col2}"
                })
    
    # Look for outliers
    for col in df.select_dtypes(include=[np.number]).columns:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = df.index[z_scores > 3].tolist()
        if outliers:
            insights.append({
                "type": "outlier",
                "finding": f"Found {len(outliers)} outliers in column {col}"
            })
    
    # Look for trends in time series
    if 'date' in df.columns:
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'date':
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                
                # Sort by date
                df_sorted = df.sort_values(by='date')
                
                # Simple trend check
                n = len(df_sorted)
                if n >= 9:  # Need at least 9 points for three segments
                    first_third = df_sorted[col].iloc[:n//3].mean()
                    last_third = df_sorted[col].iloc[-n//3:].mean()
                    
                    percent_change = ((last_third - first_third) / first_third * 100) if first_third != 0 else 0
                    
                    if abs(percent_change) >= 10:  # 10% change threshold
                        direction = "increasing" if percent_change > 0 else "decreasing"
                        insights.append({
                            "type": "trend",
                            "finding": f"{col} is {direction} over time (by {abs(round(percent_change, 2))}%)"
                        })
    
    # Save insights
    insights_file = os.path.join(output_dir, "insights.json")
    with open(insights_file, 'w') as f:
        json.dump({"insights": insights}, f, indent=2)
    
    # Create summary
    summary = [
        f"Analysis of {data_path}",
        f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns",
        "",
        "Key Insights:"
    ]
    
    for insight in insights:
        summary.append(f"- {insight['finding']}")
    
    summary.append("")
    summary.append(f"Files created in {output_dir}:")
    summary.append(f"- Statistics: {stats_file}")
    summary.append(f"- Correlations: {corr_file}")
    summary.append(f"- Visualizations: scatter_plot.png, correlation_heatmap.png")
    summary.append(f"- Insights: {insights_file}")
    
    summary_text = "\n".join(summary)
    
    # Save summary
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    
    return {
        "status": "success",
        "summary": summary_text,
        "insights": insights,
        "output_dir": output_dir
    }

# Main function
def main():
    # Create test dataset
    data_path = create_sample_dataset()
    
    # Analyze data
    result = analyze_data(data_path)
    
    # Print result
    if result["status"] == "success":
        print("\nAnalysis completed successfully!")
        print("\nSummary of findings:")
        print(result["summary"])
    else:
        print(f"\nAnalysis failed: {result.get('error_message', 'Unknown error')}")
    
    # Clean up
    if os.path.exists(data_path):
        os.remove(data_path)

if __name__ == "__main__":
    main()