# simplified_example.py

import pandas as pd
import numpy as np
import os
from data_analysis_system.simplified_main import run_data_analysis

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    # Create a directory for sample data
    if not os.path.exists("sample_data"):
        os.makedirs("sample_data")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample dates
    date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create customer segments
    segments = np.random.choice(['High Value', 'Medium Value', 'Low Value'], 100)
    
    # Create marketing spend with a trend
    marketing_spend = np.linspace(1000, 2000, 100) + np.random.normal(0, 100, 100)
    
    # Create features with clear relationships
    price = np.random.normal(100, 20, 100)
    competitor_price = price * 0.9 + np.random.normal(0, 5, 100)
    
    # Create sales with dependencies on other variables
    # Sales = f(marketing_spend, price, segment, seasonality)
    base_sales = 500
    marketing_effect = 0.5 * marketing_spend / 1000
    price_effect = -2 * (price - 100) / 20
    segment_effect = np.where(segments == 'High Value', 100, 
                             np.where(segments == 'Medium Value', 50, 0))
    seasonality = 50 * np.sin(np.linspace(0, 4*np.pi, 100))
    
    sales = base_sales + marketing_effect + price_effect + segment_effect + seasonality + np.random.normal(0, 20, 100)
    
    # Add some outliers
    sales[10] = 1000  # Unusually high sales day
    sales[50] = 100   # Unusually low sales day
    
    # Create the DataFrame
    data = {
        'date': date_range,
        'marketing_spend': marketing_spend,
        'price': price,
        'competitor_price': competitor_price,
        'sales': sales,
        'customer_segment': segments
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    file_path = "sample_data/sales_data.csv"
    df.to_csv(file_path, index=False)
    
    print(f"Sample dataset created at: {file_path}")
    print(f"Dataset shape: {df.shape}")
    print("Dataset preview:")
    print(df.head())
    
    return file_path

def run_example():
    """Run an example analysis on the sample dataset."""
    # Create sample dataset
    data_path = create_sample_dataset()
    
    # Set up arguments
    class Args:
        source_type = "file"
        data_source = data_path
        objective = "Analyze factors influencing sales performance and identify opportunities for improvement"
        business_context = "Retail sales and marketing analysis"
        target_column = "sales"
        output_dir = "example_outputs"
    
    args = Args()
    
    # Set up configuration
    config = {
        "verbose": True,
        "output_dir": args.output_dir
    }
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"\nStarting analysis with objective: {args.objective}")
    print(f"Target column: {args.target_column}")
    print(f"Business context: {args.business_context}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    
    # Run the analysis
    result = run_data_analysis(args, config)
    
    # Print result
    if result["status"] == "success":
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {os.path.abspath(args.output_dir)}")
        
        # Print summary if available
        if "summary" in result:
            print("\nSummary of findings:")
            print(result["summary"])
    else:
        print(f"\nAnalysis failed: {result['error_message']}")

if __name__ == "__main__":
    run_example()