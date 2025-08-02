# data_analysis_system/main.py

import os
import argparse
import json
from crewai import Process
from data_analysis_system.orchestration import DataAnalysisOrchestrator

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data Analysis Agentic System")
    
    # Required arguments
    parser.add_argument(
        "--data-source", 
        type=str, 
        required=True,
        help="Path to data file or URL"
    )
    
    # Optional arguments
    parser.add_argument(
        "--source-type",
        type=str,
        default="file",
        choices=["file", "api", "url"],
        help="Type of data source (default: file)"
    )
    parser.add_argument(
        "--objective", 
        type=str, 
        default="Identify key patterns and insights in the data",
        help="Analysis objective (default: Identify key patterns and insights in the data)"
    )
    parser.add_argument(
        "--business-context", 
        type=str, 
        default=None,
        help="Business context for the analysis"
    )
    parser.add_argument(
        "--target-column", 
        type=str, 
        default=None,
        help="Target column for predictive insights"
    )
    parser.add_argument(
        "--process-type",
        type=str,
        default="sequential",
        choices=["sequential", "parallel", "hierarchical"],
        help="Type of process execution (default: sequential)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs (default: outputs)"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to configuration file (JSON)"
    )
    
    return parser.parse_args()

def load_config(config_file):
    """Load configuration from a JSON file."""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}

def map_process_type(process_type_str):
    """Map process type string to CrewAI Process enum."""
    process_map = {
        "sequential": Process.sequential,
        "parallel": Process.parallel,
        "hierarchical": Process.hierarchical
    }
    return process_map.get(process_type_str, Process.sequential)

def run_data_analysis(args, config):
    """Run the data analysis workflow."""
    # Create data source dictionary
    data_source = {
        "source_type": args.source_type,
        "source_path": args.data_source
    }
    
    # Create initial context
    context = {
        "data_source": data_source,
        "analysis_objective": args.objective,
        "business_context": args.business_context,
        "target_column": args.target_column
    }
    
    # Update config with command line arguments
    config.update({
        "process_type": map_process_type(args.process_type),
        "output_dir": args.output_dir
    })
    
    # Create orchestrator with config
    orchestrator = DataAnalysisOrchestrator(config)
    
    # Run the analysis
    result = orchestrator.run(context)
    
    return result

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Run the analysis
    result = run_data_analysis(args, config)
    
    # Print result status
    if result["status"] == "success":
        print(f"Analysis completed successfully. Results saved to {os.path.abspath(args.output_dir)}")
    else:
        print(f"Analysis failed: {result['error_message']}")
        print(f"Error details saved to {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()