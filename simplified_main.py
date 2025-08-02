# data_analysis_system/simplified_main.py

import os
import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd

# Import tools
from data_analysis_system.tools.data_retrieval import DataRetrievalTool
from data_analysis_system.tools.data_cleaning import DataCleaningTool
from data_analysis_system.tools.statistical_analysis import StatisticalAnalysisTool
from data_analysis_system.tools.insight_generator import InsightGeneratorTool

# Import metrics collector
from data_analysis_system.utils.metrics_collector import MetricsCollector

class SimplifiedOrchestrator:
    """Orchestrates the data analysis workflow using multiple specialized tools."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator with optional configuration."""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("data_analysis.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("SimplifiedOrchestrator")
        
        # Set default configuration if none provided
        if config is None:
            config = {}
        
        self.config = {
            "verbose": config.get("verbose", True),
            "output_dir": config.get("output_dir", "outputs"),
            "collect_metrics": config.get("collect_metrics", True),
            "metrics_dir": config.get("metrics_dir", "metrics")
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])
        
        # Initialize metrics collector if enabled
        self.metrics_collector = None
        if self.config["collect_metrics"]:
            self.metrics_collector = MetricsCollector(output_dir=self.config["metrics_dir"])
        
        # Initialize tools
        self.tools = {
            "data_retrieval": DataRetrievalTool(),
            "data_cleaning": DataCleaningTool(),
            "statistical_analysis": StatisticalAnalysisTool(),
            "insight_generator": InsightGeneratorTool()
        }
        
        self.logger.info("SimplifiedOrchestrator initialized")
    
    def run(self, context: Dict[str, Any]):
        """Run the complete data analysis workflow."""
        self.logger.info("Starting data analysis workflow...")
        
        # Start metrics collection if enabled
        if self.metrics_collector:
            self.metrics_collector.start_run()
        
        try:
            # Step 1: Data Collection
            self.logger.info("Step 1: Data Collection")
            if self.metrics_collector:
                self.metrics_collector.start_step("data_collection")
                
            data_source = context.get('data_source', {'source_type': 'file', 'source_path': 'data.csv'})
            
            retrieval_result = self.tools["data_retrieval"]._run(
                source_type=data_source["source_type"],
                source_path=data_source["source_path"]
            )
            
            retrieval_data = json.loads(retrieval_result)
            if retrieval_data["status"] != "success":
                error_msg = f"Data retrieval failed: {retrieval_data.get('message', 'Unknown error')}"
                if self.metrics_collector:
                    self.metrics_collector.record_error(Exception(error_msg))
                raise Exception(error_msg)
            
            self.logger.info(f"Data retrieved: {retrieval_data['data_path']}")
            
            # Record dataset metrics if possible
            if self.metrics_collector and "data_path" in retrieval_data:
                try:
                    df = pd.read_csv(retrieval_data["data_path"])
                    self.metrics_collector.record_dataset_metrics(df)
                except Exception as e:
                    self.logger.warning(f"Could not record dataset metrics: {str(e)}")
            
            if self.metrics_collector:
                self.metrics_collector.end_step("data_collection")
            
            # Step 2: Data Processing
            self.logger.info("Step 2: Data Processing")
            if self.metrics_collector:
                self.metrics_collector.start_step("data_processing")
                
            data_path = retrieval_data["data_path"]
            
            cleaning_result = self.tools["data_cleaning"]._run(
                data_path=data_path,
                operations=["handle_missing", "remove_duplicates", "remove_outliers", "normalize"]
            )
            
            cleaning_data = json.loads(cleaning_result)
            if cleaning_data["status"] != "success":
                error_msg = f"Data cleaning failed: {cleaning_data.get('message', 'Unknown error')}"
                if self.metrics_collector:
                    self.metrics_collector.record_error(Exception(error_msg))
                raise Exception(error_msg)
            
            cleaned_data_path = cleaning_data["cleaned_data_path"]
            self.logger.info(f"Data cleaned: {cleaned_data_path}")
            
            if self.metrics_collector:
                self.metrics_collector.end_step("data_processing")
            
            # Step 3: Statistical Analysis
            self.logger.info("Step 3: Statistical Analysis")
            if self.metrics_collector:
                self.metrics_collector.start_step("statistical_analysis")
                
            target_column = context.get('target_column')
            
            analysis_result = self.tools["statistical_analysis"]._run(
                data_path=cleaned_data_path,
                analysis_types=["descriptive", "correlation", "regression"],
                target_column=target_column
            )
            
            analysis_data = json.loads(analysis_result)
            if analysis_data["status"] != "success":
                error_msg = f"Statistical analysis failed: {analysis_data.get('message', 'Unknown error')}"
                if self.metrics_collector:
                    self.metrics_collector.record_error(Exception(error_msg))
                raise Exception(error_msg)
            
            analysis_results_path = analysis_data["analysis_results_path"]
            self.logger.info(f"Analysis complete: {analysis_results_path}")
            
            # Record correlation and regression metrics
            if self.metrics_collector and "analyses" in analysis_data:
                if "correlation" in analysis_data["analyses"]:
                    corr_data = analysis_data["analyses"]["correlation"]
                    if "matrix" in corr_data:
                        # Convert correlation matrix back to DataFrame
                        corr_matrix = pd.DataFrame(corr_data["matrix"])
                        self.metrics_collector.record_correlation_metrics(corr_matrix)
                
                if "regression" in analysis_data["analyses"]:
                    self.metrics_collector.record_regression_metrics(
                        analysis_data["analyses"]["regression"]
                    )
            
            if self.metrics_collector:
                self.metrics_collector.end_step("statistical_analysis")
            
            # Step 4: Insight Generation
            self.logger.info("Step 4: Insight Generation")
            if self.metrics_collector:
                self.metrics_collector.start_step("insight_generation")
                
            business_context = context.get('business_context')
            
            insight_result = self.tools["insight_generator"]._run(
                data_path=cleaned_data_path,
                insight_types=["trends", "outliers", "correlations", "predictions", "segments"],
                analysis_results_path=analysis_results_path,
                target_column=target_column,
                business_context=business_context
            )
            
            insight_data = json.loads(insight_result)
            if insight_data["status"] != "success":
                error_msg = f"Insight generation failed: {insight_data.get('message', 'Unknown error')}"
                if self.metrics_collector:
                    self.metrics_collector.record_error(Exception(error_msg))
                raise Exception(error_msg)
            
            insights_path = insight_data["insights_path"]
            self.logger.info(f"Insights generated: {insights_path}")
            
            # Record insight and recommendation metrics
            if self.metrics_collector:
                if "insights" in insight_data:
                    self.metrics_collector.record_insight_metrics(insight_data["insights"])
                
                if "recommended_next_steps" in insight_data:
                    self.metrics_collector.record_recommendation_metrics(
                        insight_data["recommended_next_steps"]
                    )
            
            if self.metrics_collector:
                self.metrics_collector.end_step("insight_generation")
            
            # Create final report
            self.logger.info("Creating final report")
            if self.metrics_collector:
                self.metrics_collector.start_step("report_generation")
            
            final_report = {
                "status": "success",
                "analysis_objective": context.get('analysis_objective', 'Data analysis'),
                "data_summary": {
                    "original_data": retrieval_data.get("data_shape", []),
                    "cleaned_data": cleaning_data.get("cleaned_data_shape", [])
                },
                "key_findings": self._extract_key_findings(insight_data),
                "recommendations": insight_data.get("recommended_next_steps", []),
                "file_paths": {
                    "original_data": data_path,
                    "cleaned_data": cleaned_data_path,
                    "analysis_results": analysis_results_path,
                    "insights": insights_path
                }
            }
            
            # Save final report
            report_path = os.path.join(self.config["output_dir"], "final_report.json")
            with open(report_path, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            self.logger.info(f"Final report created: {report_path}")
            
            if self.metrics_collector:
                self.metrics_collector.end_step("report_generation")
            
            # Create a summary for console output
            summary = self._create_summary(final_report)
            final_report["summary"] = summary
            
            # Record successful run
            if self.metrics_collector:
                self.metrics_collector.record_success()
                self.metrics_collector.end_run()
                
                # Save metrics
                metrics_path = self.metrics_collector.save_metrics()
                self.logger.info(f"Metrics saved to: {metrics_path}")
                
                # Add metrics summary to report
                final_report["metrics_summary"] = self.metrics_collector.get_metrics_summary()
            
            return final_report
            
        except Exception as e:
            self.logger.exception("Error in workflow execution")
            
            # Record error metrics if enabled
            if self.metrics_collector:
                self.metrics_collector.record_error(e)
                self.metrics_collector.end_run()
                
                # Try to save metrics even on error
                try:
                    metrics_path = self.metrics_collector.save_metrics()
                    self.logger.info(f"Error metrics saved to: {metrics_path}")
                except:
                    self.logger.warning("Could not save metrics after error")
            
            # Create error output
            error_output = {
                "status": "error",
                "error_message": str(e),
                "error_type": type(e).__name__
            }
            
            # Save error output
            error_path = os.path.join(self.config["output_dir"], "error_output.json")
            with open(error_path, 'w') as f:
                json.dump(error_output, f, indent=2)
            
            self.logger.info(f"Error details saved to {error_path}")
            return error_output
    
    def _extract_key_findings(self, insight_data):
        """Extract key findings from insight data."""
        key_findings = []
        
        if "insights" not in insight_data:
            return key_findings
        
        insights = insight_data["insights"]
        
        # Extract trend insights
        if "trends" in insights and insights["trends"]:
            for trend in insights["trends"][:2]:  # Top 2 trends
                if "insight" in trend:
                    key_findings.append({
                        "type": "trend",
                        "finding": trend["insight"]
                    })
        
        # Extract correlation insights
        if "correlations" in insights and insights["correlations"]:
            for corr in insights["correlations"][:2]:  # Top 2 correlations
                if "insight" in corr:
                    key_findings.append({
                        "type": "correlation",
                        "finding": corr["insight"]
                    })
        
        # Extract prediction insights
        if "predictions" in insights and "insights" in insights["predictions"]:
            for pred in insights["predictions"]["insights"][:2]:  # Top 2 predictions
                if "insight" in pred:
                    key_findings.append({
                        "type": "prediction",
                        "finding": pred["insight"]
                    })
        
        # Extract segment insights
        if "segments" in insights and insights["segments"]:
            for segment in insights["segments"][:2]:  # Top 2 segments
                if "insight" in segment:
                    key_findings.append({
                        "type": "segment",
                        "finding": segment["insight"]
                    })
        
        return key_findings
    
    def _create_summary(self, report):
        """Create a summary of the analysis results."""
        summary = []
        
        # Add objective
        summary.append(f"Analysis Objective: {report['analysis_objective']}")
        summary.append("")
        
        # Add key findings
        if "key_findings" in report and report["key_findings"]:
            summary.append("Key Findings:")
            for finding in report["key_findings"]:
                summary.append(f"- {finding['finding']}")
            summary.append("")
        
        # Add recommendations
        if "recommendations" in report and report["recommendations"]:
            summary.append("Recommendations:")
            for rec in report["recommendations"]:
                if "recommendation" in rec and "priority" in rec:
                    summary.append(f"- [{rec['priority']}] {rec['recommendation']}")
            summary.append("")
        
        # Add file paths
        if "file_paths" in report:
            summary.append("Output Files:")
            for name, path in report["file_paths"].items():
                summary.append(f"- {name}: {path}")
        
        return "\n".join(summary)

def run_data_analysis(args, config=None):
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
    
    # Create config if not provided
    if config is None:
        config = {
            "output_dir": getattr(args, "output_dir", "outputs"),
            "collect_metrics": True
        }
    
    # Create orchestrator with config
    orchestrator = SimplifiedOrchestrator(config)
    
    # Run the analysis
    result = orchestrator.run(context)
    
    return result

# For command-line usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Analysis System")
    parser.add_argument("--data-source", required=True, help="Path to data file")
    parser.add_argument("--source-type", default="file", choices=["file", "api", "url"], help="Source type")
    parser.add_argument("--objective", default="Identify key patterns and insights", help="Analysis objective")
    parser.add_argument("--business-context", help="Business context")
    parser.add_argument("--target-column", help="Target column for predictions")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--metrics-dir", default="metrics", help="Metrics directory")
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics collection")
    
    args = parser.parse_args()
    
    config = {
        "output_dir": args.output_dir,
        "collect_metrics": not args.no_metrics,
        "metrics_dir": args.metrics_dir
    }
    
    result = run_data_analysis(args, config)
    
    if result["status"] == "success":
        print("\nAnalysis completed successfully!")
        if "summary" in result:
            print("\n" + result["summary"])
        
        if "metrics_summary" in result:
            print("\nPerformance Metrics Summary:")
            metrics = result["metrics_summary"]
            
            if "efficiency" in metrics:
                efficiency = metrics["efficiency"]
                if "avg_runtime" in efficiency:
                    print(f"Total runtime: {efficiency['avg_runtime']:.2f} seconds")
                
                for step, runtime in efficiency.items():
                    if step.startswith("avg_") and step != "avg_runtime" and step != "avg_memory_usage" and step != "avg_cpu_usage":
                        print(f"  {step.replace('avg_', '').replace('_runtime', '')}: {runtime:.2f} seconds")
                
                if "avg_memory_usage" in efficiency:
                    print(f"Memory usage: {efficiency['avg_memory_usage']:.2f} MB")
                
                if "avg_cpu_usage" in efficiency:
                    print(f"CPU usage: {efficiency['avg_cpu_usage']:.2f}%")
            
            if "system" in metrics:
                system = metrics["system"]
                if "avg_insight_count" in system:
                    print(f"Insights generated: {system['avg_insight_count']:.0f}")
                
                if "avg_recommendation_count" in system:
                    print(f"Recommendations provided: {system['avg_recommendation_count']:.0f}")
    else:
        print(f"\nAnalysis failed: {result['error_message']}")