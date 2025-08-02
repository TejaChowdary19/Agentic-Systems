# data_analysis_system/utils/metrics_collector.py

import time
import json
import os
import numpy as np
import pandas as pd
import psutil
import traceback
from datetime import datetime

class MetricsCollector:
    """Collects and analyzes performance metrics for the data analysis system."""
    
    def __init__(self, output_dir="metrics"):
        """Initialize the metrics collector."""
        self.metrics = {
            # Accuracy metrics
            "accuracy": {
                "correlation_significance": [],  # p-values for correlations
                "regression_r_squared": [],      # R² values for regression models
                "outlier_detection": [],         # Precision/recall for outlier detection
                "insight_relevance": []          # User ratings of insight relevance
            },
            
            # Efficiency metrics
            "efficiency": {
                "total_runtime": [],             # Total execution time
                "step_runtimes": {},             # Runtime for each processing step
                "memory_usage": [],              # Peak memory usage
                "cpu_usage": []                  # Average CPU usage
            },
            
            # Reliability metrics
            "reliability": {
                "error_rate": [],                # Percentage of runs with errors
                "error_types": {},               # Types of errors encountered
                "recovery_rate": [],             # Percentage of errors recovered from
                "consistency": []                # Variance in results across runs
            },
            
            # System metrics
            "system": {
                "dataset_size": [],              # Size of processed datasets
                "feature_count": [],             # Number of features/columns
                "insight_count": [],             # Number of insights generated
                "recommendation_count": []       # Number of recommendations
            }
        }
        
        # Create metrics directory if it doesn't exist
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize run ID
        self.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize timers
        self.start_time = None
        self.step_start_times = {}
        
        # Initialize memory tracking
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def start_run(self):
        """Start timing a new run."""
        self.start_time = time.time()
        self.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Reset peak memory usage
        self.peak_memory = self.initial_memory
        
        # Start CPU monitoring
        self.cpu_readings = []
        
        return self.current_run_id
    
    def start_step(self, step_name):
        """Start timing a step within the current run."""
        self.step_start_times[step_name] = time.time()
        
        # Initialize step runtime dict if not exists
        if step_name not in self.metrics["efficiency"]["step_runtimes"]:
            self.metrics["efficiency"]["step_runtimes"][step_name] = []
        
        # Track memory at step start
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
        # Track CPU
        self.cpu_readings.append(psutil.cpu_percent(interval=0.1))
        
        return step_name
    
    def end_step(self, step_name):
        """End timing for a step and record the runtime."""
        if step_name in self.step_start_times:
            step_runtime = time.time() - self.step_start_times[step_name]
            self.metrics["efficiency"]["step_runtimes"][step_name].append(step_runtime)
            
            # Track memory at step end
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, current_memory)
            
            # Track CPU
            self.cpu_readings.append(psutil.cpu_percent(interval=0.1))
            
            return step_runtime
        
        return None
    
    def end_run(self):
        """End timing for the current run and record total runtime."""
        if self.start_time:
            total_runtime = time.time() - self.start_time
            self.metrics["efficiency"]["total_runtime"].append(total_runtime)
            
            # Record memory usage
            memory_used = self.peak_memory - self.initial_memory
            self.metrics["efficiency"]["memory_usage"].append(memory_used)
            
            # Record CPU usage
            if self.cpu_readings:
                avg_cpu = sum(self.cpu_readings) / len(self.cpu_readings)
                self.metrics["efficiency"]["cpu_usage"].append(avg_cpu)
            
            return total_runtime
        
        return None
    
    def record_dataset_metrics(self, df):
        """Record metrics about the dataset."""
        if isinstance(df, pd.DataFrame):
            # Record dataset size
            self.metrics["system"]["dataset_size"].append(df.shape[0])
            
            # Record feature count
            self.metrics["system"]["feature_count"].append(df.shape[1])
            
            return {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            }
        
        return None
    
    def record_correlation_metrics(self, corr_matrix, p_values=None):
        """Record metrics about correlation analysis."""
        if isinstance(corr_matrix, pd.DataFrame):
            # Extract correlation values (excluding diagonal)
            corr_values = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_values.append(abs(corr_matrix.loc[col1, col2]))
            
            # Record significance if p-values provided
            if p_values is not None and isinstance(p_values, pd.DataFrame):
                significant_correlations = 0
                total_correlations = 0
                
                for i in range(len(p_values.columns)):
                    for j in range(i+1, len(p_values.columns)):
                        col1, col2 = p_values.columns[i], p_values.columns[j]
                        if p_values.loc[col1, col2] < 0.05:  # Significance threshold
                            significant_correlations += 1
                        total_correlations += 1
                
                if total_correlations > 0:
                    significance_ratio = significant_correlations / total_correlations
                    self.metrics["accuracy"]["correlation_significance"].append(significance_ratio)
            
            return {
                "average_correlation": np.mean(corr_values) if corr_values else 0,
                "max_correlation": np.max(corr_values) if corr_values else 0,
                "correlation_count": len(corr_values)
            }
        
        return None
    
    def record_regression_metrics(self, regression_results):
        """Record metrics about regression analysis."""
        if isinstance(regression_results, dict):
            r_squared_values = []
            
            for feature, stats in regression_results.items():
                if "r_squared" in stats:
                    r_squared_values.append(stats["r_squared"])
            
            if r_squared_values:
                self.metrics["accuracy"]["regression_r_squared"].extend(r_squared_values)
                
                return {
                    "average_r_squared": np.mean(r_squared_values),
                    "max_r_squared": np.max(r_squared_values),
                    "regression_count": len(r_squared_values)
                }
        
        return None
    
    def record_insight_metrics(self, insights):
        """Record metrics about generated insights."""
        if isinstance(insights, dict):
            insight_count = 0
            insight_types = {}
            
            # Count insights by type
            for insight_type, insights_list in insights.items():
                if isinstance(insights_list, list):
                    count = len(insights_list)
                    insight_count += count
                    insight_types[insight_type] = count
            
            self.metrics["system"]["insight_count"].append(insight_count)
            
            return {
                "total_insights": insight_count,
                "insight_types": insight_types
            }
        
        return None
    
    def record_recommendation_metrics(self, recommendations):
        """Record metrics about generated recommendations."""
        if isinstance(recommendations, list):
            self.metrics["system"]["recommendation_count"].append(len(recommendations))
            
            # Count recommendations by priority
            priority_counts = {"high": 0, "medium": 0, "low": 0}
            
            for rec in recommendations:
                if isinstance(rec, dict) and "priority" in rec:
                    priority = rec["priority"].lower()
                    if priority in priority_counts:
                        priority_counts[priority] += 1
            
            return {
                "total_recommendations": len(recommendations),
                "by_priority": priority_counts
            }
        
        return None
    
    def record_error(self, error, recovered=False):
        """Record an error that occurred during processing."""
        error_type = type(error).__name__
        
        # Increment error count for this type
        if error_type not in self.metrics["reliability"]["error_types"]:
            self.metrics["reliability"]["error_types"][error_type] = 0
        
        self.metrics["reliability"]["error_types"][error_type] += 1
        
        # Update recovery rate
        self.metrics["reliability"]["recovery_rate"].append(1 if recovered else 0)
        
        # Update error rate (assuming this is called on every run)
        self.metrics["reliability"]["error_rate"].append(1)  # 1 = error occurred
        
        return {
            "error_type": error_type,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "recovered": recovered
        }
    
    def record_success(self):
        """Record a successful run without errors."""
        # Update error rate (0 = no error)
        self.metrics["reliability"]["error_rate"].append(0)
    
    def record_consistency(self, results1, results2):
        """Record consistency between two result sets."""
        # Simple consistency metric: percentage of matching insights
        if (isinstance(results1, dict) and isinstance(results2, dict) and
            "insights" in results1 and "insights" in results2):
            
            insights1 = self._flatten_insights(results1["insights"])
            insights2 = self._flatten_insights(results2["insights"])
            
            # Calculate Jaccard similarity
            if insights1 or insights2:  # Avoid division by zero
                intersection = len(set(insights1) & set(insights2))
                union = len(set(insights1) | set(insights2))
                similarity = intersection / union
                
                self.metrics["reliability"]["consistency"].append(similarity)
                
                return {
                    "similarity": similarity,
                    "insights1_count": len(insights1),
                    "insights2_count": len(insights2),
                    "common_insights": intersection
                }
        
        return None
    
    def _flatten_insights(self, insights_dict):
        """Convert nested insights dict to a flat list of insight strings."""
        flattened = []
        
        for insight_type, insights_list in insights_dict.items():
            if isinstance(insights_list, list):
                for insight in insights_list:
                    if isinstance(insight, dict) and "insight" in insight:
                        flattened.append(insight["insight"])
        
        return flattened
    
    def record_user_feedback(self, insight_id, relevance_score):
        """Record user feedback on insight relevance."""
        # Relevance score should be 1-5 (5 being most relevant)
        if 1 <= relevance_score <= 5:
            self.metrics["accuracy"]["insight_relevance"].append(relevance_score)
            
            return {
                "insight_id": insight_id,
                "relevance_score": relevance_score
            }
        
        return None
    
    def get_metrics_summary(self):
        """Generate a summary of collected metrics."""
        summary = {
            "accuracy": {},
            "efficiency": {},
            "reliability": {},
            "system": {}
        }
        
        # Accuracy metrics
        if self.metrics["accuracy"]["correlation_significance"]:
            summary["accuracy"]["correlation_significance"] = np.mean(self.metrics["accuracy"]["correlation_significance"])
        
        if self.metrics["accuracy"]["regression_r_squared"]:
            summary["accuracy"]["avg_r_squared"] = np.mean(self.metrics["accuracy"]["regression_r_squared"])
        
        if self.metrics["accuracy"]["insight_relevance"]:
            summary["accuracy"]["avg_insight_relevance"] = np.mean(self.metrics["accuracy"]["insight_relevance"])
        
        # Efficiency metrics
        if self.metrics["efficiency"]["total_runtime"]:
            summary["efficiency"]["avg_runtime"] = np.mean(self.metrics["efficiency"]["total_runtime"])
        
        for step, runtimes in self.metrics["efficiency"]["step_runtimes"].items():
            if runtimes:
                summary["efficiency"][f"avg_{step}_runtime"] = np.mean(runtimes)
        
        if self.metrics["efficiency"]["memory_usage"]:
            summary["efficiency"]["avg_memory_usage"] = np.mean(self.metrics["efficiency"]["memory_usage"])
        
        if self.metrics["efficiency"]["cpu_usage"]:
            summary["efficiency"]["avg_cpu_usage"] = np.mean(self.metrics["efficiency"]["cpu_usage"])
        
        # Reliability metrics
        if self.metrics["reliability"]["error_rate"]:
            summary["reliability"]["error_rate"] = np.mean(self.metrics["reliability"]["error_rate"])
        
        if self.metrics["reliability"]["recovery_rate"]:
            summary["reliability"]["recovery_rate"] = np.mean(self.metrics["reliability"]["recovery_rate"])
        
        if self.metrics["reliability"]["consistency"]:
            summary["reliability"]["avg_consistency"] = np.mean(self.metrics["reliability"]["consistency"])
        
        summary["reliability"]["error_types"] = self.metrics["reliability"]["error_types"]
        
        # System metrics
        if self.metrics["system"]["dataset_size"]:
            summary["system"]["avg_dataset_size"] = np.mean(self.metrics["system"]["dataset_size"])
        
        if self.metrics["system"]["feature_count"]:
            summary["system"]["avg_feature_count"] = np.mean(self.metrics["system"]["feature_count"])
        
        if self.metrics["system"]["insight_count"]:
            summary["system"]["avg_insight_count"] = np.mean(self.metrics["system"]["insight_count"])
        
        if self.metrics["system"]["recommendation_count"]:
            summary["system"]["avg_recommendation_count"] = np.mean(self.metrics["system"]["recommendation_count"])
        
        return summary
    
    def save_metrics(self, filename=None):
        """Save collected metrics to a JSON file."""
        if filename is None:
            filename = f"metrics_{self.current_run_id}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                "metrics": self.metrics,
                "summary": self.get_metrics_summary(),
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        return filepath
    
    def load_metrics(self, filepath):
        """Load metrics from a JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                if "metrics" in data:
                    self.metrics = data["metrics"]
                    return True
        
        return False