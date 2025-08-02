# evaluation_metrics.py
# A script to evaluate the accuracy of your Data Analysis Agentic System 

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from data_analysis_system.simplified_main import SimplifiedOrchestrator

class MetricsEvaluator:
    """Class to evaluate the performance metrics of the Data Analysis Agentic System."""
    
    def __init__(self, output_dir="evaluation_results"):
        """Initialize the evaluator with output directory."""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create a subdirectory for synthetic datasets
        self.data_dir = os.path.join(output_dir, "datasets")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Initialize results dictionary
        self.results = {
            "correlation_identification": {},
            "trend_detection": {},
            "outlier_detection": {},
            "segment_difference_detection": {},
            "overall": {}
        }
        
        # Initialize the orchestrator
        self.orchestrator = SimplifiedOrchestrator({"output_dir": output_dir})
    
    def generate_correlation_test_data(self, n=100, noise_level=0.3):
        """
        Generate dataset with known correlations.
        
        Parameters:
        - n: Number of samples
        - noise_level: Amount of noise to add (0-1)
        
        Returns:
        - Path to the generated dataset
        - Dictionary of ground truth correlations
        """
        np.random.seed(42)
        
        # Create independent variables
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # Create dependent variables with known correlations
        # Strong positive correlation with x1
        y1 = 0.8 * x1 + noise_level * np.random.normal(0, 1, n)
        
        # Moderate negative correlation with x2
        y2 = -0.6 * x2 + noise_level * np.random.normal(0, 1, n)
        
        # Weak correlation with both x1 and x2
        y3 = 0.3 * x1 - 0.2 * x2 + noise_level * np.random.normal(0, 1, n)
        
        # No correlation with x1 or x2
        y4 = np.random.normal(0, 1, n)
        
        # Create dataframe
        df = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'strong_positive_y1': y1,
            'moderate_negative_y2': y2,
            'weak_mixed_y3': y3,
            'uncorrelated_y4': y4
        })
        
        # Calculate actual correlations for ground truth
        corr_matrix = df.corr()
        
        # Define expected significant correlations
        ground_truth = {
            ('x1', 'strong_positive_y1'): 'strong positive',
            ('x2', 'moderate_negative_y2'): 'moderate negative',
            ('x1', 'weak_mixed_y3'): 'weak positive',
            ('x2', 'weak_mixed_y3'): 'weak negative'
        }
        
        # Save the dataset
        file_path = os.path.join(self.data_dir, "correlation_test_data.csv")
        df.to_csv(file_path, index=False)
        
        return file_path, ground_truth
    
    def generate_trend_test_data(self, n=100, noise_level=0.3):
        """
        Generate dataset with known trends over time.
        
        Parameters:
        - n: Number of samples
        - noise_level: Amount of noise to add (0-1)
        
        Returns:
        - Path to the generated dataset
        - Dictionary of ground truth trends
        """
        np.random.seed(43)
        
        # Create date range
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
        
        # Create trend variables
        t = np.linspace(0, 1, n)  # Time variable from 0 to 1
        
        # Strong increasing trend
        increasing = 10 + 5 * t + noise_level * np.random.normal(0, 1, n)
        
        # Strong decreasing trend
        decreasing = 15 - 8 * t + noise_level * np.random.normal(0, 1, n)
        
        # No trend (flat)
        flat = 12 + noise_level * np.random.normal(0, 1, n)
        
        # Non-linear trend (U-shaped)
        nonlinear = 10 + 10 * (t - 0.5)**2 + noise_level * np.random.normal(0, 1, n)
        
        # Seasonal trend
        seasonal = 12 + 3 * np.sin(t * 4 * np.pi) + noise_level * np.random.normal(0, 1, n)
        
        # Create dataframe
        df = pd.DataFrame({
            'date': dates,
            'increasing_trend': increasing,
            'decreasing_trend': decreasing,
            'flat_no_trend': flat,
            'nonlinear_trend': nonlinear,
            'seasonal_trend': seasonal
        })
        
        # Define ground truth trends
        ground_truth = {
            'increasing_trend': 'increasing',
            'decreasing_trend': 'decreasing',
            'flat_no_trend': 'no trend',
            'nonlinear_trend': 'nonlinear',
            'seasonal_trend': 'seasonal'
        }
        
        # Save the dataset
        file_path = os.path.join(self.data_dir, "trend_test_data.csv")
        df.to_csv(file_path, index=False)
        
        return file_path, ground_truth
    
    def generate_outlier_test_data(self, n=100, outlier_percent=0.05):
        """
        Generate dataset with known outliers.
        
        Parameters:
        - n: Number of samples
        - outlier_percent: Percentage of data points that are outliers
        
        Returns:
        - Path to the generated dataset
        - Dictionary mapping row indices to outlier status
        """
        np.random.seed(44)
        
        # Calculate number of outliers
        n_outliers = int(n * outlier_percent)
        
        # Generate normal data
        x1 = np.random.normal(10, 2, n)
        x2 = np.random.normal(50, 5, n)
        
        # Track ground truth outliers
        outlier_indices = {}
        
        # Insert outliers in x1
        for i in range(n_outliers):
            idx = np.random.randint(0, n)
            # Make it an outlier (±4-6 std from mean)
            direction = 1 if np.random.random() > 0.5 else -1
            x1[idx] = 10 + direction * (8 + 4 * np.random.random())
            outlier_indices[idx] = 'x1'
        
        # Insert outliers in x2
        for i in range(n_outliers):
            idx = np.random.randint(0, n)
            # Make it an outlier (±4-6 std from mean)
            direction = 1 if np.random.random() > 0.5 else -1
            x2[idx] = 50 + direction * (20 + 10 * np.random.random())
            outlier_indices[idx] = outlier_indices.get(idx, '') + (' and ' if idx in outlier_indices else '') + 'x2'
        
        # Create dataframe
        df = pd.DataFrame({
            'id': range(n),
            'x1': x1,
            'x2': x2,
            'normal_col': np.random.normal(0, 1, n)  # Control column without outliers
        })
        
        # Save the dataset
        file_path = os.path.join(self.data_dir, "outlier_test_data.csv")
        df.to_csv(file_path, index=False)
        
        # Convert outlier_indices to a format that's easier to compare with results
        ground_truth = {
            'x1_outliers': [idx for idx, col in outlier_indices.items() if 'x1' in col],
            'x2_outliers': [idx for idx, col in outlier_indices.items() if 'x2' in col],
            'total_outliers': len(outlier_indices)
        }
        
        return file_path, ground_truth
    
    def generate_segment_test_data(self, n=100):
        """
        Generate dataset with known segment differences.
        
        Returns:
        - Path to the generated dataset
        - Dictionary of ground truth segment differences
        """
        np.random.seed(45)
        
        # Create segment categories
        segments = ['A', 'B', 'C']
        segment_column = np.random.choice(segments, n)
        
        # Create variables with known segment differences
        # Variable with large segment differences
        large_diff = np.zeros(n)
        for i, segment in enumerate(segment_column):
            if segment == 'A':
                large_diff[i] = np.random.normal(100, 10)
            elif segment == 'B':
                large_diff[i] = np.random.normal(50, 10)
            else:  # C
                large_diff[i] = np.random.normal(75, 10)
        
        # Variable with moderate segment differences
        medium_diff = np.zeros(n)
        for i, segment in enumerate(segment_column):
            if segment == 'A':
                medium_diff[i] = np.random.normal(20, 5)
            elif segment == 'B':
                medium_diff[i] = np.random.normal(15, 5)
            else:  # C
                medium_diff[i] = np.random.normal(25, 5)
        
        # Variable with no segment differences
        no_diff = np.random.normal(30, 5, n)
        
        # Create another categorical column
        category2 = np.random.choice(['X', 'Y'], n)
        
        # Create dataframe
        df = pd.DataFrame({
            'segment': segment_column,
            'category': category2,
            'large_segment_diff': large_diff,
            'medium_segment_diff': medium_diff,
            'no_segment_diff': no_diff
        })
        
        # Define ground truth segment differences
        ground_truth = {
            'large_segment_diff': {
                'segment_column': 'segment',
                'largest_diff': ('A', 'B'),
                'diff_percentage': 100  # Approx (100-50)/50 * 100
            },
            'medium_segment_diff': {
                'segment_column': 'segment',
                'largest_diff': ('C', 'B'),
                'diff_percentage': 67  # Approx (25-15)/15 * 100
            }
        }
        
        # Save the dataset
        file_path = os.path.join(self.data_dir, "segment_test_data.csv")
        df.to_csv(file_path, index=False)
        
        return file_path, ground_truth
    
    def evaluate_correlation_identification(self):
        """
        Evaluate the system's ability to identify correlations.
        
        Returns:
        - Dictionary with evaluation results
        """
        print("Evaluating correlation identification...")
        
        # Generate test data
        file_path, ground_truth = self.generate_correlation_test_data()
        
        # Run the system
        context = {
            "data_source": {
                "source_type": "file",
                "source_path": file_path
            },
            "analysis_objective": "Identify correlations between variables",
            "business_context": None
        }
        
        result = self.orchestrator.run(context)
        
        # Check if the analysis was successful
        if result["status"] != "success":
            print(f"Analysis failed: {result.get('error_message', 'Unknown error')}")
            return {"accuracy": 0, "error": result.get('error_message', 'Unknown error')}
        
        # Load the insights
        try:
            insights_path = result.get("file_paths", {}).get("insights")
            if not insights_path or not os.path.exists(insights_path):
                print("No insights file found")
                return {"accuracy": 0, "error": "No insights file found"}
            
            with open(insights_path, 'r') as f:
                insights_data = json.load(f)
            
            # Extract correlations from insights
            if "insights" not in insights_data or "correlations" not in insights_data["insights"]:
                print("No correlation insights found")
                return {"accuracy": 0, "error": "No correlation insights found"}
            
            correlations = insights_data["insights"]["correlations"]
            
            # Count correct identifications
            correct = 0
            total = len(ground_truth)
            found_correlations = []
            
            for corr in correlations:
                # Check if correlation pair matches any ground truth pair
                col1, col2 = corr["column1"], corr["column2"]
                corr_value = corr["correlation"]
                strength = corr["strength"]
                relationship = corr["relationship"]
                
                # Try both orders of columns
                pair1 = (col1, col2)
                pair2 = (col2, col1)
                
                if pair1 in ground_truth or pair2 in ground_truth:
                    # Get the expected relationship
                    expected = ground_truth.get(pair1) or ground_truth.get(pair2)
                    actual = f"{strength} {relationship}"
                    
                    # Check if strength and direction match expected
                    if expected.lower() in actual.lower():
                        correct += 1
                        found_correlations.append(pair1 if pair1 in ground_truth else pair2)
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            
            # Find missing correlations
            missing = [pair for pair in ground_truth if pair not in found_correlations]
            
            # Prepare results
            evaluation_result = {
                "accuracy": accuracy,
                "correct_identifications": correct,
                "total_ground_truth": total,
                "found_correlations": found_correlations,
                "missing_correlations": missing
            }
            
            self.results["correlation_identification"] = evaluation_result
            return evaluation_result
            
        except Exception as e:
            print(f"Error evaluating correlation identification: {str(e)}")
            return {"accuracy": 0, "error": str(e)}
    
    def evaluate_trend_detection(self):
        """
        Evaluate the system's ability to detect trends.
        
        Returns:
        - Dictionary with evaluation results
        """
        print("Evaluating trend detection...")
        
        # Generate test data
        file_path, ground_truth = self.generate_trend_test_data()
        
        # Run the system
        context = {
            "data_source": {
                "source_type": "file",
                "source_path": file_path
            },
            "analysis_objective": "Identify trends in time series data",
            "business_context": None
        }
        
        result = self.orchestrator.run(context)
        
        # Check if the analysis was successful
        if result["status"] != "success":
            print(f"Analysis failed: {result.get('error_message', 'Unknown error')}")
            return {"accuracy": 0, "error": result.get('error_message', 'Unknown error')}
        
        # Load the insights
        try:
            insights_path = result.get("file_paths", {}).get("insights")
            if not insights_path or not os.path.exists(insights_path):
                print("No insights file found")
                return {"accuracy": 0, "error": "No insights file found"}
            
            with open(insights_path, 'r') as f:
                insights_data = json.load(f)
            
            # Extract trends from insights
            if "insights" not in insights_data or "trends" not in insights_data["insights"]:
                print("No trend insights found")
                return {"accuracy": 0, "error": "No trend insights found"}
            
            trends = insights_data["insights"]["trends"]
            
            # Count correct identifications
            correct = 0
            total = len([k for k, v in ground_truth.items() if v in ['increasing', 'decreasing']])
            found_trends = []
            
            for trend in trends:
                column = trend["column"]
                direction = trend["direction"]
                
                if column in ground_truth:
                    expected = ground_truth[column]
                    
                    # For simplicity, we only check increasing/decreasing trends
                    if expected in ['increasing', 'decreasing']:
                        if expected == direction:
                            correct += 1
                            found_trends.append(column)
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            
            # Find missing trends
            missing = [col for col, dir in ground_truth.items() 
                      if dir in ['increasing', 'decreasing'] and col not in found_trends]
            
            # Prepare results
            evaluation_result = {
                "accuracy": accuracy,
                "correct_identifications": correct,
                "total_ground_truth": total,
                "found_trends": found_trends,
                "missing_trends": missing
            }
            
            self.results["trend_detection"] = evaluation_result
            return evaluation_result
            
        except Exception as e:
            print(f"Error evaluating trend detection: {str(e)}")
            return {"accuracy": 0, "error": str(e)}
    
    def evaluate_outlier_detection(self):
        """
        Evaluate the system's ability to detect outliers.
        
        Returns:
        - Dictionary with evaluation results
        """
        print("Evaluating outlier detection...")
        
        # Generate test data
        file_path, ground_truth = self.generate_outlier_test_data()
        
        # Run the system
        context = {
            "data_source": {
                "source_type": "file",
                "source_path": file_path
            },
            "analysis_objective": "Identify outliers in the data",
            "business_context": None
        }
        
        result = self.orchestrator.run(context)
        
        # Check if the analysis was successful
        if result["status"] != "success":
            print(f"Analysis failed: {result.get('error_message', 'Unknown error')}")
            return {"accuracy": 0, "error": result.get('error_message', 'Unknown error')}
        
        # Load the insights
        try:
            insights_path = result.get("file_paths", {}).get("insights")
            if not insights_path or not os.path.exists(insights_path):
                print("No insights file found")
                return {"accuracy": 0, "error": "No insights file found"}
            
            with open(insights_path, 'r') as f:
                insights_data = json.load(f)
            
            # Extract outliers from insights
            if "insights" not in insights_data or "outliers" not in insights_data["insights"]:
                print("No outlier insights found")
                return {"accuracy": 0, "error": "No outlier insights found"}
            
            outliers = insights_data["insights"]["outliers"]
            
            # Count detected outliers
            detected_outliers = []
            for outlier in outliers:
                column = outlier["column"]
                if column in ['x1', 'x2']:
                    detected_outliers.append((outlier.get("context", {}).get("id"), column))
            
            # Count correct identifications
            true_positives = 0
            
            # Check x1 outliers
            for idx in ground_truth['x1_outliers']:
                if (idx, 'x1') in detected_outliers:
                    true_positives += 1
            
            # Check x2 outliers
            for idx in ground_truth['x2_outliers']:
                if (idx, 'x2') in detected_outliers:
                    true_positives += 1
            
            total_true_outliers = len(ground_truth['x1_outliers']) + len(ground_truth['x2_outliers'])
            total_detected = len(detected_outliers)
            
            # Calculate precision and recall
            precision = true_positives / total_detected if total_detected > 0 else 0
            recall = true_positives / total_true_outliers if total_true_outliers > 0 else 0
            
            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Prepare results
            evaluation_result = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": true_positives,
                "total_true_outliers": total_true_outliers,
                "total_detected": total_detected
            }
            
            self.results["outlier_detection"] = evaluation_result
            return evaluation_result
            
        except Exception as e:
            print(f"Error evaluating outlier detection: {str(e)}")
            return {"accuracy": 0, "error": str(e)}
    
    def evaluate_segment_difference_detection(self):
        """
        Evaluate the system's ability to detect segment differences.
        
        Returns:
        - Dictionary with evaluation results
        """
        print("Evaluating segment difference detection...")
        
        # Generate test data
        file_path, ground_truth = self.generate_segment_test_data()
        
        # Run the system
        context = {
            "data_source": {
                "source_type": "file",
                "source_path": file_path
            },
            "analysis_objective": "Identify differences between segments",
            "business_context": None
        }
        
        result = self.orchestrator.run(context)
        
        # Check if the analysis was successful
        if result["status"] != "success":
            print(f"Analysis failed: {result.get('error_message', 'Unknown error')}")
            return {"accuracy": 0, "error": result.get('error_message', 'Unknown error')}
        
        # Load the insights
        try:
            insights_path = result.get("file_paths", {}).get("insights")
            if not insights_path or not os.path.exists(insights_path):
                print("No insights file found")
                return {"accuracy": 0, "error": "No insights file found"}
            
            with open(insights_path, 'r') as f:
                insights_data = json.load(f)
            
            # Extract segments from insights
            if "insights" not in insights_data or "segments" not in insights_data["insights"]:
                print("No segment insights found")
                return {"accuracy": 0, "error": "No segment insights found"}
            
            segments = insights_data["insights"]["segments"]
            
            # Count correct identifications
            correct = 0
            total = len(ground_truth)
            found_segments = []
            
            for segment in segments:
                value_column = segment["value_column"]
                segment_column = segment["segment_column"]
                top_segment = segment["top_segment"]
                bottom_segment = segment["bottom_segment"]
                
                if value_column in ground_truth:
                    truth = ground_truth[value_column]
                    
                    # Check if segment column is correct
                    if truth['segment_column'] == segment_column:
                        # Check if the largest difference pair is identified correctly
                        # (allowing for either order)
                        expected_pair = truth['largest_diff']
                        actual_pair = (top_segment, bottom_segment)
                        
                        if (expected_pair == actual_pair or 
                            (expected_pair[1], expected_pair[0]) == actual_pair):
                            correct += 1
                            found_segments.append(value_column)
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            
            # Find missing segments
            missing = [col for col in ground_truth if col not in found_segments]
            
            # Prepare results
            evaluation_result = {
                "accuracy": accuracy,
                "correct_identifications": correct,
                "total_ground_truth": total,
                "found_segments": found_segments,
                "missing_segments": missing
            }
            
            self.results["segment_difference_detection"] = evaluation_result
            return evaluation_result
            
        except Exception as e:
            print(f"Error evaluating segment difference detection: {str(e)}")
            return {"accuracy": 0, "error": str(e)}
    
    def run_all_evaluations(self):
        """Run all evaluation metrics and calculate overall performance."""
        print("Starting comprehensive system evaluation...")
        
        # Run each evaluation
        correlation_results = self.evaluate_correlation_identification()
        trend_results = self.evaluate_trend_detection()
        outlier_results = self.evaluate_outlier_detection()
        segment_results = self.evaluate_segment_difference_detection()
        
        # Calculate overall accuracy
        metrics = [
            correlation_results.get("accuracy", 0),
            trend_results.get("accuracy", 0),
            # For outliers, use F1 score
            outlier_results.get("f1_score", 0),
            segment_results.get("accuracy", 0)
        ]
        
        overall_accuracy = sum(metrics) / len(metrics)
        
        # Create overall summary
        self.results["overall"] = {
            "overall_accuracy": overall_accuracy,
            "correlation_identification_accuracy": correlation_results.get("accuracy", 0),
            "trend_detection_accuracy": trend_results.get("accuracy", 0),
            "outlier_detection_f1": outlier_results.get("f1_score", 0),
            "segment_difference_detection_accuracy": segment_results.get("accuracy", 0)
        }
        
        # Save results to file
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Evaluation complete. Results saved to {results_path}")
        
        # Generate visualizations
        self.generate_evaluation_visualizations()
        
        return self.results
    
    def generate_evaluation_visualizations(self):
        """Generate visualizations of the evaluation results."""
        # Set up the figure
        plt.figure(figsize=(12, 10))
        
        # Plot accuracy metrics
        metrics = [
            self.results["correlation_identification"].get("accuracy", 0),
            self.results["trend_detection"].get("accuracy", 0),
            self.results["outlier_detection"].get("f1_score", 0),
            self.results["segment_difference_detection"].get("accuracy", 0)
        ]
        
        labels = [
            "Correlation\nIdentification",
            "Trend\nDetection",
            "Outlier\nDetection (F1)",
            "Segment Difference\nDetection"
        ]
        
        # Create bar chart
        plt.subplot(2, 1, 1)
        bars = plt.bar(labels, metrics, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.title('System Performance Metrics', fontsize=16)
        plt.ylabel('Accuracy / F1 Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Create radar chart
        plt.subplot(2, 1, 2, polar=True)
        
        # Repeat the first value to close the polygon
        values = metrics + [metrics[0]]
        angles = np.linspace(0, 2*np.pi, len(labels) + 1, endpoint=True)
        
        plt.polar(angles, values, 'o-', linewidth=2)
        plt.fill(angles, values, alpha=0.25)
        
        # Set the labels
        plt.xticks(angles[:-1], labels)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], color='gray')
        plt.ylim(0, 1)
        
        plt.title('Radar Chart of System Performance', y=1.1, fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "evaluation_metrics.png"))
        plt.close()
        
        print(f"Evaluation visualizations saved to {self.output_dir}/evaluation_metrics.png")

# If run directly, perform the evaluation
if __name__ == "__main__":
    evaluator = MetricsEvaluator()
    results = evaluator.run_all_evaluations()
    
    # Print summary of results
    print("\nEvaluation Summary:")
    print(f"Overall Accuracy: {results['overall']['overall_accuracy']:.2f}")
    print(f"Correlation Identification: {results['overall']['correlation_identification_accuracy']:.2f}")
    print(f"Trend Detection: {results['overall']['trend_detection_accuracy']:.2f}")
    print(f"Outlier Detection (F1): {results['overall']['outlier_detection_f1']:.2f}")
    print(f"Segment Difference Detection: {results['overall']['segment_difference_detection_accuracy']:.2f}")