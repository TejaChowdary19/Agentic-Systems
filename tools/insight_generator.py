# data_analysis_system/tools/insight_generator.py

import json
import os
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Type, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

class InsightGeneratorInput(BaseModel):
    """Input for the custom insight generator tool."""
    data_path: str = Field(
        description="Path to the data file to analyze"
    )
    analysis_results_path: Optional[str] = Field(
        default=None,
        description="Path to JSON file with previous analysis results"
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Target column for predictive insights"
    )
    insight_types: List[str] = Field(
        description="Types of insights to generate (trends, outliers, correlations, predictions, segments)"
    )
    business_context: Optional[str] = Field(
        default=None,
        description="Business context to frame the insights"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for specific insight types"
    )

class InsightGeneratorTool(BaseTool):
    name = "Custom Insight Generator"
    description = "Automatically identifies patterns, trends, correlations, and actionable insights from data. Generates natural language explanations prioritized by significance."
    args_schema: Type[BaseModel] = InsightGeneratorInput
    
    def _run(self, data_path: str, insight_types: List[str], analysis_results_path: Optional[str] = None,
             target_column: Optional[str] = None, business_context: Optional[str] = None,
             parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate insights from data and analysis results."""
        try:
            # Load the data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                df = pd.read_excel(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Unsupported file format: {data_path}"
                })
            
            # Load previous analysis results if available
            previous_analysis = None
            if analysis_results_path is not None and os.path.exists(analysis_results_path):
                try:
                    with open(analysis_results_path, 'r') as f:
                        previous_analysis = json.load(f)
                except Exception as load_error:
                    return json.dumps({
                        "status": "error",
                        "message": f"Failed to load analysis results: {str(load_error)}",
                        "analysis_path": analysis_results_path
                    })
            
            # Initialize results
            insights = {
                "status": "success", 
                "business_context": business_context, 
                "insights": {},
                "data_summary": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
                }
            }
            
            # Generate insights based on requested types
            for insight_type in insight_types:
                if insight_type.lower() == "trends":
                    # Identify trends in time series data
                    trends = self._identify_trends(df, previous_analysis, parameters)
                    insights["insights"]["trends"] = trends
                
                elif insight_type.lower() == "outliers":
                    # Identify interesting outliers
                    outliers = self._identify_outliers(df, previous_analysis, parameters)
                    insights["insights"]["outliers"] = outliers
                
                elif insight_type.lower() == "correlations":
                    # Identify key correlations and potential causal relationships
                    correlations = self._identify_correlations(df, target_column, previous_analysis, parameters)
                    insights["insights"]["correlations"] = correlations
                
                elif insight_type.lower() == "predictions":
                    # Generate predictive insights
                    if target_column is not None:
                        predictions = self._generate_predictions(df, target_column, previous_analysis, parameters)
                        insights["insights"]["predictions"] = predictions
                    else:
                        insights["insights"]["predictions"] = {
                            "error": "Target column must be specified for predictive insights"
                        }
                
                elif insight_type.lower() == "segments":
                    # Identify interesting segments or clusters
                    segments = self._identify_segments(df, previous_analysis, parameters)
                    insights["insights"]["segments"] = segments
            
            # Generate follow-up recommendations
            insights["recommended_next_steps"] = self._recommend_next_steps(df, insights["insights"], business_context)
            
            # Save insights to file
            insights_path = os.path.splitext(data_path)[0] + "_insights.json"
            with open(insights_path, 'w') as f:
                json.dump(insights, f, indent=2)
            
            # Add file path to results
            insights["insights_path"] = insights_path
            
            return json.dumps(insights)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e),
                "data_path": data_path
            })
    
    def _arun(self, data_path: str, insight_types: List[str], analysis_results_path: Optional[str] = None,
              target_column: Optional[str] = None, business_context: Optional[str] = None,
              parameters: Optional[Dict[str, Any]] = None) -> str:
        """Async implementation of the tool."""
        # For simplicity, we're just calling the sync version
        return self._run(data_path, insight_types, analysis_results_path, target_column, business_context, parameters)
    
    def _identify_trends(self, df, previous_analysis, parameters):
        """Identify trends in the data."""
        trends = []
        
        # Check if there's a date/time column
        date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 
                        ('date' in col.lower() or 'time' in col.lower() or 'day' in col.lower() or 
                         'month' in col.lower() or 'year' in col.lower())]
        
        if date_columns:
            # For each date column, check for trends in numeric columns
            for date_col in date_columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                    except:
                        continue
                
                # Sort by date
                df_sorted = df.sort_values(by=date_col)
                
                # Check numeric columns for trends
                for col in df.select_dtypes(include=np.number).columns:
                    if col == date_col:
                        continue
                    
                    # Simple trend check: compare first third vs last third
                    n = len(df_sorted)
                    if n >= 9:  # Need at least 9 points for three segments
                        first_third = df_sorted[col].iloc[:n//3].mean()
                        last_third = df_sorted[col].iloc[-n//3:].mean()
                        
                        percent_change = ((last_third - first_third) / first_third * 100) if first_third != 0 else 0
                        
                        if abs(percent_change) >= 10:  # 10% change threshold
                            direction = "increasing" if percent_change > 0 else "decreasing"
                            trends.append({
                                "column": col,
                                "date_column": date_col,
                                "direction": direction,
                                "percent_change": round(percent_change, 2),
                                "start_value": round(float(first_third), 2),
                                "end_value": round(float(last_third), 2),
                                "insight": f"{col} is {direction} over time (by {abs(round(percent_change, 2))}%)"
                            })
        
        return trends
    
    def _identify_outliers(self, df, previous_analysis, parameters):
        """Identify interesting outliers in the data."""
        outliers = []
        
        # Check numeric columns for outliers
        for col in df.select_dtypes(include=np.number).columns:
            # Skip columns with too many missing values
            if df[col].isna().sum() > 0.5 * len(df):
                continue
                
            # Use Z-score method to identify outliers
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            z_threshold = 3.0  # Default threshold
            if parameters and 'z_threshold' in parameters:
                z_threshold = parameters['z_threshold']
            
            outlier_indices = df[col].dropna().index[z_scores > z_threshold]
            
            if len(outlier_indices) > 0:
                # Get outlier values
                outlier_values = df.loc[outlier_indices, col].tolist()
                
                # Check if outliers are interesting (significantly different from mean)
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:
                    # Calculate how many standard deviations from the mean
                    deviations = [(val - mean) / std for val in outlier_values]
                    
                    # Get the most extreme outliers (limited to top 3)
                    sorted_indices = np.argsort(np.abs(deviations))[-3:]
                    
                    for idx in sorted_indices:
                        outlier_value = outlier_values[idx]
                        deviation = deviations[idx]
                        
                        # Get the full row of data for context
                        outlier_row = df.loc[outlier_indices[idx]].to_dict()
                        
                        # Convert numpy types to Python native types for JSON serialization
                        for k, v in outlier_row.items():
                            if isinstance(v, (np.integer, np.floating)):
                                outlier_row[k] = float(v)
                            elif pd.isna(v):
                                outlier_row[k] = None
                            else:
                                outlier_row[k] = str(v)
                        
                        # Create insight
                        direction = "above" if deviation > 0 else "below"
                        std_count = abs(round(deviation, 1))
                        
                        outliers.append({
                            "column": col,
                            "value": float(outlier_value),
                            "std_deviations": float(std_count),
                            "direction": direction,
                            "context": outlier_row,
                            "insight": f"Found outlier in {col}: value {round(outlier_value, 2)} is {std_count} standard deviations {direction} the mean"
                        })
        
        return outliers
    
    def _identify_correlations(self, df, target_column, previous_analysis, parameters):
        """Identify key correlations in the data."""
        correlations = []
        
        # Try to use previous correlation analysis if available
        if previous_analysis and "analyses" in previous_analysis and "correlation" in previous_analysis["analyses"]:
            corr_data = previous_analysis["analyses"]["correlation"]
            
            # Use highest correlations
            if "highest_correlations" in corr_data:
                highest_corrs = corr_data["highest_correlations"]
                
                # Convert from dict to list of tuples for easier sorting
                corr_list = []
                for pair, value in highest_corrs.items():
                    col1, col2 = pair.split("__")
                    corr_list.append((col1, col2, value))
                
                # Sort by absolute correlation
                corr_list.sort(key=lambda x: abs(x[2]), reverse=True)
                
                for col1, col2, corr_value in corr_list[:5]:  # Top 5 correlations
                    relationship = "positive" if corr_value > 0 else "negative"
                    strength = "strong" if abs(corr_value) >= 0.7 else "moderate" if abs(corr_value) >= 0.3 else "weak"
                    
                    insight = f"{strength.capitalize()} {relationship} correlation ({round(corr_value, 2)}) between {col1} and {col2}"
                    
                    # Add context if target column is one of the correlated columns
                    if target_column and (col1 == target_column or col2 == target_column):
                        other_col = col2 if col1 == target_column else col1
                        if corr_value > 0:
                            insight += f". As {other_col} increases, {target_column} tends to increase."
                        else:
                            insight += f". As {other_col} increases, {target_column} tends to decrease."
                    
                    correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": round(float(corr_value), 3),
                        "relationship": relationship,
                        "strength": strength,
                        "insight": insight
                    })
        
        # Calculate correlation matrix if previous analysis not available
        else:
            numeric_df = df.select_dtypes(include=np.number)
            if len(numeric_df.columns) >= 2:
                corr_matrix = numeric_df.corr()
                
                # Convert to dictionary for easier manipulation
                corr_dict = corr_matrix.to_dict()
                
                # Find strongest correlations (positive and negative)
                strong_correlations = []
                
                for col1 in corr_dict:
                    for col2 in corr_dict[col1]:
                        if col1 != col2:
                            corr_value = corr_dict[col1][col2]
                            if not np.isnan(corr_value) and abs(corr_value) >= 0.3:  # Threshold for meaningful correlation
                                strong_correlations.append({
                                    "column1": col1,
                                    "column2": col2,
                                    "correlation": round(float(corr_value), 3)
                                })
                
                # Sort by absolute correlation strength
                strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                
                # Take top correlations (limited to prevent overwhelming)
                top_correlations = strong_correlations[:5]
                
                for corr in top_correlations:
                    col1 = corr["column1"]
                    col2 = corr["column2"]
                    corr_value = corr["correlation"]
                    
                    # Create insight
                    relationship = "positive" if corr_value > 0 else "negative"
                    strength = "strong" if abs(corr_value) >= 0.7 else "moderate" if abs(corr_value) >= 0.3 else "weak"
                    
                    insight = f"{strength.capitalize()} {relationship} correlation ({round(corr_value, 2)}) between {col1} and {col2}"
                    
                    # Add context if target column is one of the correlated columns
                    if target_column and (col1 == target_column or col2 == target_column):
                        other_col = col2 if col1 == target_column else col1
                        if corr_value > 0:
                            insight += f". As {other_col} increases, {target_column} tends to increase."
                        else:
                            insight += f". As {other_col} increases, {target_column} tends to decrease."
                    
                    correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": round(float(corr_value), 3),
                        "relationship": relationship,
                        "strength": strength,
                        "insight": insight
                    })
        
        return correlations
    
    def _generate_predictions(self, df, target_column, previous_analysis, parameters):
        """Generate predictive insights."""
        predictions = {
            "target": target_column,
            "insights": []
        }
        
        # Check if target column exists and is numeric
        if target_column not in df.columns:
            predictions["error"] = f"Target column '{target_column}' not found in data"
            return predictions
        
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            predictions["error"] = f"Target column '{target_column}' must be numeric for predictions"
            return predictions
        
        # Try to use previous regression analysis if available
        if previous_analysis and "analyses" in previous_analysis and "regression" in previous_analysis["analyses"]:
            regression_results = previous_analysis["analyses"]["regression"]
            
            # Convert to list of tuples for easier sorting
            reg_list = []
            for feature, stats in regression_results.items():
                if "r_squared" in stats and "p_value" in stats and "slope" in stats:
                    reg_list.append((feature, stats["r_squared"], stats["p_value"], stats["slope"]))
            
            # Sort by R-squared (higher is better)
            reg_list.sort(key=lambda x: x[1], reverse=True)
            
            for feature, r_squared, p_value, slope in reg_list[:3]:  # Top 3 predictors
                if p_value < 0.05:  # Statistically significant
                    direction = "increases" if slope > 0 else "decreases"
                    opposite = "increases" if slope < 0 else "decreases"
                    
                    impact = "strong" if r_squared >= 0.5 else "moderate" if r_squared >= 0.2 else "weak"
                    
                    insight = f"{feature} has a {impact} impact on {target_column} (R² = {round(r_squared, 2)}). "
                    insight += f"When {feature} {direction}, {target_column} typically {direction}."
                    
                    # Add more context with data evidence
                    feature_mean = df[feature].mean()
                    
                    above_avg = df[df[feature] > feature_mean]
                    below_avg = df[df[feature] <= feature_mean]
                    
                    if len(above_avg) > 0 and len(below_avg) > 0:
                        above_avg_target = above_avg[target_column].mean()
                        below_avg_target = below_avg[target_column].mean()
                        
                        percent_diff = ((above_avg_target - below_avg_target) / below_avg_target * 100) if below_avg_target != 0 else 0
                        
                        evidence = f"When {feature} is above average, {target_column} is on average {round(abs(percent_diff), 1)}% "
                        evidence += "higher" if percent_diff > 0 else "lower"
                        
                        predictions["insights"].append({
                            "feature": feature,
                            "r_squared": float(r_squared),
                            "p_value": float(p_value),
                            "direction": "positive" if slope > 0 else "negative",
                            "impact": impact,
                            "insight": insight,
                            "evidence": evidence
                        })
        
        # Simple predictive insights based on correlations if regression not available
        else:
            numeric_df = df.select_dtypes(include=np.number)
            target_correlations = []
            
            for col in numeric_df.columns:
                if col != target_column:
                    corr = df[[target_column, col]].corr().iloc[0, 1]
                    if not np.isnan(corr) and abs(corr) >= 0.3:  # Threshold for meaningful correlation
                        target_correlations.append({
                            "column": col,
                            "correlation": round(float(corr), 3)
                        })
            
            # Sort by absolute correlation strength
            target_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            # Generate insights for top influencers
            top_influencers = target_correlations[:3]
            
            for infl in top_influencers:
                col = infl["column"]
                corr = infl["correlation"]
                
                direction = "increases" if corr > 0 else "decreases"
                opposite_direction = "increases" if corr < 0 else "decreases"
                
                impact = "strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.5 else "weak"
                
                insight = f"{col} has a {impact} correlation with {target_column} (r = {round(corr, 2)}). "
                insight += f"When {col} {direction}, {target_column} typically {direction}."
                
                # Add more context with data evidence
                col_mean = df[col].mean()
                
                above_avg = df[df[col] > col_mean]
                below_avg = df[df[col] <= col_mean]
                
                if len(above_avg) > 0 and len(below_avg) > 0:
                    above_avg_target = above_avg[target_column].mean()
                    below_avg_target = below_avg[target_column].mean()
                    
                    percent_diff = ((above_avg_target - below_avg_target) / below_avg_target * 100) if below_avg_target != 0 else 0
                    
                    evidence = f"When {col} is above average, {target_column} is on average {round(abs(percent_diff), 1)}% "
                    evidence += "higher" if percent_diff > 0 else "lower"
                    
                    predictions["insights"].append({
                        "feature": col,
                        "correlation": float(corr),
                        "direction": "positive" if corr > 0 else "negative",
                        "impact": impact,
                        "insight": insight,
                        "evidence": evidence
                    })
        
        return predictions
    
    def _identify_segments(self, df, previous_analysis, parameters):
        """Identify interesting segments or clusters in the data."""
        segments = []
        
        # Look for categorical columns to identify segments
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Add boolean and int columns with few unique values (likely categorical)
        for col in df.select_dtypes(include=['bool', 'int', 'int64']).columns:
            if df[col].nunique() <= 10:  # Threshold for "categorical-like"
                categorical_columns.append(col)
        
        # For each categorical column, check if there are interesting differences in numeric columns
        for cat_col in categorical_columns:
            category_values = df[cat_col].unique()
            
            # Skip if too many categories or too few
            if len(category_values) > 10 or len(category_values) < 2:
                continue
            
            # Check differences in numeric columns across categories
            for num_col in df.select_dtypes(include=np.number).columns:
                if num_col == cat_col:
                    continue
                
                # Calculate mean for each category
                category_means = {}
                for cat in category_values:
                    subset = df[df[cat_col] == cat]
                    if len(subset) > 0:
                        category_means[cat] = subset[num_col].mean()
                
                # Find min and max means
                if len(category_means) >= 2:
                    min_cat = min(category_means, key=category_means.get)
                    max_cat = max(category_means, key=category_means.get)
                    
                    min_val = category_means[min_cat]
                    max_val = category_means[max_cat]
                    
                    # Calculate percent difference
                    percent_diff = ((max_val - min_val) / min_val * 100) if min_val != 0 else 0
                    
                    # If significant difference, add insight
                    if abs(percent_diff) >= 20:  # 20% threshold for "interesting" difference
                        segments.append({
                            "segment_column": cat_col,
                            "value_column": num_col,
                            "top_segment": str(max_cat),
                            "bottom_segment": str(min_cat),
                            "top_value": float(max_val),
                            "bottom_value": float(min_val),
                            "percent_difference": round(float(percent_diff), 2),
                            "insight": f"{cat_col} segment '{max_cat}' has {round(abs(percent_diff), 1)}% higher {num_col} than segment '{min_cat}'"
                        })
        
        # Sort segments by percent difference
        segments.sort(key=lambda x: abs(x["percent_difference"]), reverse=True)
        
        return segments[:5]  # Limit to top 5 insights
    
    def _recommend_next_steps(self, df, insights, business_context):
        """Generate recommended next steps based on insights."""
        recommendations = []
        
        # Based on trends
        if "trends" in insights and insights["trends"]:
            for trend in insights["trends"][:2]:  # Consider top 2 trends
                col = trend["column"]
                direction = trend["direction"]
                
                recommendations.append({
                    "type": "trend_analysis",
                    "recommendation": f"Conduct deeper time-series analysis on {col} to understand factors driving the {direction} trend",
                    "priority": "high" if abs(trend["percent_change"]) > 25 else "medium"
                })
        
        # Based on correlations
        if "correlations" in insights and insights["correlations"]:
            strongest_corr = insights["correlations"][0] if insights["correlations"] else None
            if strongest_corr and abs(strongest_corr["correlation"]) > 0.7:
                col1 = strongest_corr["column1"]
                col2 = strongest_corr["column2"]
                
                recommendations.append({
                    "type": "correlation_investigation",
                    "recommendation": f"Investigate potential causal relationship between {col1} and {col2}",
                    "priority": "high"
                })
        
        # Based on segments
        if "segments" in insights and insights["segments"]:
            top_segment = insights["segments"][0] if insights["segments"] else None
            if top_segment and abs(top_segment["percent_difference"]) > 30:
                segment_col = top_segment["segment_column"]
                value_col = top_segment["value_column"]
                top_seg = top_segment["top_segment"]
                
                recommendations.append({
                    "type": "segment_analysis",
                    "recommendation": f"Analyze factors contributing to high {value_col} in {segment_col} segment '{top_seg}'",
                    "priority": "high"
                })
        
        # Based on outliers
        if "outliers" in insights and insights["outliers"]:
            recommendations.append({
                "type": "outlier_investigation",
                "recommendation": "Investigate identified outliers to determine if they represent data quality issues or valuable business exceptions",
                "priority": "medium"
            })
        
        # Data collection recommendations
        missing_columns = df.columns[df.isna().any()].tolist()
        if missing_columns:
            recommendations.append({
                "type": "data_quality",
                "recommendation": f"Improve data collection for columns with missing values: {', '.join(missing_columns[:3])}{'...' if len(missing_columns) > 3 else ''}",
                "priority": "medium"
            })
        
        # Add business-context specific recommendations if provided
        if business_context:
            lower_context = business_context.lower()
            
            if "sales" in lower_context or "revenue" in lower_context:
                recommendations.append({
                    "type": "business_specific",
                    "recommendation": "Segment analysis by customer demographics to identify highest-value customer profiles",
                    "priority": "high"
                })
            
            if "marketing" in lower_context:
                recommendations.append({
                    "type": "business_specific",
                    "recommendation": "Analyze conversion rates across different marketing channels to optimize campaign allocation",
                    "priority": "high"
                })
        
        return recommendations