# data_analysis_system/tools/statistical_analysis.py

import json
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Type, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

class StatisticalAnalysisInput(BaseModel):
    """Input for the statistical analysis tool."""
    data_path: str = Field(
        description="Path to the data file to analyze"
    )
    analysis_types: List[str] = Field(
        description="Types of analyses to perform (descriptive, correlation, regression, hypothesis_test)"
    )
    target_column: Optional[str] = Field(
        default=None,
        description="Target column for regression or hypothesis testing"
    )
    feature_columns: Optional[List[str]] = Field(
        default=None,
        description="Feature columns for analysis (if None, uses all applicable columns)"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for specific analyses"
    )

class StatisticalAnalysisTool(BaseTool):
    name = "Statistical Analysis Tool"
    description = "Performs statistical analyses on data, including descriptive statistics, correlation analysis, regression, and hypothesis testing."
    args_schema: Type[BaseModel] = StatisticalAnalysisInput
    
    def _run(self, data_path: str, analysis_types: List[str], target_column: Optional[str] = None, 
             feature_columns: Optional[List[str]] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Perform statistical analyses on the data."""
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
            
            # Initialize results dictionary
            results = {"status": "success", "analyses": {}}
            
            # Perform requested analyses
            for analysis_type in analysis_types:
                if analysis_type.lower() == "descriptive":
                    # Descriptive statistics
                    cols_to_analyze = feature_columns if feature_columns is not None else df.select_dtypes(include=np.number).columns
                    desc_stats = df[cols_to_analyze].describe().to_dict()
                    
                    # Add additional metrics
                    for col in cols_to_analyze:
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                            desc_stats[col]['skew'] = float(df[col].skew())
                            desc_stats[col]['kurtosis'] = float(df[col].kurtosis())
                    
                    results["analyses"]["descriptive"] = desc_stats
                
                elif analysis_type.lower() == "correlation":
                    # Correlation analysis
                    cols_to_analyze = feature_columns if feature_columns is not None else df.select_dtypes(include=np.number).columns
                    corr_matrix = df[cols_to_analyze].corr().to_dict()
                    
                    # Convert numpy types to Python native types for JSON serialization
                    for col1 in corr_matrix:
                        for col2 in corr_matrix[col1]:
                            if isinstance(corr_matrix[col1][col2], (np.integer, np.floating)):
                                corr_matrix[col1][col2] = float(corr_matrix[col1][col2])
                    
                    # Find highest correlations
                    corr_df = df[cols_to_analyze].corr().unstack().sort_values(ascending=False)
                    # Remove self-correlations (which are always 1.0)
                    corr_df = corr_df[corr_df < 1.0]
                    highest_corrs = {}
                    for (col1, col2), value in corr_df.head(5).items():
                        key = f"{col1}__{col2}"
                        highest_corrs[key] = float(value)
                    
                    results["analyses"]["correlation"] = {
                        "matrix": corr_matrix,
                        "highest_correlations": highest_corrs
                    }
                
                elif analysis_type.lower() == "regression" and target_column is not None:
                    # Simple linear regression
                    if target_column not in df.columns or not pd.api.types.is_numeric_dtype(df[target_column]):
                        results["analyses"]["regression"] = {
                            "error": f"Target column '{target_column}' must be numeric"
                        }
                        continue
                    
                    cols_to_analyze = feature_columns if feature_columns is not None else df.select_dtypes(include=np.number).columns
                    cols_to_analyze = [col for col in cols_to_analyze if col != target_column and col in df.columns]
                    
                    regression_results = {}
                    for col in cols_to_analyze:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            # Run simple linear regression
                            x = df[col].values.reshape(-1, 1)
                            y = df[target_column].values
                            
                            # Handle missing values
                            mask = ~np.isnan(x.flatten()) & ~np.isnan(y)
                            if mask.sum() < 2:
                                continue
                                
                            x = x[mask].reshape(-1, 1)
                            y = y[mask]
                            
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)
                            
                            regression_results[col] = {
                                "slope": float(slope),
                                "intercept": float(intercept),
                                "r_squared": float(r_value**2),
                                "p_value": float(p_value),
                                "std_error": float(std_err)
                            }
                    
                    results["analyses"]["regression"] = regression_results
                
                elif analysis_type.lower() == "hypothesis_test" and target_column is not None:
                    # Basic hypothesis testing
                    if parameters and 'test_type' in parameters:
                        test_type = parameters['test_type']
                        
                        if test_type == 't_test' and 'group_column' in parameters:
                            # Perform t-test between two groups
                            group_col = parameters['group_column']
                            if group_col in df.columns:
                                groups = df[group_col].unique()
                                if len(groups) >= 2:
                                    group1, group2 = groups[0], groups[1]
                                    
                                    group1_data = df[df[group_col] == group1][target_column].dropna()
                                    group2_data = df[df[group_col] == group2][target_column].dropna()
                                    
                                    if len(group1_data) > 0 and len(group2_data) > 0:
                                        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                                        
                                        results["analyses"]["hypothesis_test"] = {
                                            "test_type": "t_test",
                                            "groups": [str(group1), str(group2)],
                                            "t_statistic": float(t_stat),
                                            "p_value": float(p_value),
                                            "interpretation": "Significant difference" if p_value < 0.05 else "No significant difference"
                                        }
                    else:
                        results["analyses"]["hypothesis_test"] = {
                            "error": "Hypothesis test requires parameters['test_type'] to be specified"
                        }
            
            # Save the results to a file
            results_path = data_path.replace('.', '_analysis_results.')
            if not results_path.endswith('.json'):
                results_path = results_path + '.json'
                
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            results["analysis_results_path"] = results_path
            
            return json.dumps(results)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e),
                "data_path": data_path
            })
    
    def _arun(self, data_path: str, analysis_types: List[str], target_column: Optional[str] = None, 
              feature_columns: Optional[List[str]] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Async implementation of the tool."""
        # For simplicity, we're just calling the sync version
        return self._run(data_path, analysis_types, target_column, feature_columns, parameters)