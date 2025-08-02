# data_analysis_system/tools/data_cleaning.py

import json
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Type, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

class DataCleaningInput(BaseModel):
    """Input for the data cleaning tool."""
    data_path: str = Field(
        description="Path to the data file to clean"
    )
    operations: List[str] = Field(
        description="List of cleaning operations to perform (handle_missing, remove_duplicates, remove_outliers, normalize)"
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="List of columns to apply operations to (if None, applies to all applicable columns)"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for specific cleaning operations"
    )

class DataCleaningTool(BaseTool):
    name = "Data Cleaning Tool"
    description = "Cleans and preprocesses data by handling missing values, duplicates, outliers, and normalization."
    args_schema: Type[BaseModel] = DataCleaningInput
    
    def _run(self, data_path: str, operations: List[str], columns: Optional[List[str]] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Clean the data with specified operations."""
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
            
            # Keep track of operations performed
            operations_performed = []
            
            # Apply specified cleaning operations
            for operation in operations:
                if operation.lower() == "handle_missing":
                    # Handle missing values
                    cols_to_process = columns if columns is not None else df.columns
                    for col in cols_to_process:
                        if col in df.columns:
                            missing_count = df[col].isna().sum()
                            if missing_count > 0:
                                # Default strategy is to fill with mean for numeric, mode for categorical
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    df[col].fillna(df[col].mean(), inplace=True)
                                    operations_performed.append(f"Filled {missing_count} missing values in '{col}' with mean")
                                else:
                                    df[col].fillna(df[col].mode()[0], inplace=True)
                                    operations_performed.append(f"Filled {missing_count} missing values in '{col}' with mode")
                
                elif operation.lower() == "remove_duplicates":
                    # Remove duplicate rows
                    dup_count = df.duplicated().sum()
                    if dup_count > 0:
                        df.drop_duplicates(inplace=True)
                        operations_performed.append(f"Removed {dup_count} duplicate rows")
                
                elif operation.lower() == "remove_outliers":
                    # Remove outliers using Z-score method
                    cols_to_process = columns if columns is not None else df.select_dtypes(include=np.number).columns
                    for col in cols_to_process:
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                            # Skip columns with too many missing values
                            if df[col].isna().sum() > 0.5 * len(df):
                                continue
                                
                            z_scores = np.abs(stats.zscore(df[col].dropna()))
                            z_threshold = 3.0  # Default threshold
                            if parameters and 'z_threshold' in parameters:
                                z_threshold = parameters['z_threshold']
                            
                            outlier_indices = df[col].dropna().index[z_scores > z_threshold]
                            outlier_count = len(outlier_indices)
                            
                            if outlier_count > 0:
                                # Replace outliers with NaN and then fill with mean
                                df.loc[outlier_indices, col] = np.nan
                                df[col].fillna(df[col].mean(), inplace=True)
                                operations_performed.append(f"Replaced {outlier_count} outliers in '{col}' (Z-score > {z_threshold})")
                
                elif operation.lower() == "normalize":
                    # Normalize numeric columns to 0-1 range
                    cols_to_process = columns if columns is not None else df.select_dtypes(include=np.number).columns
                    for col in cols_to_process:
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                            # Skip columns with too many missing values
                            if df[col].isna().sum() > 0.5 * len(df):
                                continue
                                
                            # Skip columns with min = max (constant columns)
                            if df[col].min() == df[col].max():
                                operations_performed.append(f"Skipped normalizing '{col}' (constant column)")
                                continue
                                
                            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                            operations_performed.append(f"Normalized '{col}' to 0-1 range")
            
            # Save the cleaned data
            cleaned_path = data_path.replace('.', '_cleaned.')
            if cleaned_path.endswith('.csv'):
                df.to_csv(cleaned_path, index=False)
            elif cleaned_path.endswith('.xlsx') or cleaned_path.endswith('.xls'):
                df.to_excel(cleaned_path, index=False)
            elif cleaned_path.endswith('.json'):
                df.to_json(cleaned_path, orient='records')
            
            # Return summary of cleaning operations
            return json.dumps({
                "status": "success",
                "operations_performed": operations_performed,
                "cleaned_data_shape": df.shape,
                "cleaned_data_path": cleaned_path,
                "sample": df.head(3).to_dict(orient='records')
            })
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e),
                "data_path": data_path
            })
    
    def _arun(self, data_path: str, operations: List[str], columns: Optional[List[str]] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Async implementation of the tool."""
        # For simplicity, we're just calling the sync version
        return self._run(data_path, operations, columns, parameters)