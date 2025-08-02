# data_analysis_system/tools/data_retrieval.py

import json
import pandas as pd
import requests
from io import StringIO
from typing import Dict, List, Optional, Type, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

class DataRetrievalInput(BaseModel):
    """Input for the data retrieval tool."""
    source_type: str = Field(
        description="Type of data source (file, api, url)"
    )
    source_path: str = Field(
        description="Path or URL to the data source"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for data retrieval (API keys, query parameters, etc.)"
    )

class DataRetrievalTool(BaseTool):
    name = "Data Retrieval Tool"
    description = "Retrieves data from various sources including files, APIs, and URLs."
    args_schema: Type[BaseModel] = DataRetrievalInput
    
    def _run(self, source_type: str, source_path: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Retrieve data from the specified source."""
        try:
            if source_type.lower() == "file":
                # Load data from a local file
                if source_path.endswith('.csv'):
                    df = pd.read_csv(source_path)
                elif source_path.endswith('.xlsx') or source_path.endswith('.xls'):
                    df = pd.read_excel(source_path)
                elif source_path.endswith('.json'):
                    df = pd.read_json(source_path)
                else:
                    return json.dumps({
                        "status": "error",
                        "message": f"Unsupported file format: {source_path}"
                    })
                
                # Return basic information about the retrieved data
                return json.dumps({
                    "status": "success",
                    "data_shape": df.shape,
                    "columns": df.columns.tolist(),
                    "sample": df.head(5).to_dict(orient='records'),
                    "data_path": source_path
                })
            
            elif source_type.lower() == "api":
                # Make an API request
                if parameters is None:
                    parameters = {}
                
                response = requests.get(source_path, params=parameters)
                if response.status_code == 200:
                    try:
                        # Try to parse as JSON
                        data = response.json()
                        return json.dumps({
                            "status": "success",
                            "data": data,
                            "source": source_path
                        })
                    except:
                        # Return raw content if not JSON
                        return json.dumps({
                            "status": "success",
                            "data": response.text[:1000] + "...",  # Truncate for readability
                            "source": source_path
                        })
                else:
                    return json.dumps({
                        "status": "error",
                        "message": f"API request failed with status code {response.status_code}",
                        "source": source_path
                    })
            
            elif source_type.lower() == "url":
                # Fetch data from a URL (assuming CSV format)
                response = requests.get(source_path)
                if response.status_code == 200:
                    try:
                        df = pd.read_csv(StringIO(response.text))
                        return json.dumps({
                            "status": "success",
                            "data_shape": df.shape,
                            "columns": df.columns.tolist(),
                            "sample": df.head(5).to_dict(orient='records'),
                            "source": source_path
                        })
                    except:
                        return json.dumps({
                            "status": "error",
                            "message": "Failed to parse URL content as CSV",
                            "source": source_path
                        })
                else:
                    return json.dumps({
                        "status": "error",
                        "message": f"URL request failed with status code {response.status_code}",
                        "source": source_path
                    })
            
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Unsupported source type: {source_type}"
                })
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e),
                "source": source_path
            })
    
    def _arun(self, source_type: str, source_path: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Async implementation of the tool."""
        # For simplicity, we're just calling the sync version
        return self._run(source_type, source_path, parameters)