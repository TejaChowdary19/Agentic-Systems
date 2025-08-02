# data_analysis_system/agents/visualization.py

from crewai import Agent
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from langchain.tools import BaseTool
from typing import Dict, List, Optional, Type, Any
from pydantic import BaseModel, Field

# Create a Visualization Tool for the Visualization Agent
class VisualizationInput(BaseModel):
    """Input for the visualization tool."""
    data_path: str = Field(
        description="Path to the data file to visualize"
    )
    plot_types: List[str] = Field(
        description="Types of plots to create (bar, line, scatter, histogram, boxplot, heatmap)"
    )
    x_column: Optional[str] = Field(
        default=None,
        description="Column to use for x-axis (required for some plot types)"
    )
    y_column: Optional[str] = Field(
        default=None,
        description="Column to use for y-axis (required for some plot types)"
    )
    group_column: Optional[str] = Field(
        default=None,
        description="Column to use for grouping or coloring points"
    )
    output_dir: Optional[str] = Field(
        default="visualizations",
        description="Directory to save visualizations"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional parameters for specific plot types"
    )

class VisualizationTool(BaseTool):
    name = "Visualization Tool"
    description = "Creates data visualizations including bar charts, line plots, scatter plots, histograms, box plots, and heatmaps."
    args_schema: Type[BaseModel] = VisualizationInput
    
    def _run(self, data_path: str, plot_types: List[str], x_column: Optional[str] = None, 
             y_column: Optional[str] = None, group_column: Optional[str] = None,
             output_dir: str = "visualizations", parameters: Optional[Dict[str, Any]] = None) -> str:
        """Create visualizations of the data."""
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
            
            # Initialize results
            results = {"status": "success", "visualizations": {}}
            plots_created = []
            
            # Create base directory for plots if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate plots based on requested types
            for plot_type in plot_types:
                try:
                    plt.figure(figsize=(10, 6))
                    
                    if plot_type.lower() == "bar" and x_column is not None and y_column is not None:
                        # Bar chart
                        sns.barplot(x=x_column, y=y_column, data=df, hue=group_column)
                        
                        plt.title(f"Bar Chart: {y_column} by {x_column}")
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        plot_path = f"{output_dir}/bar_chart_{x_column}_{y_column}.png"
                        plt.savefig(plot_path)
                        plots_created.append({"type": "bar", "path": plot_path})
                    
                    elif plot_type.lower() == "line" and x_column is not None and y_column is not None:
                        # Line plot
                        plt.figure(figsize=(10, 6))
                        
                        if group_column is not None:
                            for group in df[group_column].unique():
                                group_data = df[df[group_column] == group]
                                plt.plot(group_data[x_column], group_data[y_column], label=str(group))
                            plt.legend()
                        else:
                            plt.plot(df[x_column], df[y_column])
                        
                        plt.title(f"Line Plot: {y_column} by {x_column}")
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                        plt.tight_layout()
                        
                        plot_path = f"{output_dir}/line_plot_{x_column}_{y_column}.png"
                        plt.savefig(plot_path)
                        plots_created.append({"type": "line", "path": plot_path})
                    
                    elif plot_type.lower() == "scatter" and x_column is not None and y_column is not None:
                        # Scatter plot
                        plt.figure(figsize=(10, 6))
                        
                        scatter = sns.scatterplot(x=x_column, y=y_column, data=df, hue=group_column)
                        
                        plt.title(f"Scatter Plot: {y_column} vs {x_column}")
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                        
                        if parameters and 'add_regression_line' in parameters and parameters['add_regression_line']:
                            # Add regression line
                            sns.regplot(x=x_column, y=y_column, data=df, scatter=False, ax=scatter.axes)
                        
                        plt.tight_layout()
                        
                        plot_path = f"{output_dir}/scatter_plot_{x_column}_{y_column}.png"
                        plt.savefig(plot_path)
                        plots_created.append({"type": "scatter", "path": plot_path})
                    
                    elif plot_type.lower() == "histogram" and x_column is not None:
                        # Histogram
                        plt.figure(figsize=(10, 6))
                        
                        bins = 10  # Default number of bins
                        if parameters and 'bins' in parameters:
                            bins = parameters['bins']
                        
                        sns.histplot(df[x_column], bins=bins, kde=True)
                        
                        plt.title(f"Histogram of {x_column}")
                        plt.xlabel(x_column)
                        plt.ylabel("Frequency")
                        plt.tight_layout()
                        
                        plot_path = f"{output_dir}/histogram_{x_column}.png"
                        plt.savefig(plot_path)
                        plots_created.append({"type": "histogram", "path": plot_path})
                    
                    elif plot_type.lower() == "boxplot" and x_column is not None:
                        # Box plot
                        plt.figure(figsize=(10, 6))
                        
                        if y_column is not None:
                            # Box plot with categories
                            sns.boxplot(x=x_column, y=y_column, data=df, hue=group_column)
                            plt.title(f"Box Plot: {y_column} by {x_column}")
                            plt.xlabel(x_column)
                            plt.ylabel(y_column)
                        else:
                            # Simple box plot
                            sns.boxplot(x=df[x_column])
                            plt.title(f"Box Plot of {x_column}")
                            plt.xlabel(x_column)
                            plt.ylabel("Value")
                        
                        plt.tight_layout()
                        
                        plot_path = f"{output_dir}/boxplot_{x_column}.png"
                        plt.savefig(plot_path)
                        plots_created.append({"type": "boxplot", "path": plot_path})
                    
                    elif plot_type.lower() == "heatmap":
                        # Correlation heatmap
                        plt.figure(figsize=(12, 10))
                        
                        if parameters and 'correlation_columns' in parameters:
                            cols = parameters['correlation_columns']
                        else:
                            cols = df.select_dtypes(include=np.number).columns
                        
                        corr_matrix = df[cols].corr()
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
                        
                        plt.title("Correlation Heatmap")
                        plt.tight_layout()
                        
                        plot_path = f"{output_dir}/heatmap_correlation.png"
                        plt.savefig(plot_path)
                        plots_created.append({"type": "heatmap", "path": plot_path})
                    
                    plt.close()
                    
                except Exception as plot_error:
                    results["visualizations"][plot_type] = {
                        "error": str(plot_error)
                    }
            
            results["visualizations"]["plots_created"] = plots_created
            return json.dumps(results)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e),
                "data_path": data_path
            })
    
    def _arun(self, data_path: str, plot_types: List[str], x_column: Optional[str] = None, 
              y_column: Optional[str] = None, group_column: Optional[str] = None,
              output_dir: str = "visualizations", parameters: Optional[Dict[str, Any]] = None) -> str:
        """Async implementation of the tool."""
        return self._run(data_path, plot_types, x_column, y_column, group_column, output_dir, parameters)

class VisualizationAgent:
    """Agent responsible for creating data visualizations."""
    
    @staticmethod
    def create():
        """Create the Visualization Agent with appropriate tools and configuration."""
        # Initialize the tools
        visualization_tool = VisualizationTool()
        
        # Create the agent
        agent = Agent(
            role="Data Visualization Specialist",
            goal="Create clear, informative visualizations that communicate key insights",
            backstory="""You are a data visualization expert who knows how to translate 
            complex data into compelling visual stories. You understand the principles of 
            effective data visualization and can choose the right chart types for different 
            data patterns. You know how to highlight key insights through visual emphasis, 
            use color effectively, and design visualizations for different audiences. Your 
            visualizations are both informative and visually appealing.""",
            verbose=True,
            tools=[visualization_tool],
            allow_delegation=False,
            memory=True  # Enable memory for contextual awareness
        )
        
        return agent
    
    @staticmethod
    def create_tasks(agent, context=None):
        """Create tasks for the Visualization Agent based on context."""
        from crewai import Task
        
        # Default context if none provided
        if context is None:
            context = {}
        
        # Extract relevant information from context
        data_path = context.get('cleaned_data_path', 'cleaned_data.csv')
        analysis_results = context.get('analysis_results', {})
        insights = context.get('insights', {})
        
        # Create the visualization task
        visualization_task = Task(
            description=f"""
            Create informative visualizations for the data: {data_path}.
            
            Use the analysis results and insights to determine which relationships
            and patterns to visualize. Focus on creating visualizations that effectively
            communicate the key findings from the analysis.
            
            Your responsibilities include:
            1. Creating descriptive visualizations of key variables
            2. Visualizing important relationships identified in the correlation analysis
            3. Creating visualizations that highlight key insights
            4. Designing visualizations that support the main findings
            5. Ensuring all visualizations are clear, properly labeled, and professional
            
            Choose appropriate visualization types based on the data characteristics
            and the specific insights you want to communicate.
            """,
            expected_output="""A visualization report containing:
            1. List of created visualizations with their paths
            2. Description of what each visualization shows
            3. Key insights that can be derived from each visualization
            4. Recommendations for how the visualizations could be used in decision-making""",
            agent=agent,
            context=context
        )
        
        return [visualization_task]
    
    @staticmethod
    def handle_error(error, context=None):
        """Handle errors that occur during the agent's execution."""
        # Create a standardized error response
        error_response = {
            "status": "error",
            "agent": "Visualization Agent",
            "error_message": str(error),
            "error_type": type(error).__name__,
            "recovery_recommendation": "Check if the data contains appropriate variables for the requested visualizations."
        }
        
        # Add context-specific recovery recommendations if available
        if context and 'plot_types' in context:
            plot_types = context['plot_types']
            if "heatmap" in plot_types:
                error_response["recovery_recommendation"] = "Ensure the dataset contains multiple numeric variables for the correlation heatmap."
            elif any(x in plot_types for x in ["scatter", "line", "bar"]):
                error_response["recovery_recommendation"] = "Verify that the specified x and y columns exist and contain appropriate data types."
        
        return error_response