# data_analysis_system/agents/data_processing.py

from crewai import Agent
from data_analysis_system.tools import DataCleaningTool

class DataProcessingAgent:
    """Agent responsible for data cleaning and preprocessing."""
    
    @staticmethod
    def create():
        """Create the Data Processing Agent with appropriate tools and configuration."""
        # Initialize the tools
        data_cleaning_tool = DataCleaningTool()
        
        # Create the agent
        agent = Agent(
            role="Data Processing Specialist",
            goal="Clean and transform raw data into analysis-ready datasets",
            backstory="""You are a meticulous data engineer who excels at identifying 
            and fixing data quality issues. You're skilled in data cleaning, normalization, 
            and transformation techniques. Your attention to detail ensures that datasets 
            are properly prepared for analysis, with all issues like missing values, outliers, 
            and inconsistencies appropriately addressed. You document all your processing 
            steps to ensure transparency and reproducibility.""",
            verbose=True,
            tools=[data_cleaning_tool],
            allow_delegation=False,
            memory=True  # Enable memory for contextual awareness
        )
        
        return agent
    
    @staticmethod
    def create_tasks(agent, context=None):
        """Create tasks for the Data Processing Agent based on context."""
        from crewai import Task
        
        # Default context if none provided
        if context is None:
            context = {}
        
        # Extract relevant information from context
        data_path = context.get('data_path', 'data.csv')
        collection_report = context.get('collection_report', {})
        
        # Determine needed cleaning operations based on collection report
        cleaning_operations = ["handle_missing", "remove_duplicates"]
        if collection_report and 'data_quality_issues' in collection_report:
            issues = collection_report['data_quality_issues']
            if 'outliers_detected' in issues and issues['outliers_detected']:
                cleaning_operations.append("remove_outliers")
            if 'normalization_needed' in issues and issues['normalization_needed']:
                cleaning_operations.append("normalize")
        
        # Create the data processing task
        processing_task = Task(
            description=f"""
            Clean and preprocess the data from: {data_path}.
            
            Based on the collection report, perform these cleaning operations:
            {', '.join(cleaning_operations)}
            
            Your responsibilities include:
            1. Handling missing values appropriately
            2. Removing duplicate records
            3. Addressing outliers if present
            4. Normalizing or transforming variables as needed
            5. Documenting all transformations applied to the data
            
            Use your expertise to determine the best approach for each cleaning operation.
            For example, missing values might be filled with mean, median, or mode depending
            on the data distribution.
            """,
            expected_output="""A detailed processing report containing:
            1. Status of the data cleaning (success/failure)
            2. List of operations performed and their impact
            3. Statistics on how the data changed (e.g., # of missing values filled)
            4. Path to the cleaned data
            5. Any recommendations for further processing""",
            agent=agent,
            context=context
        )
        
        return [processing_task]
    
    @staticmethod
    def handle_error(error, context=None):
        """Handle errors that occur during the agent's execution."""
        # Create a standardized error response
        error_response = {
            "status": "error",
            "agent": "Data Processing Agent",
            "error_message": str(error),
            "error_type": type(error).__name__,
            "recovery_recommendation": "Check the data format and compatibility with the cleaning operations."
        }
        
        # Add context-specific recovery recommendations if available
        if context and 'cleaning_operations' in context:
            operations = context['cleaning_operations']
            if "normalize" in operations:
                error_response["recovery_recommendation"] = "Ensure numeric columns don't have extreme values or zeros that could cause division issues during normalization."
            elif "remove_outliers" in operations:
                error_response["recovery_recommendation"] = "Verify that numeric columns have sufficient non-missing values for outlier detection."
        
        return error_response