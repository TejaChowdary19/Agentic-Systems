# data_analysis_system/agents/data_collection.py

from crewai import Agent
from data_analysis_system.tools import DataRetrievalTool

class DataCollectionAgent:
    """Agent responsible for data collection and initial validation."""
    
    @staticmethod
    def create():
        """Create the Data Collection Agent with appropriate tools and configuration."""
        # Initialize the tools
        data_retrieval_tool = DataRetrievalTool()
        
        # Create the agent
        agent = Agent(
            role="Data Collection Specialist",
            goal="Gather high-quality data from various sources and validate its structure",
            backstory="""You are an expert in data acquisition with years of experience 
            in retrieving and validating data from diverse sources. You know how to 
            access APIs, databases, and web resources efficiently. You have a keen eye 
            for data quality issues and always perform initial validation to ensure the 
            data is suitable for analysis.""",
            verbose=True,
            tools=[data_retrieval_tool],
            allow_delegation=False,
            memory=True  # Enable memory for contextual awareness
        )
        
        return agent
    
    @staticmethod
    def create_tasks(agent, context=None):
        """Create tasks for the Data Collection Agent based on context."""
        from crewai import Task
        
        # Default context if none provided
        if context is None:
            context = {}
        
        # Extract relevant information from context
        data_source = context.get('data_source', {'source_type': 'file', 'source_path': 'data.csv'})
        
        # Create the data collection task
        collection_task = Task(
            description=f"""
            Retrieve data from the specified source: {data_source}.
            
            Your responsibilities include:
            1. Validating the source is accessible
            2. Retrieving the data completely and accurately
            3. Checking for obvious data quality issues
            4. Providing a summary of the retrieved data
            5. Documenting any potential concerns about the data
            
            If the source is inaccessible or the data has critical issues, 
            provide clear diagnostics and recommendations for resolution.
            """,
            expected_output="""A detailed report containing:
            1. Status of the data retrieval (success/failure)
            2. Data shape and structure summary
            3. Initial data quality assessment
            4. Path to the retrieved data
            5. Recommendations for data cleaning if needed""",
            agent=agent,
            context=context
        )
        
        return [collection_task]
    
    @staticmethod
    def handle_error(error, context=None):
        """Handle errors that occur during the agent's execution."""
        # Create a standardized error response
        error_response = {
            "status": "error",
            "agent": "Data Collection Agent",
            "error_message": str(error),
            "error_type": type(error).__name__,
            "recovery_recommendation": "Verify the data source accessibility and format compatibility."
        }
        
        # Add context-specific recovery recommendations if available
        if context and 'data_source' in context:
            source_type = context['data_source'].get('source_type', '')
            if source_type == 'file':
                error_response["recovery_recommendation"] = "Verify the file exists and has the correct format."
            elif source_type == 'api':
                error_response["recovery_recommendation"] = "Check API credentials and endpoint availability."
            elif source_type == 'url':
                error_response["recovery_recommendation"] = "Verify the URL is accessible and returns data in the expected format."
        
        return error_response