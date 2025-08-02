# data_analysis_system/agents/analysis.py

from crewai import Agent
from data_analysis_system.tools import StatisticalAnalysisTool, InsightGeneratorTool

class AnalysisAgent:
    """Agent responsible for data analysis and insight generation."""
    
    @staticmethod
    def create():
        """Create the Analysis Agent with appropriate tools and configuration."""
        # Initialize the tools
        statistical_analysis_tool = StatisticalAnalysisTool()
        insight_generator_tool = InsightGeneratorTool()
        
        # Create the agent
        agent = Agent(
            role="Data Analyst",
            goal="Discover meaningful patterns and insights in data",
            backstory="""You are a skilled data analyst with a strong statistical background. 
            You excel at identifying trends, correlations, and outliers in complex datasets. 
            Your analytical mindset helps you uncover hidden patterns and translate them into 
            actionable insights. You understand not just how to perform analyses, but also how 
            to interpret results in business context. You're experienced in predictive modeling
            and can identify key factors influencing target variables.""",
            verbose=True,
            tools=[statistical_analysis_tool, insight_generator_tool],
            allow_delegation=False,
            memory=True  # Enable memory for contextual awareness
        )
        
        return agent
    
    @staticmethod
    def create_tasks(agent, context=None):
        """Create tasks for the Analysis Agent based on context."""
        from crewai import Task
        
        # Default context if none provided
        if context is None:
            context = {}
        
        # Extract relevant information from context
        data_path = context.get('cleaned_data_path', 'cleaned_data.csv')
        analysis_objective = context.get('analysis_objective', 'Identify key patterns and insights')
        target_column = context.get('target_column', None)
        business_context = context.get('business_context', None)
        
        # Create the statistical analysis task
        analysis_task = Task(
            description=f"""
            Perform statistical analysis on the cleaned data: {data_path}.
            
            Your objective is: {analysis_objective}
            
            Your responsibilities include:
            1. Conducting descriptive statistical analysis
            2. Identifying significant correlations between variables
            3. Performing regression analysis if appropriate
            4. Generating hypotheses based on initial findings
            5. Documenting all results with statistical significance
            
            Use your expertise to select the most appropriate analyses based on
            the data characteristics and analysis objective.
            """,
            expected_output="""A comprehensive analysis report containing:
            1. Summary statistics for key variables
            2. Correlation analysis results
            3. Regression analysis results (if applicable)
            4. Statistical significance of findings
            5. Path to the analysis results file""",
            agent=agent,
            context=context
        )
        
        # Create the insight generation task
        insight_task = Task(
            description=f"""
            Generate actionable insights from the analysis results.
            
            Analysis objective: {analysis_objective}
            {"Target column: " + target_column if target_column else ""}
            {"Business context: " + business_context if business_context else ""}
            
            Your responsibilities include:
            1. Identifying key trends in the data
            2. Highlighting significant correlations and potential causal relationships
            3. Detecting and explaining outliers
            4. Generating predictive insights if a target variable is specified
            5. Identifying interesting segments in the data
            6. Providing actionable recommendations based on the insights
            
            Focus on insights that are both statistically significant and business relevant.
            Prioritize findings that can lead to actionable decisions.
            """,
            expected_output="""A detailed insights report containing:
            1. Key trends identified in the data
            2. Significant correlations with explanations
            3. Outlier analysis and business implications
            4. Predictive insights (if applicable)
            5. Segment analysis with business relevance
            6. Prioritized recommendations for next steps
            7. Path to the insights file""",
            agent=agent,
            context=context
        )
        
        return [analysis_task, insight_task]
    
    @staticmethod
    def handle_error(error, context=None):
        """Handle errors that occur during the agent's execution."""
        # Create a standardized error response
        error_response = {
            "status": "error",
            "agent": "Analysis Agent",
            "error_message": str(error),
            "error_type": type(error).__name__,
            "recovery_recommendation": "Verify the data quality and format for analysis compatibility."
        }
        
        # Add context-specific recovery recommendations if available
        if context:
            if 'target_column' in context and context['target_column']:
                error_response["recovery_recommendation"] = f"Ensure the target column '{context['target_column']}' exists and contains valid data for analysis."
            elif 'analysis_types' in context:
                analysis_types = context['analysis_types']
                if "regression" in analysis_types:
                    error_response["recovery_recommendation"] = "Check that the dataset contains sufficient numeric variables for regression analysis."
                elif "correlation" in analysis_types:
                    error_response["recovery_recommendation"] = "Ensure the dataset contains multiple numeric variables for correlation analysis."
        
        return error_response