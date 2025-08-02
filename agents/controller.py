# data_analysis_system/agents/controller.py

from crewai import Agent, Task
import json
import os

class ControllerAgent:
    """Agent responsible for orchestrating the entire data analysis workflow."""
    
    @staticmethod
    def create():
        """Create the Controller Agent with appropriate configuration."""
        # Create the agent
        agent = Agent(
            role="Data Analysis Coordinator",
            goal="Orchestrate the end-to-end data analysis process and synthesize results",
            backstory="""You are an experienced project manager specializing in data analytics. 
            You excel at breaking down complex analytical questions into clear tasks and 
            coordinating the work of specialized team members. You ensure that all analysis 
            activities align with the core business objectives and that the final deliverables 
            provide actionable intelligence. You have a talent for synthesizing information 
            from multiple sources into coherent, focused reports.""",
            verbose=True,
            allow_delegation=True,  # Controller can delegate tasks to other agents
            memory=True  # Enable memory for contextual awareness
        )
        
        return agent
    
    @staticmethod
    def create_tasks(agent, context=None):
        """Create tasks for the Controller Agent based on context."""
        # Default context if none provided
        if context is None:
            context = {}
        
        # Extract relevant information from context
        analysis_objective = context.get('analysis_objective', 'Identify key patterns and insights')
        business_context = context.get('business_context', 'General business analysis')
        
        # Create the planning task
        planning_task = Task(
            description=f"""
            Plan the data analysis workflow for the objective: {analysis_objective}
            
            Business context: {business_context}
            
            Your responsibilities include:
            1. Breaking down the analysis objective into specific questions to answer
            2. Determining which specialized agents need to be involved
            3. Planning the sequence of tasks and their dependencies
            4. Identifying potential challenges and mitigation strategies
            5. Defining success criteria for the analysis
            
            Create a comprehensive plan that will guide the entire analysis process.
            """,
            expected_output="""A detailed analysis plan containing:
            1. Key questions to be answered
            2. Required data and analyses
            3. Task sequence and dependencies
            4. Risk assessment and mitigation strategies
            5. Success criteria and evaluation metrics""",
            agent=agent,
            context=context
        )
        
        # Create the final report task
        report_task = Task(
            description=f"""
            Compile the final analysis report based on the outputs from all specialized agents.
            
            Analysis objective: {analysis_objective}
            Business context: {business_context}
            
            Your responsibilities include:
            1. Synthesizing findings from data collection, processing, analysis, and visualization
            2. Organizing insights in order of business relevance and actionability
            3. Ensuring all conclusions are supported by data
            4. Providing clear, actionable recommendations
            5. Creating an executive summary that highlights key findings
            
            The report should tell a coherent story that addresses the original analysis objective
            and provides actionable intelligence.
            """,
            expected_output="""A comprehensive analysis report containing:
            1. Executive summary with key findings
            2. Introduction with analysis objective and approach
            3. Methodology section describing data sources and analysis techniques
            4. Results section with visualizations and key insights
            5. Discussion of implications and limitations
            6. Actionable recommendations
            7. Appendices with supporting details""",
            agent=agent,
            context=context
        )
        
        return [planning_task, report_task]
    
    @staticmethod
    def handle_error(error, context=None):
        """Handle errors that occur during the agent's execution."""
        # Create a standardized error response
        error_response = {
            "status": "error",
            "agent": "Controller Agent",
            "error_message": str(error),
            "error_type": type(error).__name__,
            "recovery_recommendation": "Review the analysis plan and ensure all required information is provided."
        }
        
        # Add fallback strategies
        error_response["fallback_strategies"] = [
            "Simplify the analysis objective to focus on core questions",
            "Use available partial results to generate preliminary insights",
            "Request additional information or clarification on the analysis requirements"
        ]
        
        return error_response
    
    @staticmethod
    def create_memory(initial_context=None):
        """Create a memory store for the controller agent to maintain context across tasks."""
        # Initialize memory with context if provided
        memory = initial_context.copy() if initial_context else {}
        
        # Add a function to update memory
        def update_memory(key, value):
            memory[key] = value
            return memory
        
        # Add a function to retrieve from memory
        def get_from_memory(key, default=None):
            return memory.get(key, default)
        
        # Add a function to save memory to disk
        def save_memory(filepath='agent_memory.json'):
            with open(filepath, 'w') as f:
                json.dump(memory, f, indent=2)
            return filepath
        
        # Add a function to load memory from disk
        def load_memory(filepath='agent_memory.json'):
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    loaded_memory = json.load(f)
                    for key, value in loaded_memory.items():
                        memory[key] = value
                return True
            return False
        
        # Package memory functions
        memory_functions = {
            "update": update_memory,
            "get": get_from_memory,
            "save": save_memory,
            "load": load_memory,
            "current": memory
        }
        
        return memory_functions