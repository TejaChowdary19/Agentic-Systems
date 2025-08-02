# data_analysis_system/orchestration.py

from crewai import Crew, Process
import logging
import json
import os
import time
from typing import Dict, List, Any, Optional

from data_analysis_system.agents import (
    ControllerAgent,
    DataCollectionAgent,
    DataProcessingAgent,
    AnalysisAgent,
    VisualizationAgent
)

class DataAnalysisOrchestrator:
    """Orchestrates the data analysis workflow using multiple specialized agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator with optional configuration."""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("data_analysis.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("DataAnalysisOrchestrator")
        
        # Set default configuration if none provided
        if config is None:
            config = {}
        
        self.config = {
            "process_type": config.get("process_type", Process.sequential),
            "memory_enabled": config.get("memory_enabled", True),
            "verbose": config.get("verbose", True),
            "max_iterations": config.get("max_iterations", 1),
            "output_dir": config.get("output_dir", "outputs"),
            "error_handling": config.get("error_handling", "continue")  # 'continue', 'retry', or 'fail'
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.config["output_dir"]):
            os.makedirs(self.config["output_dir"])
        
        # Initialize agents
        self.agents = {
            "controller": None,
            "data_collection": None,
            "data_processing": None,
            "analysis": None,
            "visualization": None
        }
        
        # Initialize tasks
        self.tasks = {
            "controller": [],
            "data_collection": [],
            "data_processing": [],
            "analysis": [],
            "visualization": []
        }
        
        # Initialize memory store for context passing
        self.memory = {}
        
        self.logger.info("DataAnalysisOrchestrator initialized with config: %s", self.config)
    
    def initialize_agents(self):
        """Initialize all agents needed for the workflow."""
        self.logger.info("Initializing agents...")
        
        # Create all agents
        self.agents["controller"] = ControllerAgent.create()
        self.agents["data_collection"] = DataCollectionAgent.create()
        self.agents["data_processing"] = DataProcessingAgent.create()
        self.agents["analysis"] = AnalysisAgent.create()
        self.agents["visualization"] = VisualizationAgent.create()
        
        self.logger.info("All agents initialized successfully")
        return self.agents
    
    def create_tasks(self, context: Dict[str, Any]):
        """Create tasks for all agents based on context."""
        self.logger.info("Creating tasks with context...")
        
        # Create controller tasks
        self.tasks["controller"] = ControllerAgent.create_tasks(
            self.agents["controller"], context
        )
        
        # Create data collection tasks
        self.tasks["data_collection"] = DataCollectionAgent.create_tasks(
            self.agents["data_collection"], context
        )
        
        # Create data processing tasks
        self.tasks["data_processing"] = DataProcessingAgent.create_tasks(
            self.agents["data_processing"], context
        )
        
        # Create analysis tasks
        self.tasks["analysis"] = AnalysisAgent.create_tasks(
            self.agents["analysis"], context
        )
        
        # Create visualization tasks
        self.tasks["visualization"] = VisualizationAgent.create_tasks(
            self.agents["visualization"], context
        )
        
        # Flatten all tasks into a single list
        all_tasks = []
        for task_list in self.tasks.values():
            all_tasks.extend(task_list)
        
        self.logger.info(f"Created {len(all_tasks)} tasks across all agents")
        return all_tasks
    
    def create_crew(self, tasks):
        """Create a crew with all agents and tasks."""
        self.logger.info("Creating crew...")
        
        # Create agent list
        agents = list(self.agents.values())
        
        # Create the crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=self.config["verbose"],
            process=self.config["process_type"],
            max_iterations=self.config["max_iterations"]
        )
        
        self.logger.info("Crew created successfully")
        return crew
    
    def handle_task_error(self, agent_type, error, context):
        """Handle errors that occur during task execution."""
        self.logger.error(f"Error in {agent_type} task: {str(error)}")
        
        # Call the agent's error handler
        if agent_type == "controller":
            error_response = ControllerAgent.handle_error(error, context)
        elif agent_type == "data_collection":
            error_response = DataCollectionAgent.handle_error(error, context)
        elif agent_type == "data_processing":
            error_response = DataProcessingAgent.handle_error(error, context)
        elif agent_type == "analysis":
            error_response = AnalysisAgent.handle_error(error, context)
        elif agent_type == "visualization":
            error_response = VisualizationAgent.handle_error(error, context)
        else:
            error_response = {
                "status": "error",
                "agent": "Unknown",
                "error_message": str(error),
                "error_type": type(error).__name__,
                "recovery_recommendation": "Check system configuration and retry."
            }
        
        # Log the error response
        self.logger.info(f"Error response: {error_response}")
        
        # Determine action based on error handling config
        if self.config["error_handling"] == "fail":
            raise Exception(f"Task failed: {error_response['error_message']}")
        elif self.config["error_handling"] == "retry":
            self.logger.info("Will retry the task after a short delay")
            time.sleep(2)  # Small delay before retry
            return "retry"
        else:  # "continue"
            self.logger.info("Continuing workflow with partial results")
            return error_response
    
    def update_memory(self, key, value):
        """Update the memory store with new information."""
        self.memory[key] = value
        self.logger.debug(f"Memory updated: {key}")
        return self.memory
    
    def get_from_memory(self, key, default=None):
        """Retrieve information from the memory store."""
        return self.memory.get(key, default)
    
    def save_memory(self, filepath=None):
        """Save the current memory to a file."""
        if filepath is None:
            filepath = os.path.join(self.config["output_dir"], "memory.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.memory, f, indent=2)
        
        self.logger.info(f"Memory saved to {filepath}")
        return filepath
    
    def load_memory(self, filepath=None):
        """Load memory from a file."""
        if filepath is None:
            filepath = os.path.join(self.config["output_dir"], "memory.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                loaded_memory = json.load(f)
                for key, value in loaded_memory.items():
                    self.memory[key] = value
            
            self.logger.info(f"Memory loaded from {filepath}")
            return True
        
        self.logger.warning(f"Memory file not found: {filepath}")
        return False
    
    def create_feedback_loop(self, result, context):
        """Create a feedback loop to improve agent performance."""
        self.logger.info("Processing feedback loop...")
        
        # Extract results and update context
        updated_context = context.copy()
        
        # Check if result is a dictionary (typical format)
        if isinstance(result, dict):
            # Extract success/failure indicators
            if "status" in result and result["status"] == "error":
                self.logger.warning("Task returned an error status")
                updated_context["last_error"] = result
            
            # Add results to context for next iteration
            if "insights" in result:
                updated_context["insights"] = result["insights"]
            
            if "recommendations" in result:
                updated_context["recommendations"] = result["recommendations"]
        
        # If result is a string, try to parse as JSON
        elif isinstance(result, str):
            try:
                result_dict = json.loads(result)
                # Recursively call with parsed dict
                return self.create_feedback_loop(result_dict, context)
            except json.JSONDecodeError:
                # Not JSON, just use as is
                updated_context["last_result_text"] = result
        
        # Update feedback metrics if available
        if "metrics" in updated_context:
            metrics = updated_context["metrics"]
            
            # Calculate accuracy if ground truth available
            if "ground_truth" in updated_context and "predictions" in result:
                # Simple accuracy calculation - would be more sophisticated in real system
                correct = sum(1 for a, b in zip(updated_context["ground_truth"], result["predictions"]) if a == b)
                total = len(updated_context["ground_truth"])
                metrics["accuracy"] = correct / total if total > 0 else 0
            
            # Track completion time
            if "start_time" in updated_context:
                metrics["execution_time"] = time.time() - updated_context["start_time"]
            
            updated_context["metrics"] = metrics
        
        # Save feedback for learning
        feedback_path = os.path.join(self.config["output_dir"], "feedback.json")
        with open(feedback_path, 'w') as f:
            json.dump({
                "context": updated_context,
                "result": result,
                "timestamp": time.time()
            }, f, indent=2)
        
        self.logger.info(f"Feedback saved to {feedback_path}")
        return updated_context
    
    def run(self, context: Dict[str, Any]):
        """Run the complete data analysis workflow."""
        self.logger.info("Starting data analysis workflow...")
        
        try:
            # Initialize agents if not already done
            if any(agent is None for agent in self.agents.values()):
                self.initialize_agents()
            
            # Add start time to context for metrics
            context["start_time"] = time.time()
            
            # Initialize metrics in context
            if "metrics" not in context:
                context["metrics"] = {}
            
            # Create tasks based on context
            tasks = self.create_tasks(context)
            
            # Create crew
            crew = self.create_crew(tasks)
            
            # Initialize memory with context
            self.memory = context.copy()
            
            # Execute the crew
            self.logger.info("Executing crew...")
            result = crew.kickoff(inputs=context)
            
            # Process feedback loop
            updated_context = self.create_feedback_loop(result, context)
            
            # Save final memory state
            self.memory.update(updated_context)
            memory_path = self.save_memory()
            
            # Create final output
            output = {
                "status": "success",
                "result": result,
                "memory_path": memory_path,
                "metrics": updated_context.get("metrics", {})
            }
            
            # Save final output
            output_path = os.path.join(self.config["output_dir"], "final_output.json")
            with open(output_path, 'w') as f:
                # Convert result to string if it's not serializable
                if not isinstance(result, (dict, list, str, int, float, bool, type(None))):
                    output["result"] = str(result)
                json.dump(output, f, indent=2)
            
            self.logger.info(f"Workflow completed successfully. Results saved to {output_path}")
            return output
            
        except Exception as e:
            self.logger.exception("Error in workflow execution")
            
            # Create error output
            error_output = {
                "status": "error",
                "error_message": str(e),
                "error_type": type(e).__name__
            }
            
            # Save error output
            error_path = os.path.join(self.config["output_dir"], "error_output.json")
            with open(error_path, 'w') as f:
                json.dump(error_output, f, indent=2)
            
            self.logger.info(f"Error details saved to {error_path}")
            return error_output