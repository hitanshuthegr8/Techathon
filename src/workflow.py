"""
LangGraph Workflow - Orchestrates the multi-agent predictive maintenance pipeline
"""
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.prediction_agent import prediction_node
from src.agents.diagnosis_agent import diagnosis_node
from src.agents.risk_agent import risk_node
from src.agents.scheduling_agent import scheduling_node
from src.agents.explanation_agent import explanation_node
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MaintenanceState(TypedDict):
    """
    State schema for the maintenance workflow graph.
    """
    # Input
    raw_observation: Any
    
    # Agent outputs
    predictions: Dict[str, Any]
    diagnosis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    maintenance_schedule: Dict[str, Any]
    final_report: Dict[str, Any]
    
    # Metadata
    agent_outputs: Dict[str, Any]
    error: str


def create_maintenance_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for predictive maintenance.
    
    Returns:
        Configured StateGraph
    """
    logger.info("Creating maintenance workflow graph...")
    
    # Initialize the graph with state schema
    workflow = StateGraph(MaintenanceState)
    
    # Add nodes for each agent
    workflow.add_node("prediction", prediction_node)
    workflow.add_node("diagnosis", diagnosis_node)
    workflow.add_node("risk", risk_node)
    workflow.add_node("scheduling", scheduling_node)
    workflow.add_node("explanation", explanation_node)
    
    # Define the workflow edges (sequential execution)
    workflow.set_entry_point("prediction")
    workflow.add_edge("prediction", "diagnosis")
    workflow.add_edge("diagnosis", "risk")
    workflow.add_edge("risk", "scheduling")
    workflow.add_edge("scheduling", "explanation")
    workflow.add_edge("explanation", END)
    
    logger.info("Workflow graph created successfully")
    
    return workflow


def compile_workflow() -> Any:
    """
    Compile the workflow graph for execution.
    
    Returns:
        Compiled graph ready for execution
    """
    workflow = create_maintenance_workflow()
    compiled_graph = workflow.compile()
    logger.info("Workflow compiled successfully")
    return compiled_graph


def run_maintenance_analysis(
    raw_observation: Any,
    compiled_graph: Any = None
) -> Dict[str, Any]:
    """
    Run the complete maintenance analysis workflow.
    
    Args:
        raw_observation: Raw sensor observation data
        compiled_graph: Pre-compiled graph (will compile if None)
        
    Returns:
        Final state with all analysis results
    """
    logger.info("=" * 80)
    logger.info("Starting maintenance analysis workflow")
    logger.info("=" * 80)
    
    try:
        # Compile graph if not provided
        if compiled_graph is None:
            compiled_graph = compile_workflow()
        
        # Initialize state
        initial_state = {
            "raw_observation": raw_observation,
            "predictions": {},
            "diagnosis": {},
            "risk_assessment": {},
            "maintenance_schedule": {},
            "final_report": {},
            "agent_outputs": {},
            "error": ""
        }
        
        # Execute the workflow
        logger.info("Executing workflow...")
        final_state = compiled_graph.invoke(initial_state)
        
        # Check for errors
        if final_state.get("error"):
            logger.error(f"Workflow completed with errors: {final_state['error']}")
        else:
            logger.info("Workflow completed successfully")
        
        logger.info("=" * 80)
        
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        raise


def print_analysis_results(state: Dict[str, Any]) -> None:
    """
    Pretty print the analysis results.
    
    Args:
        state: Final workflow state
    """
    print("\n" + "=" * 80)
    print("MAINTENANCE ANALYSIS RESULTS")
    print("=" * 80)
    
    # Print summary
    risk = state.get("risk_assessment", {})
    diagnosis = state.get("diagnosis", {})
    schedule = state.get("maintenance_schedule", {})
    
    print(f"\nRisk Level: {risk.get('risk_level', 'UNKNOWN')}")
    print(f"Component: {diagnosis.get('probable_component', 'Unknown')}")
    print(f"RUL: {risk.get('avg_rul', 0):.0f} cycles")
    print(f"Maintenance Window: {schedule.get('maintenance_window', 'ROUTINE')}")
    
    # Print full narrative report if available
    final_report = state.get("final_report", {})
    if "narrative" in final_report:
        print("\n" + "=" * 80)
        print(final_report["narrative"])
    
    print("\n" + "=" * 80)


def get_workflow_visualization() -> str:
    """
    Get a text representation of the workflow structure.
    
    Returns:
        Workflow visualization string
    """
    visualization = """
    Predictive Maintenance Workflow
    ================================
    
    [Raw Observation]
           |
           v
    [1. PredictionAgent]
       - Calls unified_inference()
       - Returns FD001/FD002/FD003 predictions
           |
           v
    [2. DiagnosisAgent]
       - Vector DB similarity search
       - Identifies probable component
       - Finds similar past failures
           |
           v
    [3. RiskAgent]
       - Computes risk level (HIGH/MEDIUM/LOW)
       - Analyzes ensemble predictions
       - Generates risk score
           |
           v
    [4. SchedulingAgent]
       - Determines maintenance window
       - IMMEDIATE / SOON / ROUTINE
       - Provides timeline & recommendations
           |
           v
    [5. ExplanationAgent]
       - Generates comprehensive report
       - Natural language summary
       - Technical details & action items
           |
           v
    [Final Diagnostic Report]
    """
    return visualization


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Setup logging
    from src.utils.logger import setup_logging
    setup_logging(log_level="INFO")
    
    # Print workflow structure
    print(get_workflow_visualization())
    
    # Create sample observation for testing
    sample_observation = np.random.randn(24)  # 24 sensor readings
    
    # Run workflow
    print("\nRunning test workflow with sample data...")
    try:
        final_state = run_maintenance_analysis(sample_observation)
        print_analysis_results(final_state)
    except Exception as e:
        print(f"Test failed: {str(e)}")
