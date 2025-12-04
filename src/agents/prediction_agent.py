"""
Prediction Agent - Calls unified_inference() and returns all model outputs
"""
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.inference.unified_inference import unified_inference
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionAgent:
    """
    Agent responsible for running predictions using the unified inference pipeline.
    """
    
    def __init__(self):
        self.name = "PredictionAgent"
        logger.info(f"{self.name} initialized")
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute predictions on raw observation data.
        
        Args:
            state: Graph state containing 'raw_observation'
            
        Returns:
            Updated state with 'predictions' field
        """
        logger.info(f"{self.name} running predictions...")
        
        try:
            raw_observation = state.get("raw_observation")
            
            if raw_observation is None:
                raise ValueError("No raw_observation found in state")
            
            # Call unified inference
            predictions = unified_inference(raw_observation)
            
            logger.info(f"Predictions completed: {predictions}")
            
            # Update state
            state["predictions"] = predictions
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["prediction"] = {
                "agent": self.name,
                "status": "success",
                "data": predictions
            }
            
            return state
            
        except Exception as e:
            logger.error(f"{self.name} failed: {str(e)}")
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["prediction"] = {
                "agent": self.name,
                "status": "error",
                "error": str(e)
            }
            state["error"] = str(e)
            return state


def prediction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for PredictionAgent.
    """
    agent = PredictionAgent()
    return agent.run(state)