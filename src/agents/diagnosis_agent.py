"""
Diagnosis Agent - Uses vector DB to find similar past failures and identify root cause
"""
from typing import Dict, Any, List
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.vector_db.query import query_similar_failures
from src.utils.logger import get_logger
from src.utils.llm_service import llm_service

logger = get_logger(__name__)


class DiagnosisAgent:
    """
    Agent responsible for diagnosing failures using historical similarity search.
    """
    
    def __init__(self, top_k: int = 5):
        self.name = "DiagnosisAgent"
        self.top_k = top_k
        logger.info(f"{self.name} initialized with top_k={top_k}")
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Diagnose potential failures using vector similarity search.
        
        Args:
            state: Graph state containing 'predictions' and 'raw_observation'
            
        Returns:
            Updated state with 'diagnosis' field
        """
        logger.info(f"{self.name} running diagnosis...")
        
        try:
            predictions = state.get("predictions")
            raw_observation = state.get("raw_observation")
            
            if predictions is None:
                raise ValueError("No predictions found in state")
            
            # Extract component prediction from FD003 if available
            fd003_data = predictions.get("fd003", {})
            predicted_component = fd003_data.get("predicted_component", "Unknown")
            component_probs = fd003_data.get("component_probs", {})
            
            # Query vector database for similar past failures
            similar_cases = query_similar_failures(
                observation=raw_observation,
                predictions=predictions,
                top_k=self.top_k
            )
            
            # Analyze similar cases to determine probable component
            probable_component = self._determine_probable_component(
                predicted_component,
                component_probs,
                similar_cases
            )
            
            # Generate reasoning
            reason = self._generate_reasoning(
                probable_component,
                predicted_component,
                similar_cases
            )
            
            diagnosis = {
                "probable_component": probable_component,
                "predicted_component": predicted_component,
                "component_probabilities": component_probs,
                "similar_cases": similar_cases,
                "reason": reason,
                "confidence": self._calculate_confidence(similar_cases, component_probs)
            }
            
            logger.info(f"Diagnosis completed: {probable_component}")
            
            # Update state
            state["diagnosis"] = diagnosis
            state["agent_outputs"]["diagnosis"] = {
                "agent": self.name,
                "status": "success",
                "data": diagnosis
            }
            
            return state
            
        except Exception as e:
            logger.error(f"{self.name} failed: {str(e)}")
            state["agent_outputs"]["diagnosis"] = {
                "agent": self.name,
                "status": "error",
                "error": str(e)
            }
            return state
    
    def _determine_probable_component(
        self,
        predicted_component: str,
        component_probs: Dict[str, float],
        similar_cases: List[Dict]
    ) -> str:
        """
        Determine most probable failing component based on predictions and history.

        Guardrails:
        - Only trust the model prediction when its probability is high.
        - Prefer majority vote from similar historical cases when available.
        - When confidence is low and history is weak, fall back to a general diagnosis
          instead of over-committing to a specific component (e.g., always '2').
        """
        max_model_prob = 0.0
        if component_probs:
            try:
                max_model_prob = float(max(component_probs.values()))
            except (TypeError, ValueError):
                max_model_prob = 0.0

        # If we have a high-confidence prediction from the classifier, use it
        if max_model_prob >= 0.7 and predicted_component != "Unknown":
            return predicted_component
        
        # Otherwise, use majority vote from similar cases when available
        if similar_cases:
            component_counts: Dict[str, int] = {}
            for case in similar_cases:
                comp = case.get("component", "Unknown")
                component_counts[comp] = component_counts.get(comp, 0) + 1
            
            if component_counts:
                return max(component_counts.items(), key=lambda x: x[1])[0]
        
        # Low-confidence guardrail: if the model itself is low-confidence and
        # history is inconclusive, avoid over-confident specific labels.
        if max_model_prob < 0.5 or not component_probs:
            return "General"
        
        # Fallback to predicted component when it's at least somewhat supported
        return predicted_component if predicted_component != "Unknown" else "General"
    
    def _generate_reasoning(
        self,
        probable_component: str,
        predicted_component: str,
        similar_cases: List[Dict]
    ) -> str:
        """
        Generate human-readable reasoning for the diagnosis using Gemini.
        """
        prompt = f"""
        You are a Diagnostic Agent. Explain the reasoning for identifying '{probable_component}' as the failing component.
        
        CONTEXT:
        - Model Prediction: {predicted_component}
        - Historical Similar Cases: {similar_cases}
        
        INSTRUCTIONS:
        - Explain why {probable_component} was chosen.
        - Cite the historical data evidence (similarity scores).
        - If the model prediction differs from history, explain how you resolved the conflict (usually history/majority vote wins).
        - Keep it concise (2-3 sentences).
        """
        
        return llm_service.generate_text(prompt)
    
    def _calculate_confidence(
        self,
        similar_cases: List[Dict],
        component_probs: Dict[str, float]
    ) -> float:
        """
        Calculate confidence score based on model probabilities and similarity
        of retrieved cases.

        - Primary signal: maximum component probability from the classifier.
        - Secondary signal: average similarity of retrieved cases.

        Both are clamped into [0, 1] and combined to avoid NaNs.
        """
        max_prob = 0.0
        if component_probs:
            try:
                max_prob = float(max(component_probs.values()))
                if not math.isfinite(max_prob):
                    max_prob = 0.0
            except (TypeError, ValueError):
                max_prob = 0.0

        # Historical similarity signal
        if similar_cases:
            avg_similarity = sum(
                float(c.get("similarity", 0.0)) for c in similar_cases
            ) / max(len(similar_cases), 1)
        else:
            avg_similarity = 0.5  # neutral baseline when no history

        # Clamp signals to [0, 1]
        max_prob = max(0.0, min(1.0, max_prob))
        avg_similarity = max(0.0, min(1.0, avg_similarity))

        # Weighted combination – emphasize classifier probability but keep
        # some contribution from history.
        confidence = 0.7 * max_prob + 0.3 * avg_similarity
        confidence = max(0.0, min(1.0, confidence))

        return float(confidence)


def diagnosis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for DiagnosisAgent.
    """
    agent = DiagnosisAgent(top_k=5)
    return agent.run(state)