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
from src.utils.constants import ComponentType

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
            
            # Infer human-readable anomalies from the probable component
            anomalies = self._infer_anomalies(probable_component, component_probs)

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
                "anomalies": anomalies,
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

        This agent is grounded in the FD003 fault-mode mapping:
        - Component 0 → Healthy (no failure)
        - Component 1 → HPC Degradation (High-Pressure Compressor)
        - Component 2 → Fan Degradation (turbofan fan blades)
        """
        fd003_mapping = {
            "0": "Healthy (no component failure)",
            "1": "HPC Degradation (High-Pressure Compressor)",
            "2": "Fan Degradation (turbofan fan blades)",
        }
        probable_desc = fd003_mapping.get(str(probable_component), str(probable_component))
        predicted_desc = fd003_mapping.get(str(predicted_component), str(predicted_component))

        prompt = f"""
        You are a Diagnostic Agent for NASA's CMAPSS FD003 turbofan dataset.

        FAULT MODE MAPPING (FD003):
        - 0 → Healthy (no component failure)
        - 1 → HPC Degradation (High-Pressure Compressor losing efficiency)
        - 2 → Fan Degradation (turbofan fan blades degrading or damaged)

        CONTEXT:
        - Probable Component (decision): {probable_component} → {probable_desc}
        - Raw Model Prediction (classifier argmax): {predicted_component} → {predicted_desc}
        - Retrieved Similar Cases: {similar_cases}

        TASK:
        - Explain clearly why the system decided on {probable_desc} as the failing mode.
        - Explicitly mention whether this corresponds to Healthy, HPC degradation, or Fan degradation.
        - Use retrieved cases (and their similarity scores) as evidence when possible.
        - If the model prediction (argmax) differs from history, describe how the conflict was resolved.
        - Keep the explanation compact (2–4 sentences) but technically precise.
        """

        return llm_service.generate_text(prompt)

    def _infer_anomalies(
        self,
        probable_component: str,
        component_probs: Dict[str, float]
    ) -> list[str]:
        """
        Heuristically infer human-readable anomaly tags from the FD003 component prediction.

        Component interpretation (FD003):
        - 0 → Healthy (no anomaly)
        - 1 → HPC Degradation
        - 2 → Fan Degradation
        """
        anomalies: list[str] = []

        label = str(probable_component)
        max_prob = 0.0
        if component_probs:
            try:
                max_prob = float(max(component_probs.values()))
            except (TypeError, ValueError):
                max_prob = 0.0

        # Only emit anomaly tags when the classifier has at least moderate confidence
        if max_prob < 0.4:
            return anomalies

        if label in ("1", ComponentType.HPC, "HPC"):
            anomalies.append("HPC degradation signature detected")
        elif label in ("2", ComponentType.FAN, "Fan"):
            anomalies.append("Fan degradation signature detected")

        return anomalies
    
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