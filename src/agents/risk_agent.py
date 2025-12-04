"""
Risk Agent - Computes risk level based on ensemble predictions
"""
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.constants import RiskLevel

logger = get_logger(__name__)


class RiskAgent:
    """
    Agent responsible for computing risk levels from ensemble predictions.
    """
    
    def __init__(
        self,
        high_failure_threshold: float = 0.5,
        critical_rul_threshold: int = 30,
        medium_rul_threshold: int = 60
    ):
        self.name = "RiskAgent"
        self.high_failure_threshold = high_failure_threshold
        self.critical_rul_threshold = critical_rul_threshold
        self.medium_rul_threshold = medium_rul_threshold
        logger.info(f"{self.name} initialized")
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute risk level based on predictions.
        
        Args:
            state: Graph state containing 'predictions'
            
        Returns:
            Updated state with 'risk_assessment' field
        """
        logger.info(f"{self.name} computing risk level...")
        
        try:
            predictions = state.get("predictions")
            
            if predictions is None:
                raise ValueError("No predictions found in state")
            
            ensemble = predictions.get("ensemble", {})
            avg_rul = ensemble.get("avg_rul", 100)
            max_failure_prob = ensemble.get("max_failure_probability", 0)
            
            # Compute risk level
            risk_level, risk_score = self._compute_risk_level(
                avg_rul,
                max_failure_prob
            )
            
            # Generate justification
            justification = self._generate_justification(
                risk_level,
                avg_rul,
                max_failure_prob,
                predictions
            )
            
            # Compute detailed metrics
            risk_factors = self._identify_risk_factors(
                avg_rul,
                max_failure_prob,
                predictions
            )
            
            risk_assessment = {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "avg_rul": avg_rul,
                "max_failure_probability": max_failure_prob,
                "justification": justification,
                "risk_factors": risk_factors,
                # Simple confidence heuristic: 1 - risk_score, clamped to [0, 1]
                "confidence_score": max(0.0, min(1.0, 1.0 - float(risk_score))),
            }
            
            logger.info(f"Risk assessment completed: {risk_level} (score: {risk_score:.2f})")
            
            # Update state
            state["risk_assessment"] = risk_assessment
            state["agent_outputs"]["risk"] = {
                "agent": self.name,
                "status": "success",
                "data": risk_assessment
            }
            
            return state
            
        except Exception as e:
            logger.error(f"{self.name} failed: {str(e)}")
            state["agent_outputs"]["risk"] = {
                "agent": self.name,
                "status": "error",
                "error": str(e)
            }
            return state
    
    def _compute_risk_level(
        self,
        avg_rul: float,
        max_failure_prob: float
    ) -> tuple[str, float]:
        """
        Compute risk level and numerical risk score (0-1).
        
        Risk Logic:
        - HIGH: max_failure_prob > 0.5 OR avg_rul < 30
        - MEDIUM: 30 <= avg_rul < 60
        - LOW: otherwise
        """
        # Compute numerical risk score (0-1)
        rul_risk = max(0, min(1, 1 - (avg_rul / 100)))
        prob_risk = max_failure_prob
        risk_score = max(rul_risk, prob_risk)
        
        # Determine categorical risk level
        if max_failure_prob > self.high_failure_threshold or avg_rul < self.critical_rul_threshold:
            risk_level = RiskLevel.HIGH
        elif avg_rul < self.medium_rul_threshold:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return risk_level, risk_score
    
    def _generate_justification(
        self,
        risk_level: str,
        avg_rul: float,
        max_failure_prob: float,
        predictions: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable justification for the risk assessment.
        """
        justifications = []
        
        if max_failure_prob > self.high_failure_threshold:
            justifications.append(
                f"High failure probability detected ({max_failure_prob:.1%})"
            )
        
        if avg_rul < self.critical_rul_threshold:
            justifications.append(
                f"Critical RUL threshold reached ({avg_rul:.0f} cycles)"
            )
        elif avg_rul < self.medium_rul_threshold:
            justifications.append(
                f"RUL below medium threshold ({avg_rul:.0f} cycles)"
            )
        else:
            justifications.append(
                f"RUL within acceptable range ({avg_rul:.0f} cycles)"
            )
        
        # Add model agreement analysis
        fd001_rul = predictions.get("fd001", {}).get("rul", 0)
        fd002_rul = predictions.get("fd002", {}).get("rul", 0)
        fd003_rul = predictions.get("fd003", {}).get("rul", 0)
        
        rul_std = self._std_dev([fd001_rul, fd002_rul, fd003_rul])
        if rul_std < 10:
            justifications.append("All models show strong agreement")
        elif rul_std > 30:
            # Find outliers or disagreement pattern
            ruls = {"FD001": fd001_rul, "FD002": fd002_rul, "FD003": fd003_rul}
            min_model = min(ruls, key=ruls.get)
            max_model = max(ruls, key=ruls.get)
            justifications.append(f"Significant model disagreement ({min_model} predicts {ruls[min_model]:.0f}, {max_model} predicts {ruls[max_model]:.0f})")
        
        return f"{risk_level} risk: " + "; ".join(justifications) + "."
    
    def _identify_risk_factors(
        self,
        avg_rul: float,
        max_failure_prob: float,
        predictions: Dict[str, Any]
    ) -> list[str]:
        """
        Identify specific risk factors contributing to the assessment.
        """
        factors = []
        
        if max_failure_prob > 0.7:
            factors.append("CRITICAL_FAILURE_PROBABILITY")
        elif max_failure_prob > 0.5:
            factors.append("HIGH_FAILURE_PROBABILITY")
        
        if avg_rul < 20:
            factors.append("CRITICAL_RUL")
        elif avg_rul < 30:
            factors.append("LOW_RUL")
        
        # Check for model disagreement
        fd001_rul = predictions.get("fd001", {}).get("rul", 0)
        fd002_rul = predictions.get("fd002", {}).get("rul", 0)
        fd003_rul = predictions.get("fd003", {}).get("rul", 0)
        
        rul_std = self._std_dev([fd001_rul, fd002_rul, fd003_rul])
        if rul_std > 30:
            factors.append("MODEL_DISAGREEMENT")
        
        if not factors:
            factors.append("NORMAL_OPERATION")
        
        return factors
    
    @staticmethod
    def _std_dev(values: list[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


def risk_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for RiskAgent.
    """
    agent = RiskAgent()
    return agent.run(state)