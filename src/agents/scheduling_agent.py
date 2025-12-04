"""
Scheduling Agent - Produces maintenance window recommendations
"""
from typing import Dict, Any
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.constants import RiskLevel, MaintenanceWindow

logger = get_logger(__name__)


class SchedulingAgent:
    """
    Agent responsible for scheduling maintenance windows based on risk assessment.
    """
    
    def __init__(self):
        self.name = "SchedulingAgent"
        logger.info(f"{self.name} initialized")
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate maintenance scheduling recommendation.
        
        Args:
            state: Graph state containing 'risk_assessment' and 'diagnosis'
            
        Returns:
            Updated state with 'maintenance_schedule' field
        """
        logger.info(f"{self.name} generating maintenance schedule...")
        
        try:
            risk_assessment = state.get("risk_assessment")
            diagnosis = state.get("diagnosis", {})
            
            if risk_assessment is None:
                raise ValueError("No risk assessment found in state")
            
            risk_level = risk_assessment.get("risk_level", RiskLevel.LOW)
            avg_rul = risk_assessment.get("avg_rul", 100)
            max_failure_prob = risk_assessment.get("max_failure_probability", 0)
            
            # Determine maintenance window
            maintenance_window = self._determine_maintenance_window(
                risk_level,
                avg_rul,
                max_failure_prob
            )
            
            # Calculate recommended timeline
            timeline = self._calculate_timeline(
                maintenance_window,
                avg_rul,
                risk_level
            )
            
            # Generate scheduling rationale
            rationale = self._generate_rationale(
                maintenance_window,
                risk_level,
                avg_rul,
                diagnosis
            )
            
            # Generate actionable recommendations
            recommendations = self._generate_recommendations(
                maintenance_window,
                diagnosis.get("probable_component", "Unknown"),
                risk_assessment
            )
            
            maintenance_schedule = {
                "maintenance_window": maintenance_window,
                "timeline": timeline,
                "rationale": rationale,
                "recommendations": recommendations,
                "priority": self._compute_priority(maintenance_window)
            }
            
            logger.info(f"Maintenance schedule completed: {maintenance_window}")
            
            # Update state
            state["maintenance_schedule"] = maintenance_schedule
            state["agent_outputs"]["scheduling"] = {
                "agent": self.name,
                "status": "success",
                "data": maintenance_schedule
            }
            
            return state
            
        except Exception as e:
            logger.error(f"{self.name} failed: {str(e)}")
            state["agent_outputs"]["scheduling"] = {
                "agent": self.name,
                "status": "error",
                "error": str(e)
            }
            return state
    
    def _determine_maintenance_window(
        self,
        risk_level: str,
        avg_rul: float,
        max_failure_prob: float
    ) -> str:
        """
        Determine maintenance window based on risk factors.
        
        Logic:
        - IMMEDIATE: HIGH risk
        - SOON: MEDIUM risk
        - ROUTINE: LOW risk
        """
        if risk_level == RiskLevel.HIGH:
            return MaintenanceWindow.IMMEDIATE
        elif risk_level == RiskLevel.MEDIUM:
            return MaintenanceWindow.SOON
        else:
            return MaintenanceWindow.ROUTINE
    
    def _calculate_timeline(
        self,
        maintenance_window: str,
        avg_rul: float,
        risk_level: str
    ) -> Dict[str, Any]:
        """
        Calculate specific timeline recommendations.
        """
        now = datetime.now()
        
        if maintenance_window == MaintenanceWindow.IMMEDIATE:
            # Schedule within 24-48 hours
            target_date = now + timedelta(hours=24)
            deadline = now + timedelta(hours=48)
            buffer_cycles = min(10, int(avg_rul * 0.2))
            
        elif maintenance_window == MaintenanceWindow.SOON:
            # Schedule within 1-2 weeks
            target_date = now + timedelta(days=7)
            deadline = now + timedelta(days=14)
            buffer_cycles = min(20, int(avg_rul * 0.3))
            
        else:  # ROUTINE
            # Schedule within normal maintenance cycle (30-60 days)
            target_date = now + timedelta(days=30)
            deadline = now + timedelta(days=60)
            buffer_cycles = min(40, int(avg_rul * 0.4))
        
        return {
            "target_date": target_date.strftime("%Y-%m-%d %H:%M"),
            "deadline": deadline.strftime("%Y-%m-%d %H:%M"),
            "estimated_rul_at_maintenance": avg_rul - buffer_cycles,
            "buffer_cycles": buffer_cycles
        }
    
    def _generate_rationale(
        self,
        maintenance_window: str,
        risk_level: str,
        avg_rul: float,
        diagnosis: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable scheduling rationale.
        """
        component = diagnosis.get("probable_component", "Unknown")
        
        if maintenance_window == MaintenanceWindow.IMMEDIATE:
            return (
                f"{risk_level} risk assessment requires immediate attention. "
                f"Component '{component}' showing signs of imminent failure "
                f"with RUL of {avg_rul:.0f} cycles. "
                "Schedule maintenance within 24-48 hours to prevent unplanned downtime."
            )
        
        elif maintenance_window == MaintenanceWindow.SOON:
            return (
                f"{risk_level} risk assessment suggests proactive maintenance. "
                f"Component '{component}' degradation detected with RUL of {avg_rul:.0f} cycles. "
                "Schedule maintenance within 1-2 weeks to optimize maintenance costs."
            )
        
        else:  # ROUTINE
            return (
                f"{risk_level} risk assessment indicates normal operation. "
                f"Component '{component}' operating within normal parameters (RUL: {avg_rul:.0f} cycles). "
                "Schedule maintenance during next routine maintenance window."
            )
    
    def _generate_recommendations(
        self,
        maintenance_window: str,
        component: str,
        risk_assessment: Dict[str, Any]
    ) -> list[str]:
        """
        Generate actionable maintenance recommendations.
        """
        recommendations = []
        
        if maintenance_window == MaintenanceWindow.IMMEDIATE:
            recommendations.extend([
                f"Inspect {component} immediately for signs of failure",
                "Prepare replacement parts and tooling",
                "Schedule skilled technician availability",
                "Consider taking unit offline if failure risk is critical",
                "Document all observations for failure analysis"
            ])
        
        elif maintenance_window == MaintenanceWindow.SOON:
            recommendations.extend([
                f"Order replacement parts for {component}",
                "Schedule maintenance during next available window",
                "Monitor sensor readings for any rapid degradation",
                "Prepare maintenance procedures and safety protocols",
                "Allocate appropriate maintenance budget"
            ])
        
        else:  # ROUTINE
            recommendations.extend([
                f"Include {component} inspection in next routine maintenance",
                "Continue normal monitoring schedule",
                "Update maintenance records",
                "Plan for eventual replacement in maintenance forecast"
            ])
        
        # Add risk-specific recommendations
        risk_factors = risk_assessment.get("risk_factors", [])
        if "MODEL_DISAGREEMENT" in risk_factors:
            recommendations.append(
                "Note: Model disagreement detected - consider additional diagnostics"
            )
        
        return recommendations
    
    def _compute_priority(self, maintenance_window: str) -> int:
        """
        Compute numerical priority (1=highest, 3=lowest).
        """
        priority_map = {
            MaintenanceWindow.IMMEDIATE: 1,
            MaintenanceWindow.SOON: 2,
            MaintenanceWindow.ROUTINE: 3
        }
        return priority_map.get(maintenance_window, 3)


def scheduling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for SchedulingAgent.
    """
    agent = SchedulingAgent()
    return agent.run(state)