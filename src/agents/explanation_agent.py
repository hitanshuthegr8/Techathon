"""
Explanation Agent - Produces final natural-language diagnostic report
"""
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.llm_service import llm_service

logger = get_logger(__name__)


class ExplanationAgent:
    """
    Agent responsible for generating comprehensive natural-language diagnostic reports.
    """
    
    def __init__(self):
        self.name = "ExplanationAgent"
        logger.info(f"{self.name} initialized")
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final diagnostic report combining all agent outputs.
        
        Args:
            state: Graph state containing outputs from all previous agents
            
        Returns:
            Updated state with 'final_report' field
        """
        logger.info(f"{self.name} generating final report...")
        
        try:
            predictions = state.get("predictions", {})
            diagnosis = state.get("diagnosis", {})
            risk_assessment = state.get("risk_assessment", {})
            maintenance_schedule = state.get("maintenance_schedule", {})
            
            # Generate comprehensive report sections
            summary = self._generate_summary(
                risk_assessment,
                diagnosis,
                maintenance_schedule
            )
            
            detailed_findings = self._generate_detailed_findings(
                predictions,
                diagnosis,
                risk_assessment
            )
            
            maintenance_plan = self._generate_maintenance_plan(
                maintenance_schedule,
                diagnosis
            )
            
            technical_details = self._generate_technical_details(
                predictions,
                diagnosis,
                risk_assessment
            )
            
            # Compile full report
            final_report = {
                "summary": summary,
                "detailed_findings": detailed_findings,
                "maintenance_plan": maintenance_plan,
                "technical_details": technical_details,
                "report_timestamp": self._get_timestamp(),
                "report_id": self._generate_report_id(state)
            }
            
            # Generate human-readable narrative
            narrative = self._generate_narrative_report(final_report)
            final_report["narrative"] = narrative
            
            logger.info(f"Final report generated successfully")
            
            # Update state
            state["final_report"] = final_report
            state["agent_outputs"]["explanation"] = {
                "agent": self.name,
                "status": "success",
                "data": final_report
            }
            
            return state
            
        except Exception as e:
            logger.error(f"{self.name} failed: {str(e)}")
            state["agent_outputs"]["explanation"] = {
                "agent": self.name,
                "status": "error",
                "error": str(e)
            }
            return state
    
    def _generate_summary(
        self,
        risk_assessment: Dict[str, Any],
        diagnosis: Dict[str, Any],
        maintenance_schedule: Dict[str, Any]
    ) -> str:
        """
        Generate executive summary.
        """
        risk_level = risk_assessment.get("risk_level", "UNKNOWN")
        component = diagnosis.get("probable_component", "Unknown")
        window = maintenance_schedule.get("maintenance_window", "ROUTINE")
        avg_rul = risk_assessment.get("avg_rul", 0)
        
        return (
            f"RISK LEVEL: {risk_level}\n"
            f"Component Analysis: {component} degradation detected\n"
            f"Remaining Useful Life: {avg_rul:.0f} cycles\n"
            f"Maintenance Window: {window}\n"
            f"Recommendation: {maintenance_schedule.get('rationale', 'See detailed analysis')}"
        )
    
    def _generate_detailed_findings(
        self,
        predictions: Dict[str, Any],
        diagnosis: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> str:
        """
        Generate detailed analysis findings.
        """
        ensemble = predictions.get("ensemble", {})
        avg_rul = ensemble.get("avg_rul", 0)
        max_failure_prob = ensemble.get("max_failure_probability", 0)
        
        component = diagnosis.get("probable_component", "Unknown")
        confidence = diagnosis.get("confidence", 0)
        reason = diagnosis.get("reason", "No diagnostic reasoning available")
        
        findings = [
            "=== PREDICTIVE ANALYSIS ===",
            f"Ensemble RUL Estimate: {avg_rul:.0f} cycles",
            f"Maximum Failure Probability: {max_failure_prob:.1%}",
            "",
            "=== DIAGNOSTIC FINDINGS ===",
            f"Identified Component: {component}",
            f"Diagnostic Confidence: {confidence:.1%}",
            f"Reasoning: {reason}",
            "",
            "=== RISK FACTORS ===",
            risk_assessment.get("justification", "No risk assessment available")
        ]
        
        # Add model-specific predictions
        findings.append("")
        findings.append("=== MODEL PREDICTIONS ===")
        for model_name in ["fd001", "fd002", "fd003"]:
            if model_name in predictions:
                model_data = predictions[model_name]
                rul = model_data.get("rul", 0)
                fail_prob = model_data.get("failure_probability", 0)
                findings.append(f"{model_name.upper()}: RUL={rul:.0f}, P(failure)={fail_prob:.1%}")
        
        # Add component probabilities if available
        comp_probs = diagnosis.get("component_probabilities", {})
        if comp_probs:
            findings.append("")
            findings.append("=== COMPONENT PROBABILITIES ===")
            for comp, prob in sorted(comp_probs.items(), key=lambda x: x[1], reverse=True):
                findings.append(f"  {comp}: {prob:.1%}")
        
        return "\n".join(findings)
    
    def _generate_maintenance_plan(
        self,
        maintenance_schedule: Dict[str, Any],
        diagnosis: Dict[str, Any]
    ) -> str:
        """
        Generate maintenance action plan.
        """
        window = maintenance_schedule.get("maintenance_window", "ROUTINE")
        timeline = maintenance_schedule.get("timeline", {})
        recommendations = maintenance_schedule.get("recommendations", [])
        
        plan = [
            "=== MAINTENANCE ACTION PLAN ===",
            f"Priority Level: {maintenance_schedule.get('priority', 'N/A')}",
            f"Maintenance Window: {window}",
            f"Target Date: {timeline.get('target_date', 'Not specified')}",
            f"Deadline: {timeline.get('deadline', 'Not specified')}",
            "",
            "=== ACTIONABLE RECOMMENDATIONS ==="
        ]
        
        for i, rec in enumerate(recommendations, 1):
            plan.append(f"{i}. {rec}")
        
        # Add similar case insights
        similar_cases = diagnosis.get("similar_cases", [])
        if similar_cases:
            plan.append("")
            plan.append("=== HISTORICAL INSIGHTS ===")
            plan.append(f"Found {len(similar_cases)} similar failure patterns:")
            for i, case in enumerate(similar_cases[:3], 1):
                comp = case.get("component", "Unknown")
                sim = case.get("similarity", 0)
                plan.append(f"  {i}. {comp} failure (similarity: {sim:.1%})")
        
        return "\n".join(plan)
    
    def _generate_technical_details(
        self,
        predictions: Dict[str, Any],
        diagnosis: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate technical metadata and details.
        """
        return {
            "ensemble_metrics": predictions.get("ensemble", {}),
            "model_predictions": {
                k: v for k, v in predictions.items() if k != "ensemble"
            },
            "diagnosis_metadata": {
                "confidence": diagnosis.get("confidence", 0),
                "similar_cases_count": len(diagnosis.get("similar_cases", [])),
                "predicted_component": diagnosis.get("predicted_component", "Unknown"),
                "probable_component": diagnosis.get("probable_component", "Unknown")
            },
            "risk_metrics": {
                "risk_level": risk_assessment.get("risk_level", "UNKNOWN"),
                "risk_score": risk_assessment.get("risk_score", 0),
                "risk_factors": risk_assessment.get("risk_factors", [])
            }
        }
    
    def _generate_narrative_report(self, report: Dict[str, Any]) -> str:
        """
        Generate flowing narrative-style report using Gemini.
        """
        logger.info("Generating narrative report using Gemini...")
        
        # Construct prompt with all available data
        # We use the technical details and other sections as context
        prompt = f"""
        You are an expert Predictive Maintenance Analyst for an industrial plant. 
        Generate a comprehensive, professional diagnostic report based on the following analysis data.
        
        ANALYSIS DATA:
        {report}
        
        INSTRUCTIONS:
        1. Write a professional Executive Summary.
        2. Provide a Detailed Analysis of the findings, explaining the root causes and risk factors.
        3. Outline a clear Maintenance Action Plan.
        4. Use a professional, authoritative, yet helpful tone.
        5. Format with clear headings (e.g., "Executive Summary", "Technical Analysis", "Recommendations") and bullet points.
        6. Do NOT use markdown code blocks. Just return the formatted text.
        7. If the risk is HIGH, emphasize the urgency.
        """
        
        return llm_service.generate_text(prompt)
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def _generate_report_id(state: Dict[str, Any]) -> str:
        """Generate unique report ID."""
        from datetime import datetime
        import hashlib
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        state_hash = hashlib.md5(str(state).encode()).hexdigest()[:8]
        return f"RPT-{timestamp}-{state_hash}"


def explanation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for ExplanationAgent.
    """
    agent = ExplanationAgent()
    return agent.run(state)