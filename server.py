"""
Flask Server - Exposes the Predictive Maintenance Agents via REST API
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.workflow import run_maintenance_analysis
from src.utils.logger import setup_logging

# Setup logging
setup_logging(log_level="DEBUG")
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.DEBUG)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes and origins

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "predictive-maintenance-system"})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze sensor data using the multi-agent workflow.
    
    Expected JSON payload:
    {
        "observation": [float, float, ...]  # Array of sensor readings
    }
    """
    try:
        logger.debug(f"Request Headers: {request.headers}")
        data = request.get_json()
        logger.debug(f"Request Payload: {data}")
        
        if not data or "observation" not in data:
            return jsonify({"error": "Missing 'observation' in payload"}), 400
        
        raw_input = data["observation"]
        logger.debug(f"Raw input type: {type(raw_input)}")
        
        # Handle string input (e.g. "-0.0007, -0.0004, ...")
        if isinstance(raw_input, str):
            logger.debug("Parsing string input...")
            try:
                # Remove brackets if present and split
                cleaned = raw_input.strip("[] ")
                if not cleaned:
                    observation_list = []
                else:
                    observation_list = [float(x.strip()) for x in cleaned.split(',')]
            except ValueError as e:
                return jsonify({"error": f"Invalid number format in string input: {str(e)}"}), 400
        # Handle list input
        elif isinstance(raw_input, list):
            logger.debug("Parsing list input...")
            try:
                observation_list = [float(x) for x in raw_input]
            except ValueError as e:
                return jsonify({"error": f"Invalid number format in list input: {str(e)}"}), 400
        else:
            return jsonify({"error": f"Unsupported input type: {type(raw_input)}"}), 400

        observation = np.array(observation_list, dtype=np.float32)
        
        # Validate observation shape
        if observation.size == 0:
             return jsonify({"error": "Empty observation"}), 400
        
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
            
        if observation.shape[1] != 24:
            return jsonify({"error": f"Expected 24 sensor readings, got {observation.shape[1]}"}), 400

        logger.info(f"Parsed observation shape: {observation.shape}, dtype: {observation.dtype}")
        logger.debug(f"Observation values: {observation}")
        
        # Run the agentic workflow
        # Note: compiled_graph is cached inside the function if not provided, 
        # but for production we might want to compile it once globally.
        # For now, we let the function handle it.
        results = run_maintenance_analysis(observation)
        
        # Extract relevant serializable data for response
        response = {
            "risk_assessment": results.get("risk_assessment"),
            "diagnosis": results.get("diagnosis"),
            "maintenance_schedule": results.get("maintenance_schedule"),
            "final_report": results.get("final_report"),
            "predictions": results.get("predictions")
        }
        
        # Helper to convert numpy types to python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        return jsonify(convert_numpy(response))

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(">>> STARTING PREDICTIVE MAINTENANCE SERVER <<<")
    logger.info("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
