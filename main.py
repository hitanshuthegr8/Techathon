"""
Main Entry Point - CMAPSS Predictive Maintenance System
"""
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.utils.logger import setup_logging, get_logger
from src.utils.constants import VECTOR_DB_CONFIG
from src.vector_db.query import initialize_vector_db
from src.workflow import (
    compile_workflow,
    run_maintenance_analysis,
    print_analysis_results,
    get_workflow_visualization
)

logger = get_logger(__name__)


def initialize_system(
    vector_backend: str = "chromadb",
    log_level: str = "INFO"
) -> None:
    """
    Initialize the predictive maintenance system.
    
    Args:
        vector_backend: Vector database backend ("chromadb" or "pinecone")
        log_level: Logging level
    """
    # Setup logging
    setup_logging(
        log_level=log_level,
        log_file="logs/maintenance_system.log"
    )
    
    logger.info("=" * 80)
    logger.info("Initializing CMAPSS Predictive Maintenance System")
    logger.info("=" * 80)
    
    # Initialize vector database
    try:
        logger.info(f"Initializing vector database (backend: {vector_backend})...")
        
        if vector_backend == "chromadb":
            initialize_vector_db(
                backend="chromadb",
                embedding_dim=VECTOR_DB_CONFIG["embedding_dim"],
                collection_name=VECTOR_DB_CONFIG["chromadb_collection"],
                persist_directory=VECTOR_DB_CONFIG["chromadb_persist_dir"]
            )
        elif vector_backend == "pinecone":
            # For Pinecone, you'll need to set environment variables:
            # PINECONE_API_KEY and PINECONE_ENVIRONMENT
            initialize_vector_db(
                backend="pinecone",
                embedding_dim=VECTOR_DB_CONFIG["embedding_dim"],
                index_name=VECTOR_DB_CONFIG["pinecone_index"]
            )
        else:
            raise ValueError(f"Unknown vector backend: {vector_backend}")
        
        logger.info("Vector database initialized successfully")
        
    except Exception as e:
        logger.warning(f"Vector database initialization failed: {str(e)}")
        logger.warning("System will continue without historical failure search")
    
    logger.info("System initialization complete")
    logger.info("=" * 80)


def run_single_analysis(
    observation: np.ndarray,
    verbose: bool = True
) -> dict:
    """
    Run analysis on a single observation.
    
    Args:
        observation: Raw sensor observation
        verbose: Whether to print detailed results
        
    Returns:
        Analysis results dictionary
    """
    logger.info("Running single observation analysis...")
    
    # Compile workflow (cached after first compile)
    compiled_graph = compile_workflow()
    
    # Run analysis
    results = run_maintenance_analysis(observation, compiled_graph)
    
    # Print results if verbose
    if verbose:
        print_analysis_results(results)
    
    return results


def run_batch_analysis(
    observations: list[np.ndarray],
    output_file: str = None
) -> list[dict]:
    """
    Run analysis on multiple observations.
    
    Args:
        observations: List of sensor observations
        output_file: Optional file to save results
        
    Returns:
        List of analysis results
    """
    logger.info(f"Running batch analysis on {len(observations)} observations...")
    
    # Compile workflow once
    compiled_graph = compile_workflow()
    
    results = []
    for i, obs in enumerate(observations):
        logger.info(f"Processing observation {i+1}/{len(observations)}...")
        result = run_maintenance_analysis(obs, compiled_graph)
        results.append(result)
    
    # Save results if output file specified
    if output_file:
        import json
        from datetime import datetime
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_observations": len(observations),
            "results": []
        }
        
        for i, result in enumerate(results):
            output_data["results"].append({
                "observation_id": i,
                "risk_level": result.get("risk_assessment", {}).get("risk_level"),
                "component": result.get("diagnosis", {}).get("probable_component"),
                "rul": result.get("risk_assessment", {}).get("avg_rul"),
                "maintenance_window": result.get("maintenance_schedule", {}).get("maintenance_window"),
                "report_id": result.get("final_report", {}).get("report_id")
            })
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    return results


def main():
    """
    Main entry point with CLI interface.
    """
    parser = argparse.ArgumentParser(
        description="CMAPSS Predictive Maintenance System"
    )
    
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "visualize"],
        default="single",
        help="Operation mode"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input file (numpy array for single, CSV/NPY for batch)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for batch results"
    )
    
    parser.add_argument(
        "--vector-backend",
        choices=["chromadb", "pinecone"],
        default="chromadb",
        help="Vector database backend"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    initialize_system(
        vector_backend=args.vector_backend,
        log_level=args.log_level
    )
    
    # Handle different modes
    if args.mode == "visualize":
        print(get_workflow_visualization())
        return
    
    elif args.mode == "single":
        # Load single observation
        if args.input:
            observation = np.load(args.input)
        else:
            # Generate random test observation
            logger.warning("No input provided, using random test data")
            observation = np.random.randn(24)
        
        # Run analysis
        run_single_analysis(observation, verbose=not args.no_verbose)
    
    elif args.mode == "batch":
        # Load batch observations
        if not args.input:
            logger.error("Batch mode requires --input file")
            return
        
        # Load observations based on file type
        if args.input.endswith('.npy'):
            observations = np.load(args.input)
            if observations.ndim == 1:
                observations = [observations]
            else:
                observations = list(observations)
        elif args.input.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(args.input)
            observations = [row.values for _, row in df.iterrows()]
        else:
            logger.error("Unsupported input format. Use .npy or .csv")
            return
        
        # Run batch analysis
        run_batch_analysis(observations, output_file=args.output)


if __name__ == "__main__":
    main()