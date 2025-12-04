"""
Vector Query - High-level interface for querying similar failures
"""
import numpy as np
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.vector_db.embedder import create_embedder
from src.vector_db.store import create_vector_store
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global instances (singleton pattern)
_embedder = None
_vector_store = None


def initialize_vector_db(
    backend: str = "chromadb",
    embedding_dim: int = 128,
    **store_kwargs
) -> None:
    """
    Initialize the vector database system.
    
    Args:
        backend: Vector store backend ("chromadb" or "pinecone")
        embedding_dim: Dimension of embeddings
        **store_kwargs: Additional arguments for vector store
    """
    global _embedder, _vector_store
    
    try:
        logger.info(f"Initializing vector DB with backend: {backend}")
        
        _embedder = create_embedder(embedding_dim=embedding_dim)
        _vector_store = create_vector_store(backend=backend, **store_kwargs)
        
        logger.info("Vector DB initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize vector DB: {str(e)}")
        raise


def add_failure_case(
    observation: np.ndarray,
    predictions: Dict[str, Any],
    component: str,
    failure_type: str = "degradation",
    severity: str = "medium",
    additional_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add a failure case to the vector database.
    
    Args:
        observation: Raw sensor observation
        predictions: Prediction outputs
        component: Failed component name
        failure_type: Type of failure
        severity: Failure severity
        additional_metadata: Additional metadata to store
        
    Returns:
        ID of the stored case
    """
    global _embedder, _vector_store
    
    if _embedder is None or _vector_store is None:
        raise RuntimeError("Vector DB not initialized. Call initialize_vector_db() first.")
    
    try:
        # Create embedding
        embedding = _embedder.embed_observation(observation, predictions)
        
        # Prepare metadata
        metadata = {
            "component": component,
            "failure_type": failure_type,
            "severity": severity,
            "rul": predictions.get("ensemble", {}).get("avg_rul", 0),
            "failure_probability": predictions.get("ensemble", {}).get("max_failure_probability", 0),
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Store in vector DB
        ids = _vector_store.add(
            embeddings=np.array([embedding]),
            metadata=[metadata]
        )
        
        logger.info(f"Added failure case: {ids[0]} ({component})")
        return ids[0]
        
    except Exception as e:
        logger.error(f"Error adding failure case: {str(e)}")
        raise


def query_similar_failures(
    observation: np.ndarray,
    predictions: Dict[str, Any],
    top_k: int = 5,
    min_similarity: float = 0.5,
    component_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Query similar failure cases from the database.
    
    Args:
        observation: Current sensor observation
        predictions: Current predictions
        top_k: Number of similar cases to retrieve
        min_similarity: Minimum similarity threshold
        component_filter: Optional component name filter
        
    Returns:
        List of similar failure cases with metadata
    """
    global _embedder, _vector_store
    
    if _embedder is None or _vector_store is None:
        logger.warning("Vector DB not initialized. Returning empty results.")
        return []
    
    try:
        # Create query embedding
        query_embedding = _embedder.embed_observation(observation, predictions)
        
        # Prepare filter
        filter_dict = None
        if component_filter:
            filter_dict = {"component": component_filter}
        
        # Query vector store
        results = _vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        # Filter by minimum similarity
        filtered_results = []
        for result in results:
            similarity = result.get('similarity', 0)
            if similarity >= min_similarity:
                case = {
                    'id': result['id'],
                    'similarity': similarity,
                    'component': result['metadata'].get('component', 'Unknown'),
                    'failure_type': result['metadata'].get('failure_type', 'Unknown'),
                    'severity': result['metadata'].get('severity', 'Unknown'),
                    'rul': result['metadata'].get('rul', 0),
                    'failure_probability': result['metadata'].get('failure_probability', 0),
                    'metadata': result['metadata']
                }
                filtered_results.append(case)
        
        logger.info(f"Retrieved {len(filtered_results)} similar failure cases")
        return filtered_results
        
    except Exception as e:
        logger.error(f"Error querying similar failures: {str(e)}")
        return []


def bulk_add_failures(
    observations: List[np.ndarray],
    predictions_list: List[Dict[str, Any]],
    components: List[str],
    failure_types: Optional[List[str]] = None,
    severities: Optional[List[str]] = None
) -> List[str]:
    """
    Bulk add multiple failure cases to the database.
    
    Args:
        observations: List of sensor observations
        predictions_list: List of prediction outputs
        components: List of component names
        failure_types: Optional list of failure types
        severities: Optional list of severities
        
    Returns:
        List of stored case IDs
    """
    global _embedder, _vector_store
    
    if _embedder is None or _vector_store is None:
        raise RuntimeError("Vector DB not initialized. Call initialize_vector_db() first.")
    
    try:
        n = len(observations)
        if failure_types is None:
            failure_types = ["degradation"] * n
        if severities is None:
            severities = ["medium"] * n
        
        # Create embeddings
        embeddings = _embedder.embed_batch(observations, predictions_list)
        
        # Prepare metadata
        metadata_list = []
        for pred, comp, ft, sev in zip(predictions_list, components, failure_types, severities):
            metadata = {
                "component": comp,
                "failure_type": ft,
                "severity": sev,
                "rul": pred.get("ensemble", {}).get("avg_rul", 0),
                "failure_probability": pred.get("ensemble", {}).get("max_failure_probability", 0),
            }
            metadata_list.append(metadata)
        
        # Bulk store
        ids = _vector_store.add(
            embeddings=embeddings,
            metadata=metadata_list
        )
        
        logger.info(f"Bulk added {len(ids)} failure cases")
        return ids
        
    except Exception as e:
        logger.error(f"Error in bulk add: {str(e)}")
        raise


def get_database_stats() -> Dict[str, Any]:
    """
    Get statistics about the vector database.
    
    Returns:
        Dictionary with database statistics
    """
    global _vector_store
    
    if _vector_store is None:
        return {"initialized": False}
    
    # Basic stats (implementation depends on backend)
    return {
        "initialized": True,
        "backend": _vector_store.__class__.__name__,
    }