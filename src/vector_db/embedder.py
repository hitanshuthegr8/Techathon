"""
Vector Embedder - Creates embeddings for sensor observations and predictions
"""
import numpy as np
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FailureEmbedder:
    """
    Creates vector embeddings for failure patterns using sensor data and predictions.
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        logger.info(f"FailureEmbedder initialized with dim={embedding_dim}")
    
    def embed_observation(
        self,
        raw_observation: np.ndarray,
        predictions: Dict[str, Any] = None
    ) -> np.ndarray:
        """
        Create embedding from sensor observation and optional predictions.
        
        Args:
            raw_observation: Raw sensor data array
            predictions: Optional prediction data to enhance embedding
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Extract features from raw observation
            obs_features = self._extract_observation_features(raw_observation)
            
            # Extract prediction features if available
            if predictions:
                pred_features = self._extract_prediction_features(predictions)
                combined_features = np.concatenate([obs_features, pred_features])
            else:
                combined_features = obs_features
            
            # Project to embedding dimension if needed
            if len(combined_features) != self.embedding_dim:
                embedding = self._project_to_embedding_dim(combined_features)
            else:
                embedding = combined_features
            
            # Normalize
            embedding = self._normalize(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            # Return zero vector on error
            return np.zeros(self.embedding_dim)
    
    def _extract_observation_features(self, observation: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from raw sensor observation.
        """
        if observation is None or len(observation) == 0:
            return np.zeros(64)
        
        # Ensure 1D array
        obs_flat = observation.flatten()
        
        # Statistical features
        features = [
            np.mean(obs_flat),
            np.std(obs_flat),
            np.min(obs_flat),
            np.max(obs_flat),
            np.median(obs_flat),
            np.percentile(obs_flat, 25),
            np.percentile(obs_flat, 75),
            np.ptp(obs_flat),  # peak-to-peak
        ]
        
        # Add normalized sensor readings (take first 56 values if available)
        sensor_readings = obs_flat[:56] if len(obs_flat) >= 56 else np.pad(
            obs_flat, (0, 56 - len(obs_flat)), mode='constant'
        )
        
        features.extend(sensor_readings.tolist())
        
        return np.array(features[:64])  # Ensure fixed size
    
    def _extract_prediction_features(self, predictions: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from prediction outputs.
        """
        features = []
        
        # Ensemble features
        ensemble = predictions.get("ensemble", {})
        features.extend([
            ensemble.get("avg_rul", 0) / 100.0,  # Normalize
            ensemble.get("max_failure_probability", 0)
        ])
        
        # Individual model features
        for model_name in ["fd001", "fd002", "fd003"]:
            if model_name in predictions:
                model_data = predictions[model_name]
                features.extend([
                    model_data.get("rul", 0) / 100.0,
                    model_data.get("failure_probability", 0)
                ])
            else:
                features.extend([0.0, 0.0])
        
        # Component probabilities (FD003) – support both numeric and textual keys.
        if "fd003" in predictions:
            comp_probs = predictions["fd003"].get("component_probs", {})

            def get_prob(keys):
                for k in keys:
                    if k in comp_probs:
                        return comp_probs[k]
                return 0.0

            # 0 → Healthy, 1 → HPC, 2 → Fan
            features.append(get_prob(["Healthy", "0"]))
            features.append(get_prob(["HPC", "1"]))
            features.append(get_prob(["Fan", "2"]))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Pad to 64 dimensions
        while len(features) < 64:
            features.append(0.0)
        
        return np.array(features[:64])
    
    def _project_to_embedding_dim(self, features: np.ndarray) -> np.ndarray:
        """
        Project features to target embedding dimension using simple random projection.
        """
        current_dim = len(features)
        
        if current_dim == self.embedding_dim:
            return features
        
        # Simple approach: repeat or truncate + random projection
        if current_dim < self.embedding_dim:
            # Pad with zeros
            padded = np.pad(features, (0, self.embedding_dim - current_dim), mode='constant')
            return padded
        else:
            # Use averaging to reduce dimensions
            n_chunks = self.embedding_dim
            chunk_size = current_dim // n_chunks
            projected = []
            
            for i in range(n_chunks):
                start = i * chunk_size
                end = start + chunk_size if i < n_chunks - 1 else current_dim
                projected.append(np.mean(features[start:end]))
            
            return np.array(projected)
    
    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """
        L2 normalize the embedding vector.
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def embed_batch(
        self,
        observations: List[np.ndarray],
        predictions_list: List[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Create embeddings for a batch of observations.
        
        Args:
            observations: List of observation arrays
            predictions_list: Optional list of prediction dicts
            
        Returns:
            Array of embeddings (batch_size, embedding_dim)
        """
        if predictions_list is None:
            predictions_list = [None] * len(observations)
        
        embeddings = []
        for obs, pred in zip(observations, predictions_list):
            embedding = self.embed_observation(obs, pred)
            embeddings.append(embedding)
        
        return np.array(embeddings)


def create_embedder(embedding_dim: int = 128) -> FailureEmbedder:
    """
    Factory function to create embedder instance.
    """
    return FailureEmbedder(embedding_dim=embedding_dim)