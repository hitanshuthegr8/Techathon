"""
Data normalization utilities for sensor observations
"""
import numpy as np
from typing import Optional, Tuple
import pickle
from pathlib import Path


class SensorNormalizer:
    """
    Normalizes sensor observations using saved statistics.
    """
    
    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        min_val: Optional[np.ndarray] = None,
        max_val: Optional[np.ndarray] = None
    ):
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.fitted = False
        
        if mean is not None and std is not None:
            self.fitted = True
    
    def fit(self, data: np.ndarray) -> 'SensorNormalizer':
        """
        Fit normalizer to training data.
        
        Args:
            data: Training data array (n_samples, n_features)
            
        Returns:
            Self for chaining
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)
        
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1, self.std)
        
        self.fitted = True
        return self
    
    def transform(self, data: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Data to normalize
            method: Normalization method ("zscore" or "minmax")
            
        Returns:
            Normalized data
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        if method == "zscore":
            return (data - self.mean) / self.std
        elif method == "minmax":
            range_val = self.max_val - self.min_val
            range_val = np.where(range_val == 0, 1, range_val)
            return (data - self.min_val) / range_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def inverse_transform(self, data: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data
            method: Normalization method used
            
        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted.")
        
        if method == "zscore":
            return data * self.std + self.mean
        elif method == "minmax":
            range_val = self.max_val - self.min_val
            return data * range_val + self.min_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def save(self, filepath: str) -> None:
        """Save normalizer parameters to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'std': self.std,
                'min_val': self.min_val,
                'max_val': self.max_val
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SensorNormalizer':
        """Load normalizer parameters from file."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        return cls(**params)


def normalize_observation(
    observation: np.ndarray,
    normalizer: Optional[SensorNormalizer] = None,
    method: str = "zscore"
) -> np.ndarray:
    """
    Convenience function to normalize a single observation.
    
    Args:
        observation: Raw sensor observation
        normalizer: Fitted normalizer (if None, returns observation as-is)
        method: Normalization method
        
    Returns:
        Normalized observation
    """
    if normalizer is None:
        return observation
    
    return normalizer.transform(observation, method=method)