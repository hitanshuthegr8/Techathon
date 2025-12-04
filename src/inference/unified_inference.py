"""
Unified Inference - Interface to CMAPSS trained models
"""
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any

from src.utils.normalizer import SensorNormalizer, normalize_observation


def load_models():
    """
    Load all trained CMAPSS models.
    
    Returns:
        Dictionary of loaded models
    """
    models_dir = Path(__file__).parent.parent.parent / "models"
    
    models = {
        # FD001 models
        "fd001_regressor": joblib.load(models_dir / "fd001_regressor.pkl"),
        "fd001_failure": joblib.load(models_dir / "fd001_failure.pkl"),
        
        # FD002 models
        "fd002_regressor": joblib.load(models_dir / "fd002_regressor.pkl"),
        "fd002_failure": joblib.load(models_dir / "fd002_failure.pkl"),
        
        # FD003 models
        "fd003_regressor": joblib.load(models_dir / "fd003_regressor.pkl"),
        "fd003_failure": joblib.load(models_dir / "fd003_failure.pkl"),
        "fd003_component": joblib.load(models_dir / "fd003_component.pkl"),
    }
    
    return models


# Load models / normalizers once at module import (singleton pattern)
_MODELS = None
_FD003_NORMALIZER: SensorNormalizer | None = None


def get_models():
    """Get cached models."""
    global _MODELS
    if _MODELS is None:
        _MODELS = load_models()
    return _MODELS


def get_fd003_normalizer() -> SensorNormalizer | None:
    """
    Lazily load FD003 normalizer if available.

    This expects a normalizer saved with SensorNormalizer.save() during training,
    e.g. at models/fd003_normalizer.pkl. If the file is missing, we simply
    return None and use raw features (avoids crashing in production).
    """
    global _FD003_NORMALIZER
    if _FD003_NORMALIZER is not None:
        return _FD003_NORMALIZER

    models_dir = Path(__file__).parent.parent.parent / "models"
    normalizer_path = models_dir / "fd003_normalizer.pkl"
    if normalizer_path.exists():
        _FD003_NORMALIZER = SensorNormalizer.load(str(normalizer_path))
    else:
        _FD003_NORMALIZER = None
    return _FD003_NORMALIZER


def unified_inference(raw_observation: np.ndarray) -> Dict[str, Any]:
    """
    Unified inference across all CMAPSS models.
    
    Args:
        raw_observation: Raw sensor observation array
        
    Returns:
        Dictionary containing analysis results
    """
    models = get_models()

    # Ensure observation is 2D
    if raw_observation.ndim == 1:
        obs = raw_observation.reshape(1, -1)
    else:
        obs = raw_observation
        
    # Strict Input Validation
    if obs.shape[1] != 24:
        raise ValueError(f"Model expects 24 features, got {obs.shape[1]}")
        
    if not np.issubdtype(obs.dtype, np.number):
        raise ValueError(f"Model expects numeric input, got {obs.dtype}")
        
    if np.isnan(obs).any():
        raise ValueError("Input contains NaN values")
        
    if np.isinf(obs).any():
        raise ValueError("Input contains Infinite values")
        
    print(f"DEBUG: Inference Input Shape: {obs.shape}")
    # print(f"DEBUG: Inference Input Values: {obs}") # Commented out to reduce noise
    
    # FD001 predictions (trained on their own feature scaling; we assume
    # they were trained on raw features as provided by the current API).
    fd001_rul = max(0.0, float(models["fd001_regressor"].predict(obs)[0]))
    fd001_failure_prob = float(models["fd001_failure"].predict_proba(obs)[0, 1])

    # FD002 predictions
    fd002_rul = max(0.0, float(models["fd002_regressor"].predict(obs)[0]))
    fd002_failure_prob = float(models["fd002_failure"].predict_proba(obs)[0, 1])

    # FD003 predictions (per-unit normalized, if stats are available)
    fd003_normalizer = get_fd003_normalizer()
    if fd003_normalizer is not None:
        obs_fd003 = normalize_observation(obs, fd003_normalizer, method="zscore")
    else:
        # Fallback to raw features if normalizer is not present; this may
        # degrade FD003 performance but keeps the API operational.
        obs_fd003 = obs

    fd003_rul = max(0.0, float(models["fd003_regressor"].predict(obs_fd003)[0]))
    fd003_failure_prob = float(models["fd003_failure"].predict_proba(obs_fd003)[0, 1])

    # Component classification (FD003 only) – use the same normalized features
    component_probs = models["fd003_component"].predict_proba(obs_fd003)[0]
    component_classes = models["fd003_component"].classes_
    
    component_probs_dict = {
        str(cls): float(prob) 
        for cls, prob in zip(component_classes, component_probs)
    }
    
    predicted_component = str(component_classes[np.argmax(component_probs)])
    
    # Ensemble calculations
    avg_rul = (fd001_rul + fd002_rul + fd003_rul) / 3.0
    max_failure_prob = max(fd001_failure_prob, fd002_failure_prob, fd003_failure_prob)
    
    return {
        "fd001": {
            "rul": fd001_rul,
            "failure_probability": fd001_failure_prob
        },
        "fd002": {
            "rul": fd002_rul,
            "failure_probability": fd002_failure_prob
        },
        "fd003": {
            "rul": fd003_rul,
            "failure_probability": fd003_failure_prob,
            "predicted_component": predicted_component,
            "component_probs": component_probs_dict
        },
        "ensemble": {
            "avg_rul": avg_rul,
            "max_failure_probability": max_failure_prob
        }
    }


# For testing
if __name__ == "__main__":
    # Test with random observation
    test_obs = np.random.randn(24)
    results = unified_inference(test_obs)
    
    print("Unified Inference Test Results:")
    print("=" * 60)
    for key, value in results.items():
        print(f"\n{key.upper()}:")
        for k, v in value.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")