"""
Constants and enumerations for the predictive maintenance system
"""


class RiskLevel:
    """Risk level classifications"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class MaintenanceWindow:
    """Maintenance scheduling windows"""
    IMMEDIATE = "IMMEDIATE"
    SOON = "SOON"
    ROUTINE = "ROUTINE"


class ComponentType:
    """Known component types"""
    HEALTHY = "Healthy"
    HPC = "HPC"  # High Pressure Compressor
    FAN = "Fan"
    UNKNOWN = "Unknown"
    GENERAL = "General"


class FailureType:
    """Failure type classifications"""
    DEGRADATION = "degradation"
    SUDDEN = "sudden"
    INTERMITTENT = "intermittent"
    WEAR = "wear"


class Severity:
    """Failure severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Thresholds
RISK_THRESHOLDS = {
    "high_failure_probability": 0.5,
    "critical_rul": 30,
    "medium_rul": 60,
}

# Vector DB configuration
VECTOR_DB_CONFIG = {
    "default_backend": "chromadb",
    "embedding_dim": 128,
    "chromadb_collection": "failure_patterns",
    "chromadb_persist_dir": "./chroma_db",
    "pinecone_index": "failure-patterns",
}

# Agent configuration
AGENT_CONFIG = {
    "diagnosis_top_k": 5,
    "min_similarity": 0.5,
}