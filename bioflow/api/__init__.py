"""
BioFlow API
============
FastAPI backend bridging the Next.js UI with OpenBioMed core.

Provides:
- ModelService: Unified access to encoders and predictors
- QdrantService: Vector database operations
- DTI Predictor: Drug-Target Interaction prediction

Usage:
    from bioflow.api import get_model_service, get_qdrant_service
    
    model = get_model_service()
    emb = model.encode_molecule("CCO")
"""

__version__ = "2.0.0"

from bioflow.api.model_service import (
    ModelService,
    get_model_service,
    EncodingResult,
    PredictionResult,
    GenerationResult,
)

from bioflow.api.qdrant_service import (
    QdrantService,
    get_qdrant_service,
    SearchResult,
    IngestResult,
    CollectionType,
)

from bioflow.api.dti_predictor import (
    DeepPurposePredictor,
    get_dti_predictor,
    DTIPrediction,
)

__all__ = [
    # Model Service
    "ModelService",
    "get_model_service",
    "EncodingResult",
    "PredictionResult",
    "GenerationResult",
    # Qdrant Service
    "QdrantService",
    "get_qdrant_service",
    "SearchResult",
    "IngestResult",
    "CollectionType",
    # DTI
    "DeepPurposePredictor",
    "get_dti_predictor",
    "DTIPrediction",
]
