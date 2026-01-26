"""
BioFlow - Multimodal Biological Intelligence Framework
========================================================

A modular, open-source platform for biological discovery integrating:
- Multimodal encoders (text, molecules, proteins, images)
- Vector database memory (Qdrant)
- Prediction tools (DTI, ADMET)
- Workflow orchestration

Core Modules:
    - core: Abstract interfaces, registry, and orchestrator
    - plugins: Tool implementations (OBM, DeepPurpose, etc.)
    - workflows: YAML-based pipeline definitions

Open-Source Models Supported:
    - Text: PubMedBERT, SciBERT, Specter
    - Molecules: ChemBERTa, RDKit FP
    - Proteins: ESM-2, ProtBERT
    - Images: CLIP, BioMedCLIP
"""

__version__ = "0.2.0"
__author__ = "BioFlow Team"

# Core abstractions
from bioflow.core import (
    Modality,
    BioEncoder,
    BioPredictor,
    BioGenerator,
    BioRetriever,
    ToolRegistry,
    BioFlowOrchestrator,
    WorkflowConfig,
    NodeConfig,
)

# Legacy imports (for backward compatibility)
try:
    from bioflow.obm_wrapper import OBMWrapper
    from bioflow.qdrant_manager import QdrantManager
except ImportError:
    OBMWrapper = None
    QdrantManager = None

__all__ = [
    # Core
    "Modality",
    "BioEncoder",
    "BioPredictor", 
    "BioGenerator",
    "BioRetriever",
    "ToolRegistry",
    "BioFlowOrchestrator",
    "WorkflowConfig",
    "NodeConfig",
    # Wrappers
    "OBMWrapper",
    "QdrantManager",
]
