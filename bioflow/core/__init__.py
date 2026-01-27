"""
BioFlow Core
=============

Core abstractions and orchestration for the BioFlow platform.

Public API:
    - Modality: Enum of supported data types
    - BioEncoder, BioPredictor, BioGenerator, BioRetriever: Abstract interfaces
    - EmbeddingResult, PredictionResult, RetrievalResult: Data containers
    - ToolRegistry: Central tool management
    - BioFlowOrchestrator: Pipeline execution engine
    - WorkflowConfig, NodeConfig: Configuration classes
"""

from bioflow.core.base import (
    Modality,
    BioEncoder,
    BioPredictor,
    BioGenerator,
    BioRetriever,
    BioTool,
    EmbeddingResult,
    PredictionResult,
    RetrievalResult,
)

from bioflow.core.registry import ToolRegistry

from bioflow.core.orchestrator import (
    BioFlowOrchestrator,
    ExecutionContext,
    PipelineResult,
)

from bioflow.core.config import (
    NodeType,
    NodeConfig,
    WorkflowConfig,
    EncoderConfig,
    VectorDBConfig,
    BioFlowConfig,
)

from bioflow.core.nodes import (
    EncodeNode,
    RetrieveNode,
    PredictNode,
    IngestNode,
    FilterNode,
    TraceabilityNode,
)

__all__ = [
    # Enums
    "Modality",
    "NodeType",
    # Abstract interfaces
    "BioEncoder",
    "BioPredictor",
    "BioGenerator",
    "BioRetriever",
    "BioTool",
    # Data containers
    "EmbeddingResult",
    "PredictionResult",
    "RetrievalResult",
    # Registry
    "ToolRegistry",
    # Orchestrator
    "BioFlowOrchestrator",
    "ExecutionContext",
    "PipelineResult",
    # Config
    "NodeConfig",
    "WorkflowConfig",
    "EncoderConfig",
    "VectorDBConfig",
    "BioFlowConfig",
    # Nodes
    "EncodeNode",
    "RetrieveNode",
    "PredictNode",
    "IngestNode",
    "FilterNode",
    "TraceabilityNode",
]
