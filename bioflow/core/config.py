"""
BioFlow Configuration Schema
=============================

Dataclasses and schemas for workflow configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class NodeType(Enum):
    """Types of nodes in a BioFlow pipeline."""
    ENCODE = "encode"         # Vectorize input using encoder
    RETRIEVE = "retrieve"     # Search vector DB for neighbors
    PREDICT = "predict"       # Run prediction model
    GENERATE = "generate"     # Generate new candidates
    FILTER = "filter"         # Filter/rank candidates
    CUSTOM = "custom"         # User-defined function


@dataclass
class NodeConfig:
    """Configuration for a single pipeline node."""
    id: str
    type: NodeType
    tool: str                            # Name of registered tool
    inputs: List[str] = field(default_factory=list)  # Node IDs or "input"
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = NodeType(self.type)


@dataclass
class WorkflowConfig:
    """Configuration for an entire workflow."""
    name: str
    description: str = ""
    nodes: List[NodeConfig] = field(default_factory=list)
    output_node: str = ""                # ID of final node
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowConfig":
        """Create WorkflowConfig from dictionary (e.g., loaded YAML)."""
        nodes = [
            NodeConfig(**node) if isinstance(node, dict) else node
            for node in data.get("nodes", [])
        ]
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            nodes=nodes,
            output_node=data.get("output_node", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class EncoderConfig:
    """Configuration for an encoder."""
    name: str
    model_type: str                      # e.g., "esm2", "pubmedbert", "chemberta"
    model_path: Optional[str] = None
    device: str = "cpu"
    dimension: int = 768
    modalities: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    provider: str = "qdrant"             # qdrant, faiss, etc.
    url: Optional[str] = None
    path: Optional[str] = None
    default_collection: str = "bioflow_memory"
    distance_metric: str = "cosine"


@dataclass
class BioFlowConfig:
    """Master configuration for entire BioFlow system."""
    project_name: str = "BioFlow"
    encoders: Dict[str, EncoderConfig] = field(default_factory=dict)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    workflows: Dict[str, WorkflowConfig] = field(default_factory=dict)
    default_encoder: str = "default"
    log_level: str = "INFO"
