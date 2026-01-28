"""
BioFlow Plugins
================

Tool implementations for the BioFlow platform.

Encoders:
- OBMEncoder: Unified multimodal encoder (text, molecules, proteins)
- TextEncoder: PubMedBERT / SciBERT for biomedical text
- MoleculeEncoder: ChemBERTa for SMILES
- ProteinEncoder: ESM-2 for protein sequences

Retrievers:
- QdrantRetriever: Vector database search with Qdrant

Predictors:
- DeepPurposePredictor: Drug-Target Interaction prediction
"""

# Encoders
from bioflow.plugins.obm_encoder import OBMEncoder
from bioflow.plugins.encoders import TextEncoder, MoleculeEncoder, ProteinEncoder

# Retriever
from bioflow.plugins.qdrant_retriever import QdrantRetriever

# Predictor
from bioflow.plugins.deeppurpose_predictor import DeepPurposePredictor

__all__ = [
    # Encoders
    "OBMEncoder",
    "TextEncoder",
    "MoleculeEncoder", 
    "ProteinEncoder",
    # Retriever
    "QdrantRetriever",
    # Predictor
    "DeepPurposePredictor",
]


def register_all(registry=None):
    """
    Register all plugins with the tool registry.
    
    Args:
        registry: ToolRegistry instance (uses global if None)
        
    Returns:
        dict: Available plugin classes by category
    """
    import logging
    logger = logging.getLogger(__name__)
    
    from bioflow.core import ToolRegistry
    registry = registry or ToolRegistry
    
    available = {
        "encoders": ["OBMEncoder", "TextEncoder", "MoleculeEncoder", "ProteinEncoder"],
        "retrievers": ["QdrantRetriever"],
        "predictors": ["DeepPurposePredictor"],
    }
    
    logger.info(f"Plugins available for registration: {available}")
    return available
