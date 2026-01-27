"""
OBM Wrapper - Unified Multimodal Encoding Interface
=====================================================

This module provides a clean, high-level API for encoding biological data
(text, molecules, proteins) into a unified vector space using open-source models.

Implementation Note:
    Internally delegates to OBMEncoder from bioflow.plugins.obm_encoder
    for actual encoding. This wrapper provides backward compatibility
    and a simplified API.
"""

import os
import sys
import numpy as np
import logging
from typing import List, Union, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Supported data modalities."""
    TEXT = "text"
    MOLECULE = "molecule"
    SMILES = "smiles"
    PROTEIN = "protein"
    CELL = "cell"


@dataclass
class EmbeddingResult:
    """Container for embedding results with metadata."""
    vector: np.ndarray
    modality: ModalityType
    content: str
    content_hash: str
    dimension: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector.tolist(),
            "modality": self.modality.value,
            "content": self.content,
            "content_hash": self.content_hash,
            "dimension": self.dimension
        }


class OBMWrapper:
    """
    Unified wrapper for OpenBioMed multimodal encoding.
    
    This class provides a clean API for encoding biological data into
    a shared embedding space, enabling cross-modal similarity search.
    
    Internally uses OBMEncoder for actual encoding operations.
    
    Attributes:
        device: Computing device ('cuda' or 'cpu')
        vector_dim: Dimension of output embeddings
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize the OBM wrapper.
        
        Args:
            device: 'cuda' or 'cpu'. Auto-detects if None.
            config_path: Path to model config YAML (optional, for compatibility).
            checkpoint_path: Path to model weights (optional, for compatibility).
            
        Raises:
            RuntimeError: If encoders fail to initialize.
        """
        # Import torch lazily for device detection
        try:
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = device or "cpu"
        
        self._vector_dim = 768
        self._encoder = None
        
        # Store config for compatibility (not used internally)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        self._init_encoder()
    
    def _init_encoder(self):
        """
        Initialize the underlying OBMEncoder.
        
        Raises:
            RuntimeError: If encoder fails to load.
        """
        try:
            from bioflow.plugins.obm_encoder import OBMEncoder
            from bioflow.core import Modality
            
            self._encoder = OBMEncoder(
                device=self.device,
                lazy_load=True  # Load models on first use
            )
            self._Modality = Modality
            self._vector_dim = self._encoder.output_dim
            
            logger.info(f"OBMWrapper initialized. Device: {self.device}, Vector dim: {self._vector_dim}")
            
        except ImportError as e:
            logger.error(f"Failed to import OBMEncoder: {e}")
            raise RuntimeError(
                f"OBMEncoder not available: {e}. "
                "Ensure bioflow.plugins.obm_encoder is properly installed."
            )
        except Exception as e:
            logger.error(f"Failed to initialize encoder: {e}")
            raise RuntimeError(f"OBM encoder initialization failed: {e}")
    
    @property
    def vector_dim(self) -> int:
        """Return the embedding dimension."""
        return self._vector_dim
    
    @property
    def is_ready(self) -> bool:
        """Check if encoder is loaded and ready."""
        return self._encoder is not None
    
    def _compute_hash(self, content: str) -> str:
        """Compute content hash for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _modality_type_to_core(self, modality: ModalityType):
        """Convert ModalityType to core Modality."""
        mapping = {
            ModalityType.TEXT: self._Modality.TEXT,
            ModalityType.MOLECULE: self._Modality.SMILES,
            ModalityType.SMILES: self._Modality.SMILES,
            ModalityType.PROTEIN: self._Modality.PROTEIN,
        }
        return mapping.get(modality, self._Modality.TEXT)
    
    def encode_text(self, text: Union[str, List[str]]) -> List[EmbeddingResult]:
        """
        Encode text (abstracts, descriptions, notes) into embeddings.
        
        Args:
            text: Single string or list of strings.
            
        Returns:
            List of EmbeddingResult objects.
            
        Raises:
            RuntimeError: If encoder is not ready.
        """
        if not self.is_ready:
            raise RuntimeError("OBM encoder not initialized - cannot encode text")
            
        if isinstance(text, str):
            text = [text]
        
        results = []
        for t in text:
            emb = self._encoder.encode(t, self._Modality.TEXT)
            results.append(EmbeddingResult(
                vector=emb.vector,
                modality=ModalityType.TEXT,
                content=t[:200],  # Truncate for storage
                content_hash=self._compute_hash(t),
                dimension=len(emb.vector)
            ))
        
        return results
    
    def encode_smiles(self, smiles: Union[str, List[str]]) -> List[EmbeddingResult]:
        """
        Encode SMILES molecular representations into embeddings.
        
        Args:
            smiles: Single SMILES string or list of SMILES.
            
        Returns:
            List of EmbeddingResult objects.
            
        Raises:
            RuntimeError: If encoder is not ready.
        """
        if not self.is_ready:
            raise RuntimeError("OBM encoder not initialized - cannot encode SMILES")
            
        if isinstance(smiles, str):
            smiles = [smiles]
        
        results = []
        for s in smiles:
            emb = self._encoder.encode(s, self._Modality.SMILES)
            results.append(EmbeddingResult(
                vector=emb.vector,
                modality=ModalityType.MOLECULE,
                content=s,
                content_hash=self._compute_hash(s),
                dimension=len(emb.vector)
            ))
        
        return results
    
    def encode_protein(self, sequences: Union[str, List[str]]) -> List[EmbeddingResult]:
        """
        Encode protein sequences (FASTA format) into embeddings.
        
        Args:
            sequences: Single sequence or list of sequences.
            
        Returns:
            List of EmbeddingResult objects.
            
        Raises:
            RuntimeError: If encoder is not ready.
        """
        if not self.is_ready:
            raise RuntimeError("OBM encoder not initialized - cannot encode protein")
            
        if isinstance(sequences, str):
            sequences = [sequences]
        
        results = []
        for seq in sequences:
            emb = self._encoder.encode(seq, self._Modality.PROTEIN)
            results.append(EmbeddingResult(
                vector=emb.vector,
                modality=ModalityType.PROTEIN,
                content=seq[:100] + "..." if len(seq) > 100 else seq,
                content_hash=self._compute_hash(seq),
                dimension=len(emb.vector)
            ))
        
        return results
    
    def encode(self, content: str, modality: Union[str, ModalityType]) -> EmbeddingResult:
        """
        Universal encoding function.
        
        Args:
            content: The content to encode.
            modality: Type of content ('text', 'smiles', 'molecule', 'protein').
            
        Returns:
            Single EmbeddingResult.
        """
        if isinstance(modality, str):
            modality = ModalityType(modality.lower())
        
        if modality in [ModalityType.TEXT]:
            return self.encode_text(content)[0]
        elif modality in [ModalityType.MOLECULE, ModalityType.SMILES]:
            return self.encode_smiles(content)[0]
        elif modality == ModalityType.PROTEIN:
            return self.encode_protein(content)[0]
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def cross_modal_similarity(
        self, 
        query: str, 
        query_modality: str,
        targets: List[str],
        target_modality: str
    ) -> List[Tuple[str, float]]:
        """
        Compute cross-modal similarities.
        
        Args:
            query: Query content.
            query_modality: Modality of query.
            targets: List of target contents.
            target_modality: Modality of targets.
            
        Returns:
            List of (target, similarity_score) tuples, sorted by similarity.
        """
        query_emb = self.encode(query, query_modality)
        target_embs = []
        
        if target_modality.lower() in ['text']:
            target_embs = self.encode_text(targets)
        elif target_modality.lower() in ['smiles', 'molecule']:
            target_embs = self.encode_smiles(targets)
        elif target_modality.lower() == 'protein':
            target_embs = self.encode_protein(targets)
        
        results = []
        for emb in target_embs:
            sim = self.compute_similarity(query_emb.vector, emb.vector)
            results.append((emb.content, sim))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
