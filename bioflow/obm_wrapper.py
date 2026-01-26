"""
OBM Wrapper - Unified Multimodal Encoding Interface
=====================================================

This module provides a clean, high-level API for encoding biological data
(text, molecules, proteins) into a unified vector space using open-source models.
"""

import os
import sys
import torch
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
    
    Attributes:
        device: Computing device ('cuda' or 'cpu')
        model: Underlying open-source model
        vector_dim: Dimension of output embeddings
    """
    
    def __init__(
        self, 
        device: str = None,
        config_path: str = None,
        checkpoint_path: str = None,
        use_mock: bool = False
    ):
        """
        Initialize the OBM wrapper.
        
        Args:
            device: 'cuda' or 'cpu'. Auto-detects if None.
            config_path: Path to open-source model config YAML.
            checkpoint_path: Path to model weights.
            use_mock: If True, uses mock embeddings (for testing without GPU).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mock = use_mock
        self._model = None
        self._vector_dim = 768  # Default, updated after model load
        
        if config_path is None:
            config_path = os.path.join(ROOT_DIR, "configs/model/opensource_model.yaml")
        
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        if not use_mock:
            self._init_model()
        else:
            logger.info("Using MOCK mode - embeddings are random vectors for testing")
            self._vector_dim = 768
    
    def _init_model(self):
        """Initialize the open-source model."""
        try:
            # Placeholder for initializing open-source model
            pass
            
            self._model = None
            
            self._vector_dim = 768
            
            logger.info(f"OBM initialized. Device: {self.device}, Vector dim: {self._vector_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Falling back to MOCK mode")
            self.use_mock = True
            self._vector_dim = 768
    
    @property
    def vector_dim(self) -> int:
        """Return the embedding dimension."""
        return self._vector_dim
    
    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model is not None or self.use_mock
    
    def _compute_hash(self, content: str) -> str:
        """Compute content hash for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _mock_embed(self, content: str, modality: ModalityType) -> np.ndarray:
        """Generate deterministic mock embedding based on content hash."""
        seed = int(self._compute_hash(content), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._vector_dim).astype(np.float32)
        # Normalize
        vec = vec / np.linalg.norm(vec)
        return vec
    
    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]]) -> List[EmbeddingResult]:
        """
        Encode text (abstracts, descriptions, notes) into embeddings.
        
        Args:
            text: Single string or list of strings.
            
        Returns:
            List of EmbeddingResult objects.
        """
        if isinstance(text, str):
            text = [text]
        
        results = []
        
        if self.use_mock:
            for t in text:
                vec = self._mock_embed(t, ModalityType.TEXT)
                results.append(EmbeddingResult(
                    vector=vec,
                    modality=ModalityType.TEXT,
                    content=t[:200],  # Truncate for storage
                    content_hash=self._compute_hash(t),
                    dimension=self._vector_dim
                ))
        else:
            tokenizer = self._model.llm_tokenizer
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self._model.llm(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors = pooled.cpu().numpy()
            
            for i, t in enumerate(text):
                results.append(EmbeddingResult(
                    vector=vectors[i],
                    modality=ModalityType.TEXT,
                    content=t[:200],
                    content_hash=self._compute_hash(t),
                    dimension=self._vector_dim
                ))
        
        return results
    
    @torch.no_grad()
    def encode_smiles(self, smiles: Union[str, List[str]]) -> List[EmbeddingResult]:
        """
        Encode SMILES molecular representations into embeddings.
        
        Args:
            smiles: Single SMILES string or list of SMILES.
            
        Returns:
            List of EmbeddingResult objects.
        """
        if isinstance(smiles, str):
            smiles = [smiles]
        
        results = []
        
        if self.use_mock:
            for s in smiles:
                vec = self._mock_embed(s, ModalityType.MOLECULE)
                results.append(EmbeddingResult(
                    vector=vec,
                    modality=ModalityType.MOLECULE,
                    content=s,
                    content_hash=self._compute_hash(s),
                    dimension=self._vector_dim
                ))
        else:
            from open_biomed.data import Molecule
            from torch_scatter import scatter_mean
            
            molecules = [Molecule.from_smiles(s) for s in smiles]
            mol_feats = [self._model.featurizer.molecule_featurizer(m) for m in molecules]
            collated = self._model.collator.molecule_collator(mol_feats).to(self.device)
            
            node_feats = self._model.mol_structure_encoder(collated)
            proj_feats = self._model.proj_mol(node_feats)
            vectors = scatter_mean(proj_feats, collated.batch, dim=0).cpu().numpy()
            
            for i, s in enumerate(smiles):
                results.append(EmbeddingResult(
                    vector=vectors[i],
                    modality=ModalityType.MOLECULE,
                    content=s,
                    content_hash=self._compute_hash(s),
                    dimension=self._vector_dim
                ))
        
        return results
    
    @torch.no_grad()
    def encode_protein(self, sequences: Union[str, List[str]]) -> List[EmbeddingResult]:
        """
        Encode protein sequences (FASTA format) into embeddings.
        
        Args:
            sequences: Single sequence or list of sequences.
            
        Returns:
            List of EmbeddingResult objects.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        results = []
        
        if self.use_mock:
            for seq in sequences:
                vec = self._mock_embed(seq, ModalityType.PROTEIN)
                results.append(EmbeddingResult(
                    vector=vec,
                    modality=ModalityType.PROTEIN,
                    content=seq[:100] + "..." if len(seq) > 100 else seq,
                    content_hash=self._compute_hash(seq),
                    dimension=self._vector_dim
                ))
        else:
            tokenizer = self._model.prot_tokenizer
            inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self._model.prot_structure_encoder(**inputs)
            hidden = outputs.last_hidden_state
            proj = self._model.proj_prot(hidden)
            
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            pooled = (proj * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors = pooled.cpu().numpy()
            
            for i, seq in enumerate(sequences):
                results.append(EmbeddingResult(
                    vector=vectors[i],
                    modality=ModalityType.PROTEIN,
                    content=seq[:100] + "..." if len(seq) > 100 else seq,
                    content_hash=self._compute_hash(seq),
                    dimension=self._vector_dim
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
