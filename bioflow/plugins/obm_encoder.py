"""
OBM Encoder - Unified Multimodal Encoder
==========================================

The OBM (Open BioMed) Encoder is the central multimodal embedding engine
that unifies text, molecules, and proteins into a common vector space.

This is the "heart" of the BioFlow platform - it enables cross-modal
similarity search (e.g., find proteins similar to a text description).

Architecture:
    ┌─────────────────────────────────────────────┐
    │              OBMEncoder                      │
    │  ┌─────────┐ ┌──────────┐ ┌─────────────┐  │
    │  │  Text   │ │ Molecule │ │   Protein   │  │
    │  │ Encoder │ │  Encoder │ │   Encoder   │  │
    │  │(PubMed) │ │(ChemBERTa│ │   (ESM-2)   │  │
    │  └────┬────┘ └────┬─────┘ └──────┬──────┘  │
    │       │           │              │          │
    │       └───────────┼──────────────┘          │
    │                   ▼                         │
    │           Unified Embedding                 │
    │              (768-dim)                      │
    └─────────────────────────────────────────────┘
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

from bioflow.core import BioEncoder, Modality, EmbeddingResult

logger = logging.getLogger(__name__)


class OBMEncoder(BioEncoder):
    """
    Unified Multimodal Encoder for BioFlow.
    
    Combines specialized encoders for each modality and optionally
    projects them into a shared embedding space.
    
    Example:
        >>> obm = OBMEncoder()
        >>> 
        >>> # Encode different modalities
        >>> text_emb = obm.encode("EGFR inhibitor for lung cancer", Modality.TEXT)
        >>> mol_emb = obm.encode("CC(=O)Oc1ccccc1C(=O)O", Modality.SMILES)  # Aspirin
        >>> prot_emb = obm.encode("MKTVRQERLKSIVRILERSKEPVSG", Modality.PROTEIN)
        >>> 
        >>> # All embeddings have the same dimension
        >>> assert len(text_emb.vector) == len(mol_emb.vector) == len(prot_emb.vector)
    
    Attributes:
        text_encoder: Encoder for biomedical text
        molecule_encoder: Encoder for SMILES molecules
        protein_encoder: Encoder for protein sequences
        output_dim: Dimension of output embeddings (after projection)
    """
    
    def __init__(
        self,
        text_model: str = "pubmedbert",
        molecule_model: str = "chemberta",
        protein_model: str = "esm2_t12",
        device: str = None,
        output_dim: int = 768,
        lazy_load: bool = True
    ):
        """
        Initialize OBMEncoder.
        
        Args:
            text_model: Model for text encoding
            molecule_model: Model for molecule encoding
            protein_model: Model for protein encoding
            device: torch device (auto-detected if None)
            output_dim: Target dimension for all embeddings
            lazy_load: If True, load encoders on first use
        """
        self.text_model = text_model
        self.molecule_model = molecule_model
        self.protein_model = protein_model
        self.device = device
        self._output_dim = output_dim
        self.lazy_load = lazy_load
        
        # Encoders (lazy loaded)
        self._text_encoder = None
        self._molecule_encoder = None
        self._protein_encoder = None
        
        # Projection matrices (for dimension alignment)
        self._projections: Dict[Modality, Any] = {}
        
        if not lazy_load:
            self._load_all_encoders()
        
        logger.info(f"OBMEncoder initialized (lazy_load={lazy_load}, output_dim={output_dim})")
    
    def _load_all_encoders(self):
        """Load all encoders."""
        self._get_text_encoder()
        self._get_molecule_encoder()
        self._get_protein_encoder()
    
    def _get_text_encoder(self):
        """Get or create text encoder."""
        if self._text_encoder is None:
            from bioflow.plugins.encoders.text_encoder import TextEncoder
            self._text_encoder = TextEncoder(
                model_name=self.text_model,
                device=self.device
            )
        return self._text_encoder
    
    def _get_molecule_encoder(self):
        """Get or create molecule encoder."""
        if self._molecule_encoder is None:
            from bioflow.plugins.encoders.molecule_encoder import MoleculeEncoder
            self._molecule_encoder = MoleculeEncoder(
                backend=self.molecule_model if self.molecule_model.startswith("rdkit") else "chemberta",
                model_name=None if self.molecule_model.startswith("rdkit") else self.molecule_model,
                device=self.device
            )
        return self._molecule_encoder
    
    def _get_protein_encoder(self):
        """Get or create protein encoder."""
        if self._protein_encoder is None:
            from bioflow.plugins.encoders.protein_encoder import ProteinEncoder
            self._protein_encoder = ProteinEncoder(
                model_name=self.protein_model,
                device=self.device
            )
        return self._protein_encoder
    
    def _get_encoder_for_modality(self, modality: Modality) -> BioEncoder:
        """Get the appropriate encoder for a modality."""
        if modality == Modality.TEXT:
            return self._get_text_encoder()
        elif modality == Modality.SMILES:
            return self._get_molecule_encoder()
        elif modality == Modality.PROTEIN:
            return self._get_protein_encoder()
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def _project_embedding(self, vector: List[float], source_dim: int) -> List[float]:
        """
        Project embedding to output dimension.
        
        For simplicity, uses truncation/padding. In production,
        you would train a projection layer.
        """
        if source_dim == self._output_dim:
            return vector
        elif source_dim > self._output_dim:
            # Truncate (or use PCA in production)
            return vector[:self._output_dim]
        else:
            # Pad with zeros (or use learned projection)
            return vector + [0.0] * (self._output_dim - source_dim)
    
    @property
    def dimension(self) -> int:
        return self._output_dim
    
    @property
    def supported_modalities(self) -> List[Modality]:
        return [Modality.TEXT, Modality.SMILES, Modality.PROTEIN]
    
    def encode(self, content: Any, modality: Modality) -> EmbeddingResult:
        """
        Encode content from any supported modality.
        
        Args:
            content: Raw input (text, SMILES, or protein sequence)
            modality: Type of the input
            
        Returns:
            EmbeddingResult with unified dimension
        """
        # Get appropriate encoder
        encoder = self._get_encoder_for_modality(modality)
        
        # Encode
        result = encoder.encode(content, modality)
        
        # Project to unified dimension
        projected_vector = self._project_embedding(result.vector, encoder.dimension)
        
        return EmbeddingResult(
            vector=projected_vector,
            modality=modality,
            dimension=self._output_dim,
            metadata={
                **result.metadata,
                "source_encoder": encoder.__class__.__name__,
                "source_dim": encoder.dimension,
                "projected": encoder.dimension != self._output_dim
            }
        )
    
    def encode_auto(self, content: str) -> EmbeddingResult:
        """
        Auto-detect modality and encode.
        
        Uses heuristics to determine if input is:
        - Protein: Contains only amino acid letters (ACDEFGHIKLMNPQRSTVWY)
        - SMILES: Contains typical SMILES characters (=, #, @, etc.)
        - Text: Everything else
        """
        content = content.strip()
        
        # Check for protein (only amino acid letters)
        amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if content.isupper() and set(content).issubset(amino_acids) and len(content) > 10:
            return self.encode(content, Modality.PROTEIN)
        
        # Check for SMILES (contains typical characters)
        smiles_chars = set("=#@[]()+-.")
        if any(c in content for c in smiles_chars) or (
            len(content) < 100 and not " " in content and content[0].isupper()
        ):
            try:
                # Validate as SMILES
                return self.encode(content, Modality.SMILES)
            except:
                pass
        
        # Default to text
        return self.encode(content, Modality.TEXT)
    
    def batch_encode(
        self,
        contents: List[Any],
        modality: Modality
    ) -> List[EmbeddingResult]:
        """Batch encode multiple items of the same modality."""
        encoder = self._get_encoder_for_modality(modality)
        results = encoder.batch_encode(contents, modality)
        
        # Project all to unified dimension
        projected_results = []
        for result in results:
            projected_vector = self._project_embedding(result.vector, encoder.dimension)
            projected_results.append(EmbeddingResult(
                vector=projected_vector,
                modality=modality,
                dimension=self._output_dim,
                metadata={**result.metadata, "source_dim": encoder.dimension}
            ))
        
        return projected_results
    
    def similarity(self, emb1: EmbeddingResult, emb2: EmbeddingResult) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Useful for cross-modal similarity (e.g., text-molecule).
        """
        v1 = np.array(emb1.vector)
        v2 = np.array(emb2.vector)
        
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def get_encoder_info(self) -> Dict[str, Any]:
        """Get information about loaded encoders."""
        info = {
            "output_dim": self._output_dim,
            "device": self.device,
            "encoders": {}
        }
        
        if self._text_encoder:
            info["encoders"]["text"] = {
                "model": self._text_encoder.model_path,
                "dim": self._text_encoder.dimension
            }
        
        if self._molecule_encoder:
            info["encoders"]["molecule"] = {
                "backend": self._molecule_encoder.backend.value,
                "dim": self._molecule_encoder.dimension
            }
        
        if self._protein_encoder:
            info["encoders"]["protein"] = {
                "model": self._protein_encoder.model_path,
                "dim": self._protein_encoder.dimension
            }
        
        return info
