"""
Protein Encoder - ESM-2 / ProtBERT
===================================

Encodes protein sequences into vectors.

Models:
- facebook/esm2_t33_650M_UR50D (default, 1280-dim)
- facebook/esm2_t12_35M_UR50D (smaller, 480-dim)
- Rostlab/prot_bert (1024-dim)
"""

import logging
from typing import List, Optional

from bioflow.core import BioEncoder, Modality, EmbeddingResult

logger = logging.getLogger(__name__)

# Lazy imports
_transformers = None
_torch = None


def _load_transformers():
    global _transformers, _torch
    if _transformers is None:
        try:
            import transformers
            import torch
            _transformers = transformers
            _torch = torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )
    return _transformers, _torch


class ProteinEncoder(BioEncoder):
    """
    Encoder for protein sequences using ESM-2 or ProtBERT.
    
    Example:
        >>> encoder = ProteinEncoder(model_name="esm2_t12")
        >>> result = encoder.encode("MKTVRQERLKSIVRILERSKEPVSG", Modality.PROTEIN)
        >>> print(len(result.vector))  # 480
    """
    
    SUPPORTED_MODELS = {
        "esm2_t33": "facebook/esm2_t33_650M_UR50D",      # 1280-dim, 650M params
        "esm2_t30": "facebook/esm2_t30_150M_UR50D",      # 640-dim, 150M params
        "esm2_t12": "facebook/esm2_t12_35M_UR50D",       # 480-dim, 35M params (fast)
        "esm2_t6": "facebook/esm2_t6_8M_UR50D",          # 320-dim, 8M params (fastest)
        "protbert": "Rostlab/prot_bert",                 # 1024-dim
        "protbert_bfd": "Rostlab/prot_bert_bfd",         # 1024-dim, larger
    }
    
    def __init__(
        self,
        model_name: str = "esm2_t12",
        device: str = None,
        max_length: int = 1024,
        pooling: str = "mean"
    ):
        """
        Initialize ProteinEncoder.
        
        Args:
            model_name: Model key or HuggingFace path
            device: torch device
            max_length: Max sequence length
            pooling: Pooling strategy (mean, cls)
        """
        transformers, torch = _load_transformers()
        
        # Resolve model name
        self.model_path = self.SUPPORTED_MODELS.get(model_name.lower(), model_name)
        self.max_length = max_length
        self.pooling = pooling
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model
        logger.info(f"Loading ProteinEncoder: {self.model_path} on {self.device}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
        self.model = transformers.AutoModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self._dimension = self.model.config.hidden_size
        logger.info(f"ProteinEncoder ready (dim={self._dimension})")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def supported_modalities(self) -> List[Modality]:
        return [Modality.PROTEIN]
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess protein sequence for encoding."""
        # Remove whitespace
        sequence = sequence.strip().upper()
        
        # For ProtBERT, add spaces between amino acids
        if "prot_bert" in self.model_path.lower():
            sequence = " ".join(list(sequence))
        
        return sequence
    
    def encode(self, content: str, modality: Modality = Modality.PROTEIN) -> EmbeddingResult:
        """Encode protein sequence into a vector."""
        if modality != Modality.PROTEIN:
            raise ValueError(f"ProteinEncoder only supports PROTEIN modality, got {modality}")
        
        transformers, torch = _load_transformers()
        
        # Preprocess
        sequence = self._preprocess_sequence(content)
        
        # Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            if self.pooling == "cls":
                embedding = hidden_states[:, 0, :]
            else:  # mean
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                embedding = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
        
        vector = embedding.squeeze().cpu().numpy().tolist()
        
        return EmbeddingResult(
            vector=vector,
            modality=modality,
            dimension=self._dimension,
            metadata={
                "model": self.model_path,
                "sequence_length": len(content)
            }
        )
    
    def batch_encode(self, contents: List[str], modality: Modality = Modality.PROTEIN) -> List[EmbeddingResult]:
        """Batch encode protein sequences."""
        transformers, torch = _load_transformers()
        
        sequences = [self._preprocess_sequence(s) for s in contents]
        
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            # Apply same pooling strategy as encode()
            if self.pooling == "cls":
                embeddings = hidden_states[:, 0, :]
            else:  # mean pooling
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
        
        results = []
        for i, emb in enumerate(embeddings):
            results.append(EmbeddingResult(
                vector=emb.cpu().numpy().tolist(),
                modality=modality,
                dimension=self._dimension,
                metadata={"model": self.model_path, "sequence_length": len(contents[i])}
            ))
        
        return results
