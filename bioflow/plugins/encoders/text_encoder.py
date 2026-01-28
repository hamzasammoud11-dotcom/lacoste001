"""
Text Encoder - PubMedBERT / SciBERT
====================================

Encodes biomedical text (abstracts, clinical notes) into vectors.

Models:
- microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext (default)
- allenai/scibert_scivocab_uncased
- allenai/specter
"""

import logging
from typing import List, Optional
import numpy as np

from bioflow.core import BioEncoder, Modality, EmbeddingResult

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
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
                "transformers and torch are required for TextEncoder. "
                "Install with: pip install transformers torch"
            )
    return _transformers, _torch


class TextEncoder(BioEncoder):
    """
    Encoder for biomedical text using PubMedBERT or similar models.
    
    Example:
        >>> encoder = TextEncoder(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        >>> result = encoder.encode("EGFR mutations in lung cancer", Modality.TEXT)
        >>> print(len(result.vector))  # 768
    """
    
    SUPPORTED_MODELS = {
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "scibert": "allenai/scibert_scivocab_uncased",
        "specter": "allenai/specter",
        "biobert": "dmis-lab/biobert-base-cased-v1.2",
    }
    
    def __init__(
        self,
        model_name: str = "pubmedbert",
        device: str = None,
        max_length: int = 512,
        pooling: str = "mean"  # mean, cls, max
    ):
        """
        Initialize TextEncoder.
        
        Args:
            model_name: Model key or HuggingFace model path
            device: torch device (auto-detected if None)
            max_length: Maximum token length
            pooling: Pooling strategy for embeddings
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
        
        # Load model and tokenizer
        logger.info(f"Loading TextEncoder: {self.model_path} on {self.device}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
        self.model = transformers.AutoModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self._dimension = self.model.config.hidden_size
        logger.info(f"TextEncoder ready (dim={self._dimension})")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def supported_modalities(self) -> List[Modality]:
        return [Modality.TEXT]
    
    def encode(self, content: str, modality: Modality = Modality.TEXT) -> EmbeddingResult:
        """Encode text into a vector."""
        if modality != Modality.TEXT:
            raise ValueError(f"TextEncoder only supports TEXT modality, got {modality}")
        
        transformers, torch = _load_transformers()
        
        # Tokenize
        inputs = self.tokenizer(
            content,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            # Apply pooling
            if self.pooling == "cls":
                embedding = hidden_states[:, 0, :]
            elif self.pooling == "mean":
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                embedding = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
            elif self.pooling == "max":
                embedding = hidden_states.max(dim=1).values
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
        
        vector = embedding.squeeze().cpu().numpy().tolist()
        
        return EmbeddingResult(
            vector=vector,
            modality=modality,
            dimension=self._dimension,
            metadata={"model": self.model_path, "pooling": self.pooling}
        )
    
    def batch_encode(self, contents: List[str], modality: Modality = Modality.TEXT) -> List[EmbeddingResult]:
        """Batch encode multiple texts."""
        transformers, torch = _load_transformers()
        
        inputs = self.tokenizer(
            contents,
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
            elif self.pooling == "mean":
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
            elif self.pooling == "max":
                embeddings = hidden_states.max(dim=1).values
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
        
        results = []
        for i, emb in enumerate(embeddings):
            results.append(EmbeddingResult(
                vector=emb.cpu().numpy().tolist(),
                modality=modality,
                dimension=self._dimension,
                metadata={"model": self.model_path, "index": i}
            ))
        
        return results
