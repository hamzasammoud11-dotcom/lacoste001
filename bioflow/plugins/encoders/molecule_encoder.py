"""
Molecule Encoder - ChemBERTa / RDKit
=====================================

Encodes SMILES molecules into vectors.

Models:
- seyonec/ChemBERTa-zinc-base-v1 (default)
- DeepChem/ChemBERTa-77M-MTR
- RDKit fingerprints (fallback, no GPU needed)
"""

import logging
from typing import List, Optional
from enum import Enum

from bioflow.core import BioEncoder, Modality, EmbeddingResult

logger = logging.getLogger(__name__)

# Lazy imports
_transformers = None
_torch = None
_rdkit = None


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


def _load_rdkit():
    global _rdkit
    if _rdkit is None:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            _rdkit = (Chem, AllChem)
        except ImportError:
            raise ImportError(
                "RDKit is required for fingerprint encoding. "
                "Install with: pip install rdkit"
            )
    return _rdkit


class MoleculeEncoderBackend(Enum):
    CHEMBERTA = "chemberta"
    RDKIT_MORGAN = "rdkit_morgan"
    RDKIT_MACCS = "rdkit_maccs"


class MoleculeEncoder(BioEncoder):
    """
    Encoder for SMILES molecules using ChemBERTa or RDKit fingerprints.
    
    Example:
        >>> encoder = MoleculeEncoder(backend="chemberta")
        >>> result = encoder.encode("CCO", Modality.SMILES)  # Ethanol
        >>> print(len(result.vector))  # 768
        
        >>> encoder = MoleculeEncoder(backend="rdkit_morgan")
        >>> result = encoder.encode("CCO", Modality.SMILES)
        >>> print(len(result.vector))  # 2048
    """
    
    SUPPORTED_MODELS = {
        "chemberta": "seyonec/ChemBERTa-zinc-base-v1",
        "chemberta-77m": "DeepChem/ChemBERTa-77M-MTR",
    }
    
    def __init__(
        self,
        backend: str = "chemberta",
        model_name: str = None,
        device: str = None,
        fp_size: int = 2048,  # For RDKit fingerprints
        fp_radius: int = 2,   # For Morgan fingerprints
    ):
        """
        Initialize MoleculeEncoder.
        
        Args:
            backend: "chemberta", "rdkit_morgan", or "rdkit_maccs"
            model_name: HuggingFace model path (for chemberta)
            device: torch device
            fp_size: Fingerprint size (for RDKit)
            fp_radius: Morgan fingerprint radius
        """
        self.backend = MoleculeEncoderBackend(backend.lower())
        self.fp_size = fp_size
        self.fp_radius = fp_radius
        
        if self.backend == MoleculeEncoderBackend.CHEMBERTA:
            transformers, torch = _load_transformers()
            
            self.model_path = model_name or self.SUPPORTED_MODELS["chemberta"]
            
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            
            logger.info(f"Loading MoleculeEncoder: {self.model_path} on {self.device}")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
            self.model = transformers.AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            self._dimension = self.model.config.hidden_size
        else:
            # RDKit fingerprints
            _load_rdkit()
            self.device = "cpu"
            self.model = None
            self.tokenizer = None
            
            if self.backend == MoleculeEncoderBackend.RDKIT_MORGAN:
                self._dimension = fp_size
            else:  # MACCS
                self._dimension = 167
        
        logger.info(f"MoleculeEncoder ready (backend={backend}, dim={self._dimension})")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def supported_modalities(self) -> List[Modality]:
        return [Modality.SMILES]
    
    def _encode_chemberta(self, smiles: str) -> List[float]:
        """Encode using ChemBERTa."""
        transformers, torch = _load_transformers()
        
        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden_states = outputs.last_hidden_state
            embedding = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
        
        return embedding.squeeze().cpu().numpy().tolist()
    
    def _encode_rdkit(self, smiles: str) -> List[float]:
        """Encode using RDKit fingerprints."""
        Chem, AllChem = _load_rdkit()
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        if self.backend == MoleculeEncoderBackend.RDKIT_MORGAN:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_size)
        else:  # MACCS
            from rdkit.Chem import MACCSkeys
            fp = MACCSkeys.GenMACCSKeys(mol)
        
        return list(fp)
    
    def encode(self, content: str, modality: Modality = Modality.SMILES) -> EmbeddingResult:
        """Encode SMILES into a vector."""
        if modality != Modality.SMILES:
            raise ValueError(f"MoleculeEncoder only supports SMILES modality, got {modality}")
        
        if self.backend == MoleculeEncoderBackend.CHEMBERTA:
            vector = self._encode_chemberta(content)
        else:
            vector = self._encode_rdkit(content)
        
        return EmbeddingResult(
            vector=vector,
            modality=modality,
            dimension=self._dimension,
            metadata={"backend": self.backend.value, "smiles": content}
        )
    
    def batch_encode(self, contents: List[str], modality: Modality = Modality.SMILES) -> List[EmbeddingResult]:
        """Batch encode SMILES."""
        if self.backend == MoleculeEncoderBackend.CHEMBERTA:
            transformers, torch = _load_transformers()
            
            inputs = self.tokenizer(
                contents,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                hidden_states = outputs.last_hidden_state
                embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
            
            results = []
            for i, emb in enumerate(embeddings):
                results.append(EmbeddingResult(
                    vector=emb.cpu().numpy().tolist(),
                    modality=modality,
                    dimension=self._dimension,
                    metadata={"backend": self.backend.value, "smiles": contents[i]}
                ))
            return results
        else:
            return [self.encode(s, modality) for s in contents]
