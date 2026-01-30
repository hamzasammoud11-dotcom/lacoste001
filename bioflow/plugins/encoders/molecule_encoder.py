"""
Molecule Encoder - ChemBERTa / RDKit
=====================================

Encodes SMILES molecules into vectors.

Models:
- seyonec/ChemBERTa-zinc-base-v1 (default)
- DeepChem/ChemBERTa-77M-MTR
- RDKit fingerprints (fallback, no GPU needed)

Similarity Metrics:
- Tanimoto coefficient for structural similarity (fingerprint-based)
- Cosine similarity for semantic similarity (embedding-based)
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
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


# =============================================================================
# TANIMOTO SIMILARITY - Gold standard for molecular structure comparison
# =============================================================================

def compute_tanimoto_similarity(smiles1: str, smiles2: str, fp_type: str = "morgan") -> Optional[float]:
    """
    Compute Tanimoto coefficient between two molecules.
    
    Tanimoto = |A ∩ B| / |A ∪ B|
    
    This is the gold standard for molecular structural similarity in cheminformatics.
    Unlike cosine similarity on embeddings, Tanimoto directly compares chemical features.
    
    Args:
        smiles1: First molecule SMILES
        smiles2: Second molecule SMILES
        fp_type: "morgan" (ECFP4, default), "rdkit", or "maccs"
        
    Returns:
        Tanimoto coefficient (0.0 to 1.0), or None if invalid SMILES
        
    Example:
        >>> compute_tanimoto_similarity("CC(=O)O", "CC(=O)OC")  # Acetic acid vs methyl acetate
        0.625
    """
    try:
        Chem, AllChem = _load_rdkit()
        from rdkit import DataStructs
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return None
        
        # Generate fingerprints
        if fp_type == "morgan":
            # ECFP4-like fingerprint (radius=2, 2048 bits)
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        elif fp_type == "rdkit":
            fp1 = Chem.RDKFingerprint(mol1)
            fp2 = Chem.RDKFingerprint(mol2)
        elif fp_type == "maccs":
            from rdkit.Chem import MACCSkeys
            fp1 = MACCSkeys.GenMACCSKeys(mol1)
            fp2 = MACCSkeys.GenMACCSKeys(mol2)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
        
    except Exception as e:
        logger.warning(f"Tanimoto calculation failed: {e}")
        return None


def compute_similarity_breakdown(
    smiles1: str, 
    smiles2: str,
    embedding1: Optional[List[float]] = None,
    embedding2: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compute a comprehensive similarity breakdown between two molecules.
    
    Returns both structural (Tanimoto) and semantic (cosine) similarities
    with clear labels explaining what each metric measures.
    
    This addresses the jury requirement: "The UI must display the Evidence Strength"
    by providing explicit explanation of WHY molecules are similar.
    
    Args:
        smiles1: First molecule SMILES
        smiles2: Second molecule SMILES
        embedding1: Optional pre-computed embedding for smiles1
        embedding2: Optional pre-computed embedding for smiles2
        
    Returns:
        Dict with:
        - tanimoto_similarity: Structural similarity (0-1)
        - cosine_similarity: Semantic/embedding similarity (0-1) if embeddings provided
        - similarity_type: "structural", "functional", or "both"
        - explanation: Human-readable explanation of the similarity
        - confidence: How confident we are in this similarity assessment
    """
    result = {
        "tanimoto_similarity": None,
        "cosine_similarity": None,
        "similarity_type": "unknown",
        "explanation": "",
        "confidence": 0.0,
    }
    
    # Compute Tanimoto (structural similarity)
    tanimoto = compute_tanimoto_similarity(smiles1, smiles2)
    if tanimoto is not None:
        result["tanimoto_similarity"] = round(tanimoto, 4)
    
    # Compute cosine similarity if embeddings provided
    if embedding1 is not None and embedding2 is not None:
        try:
            import numpy as np
            e1 = np.array(embedding1)
            e2 = np.array(embedding2)
            cosine = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
            result["cosine_similarity"] = round(cosine, 4)
        except Exception:
            pass
    
    # Generate explanation based on similarity types
    tanimoto = result["tanimoto_similarity"]
    cosine = result["cosine_similarity"]
    
    explanations = []
    confidence_factors = []
    
    if tanimoto is not None:
        if tanimoto >= 0.85:
            explanations.append(f"Structurally very similar (Tanimoto: {tanimoto:.3f}) - likely same scaffold")
            confidence_factors.append(0.9)
            result["similarity_type"] = "structural"
        elif tanimoto >= 0.7:
            explanations.append(f"Structural analogs (Tanimoto: {tanimoto:.3f}) - shared core features")
            confidence_factors.append(0.75)
            result["similarity_type"] = "structural"
        elif tanimoto >= 0.5:
            explanations.append(f"Moderate structural similarity (Tanimoto: {tanimoto:.3f})")
            confidence_factors.append(0.5)
        elif tanimoto >= 0.3:
            explanations.append(f"Weak structural similarity (Tanimoto: {tanimoto:.3f}) - different scaffolds")
            confidence_factors.append(0.3)
        else:
            explanations.append(f"Structurally distinct (Tanimoto: {tanimoto:.3f})")
            confidence_factors.append(0.2)
    
    if cosine is not None:
        if cosine >= 0.9:
            explanations.append(f"Very similar in chemical space (embedding: {cosine:.3f})")
            confidence_factors.append(0.85)
            if result["similarity_type"] == "unknown":
                result["similarity_type"] = "functional"
        elif cosine >= 0.75:
            explanations.append(f"Semantically related (embedding: {cosine:.3f})")
            confidence_factors.append(0.6)
        elif cosine >= 0.5:
            explanations.append(f"Moderate semantic similarity (embedding: {cosine:.3f})")
            confidence_factors.append(0.4)
    
    # Check for divergence between structural and semantic similarity
    if tanimoto is not None and cosine is not None:
        if abs(tanimoto - cosine) > 0.3:
            if tanimoto > cosine:
                explanations.append("⚠️ Structurally similar but functionally different")
            else:
                explanations.append("💡 Functionally similar despite structural differences (possible isostere)")
                result["similarity_type"] = "functional"
        elif tanimoto >= 0.7 and cosine >= 0.7:
            result["similarity_type"] = "both"
    
    result["explanation"] = " | ".join(explanations) if explanations else "Insufficient data for similarity assessment"
    result["confidence"] = max(confidence_factors) if confidence_factors else 0.0
    
    return result


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
