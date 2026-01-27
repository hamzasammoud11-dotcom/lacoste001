"""
BioFlow Model Service
======================
Unified service for accessing OpenBioMed models.
Provides molecule encoding, protein folding, property prediction, and more.

NO FALLBACKS - Models must work or fail explicitly.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Available model types."""
    MOLECULE_ENCODER = "molecule_encoder"
    PROTEIN_ENCODER = "protein_encoder"
    TEXT_ENCODER = "text_encoder"
    PROPERTY_PREDICTOR = "property_predictor"
    DTI_PREDICTOR = "dti_predictor"
    MOLECULE_GENERATOR = "molecule_generator"
    PROTEIN_FOLDER = "protein_folder"


@dataclass
class EncodingResult:
    """Result from encoding operation."""
    vector: List[float]
    modality: str
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result from prediction operation."""
    value: float
    confidence: Optional[float] = None
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result from generation operation."""
    output: str
    score: Optional[float] = None
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelServiceError(Exception):
    """Raised when a model operation fails and cannot continue."""
    pass


class DependencyError(ModelServiceError):
    """Raised when a required dependency is not available."""
    pass


class ModelService:
    """
    Unified service for OpenBioMed model access.
    
    NO FALLBACKS - All models must be operational or they will raise errors.
    
    Example:
        >>> service = ModelService()
        >>> 
        >>> # Encode a molecule
        >>> result = service.encode_molecule("CCO")
        >>> print(f"Embedding dim: {len(result.vector)}")
        >>> 
        >>> # Predict property
        >>> pred = service.predict_property("CCO", "logP")
        >>> print(f"logP: {pred.value}")
    """
    
    def __init__(
        self,
        device: str = None,
        lazy_load: bool = True
    ):
        """
        Initialize ModelService.
        
        Args:
            device: PyTorch device (auto-detected if None)
            lazy_load: Load models on first use
        """
        self.device = device
        self.lazy_load = lazy_load
        
        # Model cache
        self._models: Dict[ModelType, Any] = {}
        self._obm_encoder = None
        self._dti_predictor = None
        
        # Validate dependencies at init
        self._validate_dependencies()
        
        logger.info(f"ModelService initialized (device={self.device})")
    
    def _validate_dependencies(self):
        """Validate that required dependencies are available."""
        # Check PyTorch
        try:
            import torch
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"PyTorch available, using device: {self.device}")
        except ImportError:
            raise DependencyError(
                "PyTorch is required but not installed. "
                "Install with: pip install torch"
            )
        
        # Check RDKit (required for molecular operations)
        try:
            from rdkit import Chem
            logger.info("RDKit available")
        except ImportError:
            raise DependencyError(
                "RDKit is required but not installed. "
                "Install with: pip install rdkit"
            )
    
    # =========================================================================
    # OBM Encoder (unified multimodal)
    # =========================================================================
    
    def get_obm_encoder(self):
        """
        Get the OBM encoder for multimodal embeddings.
        
        Returns:
            OBMEncoder instance configured for multimodal (text, molecule, protein) encoding.
        """
        return self._get_obm_encoder()
    
    def _get_obm_encoder(self):
        """Get or create OBM encoder."""
        if self._obm_encoder is None:
            try:
                from bioflow.plugins.obm_encoder import OBMEncoder
                self._obm_encoder = OBMEncoder(
                    device=self.device,
                    lazy_load=True
                )
                logger.info("OBMEncoder loaded successfully")
            except ImportError as e:
                raise DependencyError(
                    f"OBMEncoder not available: {e}. "
                    "Ensure bioflow.plugins.obm_encoder is properly installed."
                )
            except Exception as e:
                raise ModelServiceError(f"Failed to initialize OBMEncoder: {e}")
        return self._obm_encoder
    
    def encode_molecule(self, smiles: str) -> EncodingResult:
        """
        Encode a SMILES molecule to embedding vector.
        
        Args:
            smiles: SMILES string
            
        Returns:
            EncodingResult with embedding vector
            
        Raises:
            ModelServiceError: If encoding fails
        """
        try:
            encoder = self._get_obm_encoder()
            from bioflow.core import Modality
            result = encoder.encode(smiles, Modality.SMILES)
            return EncodingResult(
                vector=result.vector.tolist() if hasattr(result.vector, 'tolist') else list(result.vector),
                modality="molecule",
                model_name=encoder.molecule_model,
                metadata={"smiles": smiles}
            )
        except DependencyError:
            raise
        except Exception as e:
            raise ModelServiceError(f"Molecule encoding failed for '{smiles}': {e}")
    
    def encode_protein(self, sequence: str) -> EncodingResult:
        """
        Encode a protein sequence to embedding vector.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            EncodingResult with embedding vector
            
        Raises:
            ModelServiceError: If encoding fails
        """
        try:
            encoder = self._get_obm_encoder()
            from bioflow.core import Modality
            result = encoder.encode(sequence, Modality.PROTEIN)
            return EncodingResult(
                vector=result.vector.tolist() if hasattr(result.vector, 'tolist') else list(result.vector),
                modality="protein",
                model_name=encoder.protein_model,
                metadata={"sequence_length": len(sequence)}
            )
        except DependencyError:
            raise
        except Exception as e:
            raise ModelServiceError(f"Protein encoding failed: {e}")
    
    def encode_text(self, text: str) -> EncodingResult:
        """
        Encode biomedical text to embedding vector.
        
        Args:
            text: Natural language text
            
        Returns:
            EncodingResult with embedding vector
            
        Raises:
            ModelServiceError: If encoding fails
        """
        try:
            encoder = self._get_obm_encoder()
            from bioflow.core import Modality
            result = encoder.encode(text, Modality.TEXT)
            return EncodingResult(
                vector=result.vector.tolist() if hasattr(result.vector, 'tolist') else list(result.vector),
                modality="text",
                model_name=encoder.text_model,
                metadata={"text_length": len(text)}
            )
        except DependencyError:
            raise
        except Exception as e:
            raise ModelServiceError(f"Text encoding failed: {e}")
    
    # =========================================================================
    # DTI Prediction
    # =========================================================================
    
    def _get_dti_predictor(self):
        """Get or create DTI predictor."""
        if self._dti_predictor is None:
            try:
                from bioflow.api.dti_predictor import DeepPurposePredictor
                self._dti_predictor = DeepPurposePredictor()
                logger.info("DeepPurposePredictor loaded successfully")
            except ImportError as e:
                raise DependencyError(
                    f"DeepPurposePredictor not available: {e}. "
                    "Ensure DeepPurpose is installed."
                )
            except Exception as e:
                raise ModelServiceError(f"Failed to initialize DeepPurposePredictor: {e}")
        return self._dti_predictor
    
    def predict_dti(
        self,
        drug_smiles: str,
        target_sequence: str,
        dataset: str = "DAVIS"
    ) -> PredictionResult:
        """
        Predict drug-target interaction affinity.
        
        Args:
            drug_smiles: Drug SMILES string
            target_sequence: Target protein sequence
            dataset: Dataset for model (DAVIS, KIBA, BindingDB_Kd)
            
        Returns:
            PredictionResult with predicted affinity
            
        Raises:
            ModelServiceError: If prediction fails
        """
        predictor = self._get_dti_predictor()
        
        try:
            result = predictor.predict(drug_smiles, target_sequence)
            return PredictionResult(
                value=result.binding_affinity,
                confidence=result.confidence,
                model_name=f"DeepPurpose_{result.drug_encoding}_{result.target_encoding}",
                metadata={
                    "dataset": dataset,
                    "drug_encoding": result.drug_encoding,
                    "target_encoding": result.target_encoding
                }
            )
        except Exception as e:
            raise ModelServiceError(f"DTI prediction failed: {e}")
    
    # =========================================================================
    # Property Prediction
    # =========================================================================
    
    def predict_property(
        self,
        smiles: str,
        property_name: str
    ) -> PredictionResult:
        """
        Predict molecular property using RDKit.
        
        Args:
            smiles: SMILES string
            property_name: Property to predict (logP, TPSA, MW, QED, etc.)
            
        Returns:
            PredictionResult with predicted value
            
        Raises:
            ModelServiceError: If prediction fails
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED as QEDModule
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            prop_funcs = {
                "logp": Descriptors.MolLogP,
                "tpsa": Descriptors.TPSA,
                "mw": Descriptors.MolWt,
                "hbd": Descriptors.NumHDonors,
                "hba": Descriptors.NumHAcceptors,
                "rotatable_bonds": Descriptors.NumRotatableBonds,
                "qed": lambda m: QEDModule.qed(m),
            }
            
            prop_key = property_name.lower()
            if prop_key not in prop_funcs:
                raise ValueError(
                    f"Unknown property: {property_name}. "
                    f"Available: {list(prop_funcs.keys())}"
                )
            
            value = prop_funcs[prop_key](mol)
            return PredictionResult(
                value=float(value),
                confidence=1.0,  # RDKit is deterministic
                model_name="RDKit",
                metadata={"property": property_name}
            )
        except ImportError:
            raise DependencyError("RDKit is required for property prediction")
        except ValueError:
            raise
        except Exception as e:
            raise ModelServiceError(f"Property prediction failed: {e}")
    
    # =========================================================================
    # Molecule Generation
    # =========================================================================
    
    def generate_molecule(
        self,
        prompt: str,
        num_samples: int = 5
    ) -> List[GenerationResult]:
        """
        Generate molecules from text description.
        
        Args:
            prompt: Text description of desired molecule
            num_samples: Number of molecules to generate
            
        Returns:
            List of GenerationResult with SMILES
            
        Raises:
            ModelServiceError: If generation fails
        """
        try:
            from open_biomed.models import MODEL_REGISTRY
            # Use BioT5 or MolT5 for generation
            # TODO: Implement actual OBM-based generation
            raise NotImplementedError(
                "Molecule generation requires BioT5/MolT5 model weights. "
                "This feature is not yet fully implemented."
            )
        except ImportError:
            raise DependencyError(
                "OpenBioMed is required for molecule generation. "
                "Ensure open_biomed is properly installed with model weights."
            )
    
    # =========================================================================
    # Similarity Search
    # =========================================================================
    
    def compute_similarity(
        self,
        query: str,
        candidates: List[str],
        modality: str = "molecule"
    ) -> List[Dict[str, Any]]:
        """
        Compute similarity between query and candidates.
        
        Args:
            query: Query SMILES/sequence/text
            candidates: List of candidate SMILES/sequences/texts
            modality: Type of data (molecule, protein, text)
            
        Returns:
            List of dicts with candidate and similarity score
            
        Raises:
            ModelServiceError: If similarity computation fails
        """
        # Encode query
        if modality == "molecule":
            query_enc = self.encode_molecule(query)
        elif modality == "protein":
            query_enc = self.encode_protein(query)
        elif modality == "text":
            query_enc = self.encode_text(query)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        query_vec = np.array(query_enc.vector)
        
        results = []
        for candidate in candidates:
            try:
                if modality == "molecule":
                    cand_enc = self.encode_molecule(candidate)
                elif modality == "protein":
                    cand_enc = self.encode_protein(candidate)
                else:
                    cand_enc = self.encode_text(candidate)
                
                cand_vec = np.array(cand_enc.vector)
                
                # Cosine similarity
                similarity = float(np.dot(query_vec, cand_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(cand_vec) + 1e-8
                ))
                
                results.append({
                    "candidate": candidate,
                    "similarity": similarity,
                    "modality": modality
                })
            except Exception as e:
                # Include the error in results rather than silently skipping
                results.append({
                    "candidate": candidate,
                    "similarity": None,
                    "modality": modality,
                    "error": str(e)
                })
        
        # Sort by similarity (None values at end)
        results.sort(key=lambda x: x["similarity"] if x["similarity"] is not None else -1, reverse=True)
        return results


# ============================================================================
# Singleton Instance
# ============================================================================
_model_service: Optional[ModelService] = None


def get_model_service(
    device: str = None,
    lazy_load: bool = True,
    reset: bool = False
) -> ModelService:
    """
    Get or create the global ModelService instance.
    
    Args:
        device: PyTorch device
        lazy_load: Load models lazily
        reset: Force create new instance
        
    Returns:
        ModelService singleton
        
    Raises:
        DependencyError: If required dependencies are not available
    """
    global _model_service
    if _model_service is None or reset:
        _model_service = ModelService(
            device=device,
            lazy_load=lazy_load
        )
    return _model_service
