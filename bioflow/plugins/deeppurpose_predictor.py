"""
DeepPurpose Predictor - DTI Prediction
========================================

Implements BioPredictor interface for drug-target interaction prediction.

Note: DeepPurpose is an open-source toolkit for DTI/DDI prediction.
If DeepPurpose is not available, falls back to a simple baseline.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import warnings

from bioflow.core import BioPredictor, PredictionResult

logger = logging.getLogger(__name__)

# Lazy import
_deeppurpose = None
_deeppurpose_available = None


def _check_deeppurpose():
    global _deeppurpose, _deeppurpose_available
    
    if _deeppurpose_available is None:
        try:
            from DeepPurpose import DTI as DeepPurposeDTI
            from DeepPurpose import utils as DeepPurposeUtils
            _deeppurpose = {
                "DTI": DeepPurposeDTI,
                "utils": DeepPurposeUtils
            }
            _deeppurpose_available = True
            logger.info("DeepPurpose is available")
        except ImportError:
            _deeppurpose_available = False
            logger.warning(
                "DeepPurpose not available. Using fallback predictor. "
                "Install with: pip install DeepPurpose"
            )
    
    return _deeppurpose_available, _deeppurpose


class DeepPurposePredictor(BioPredictor):
    """
    Drug-Target Interaction predictor using DeepPurpose.
    
    Predicts binding affinity between a drug (SMILES) and target (protein sequence).
    
    Example:
        >>> predictor = DeepPurposePredictor()
        >>> result = predictor.predict(
        ...     drug="CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        ...     target="MKTVRQERLKSIVRILERSKEPVSG..."  # Target protein
        ... )
        >>> print(result.score)  # Predicted binding affinity
    
    Models (when DeepPurpose is available):
    - Transformer + CNN (default)
    - MPNN + CNN
    - Morgan + AAC (baseline)
    """
    
    AVAILABLE_MODELS = [
        "Transformer_CNN",
        "MPNN_CNN", 
        "Morgan_CNN",
        "Morgan_AAC",
    ]
    
    def __init__(
        self,
        model_type: str = "Transformer_CNN",
        pretrained: str = None,
        device: str = "cpu"
    ):
        """
        Initialize DeepPurposePredictor.
        
        Args:
            model_type: Model architecture (e.g., "Transformer_CNN")
            pretrained: Path to pretrained model
            device: torch device
        """
        self.model_type = model_type
        self.pretrained = pretrained
        self.device = device
        
        available, dp = _check_deeppurpose()
        self._use_deeppurpose = available
        self._model = None
        
        if available and pretrained:
            self._load_pretrained(pretrained)
    
    def _load_pretrained(self, path: str):
        """Load pretrained DeepPurpose model."""
        available, dp = _check_deeppurpose()
        if not available:
            return
        
        try:
            self._model = dp["DTI"].load_pretrained_model(path)
            logger.info(f"Loaded pretrained model from {path}")
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
    
    def _fallback_predict(self, drug: str, target: str) -> Tuple[float, float]:
        """
        Fallback prediction when DeepPurpose is not available.
        
        Uses simple heuristics based on molecular properties.
        This is NOT accurate - just a placeholder.
        """
        # Simple heuristics based on sequence/molecule properties
        drug_score = min(len(drug) / 50.0, 1.0)  # Longer SMILES = higher complexity
        target_score = min(len(target) / 500.0, 1.0)  # Longer protein = more binding sites
        
        # Combine with some randomness
        import random
        random.seed(hash(drug + target) % 2**32)
        base_score = (drug_score + target_score) / 2
        noise = random.uniform(-0.1, 0.1)
        
        score = max(0, min(1, base_score + noise))
        confidence = 0.3  # Low confidence for fallback
        
        return score, confidence
    
    def predict(self, drug: str, target: str) -> PredictionResult:
        """
        Predict drug-target interaction.
        
        Args:
            drug: SMILES string of drug molecule
            target: Protein sequence
            
        Returns:
            PredictionResult with binding affinity score
        """
        if self._use_deeppurpose:
            return self._predict_deeppurpose(drug, target)
        else:
            score, confidence = self._fallback_predict(drug, target)
            return PredictionResult(
                score=score,
                confidence=confidence,
                label="binding" if score > 0.5 else "non-binding",
                metadata={
                    "method": "fallback_heuristic",
                    "warning": "DeepPurpose not available, using simple heuristics"
                }
            )
    
    def _predict_deeppurpose(self, drug: str, target: str) -> PredictionResult:
        """Predict using DeepPurpose."""
        available, dp = _check_deeppurpose()
        
        try:
            # Encode drug and target
            drug_encoding = dp["utils"].drug_encoding(drug, self.model_type.split("_")[0])
            target_encoding = dp["utils"].target_encoding(target, self.model_type.split("_")[1])
            
            # Predict
            if self._model:
                y_pred = self._model.predict(drug_encoding, target_encoding)
            else:
                # Train a quick model or use default
                warnings.warn("No pretrained model loaded, predictions may be unreliable")
                y_pred = [0.5]  # Default
            
            score = float(y_pred[0]) if hasattr(y_pred, '__iter__') else float(y_pred)
            
            return PredictionResult(
                score=score,
                confidence=0.8,
                label="binding" if score > 0.5 else "non-binding",
                metadata={
                    "method": "deeppurpose",
                    "model_type": self.model_type,
                    "drug_smiles": drug[:50],
                    "target_length": len(target)
                }
            )
            
        except Exception as e:
            logger.error(f"DeepPurpose prediction failed: {e}")
            # Fallback
            score, confidence = self._fallback_predict(drug, target)
            return PredictionResult(
                score=score,
                confidence=confidence,
                label="binding" if score > 0.5 else "non-binding",
                metadata={"method": "fallback", "error": str(e)}
            )
    
    def batch_predict(self, pairs: List[Tuple[str, str]]) -> List[PredictionResult]:
        """
        Batch predict drug-target interactions.
        
        Args:
            pairs: List of (drug_smiles, target_sequence) tuples
            
        Returns:
            List of PredictionResults
        """
        return [self.predict(drug, target) for drug, target in pairs]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.model_type,
            "use_deeppurpose": self._use_deeppurpose,
            "pretrained": self.pretrained,
            "device": self.device,
            "available_models": self.AVAILABLE_MODELS
        }
