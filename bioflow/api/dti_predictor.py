"""
BioFlow DTI Predictor
======================
Drug-Target Interaction prediction using DeepPurpose.
Integrated from lacoste001/deeppurpose002.py for the hackathon.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class DTIPrediction:
    """Result of a DTI prediction."""
    drug_smiles: str
    target_sequence: str
    binding_affinity: float  # pKd or similar
    confidence: float
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug_smiles": self.drug_smiles,
            "target_sequence": self.target_sequence[:50] + "..." if len(self.target_sequence) > 50 else self.target_sequence,
            "binding_affinity": self.binding_affinity,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "metadata": self.metadata,
        }


@dataclass
class DTIMetrics:
    """Metrics from model evaluation."""
    mse: float
    rmse: float
    mae: float
    pearson: float
    spearman: float
    concordance_index: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "concordance_index": self.concordance_index,
        }


# ============================================================================
# Metric Functions (from deeppurpose002.py)
# ============================================================================
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation."""
    a = pd.Series(np.asarray(y_true, dtype=float).reshape(-1)).rank(method="average").to_numpy()
    b = pd.Series(np.asarray(y_pred, dtype=float).reshape(-1)).rank(method="average").to_numpy()
    return pearson(a, b)


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray, max_n: int = 2000, seed: int = 0) -> float:
    """
    Concordance Index (CI) - approximated for large datasets.
    Measures pairwise ranking accuracy.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = len(y_true)
    
    if n < 2:
        return float("nan")

    # Sample if too large
    if n > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        n = max_n

    conc = 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            total += 1.0
            dt = y_true[i] - y_true[j]
            dp = y_pred[i] - y_pred[j]
            prod = dt * dp
            if prod > 0:
                conc += 1.0
            elif prod == 0:
                conc += 0.5
                
    if total == 0:
        return float("nan")
    return float(conc / total)


# ============================================================================
# DeepPurpose Predictor Class
# ============================================================================
class DeepPurposePredictor:
    """
    Drug-Target Interaction predictor using DeepPurpose.
    
    Supports multiple encoding strategies:
    - Drug: Morgan, CNN, Transformer, MPNN
    - Target: CNN, Transformer, AAC
    
    Example:
        >>> predictor = DeepPurposePredictor()
        >>> result = predictor.predict("CCO", "MKTVRQERLKSIVRILERSKEPVSG")
        >>> print(result.binding_affinity)
    """
    
    def __init__(
        self,
        drug_encoding: str = "Morgan",
        target_encoding: str = "CNN",
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.drug_encoding = drug_encoding
        self.target_encoding = target_encoding
        self.model_path = model_path
        self.device = device
        self.model = None
        self._loaded = False
        
    def load_model(self) -> bool:
        """Load the DeepPurpose model."""
        try:
            from DeepPurpose import DTI as dp_models
            from DeepPurpose import utils
            
            if self.model_path and os.path.exists(self.model_path):
                # Load pre-trained model
                self.model = dp_models.model_pretrained(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            else:
                # Initialize new model (for inference with pre-trained weights)
                config = utils.generate_config(
                    drug_encoding=self.drug_encoding,
                    target_encoding=self.target_encoding,
                    cls_hidden_dims=[1024, 1024, 512],
                )
                self.model = dp_models.model_initialize(**config)
                logger.info(f"Initialized new model: {self.drug_encoding}-{self.target_encoding}")
            
            self._loaded = True
            return True
            
        except ImportError as e:
            raise ImportError(
                f"DeepPurpose is required but not installed: {e}. "
                "Install with: pip install DeepPurpose"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load DeepPurpose model: {e}")
    
    def predict(self, drug_smiles: str, target_sequence: str) -> DTIPrediction:
        """
        Predict binding affinity between drug and target.
        
        Args:
            drug_smiles: SMILES string of the drug molecule
            target_sequence: Amino acid sequence of the target protein
            
        Returns:
            DTIPrediction with binding affinity and confidence
        """
        if not self._loaded:
            self.load_model()
        
        if self.model is not None:
            try:
                from DeepPurpose import utils
                
                # Prepare data for prediction
                X_drug = [drug_smiles]
                X_target = [target_sequence]
                y_dummy = [0]  # Not used for prediction
                
                data = utils.data_process(
                    X_drug, X_target, y_dummy,
                    drug_encoding=self.drug_encoding,
                    target_encoding=self.target_encoding,
                    split_method="no_split",
                )
                
                # Get prediction
                pred = self.model.predict(data)
                affinity = float(pred[0]) if len(pred) > 0 else 0.0
                
                return DTIPrediction(
                    drug_smiles=drug_smiles,
                    target_sequence=target_sequence,
                    binding_affinity=affinity,
                    confidence=0.85,  # TODO: Implement uncertainty estimation
                    model_name=f"DeepPurpose-{self.drug_encoding}-{self.target_encoding}",
                    metadata={
                        "timestamp": datetime.utcnow().isoformat(),
                        "device": self.device,
                    }
                )
                
            except Exception as e:
                raise RuntimeError(f"DTI prediction failed: {e}")
        
        raise RuntimeError("DeepPurpose model not loaded")
    
    def batch_predict(
        self,
        drug_target_pairs: List[Tuple[str, str]],
    ) -> List[DTIPrediction]:
        """Predict for multiple drug-target pairs."""
        return [self.predict(d, t) for d, t in drug_target_pairs]
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> DTIMetrics:
        """Evaluate predictions against ground truth."""
        import math
        
        m_mse = mse(y_true, y_pred)
        
        return DTIMetrics(
            mse=m_mse,
            rmse=math.sqrt(m_mse),
            mae=mae(y_true, y_pred),
            pearson=pearson(y_true, y_pred),
            spearman=spearman(y_true, y_pred),
            concordance_index=concordance_index(y_true, y_pred),
        )


# ============================================================================
# Factory function
# ============================================================================
def get_dti_predictor(
    drug_encoding: str = "Morgan",
    target_encoding: str = "CNN",
    model_path: Optional[str] = None,
) -> DeepPurposePredictor:
    """Factory function to get a DTI predictor instance."""
    predictor = DeepPurposePredictor(
        drug_encoding=drug_encoding,
        target_encoding=target_encoding,
        model_path=model_path,
    )
    return predictor


# ============================================================================
# CLI for standalone usage
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DTI Prediction")
    parser.add_argument("--drug", required=True, help="Drug SMILES")
    parser.add_argument("--target", required=True, help="Target protein sequence")
    parser.add_argument("--drug-enc", default="Morgan", help="Drug encoding method")
    parser.add_argument("--target-enc", default="CNN", help="Target encoding method")
    
    args = parser.parse_args()
    
    predictor = get_dti_predictor(args.drug_enc, args.target_enc)
    result = predictor.predict(args.drug, args.target)
    
    print(f"\n{'='*60}")
    print("DTI Prediction Result")
    print(f"{'='*60}")
    print(f"Drug: {result.drug_smiles}")
    print(f"Target: {result.target_sequence}")
    print(f"Binding Affinity (pKd): {result.binding_affinity:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Model: {result.model_name}")
    print(f"{'='*60}\n")
