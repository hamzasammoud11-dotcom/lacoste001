"""
DeepPurpose API Endpoints
========================

Provides vector search and visualization endpoints using DeepPurpose models.
Integrates with Qdrant for drug-target interaction data.
"""
import os
import sys
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dp", tags=["deeppurpose"])

# Global state for DeepPurpose model
_dp_model = None
_dp_qdrant = None
_dp_device = None
_dp_initialized = False


class SearchRequest(BaseModel):
    query: str
    type: str  # "drug" (SMILES), "target" (Sequence), or "text"
    limit: int = 20


class PointsRequest(BaseModel):
    limit: int = 500
    view: str = "combined"  # "drug", "target", or "combined"


def _check_deeppurpose():
    """Check if DeepPurpose is available."""
    try:
        from DeepPurpose import utils, DTI as dp_models
        return True
    except ImportError:
        return False


def _init_deeppurpose():
    """Initialize DeepPurpose model and Qdrant connection."""
    global _dp_model, _dp_qdrant, _dp_device, _dp_initialized
    
    if _dp_initialized:
        return _dp_model is not None
    
    _dp_initialized = True
    
    if not _check_deeppurpose():
        logger.warning("DeepPurpose not available - endpoints will use fallback")
        return False
    
    try:
        import torch
        import pickle
        from DeepPurpose import utils, DTI as dp_models
        from qdrant_client import QdrantClient
        
        # Try to import config from bioflow
        try:
            from bioflow.deeppurpose_config import (
                BEST_MODEL_RUN, MODEL_CONFIG,
                QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
            )
        except ImportError:
            logger.warning("DeepPurpose config not found, using defaults")
            BEST_MODEL_RUN = os.path.join(ROOT_DIR, "bioflow", "runs", "20260125_104915_KIBA")
            QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
            QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
            COLLECTION_NAME = "bio_discovery"
            MODEL_CONFIG = {
                "drug_encoding": "Morgan",
                "target_encoding": "CNN",
                "cls_hidden_dims": [1024, 1024, 512],
            }
        
        logger.info("[DeepPurpose] Loading model...")
        
        # Load model config
        config_path = os.path.join(BEST_MODEL_RUN, "config.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
            config["result_folder"] = BEST_MODEL_RUN
        else:
            config = utils.generate_config(
                drug_encoding=MODEL_CONFIG["drug_encoding"],
                target_encoding=MODEL_CONFIG["target_encoding"],
                cls_hidden_dims=MODEL_CONFIG["cls_hidden_dims"],
                train_epoch=1, LR=1e-4, batch_size=256,
                result_folder=BEST_MODEL_RUN
            )
        
        _dp_model = dp_models.model_initialize(**config)
        
        # Load weights if available
        model_path = os.path.join(BEST_MODEL_RUN, "model.pt")
        if os.path.exists(model_path):
            _dp_model.load_pretrained(model_path)
            logger.info(f"[DeepPurpose] Model loaded from {model_path}")
        else:
            logger.warning(f"[DeepPurpose] No model.pt at {model_path}")
        
        # Set device
        import DeepPurpose.encoders as dp_encoders
        _dp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dp_encoders.device = _dp_device
        _dp_model.model = _dp_model.model.to(_dp_device)
        _dp_model.model.eval()
        
        # Connect to Qdrant
        try:
            _dp_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
            collections = _dp_qdrant.get_collections()
            logger.info(f"[DeepPurpose] Qdrant connected: {[c.name for c in collections.collections]}")
        except Exception as e:
            logger.warning(f"[DeepPurpose] Qdrant connection failed: {e}")
            _dp_qdrant = None
        
        return True
        
    except Exception as e:
        logger.error(f"[DeepPurpose] Init failed: {e}")
        return False


def _encode_query(query: str, query_type: str) -> List[float]:
    """Encode a drug/target query into a vector."""
    if not _dp_model:
        raise HTTPException(status_code=503, detail="DeepPurpose model not initialized")
    
    try:
        import torch
        import numpy as np
        
        if query_type == "drug":
            from DeepPurpose.utils import smiles2morgan
            from rdkit import Chem
            
            mol = Chem.MolFromSmiles(query)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {query}")
            
            morgan_fp = smiles2morgan(query, radius=2, nBits=1024)
            if morgan_fp is None:
                raise ValueError(f"Failed to compute Morgan fingerprint")
            
            v_d = torch.tensor(np.array([morgan_fp]), dtype=torch.float32)
            
            with torch.no_grad():
                vector = _dp_model.model.model_drug(v_d).cpu().numpy()[0].tolist()
            return vector
            
        elif query_type == "target":
            from DeepPurpose.utils import trans_protein
            
            target_encoding = trans_protein(query)
            if target_encoding is None:
                raise ValueError(f"Failed to encode protein sequence")
            
            MAX_SEQ_LEN = 1000
            if len(target_encoding) > MAX_SEQ_LEN:
                target_encoding = target_encoding[:MAX_SEQ_LEN]
            else:
                target_encoding = target_encoding + [0] * (MAX_SEQ_LEN - len(target_encoding))
            
            v_p = torch.tensor(np.array([target_encoding]), dtype=torch.long)
            
            with torch.no_grad():
                vector = _dp_model.model.model_protein(v_p).cpu().numpy()[0].tolist()
            return vector
        else:
            raise HTTPException(status_code=400, detail="type must be 'drug' or 'target'")
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")


@router.post("/dp-search")
async def search_vectors(req: SearchRequest):
    """Search for similar drugs/targets using DeepPurpose embeddings.
    
    Note: Use /api/search for general enhanced search with MMR diversification.
    This endpoint is specifically for DeepPurpose model-based vector search.
    """
    if not _init_deeppurpose():
        raise HTTPException(status_code=503, detail="DeepPurpose not available")
    
    if not _dp_qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not connected")
    
    try:
        from bioflow.deeppurpose_config import COLLECTION_NAME
    except ImportError:
        COLLECTION_NAME = "bio_discovery"
    
    # Text search - filter by payload
    if req.type == "text":
        return await _text_search(req.query, req.limit, COLLECTION_NAME)
    
    # Vector search
    try:
        vector = _encode_query(req.query, req.type)
    except Exception as e:
        logger.warning(f"Encoding failed ({e}), falling back to text search")
        return await _text_search(req.query, req.limit, COLLECTION_NAME)
    
    try:
        hits = _dp_qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=(req.type, vector),
            limit=req.limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    results = []
    for hit in hits:
        results.append({
            "id": hit.id,
            "score": hit.score,
            "smiles": hit.payload.get("smiles"),
            "target_seq": hit.payload.get("target_seq", "")[:100] + "...",
            "label": hit.payload.get("label_true"),
            "affinity_class": hit.payload.get("affinity_class"),
        })
    
    return {"results": results, "query_type": req.type, "count": len(results)}


async def _text_search(query: str, limit: int, collection_name: str):
    """Text-based search through payloads."""
    try:
        res, _ = _dp_qdrant.scroll(
            collection_name=collection_name,
            limit=500,
            with_payload=True,
            with_vectors=False
        )
        
        query_lower = query.lower()
        results = []
        for point in res:
            smiles = point.payload.get("smiles", "").lower()
            if query_lower in smiles:
                results.append({
                    "id": point.id,
                    "score": 0.95 if query_lower == smiles else 0.8,
                    "smiles": point.payload.get("smiles"),
                    "target_seq": point.payload.get("target_seq", "")[:100] + "...",
                    "label": point.payload.get("label_true"),
                    "affinity_class": point.payload.get("affinity_class"),
                })
                if len(results) >= limit:
                    break
        
        return {"results": results, "query_type": "text", "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text search failed: {str(e)}")


@router.get("/points")
async def get_visualization_points(limit: int = 500, view: str = "combined"):
    """Get points with pre-computed PCA for 3D visualization."""
    if not _init_deeppurpose():
        raise HTTPException(status_code=503, detail="DeepPurpose not available")
    
    if not _dp_qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not connected")
    
    try:
        from bioflow.deeppurpose_config import COLLECTION_NAME, METRICS
    except ImportError:
        COLLECTION_NAME = "bio_discovery"
        METRICS = {"BindingDB_Kd": {"CI": 0.80}}
    
    try:
        res, _ = _dp_qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            with_vectors=False,
            with_payload=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scroll failed: {str(e)}")
    
    pca_key = f"pca_{view}" if view in ["drug", "target", "combined"] else "pca_combined"
    
    points = []
    for point in res:
        pca = point.payload.get(pca_key, [0, 0, 0])
        affinity_class = point.payload.get("affinity_class", "low")
        color = {
            "high": "#10b981",
            "medium": "#f59e0b",
            "low": "#64748b"
        }.get(affinity_class, "#64748b")
        
        points.append({
            "id": point.id,
            "x": pca[0] if len(pca) > 0 else 0,
            "y": pca[1] if len(pca) > 1 else 0,
            "z": pca[2] if len(pca) > 2 else 0,
            "color": color,
            "name": (point.payload.get("smiles") or "Unknown")[:15] + "...",
            "affinity": point.payload.get("label_true", 0),
            "affinity_class": affinity_class,
            "smiles": point.payload.get("smiles"),
        })
    
    return {
        "points": points,
        "metrics": {
            "activeMolecules": len(points),
            "clusters": 3,
            "avgConfidence": METRICS.get("BindingDB_Kd", {}).get("CI", 0.80),
        },
        "view": view,
    }


@router.get("/stats")
async def get_collection_stats():
    """Get real statistics from Qdrant collection."""
    if not _init_deeppurpose():
        raise HTTPException(status_code=503, detail="DeepPurpose not available")
    
    if not _dp_qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not connected")
    
    try:
        from bioflow.deeppurpose_config import COLLECTION_NAME
    except ImportError:
        COLLECTION_NAME = "bio_discovery"
    
    try:
        collection_info = _dp_qdrant.get_collection(collection_name=COLLECTION_NAME)
        total_vectors = collection_info.vectors_count
        
        sample, _ = _dp_qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=["affinity_class", "smiles", "target_id"],
            with_vectors=False
        )
        
        unique_drugs = len(set(p.payload.get("smiles", "") for p in sample if p.payload.get("smiles")))
        unique_targets = len(set(p.payload.get("target_id", "") for p in sample if p.payload.get("target_id")))
        
        affinity_counts = {}
        for p in sample:
            aff = p.payload.get("affinity_class", "unknown")
            affinity_counts[aff] = affinity_counts.get(aff, 0) + 1
        
        return {
            "total_vectors": total_vectors,
            "sample_size": len(sample),
            "unique_drugs_sampled": unique_drugs,
            "unique_targets_sampled": unique_targets,
            "affinity_distribution": affinity_counts,
            "collection_name": COLLECTION_NAME,
            "status": collection_info.status.value if hasattr(collection_info.status, 'value') else str(collection_info.status),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats fetch failed: {str(e)}")
