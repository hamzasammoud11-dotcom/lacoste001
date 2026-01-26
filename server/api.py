"""
Phase 2: FastAPI Backend for BioDiscovery Search

Fixes applied:
- Shared config import (no duplication)
- Model caching at startup (not per-request)
- Proper error handling
- Uses pre-computed PCA from Qdrant payloads
- Valid dummy sequences instead of "M" * 10
"""
import os
os.environ["DGL_DISABLE_GRAPHBOLT"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import warnings
import pickle
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from DeepPurpose import utils, DTI as dp_models

warnings.filterwarnings("ignore")

# Import shared config
from config import (
    BEST_MODEL_RUN, MODEL_CONFIG,
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, METRICS,
    VALID_DUMMY_DRUG, VALID_DUMMY_TARGET
)

app = FastAPI(title="BioDiscovery API", version="2.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STATE (loaded once at startup) ---
_model = None
_qdrant = None
_device = None

class SearchRequest(BaseModel):
    query: str
    type: str  # "drug" (SMILES) or "target" (Sequence) or "text" (plain text search)
    limit: int = 20

class PointsRequest(BaseModel):
    limit: int = 500
    view: str = "combined"  # "drug", "target", or "combined"

@app.on_event("startup")
async def load_resources():
    """Load model and connect to Qdrant at startup (cached)."""
    global _model, _qdrant, _device
    
    print("[STARTUP] Loading DeepPurpose model...")
    
    # Load config
    config_path = os.path.join(BEST_MODEL_RUN, "config.pkl")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        # Override result_folder to current path (old path may be stale)
        config["result_folder"] = BEST_MODEL_RUN
    else:
        config = utils.generate_config(
            drug_encoding=MODEL_CONFIG["drug_encoding"], 
            target_encoding=MODEL_CONFIG["target_encoding"], 
            cls_hidden_dims=MODEL_CONFIG["cls_hidden_dims"], 
            train_epoch=1, LR=1e-4, batch_size=256,
            result_folder=BEST_MODEL_RUN
        )
    
    _model = dp_models.model_initialize(**config)
    
    model_path = os.path.join(BEST_MODEL_RUN, "model.pt")
    if os.path.exists(model_path):
        _model.load_pretrained(model_path)
        print(f"[STARTUP] Model loaded from {model_path}")
    else:
        print(f"[WARNING] No model.pt found at {model_path}")
    
    # CRITICAL FIX: Override DeepPurpose's global device variable
    # The encoders.py uses a module-level `device = torch.device('cuda' if...)` 
    # and the MLP forward does `v = v.float().to(device)` using that global!
    import DeepPurpose.encoders as dp_encoders
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dp_encoders.device = _device  # Override the global
    print(f"[STARTUP] Using device: {_device}")
    
    # Ensure model is on the correct device
    _model.model = _model.model.to(_device)
    _model.model.eval()
    
    print("[STARTUP] Connecting to Qdrant...")
    try:
        _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
        collections = _qdrant.get_collections()
        print(f"[STARTUP] Connected. Collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"[WARNING] Qdrant connection failed: {e}")
        _qdrant = None
    
    print("[STARTUP] Ready!")

def encode_query(query: str, query_type: str) -> List[float]:
    """Encode a single drug/target query into a vector using direct encoding."""
    if not _model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        if query_type == "drug":
            # Direct Morgan fingerprint encoding (avoid data_process)
            from DeepPurpose.utils import smiles2morgan
            from rdkit import Chem
            import numpy as np
            
            # Validate SMILES
            mol = Chem.MolFromSmiles(query)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {query}")
            
            # Get Morgan fingerprint
            morgan_fp = smiles2morgan(query, radius=2, nBits=1024)
            if morgan_fp is None:
                raise ValueError(f"Failed to compute Morgan fingerprint for: {query}")
            
            # Convert to tensor and encode through model's drug encoder
            v_d = torch.tensor(np.array([morgan_fp]), dtype=torch.float32)
            
            with torch.no_grad():
                vector = _model.model.model_drug(v_d).cpu().numpy()[0].tolist()
            return vector
            
        elif query_type == "target":
            # Direct CNN target encoding
            from DeepPurpose.utils import trans_protein
            import numpy as np
            
            # Encode protein sequence
            target_encoding = trans_protein(query)
            if target_encoding is None:
                raise ValueError(f"Failed to encode protein sequence")
            
            # CNN expects [batch, seq_len] input, max_len=1000 in default config
            MAX_SEQ_LEN = 1000
            if len(target_encoding) > MAX_SEQ_LEN:
                target_encoding = target_encoding[:MAX_SEQ_LEN]
            else:
                target_encoding = target_encoding + [0] * (MAX_SEQ_LEN - len(target_encoding))
            
            v_p = torch.tensor(np.array([target_encoding]), dtype=torch.long)
            
            with torch.no_grad():
                vector = _model.model.model_protein(v_p).cpu().numpy()[0].tolist()
            return vector
        else:
            raise HTTPException(status_code=400, detail="type must be 'drug' or 'target'")
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")

@app.post("/api/search")
async def search_vectors(req: SearchRequest):
    """Search for similar drugs/targets."""
    if not _qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not connected")
    
    # Text search - just filter by payload, no encoding needed
    if req.type == "text":
        return await text_search(req.query, req.limit)
    
    # Vector search - encode and search
    try:
        vector = encode_query(req.query, req.type)
    except Exception as e:
        # Fallback to text search if encoding fails
        print(f"Encoding failed ({e}), falling back to text search")
        return await text_search(req.query, req.limit)
    
    try:
        hits = _qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=(req.type, vector),  # Named vector
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


async def text_search(query: str, limit: int = 20):
    """Text-based search through payloads (fallback when encoding fails)."""
    try:
        # Scroll through and filter by SMILES containing the query
        res, _ = _qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,  # Get more to filter through
            with_payload=True,
            with_vectors=False
        )
        
        # Filter results that match query in SMILES or other fields
        query_lower = query.lower()
        results = []
        for point in res:
            smiles = point.payload.get("smiles", "").lower()
            # Match if query is substring of SMILES or SMILES contains query
            if query_lower in smiles:
                results.append({
                    "id": point.id,
                    "score": 0.95 if query_lower == smiles else 0.8,  # Higher score for exact match
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

@app.get("/api/points")
async def get_visualization_points(limit: int = 500, view: str = "combined"):
    """Get points with pre-computed PCA for 3D visualization."""
    if not _qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not connected")
    
    try:
        # Use scroll to get points (more efficient than search for bulk)
        res, _ = _qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            with_vectors=False,  # Don't need raw vectors, use PCA from payload
            with_payload=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scroll failed: {str(e)}")
    
    # Map view to correct PCA key
    pca_key = f"pca_{view}" if view in ["drug", "target", "combined"] else "pca_combined"
    
    points = []
    for point in res:
        pca = point.payload.get(pca_key, [0, 0, 0])
        
        # Determine color based on affinity class
        affinity_class = point.payload.get("affinity_class", "low")
        color = {
            "high": "#10b981",   # Green
            "medium": "#f59e0b", # Amber
            "low": "#64748b"     # Slate
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
            "clusters": 3,  # high/medium/low
            "avgConfidence": METRICS.get("BindingDB_Kd", {}).get("CI", 0.80),
        },
        "view": view,
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "qdrant_connected": _qdrant is not None,
        "metrics": METRICS,
    }

@app.get("/api/stats")
async def get_collection_stats():
    """Get real statistics from Qdrant collection for the data page."""
    if not _qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not connected")
    
    try:
        collection_info = _qdrant.get_collection(collection_name=COLLECTION_NAME)
        total_vectors = collection_info.vectors_count
        
        # Sample to count affinity classes
        sample, _ = _qdrant.scroll(
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
