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
    type: str  # "drug" (SMILES) or "target" (Sequence)
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
    
    # Use CPU for inference (avoids VRAM contention)
    _device = torch.device('cpu')
    _model.model.to(_device)
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
    """Encode a single drug/target query into a vector."""
    if not _model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        if query_type == "drug":
            # Use valid dummy target for data_process
            data = utils.data_process(
                [query], [VALID_DUMMY_TARGET], [0],
                _model.config['drug_encoding'],
                _model.config['target_encoding'],
                split_method='random', frac=[0,0,1], random_seed=1
            )[2]
            
            loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
            v_d, _, _ = next(iter(loader))
            
            with torch.no_grad():
                v_d = v_d.to(_device)
                vector = _model.model.model_drug(v_d).cpu().numpy()[0].tolist()
            return vector
            
        elif query_type == "target":
            # Use valid dummy drug for data_process
            data = utils.data_process(
                [VALID_DUMMY_DRUG], [query], [0],
                _model.config['drug_encoding'],
                _model.config['target_encoding'],
                split_method='random', frac=[0,0,1], random_seed=1
            )[2]
            
            loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
            _, v_p, _ = next(iter(loader))
            
            with torch.no_grad():
                v_p = v_p.to(_device)
                vector = _model.model.model_protein(v_p).cpu().numpy()[0].tolist()
            return vector
        else:
            raise HTTPException(status_code=400, detail="type must be 'drug' or 'target'")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")

@app.post("/api/search")
async def search_vectors(req: SearchRequest):
    """Search for similar drugs/targets."""
    if not _qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not connected")
    
    vector = encode_query(req.query, req.type)
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
