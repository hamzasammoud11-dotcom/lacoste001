"""
Phase 2: FastAPI Backend for BioDiscovery Search

- Robust /api/stats (raw REST to bypass qdrant-client pydantic parsing issues)
- Use CORS_ORIGINS from config if available
- Ensure tensors are moved to the selected device in encode_query
- Map API query types -> Qdrant named vectors (molecules_v2: molecule, text)
- target disabled (no target vectors in molecules_v2)
"""
import os
os.environ["DGL_DISABLE_GRAPHBOLT"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import numpy as np

import pickle
from typing import List

import torch
import numpy as np
import httpx

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

# Optional config (safe fallbacks)
try:
    from config import CORS_ORIGINS
except Exception:
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

try:
    from config import QDRANT_URL
except Exception:
    QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

app = FastAPI(title="BioDiscovery API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STATE (loaded once at startup) ---
_model = None
_qdrant = None
_device = None

# Map API query types -> Qdrant named vectors (molecules_v2)
VECTOR_NAME_BY_TYPE = {
    "drug": "molecule",
    # "text": "text"  # Pas de text-embedder dans ce backend => on fait payload search
}

class SearchRequest(BaseModel):
    query: str
    type: str  # "drug" | "text" | "target"
    limit: int = 20

class PointsRequest(BaseModel):
    limit: int = 500
    view: str = "combined"  # "drug" | "target" | "combined"

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

    # DeepPurpose global device override
    import DeepPurpose.encoders as dp_encoders
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp_encoders.device = _device
    print(f"[STARTUP] Using device: {_device}")

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
    if not _model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    if query_type != "drug":
        raise HTTPException(status_code=400, detail="type must be 'drug' or 'text' (target not supported)")

    try:
        from DeepPurpose.utils import smiles2morgan
        from rdkit import Chem

        mol = Chem.MolFromSmiles(query)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {query}")

        morgan_fp = smiles2morgan(query, radius=2, nBits=1024)
        if morgan_fp is None:
            raise ValueError(f"Failed to compute Morgan fingerprint for: {query}")

        v_d = torch.tensor(np.array([morgan_fp]), dtype=torch.float32).to(_device)

        with torch.no_grad():
            vec = _model.model.model_drug(v_d).detach().cpu().numpy()[0].astype(float).tolist()

        return vec

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed for type={query_type}: {str(e)}")


@app.post("/api/search")
async def search_vectors(req: SearchRequest):
    if not _qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not connected")

    # Text = payload substring (pas d’embedder texte ici)
    if req.type == "text":
        return await text_search(req.query, req.limit)

    # Drug = vector search sur le vecteur nommé "molecule" (dim=256)
    if req.type == "drug":
        vector = encode_query(req.query, "drug")  # doit sortir 256 floats
        try:
            hits = _qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=("molecule", vector),
                limit=req.limit,
                with_payload=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "id": hit.id,
                "score": hit.score,
                "smiles": payload.get("smiles") or payload.get("content"),
                "label": payload.get("label_true"),
                "affinity_class": payload.get("affinity_class", "unknown"),
            })

        return {"results": results, "query_type": req.type, "count": len(results)}

    # target non supporté (pas de vecteur target dans molecules_dp)
    raise HTTPException(status_code=400, detail="type must be 'drug' or 'text'")



async def text_search(query: str, limit: int = 20):
    """Text-based search through payloads."""
    try:
        res, _ = _qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True,
            with_vectors=False
        )

        query_lower = (query or "").lower()
        results = []
        for point in res:
            payload = point.payload or {}
            smiles = (payload.get("smiles") or "").lower()
            if query_lower in smiles:
                target_seq = payload.get("target_seq") or ""
                results.append({
                    "id": point.id,
                    "score": 0.95 if query_lower == smiles else 0.8,
                    "smiles": payload.get("smiles"),
                    "target_seq": (target_seq[:100] + "...") if len(target_seq) > 100 else target_seq,
                    "label": payload.get("label_true"),
                    "affinity_class": payload.get("affinity_class"),
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
        res, _ = _qdrant.scroll(
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
        payload = point.payload or {}
        pca = payload.get(pca_key, [0, 0, 0])

        affinity_class = payload.get("affinity_class", "low")
        color = {"high": "#10b981", "medium": "#f59e0b", "low": "#64748b"}.get(affinity_class, "#64748b")

        name = (payload.get("smiles") or "Unknown")
        points.append({
            "id": point.id,
            "x": pca[0] if len(pca) > 0 else 0,
            "y": pca[1] if len(pca) > 1 else 0,
            "z": pca[2] if len(pca) > 2 else 0,
            "color": color,
            "name": (name[:15] + "...") if len(name) > 15 else name,
            "affinity": payload.get("label_true", 0),
            "affinity_class": affinity_class,
            "smiles": payload.get("smiles"),
        })

    avg_conf = 0.80
    if isinstance(METRICS, dict):
        avg_conf = METRICS.get("BindingDB_Kd", {}).get("CI", 0.80) if isinstance(METRICS.get("BindingDB_Kd", {}), dict) else 0.80

    return {
        "points": points,
        "metrics": {"activeMolecules": len(points), "clusters": 3, "avgConfidence": avg_conf},
        "view": view,
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "qdrant_connected": _qdrant is not None,
        "metrics": METRICS,
    }

@app.get("/api/stats")
async def get_collection_stats():
    """
    Get stats via raw Qdrant REST (avoid qdrant-client parsing).
    """
    if not _qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not connected")

    try:
        url = f"{QDRANT_URL.rstrip('/')}/collections/{COLLECTION_NAME}"
        r = httpx.get(url, timeout=10.0)
        r.raise_for_status()
        payload = r.json() or {}
        result = payload.get("result", {}) or {}

        points_count = int(result.get("points_count", 0) or 0)
        indexed_vectors_count = int(result.get("indexed_vectors_count", 0) or 0)
        status = result.get("status", "unknown")

        vectors_cfg = (((result.get("config") or {}).get("params") or {}).get("vectors")) or {}
        vector_dims = {}
        if isinstance(vectors_cfg, dict):
            for name, cfg in vectors_cfg.items():
                vector_dims[name] = cfg.get("size") if isinstance(cfg, dict) else None

        sample, _ = _qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        unique_drugs = len(set((p.payload or {}).get("smiles", "") for p in sample if (p.payload or {}).get("smiles")))
        unique_targets = len(set(
            ((p.payload or {}).get("target_id") or (p.payload or {}).get("target_seq") or "")
            for p in sample
            if ((p.payload or {}).get("target_id") or (p.payload or {}).get("target_seq"))
        ))

        affinity_counts = {}
        for p in sample:
            aff = (p.payload or {}).get("affinity_class", "unknown")
            affinity_counts[aff] = affinity_counts.get(aff, 0) + 1

        return {
            "collection_name": COLLECTION_NAME,
            "status": status,
            "points_count": points_count,
            "indexed_vectors_count": indexed_vectors_count,
            "vector_dims": vector_dims,
            "sample_size": len(sample),
            "unique_drugs_sampled": unique_drugs,
            "unique_targets_sampled": unique_targets,
            "affinity_distribution": affinity_counts,
        }

    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Stats fetch failed (HTTP): {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats fetch failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
