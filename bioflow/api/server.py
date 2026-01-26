"""
BioFlow API - Main Server
==========================
FastAPI application serving the Next.js frontend.
Endpoints for discovery, prediction, and data management.
"""

import os
import sys
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# In-Memory Job Store (replace with Redis/DB in production)
# ============================================================================
JOBS: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Pydantic Models
# ============================================================================
class DiscoveryRequest(BaseModel):
    """Request for drug discovery pipeline."""
    query: str = Field(..., description="SMILES, FASTA, or natural language query")
    search_type: str = Field(default="similarity", description="similarity | binding | properties")
    database: str = Field(default="all", description="Target database")
    limit: int = Field(default=10, ge=1, le=100)


class PredictRequest(BaseModel):
    """Request for DTI prediction."""
    drug_smiles: str = Field(..., description="SMILES string of drug")
    target_sequence: str = Field(..., description="Protein sequence (FASTA)")


class IngestRequest(BaseModel):
    """Request to ingest data into vector DB."""
    content: str
    modality: str = Field(default="smiles", description="smiles | protein | text")
    metadata: Optional[Dict[str, Any]] = None


class JobStatus(BaseModel):
    """Status of an async job."""
    job_id: str
    status: str  # pending | running | completed | failed
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str


# ============================================================================
# Lifespan (startup/shutdown)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    logger.info("🚀 BioFlow API starting up...")
    # TODO: Initialize Qdrant connection, load models
    yield
    logger.info("🛑 BioFlow API shutting down...")


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="BioFlow API",
    description="AI-Powered Drug Discovery Platform API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Info
# ============================================================================
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


# ============================================================================
# Discovery Pipeline
# ============================================================================
def run_discovery_pipeline(job_id: str, request: DiscoveryRequest):
    """Background task for discovery pipeline."""
    import time
    
    try:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()
        
        # Step 1: Encode
        JOBS[job_id]["progress"] = 25
        JOBS[job_id]["current_step"] = "encode"
        time.sleep(1)  # TODO: Replace with actual encoding
        
        # Step 2: Search
        JOBS[job_id]["progress"] = 50
        JOBS[job_id]["current_step"] = "search"
        time.sleep(1)  # TODO: Replace with vector search
        
        # Step 3: Predict
        JOBS[job_id]["progress"] = 75
        JOBS[job_id]["current_step"] = "predict"
        time.sleep(1)  # TODO: Replace with DTI prediction
        
        # Step 4: Results
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["current_step"] = "complete"
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = {
            "candidates": [
                {"name": "Candidate A", "smiles": "CCO", "score": 0.95, "mw": 342.4, "logp": 2.1},
                {"name": "Candidate B", "smiles": "CC(=O)O", "score": 0.89, "mw": 298.3, "logp": 1.8},
                {"name": "Candidate C", "smiles": "c1ccccc1", "score": 0.82, "mw": 415.5, "logp": 3.2},
            ],
            "query": request.query,
            "search_type": request.search_type,
        }
        JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()
        logger.error(f"Discovery pipeline failed: {e}")


@app.post("/api/discovery")
async def start_discovery(request: DiscoveryRequest, background_tasks: BackgroundTasks):
    """Start a discovery pipeline (async)."""
    job_id = f"disc_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "current_step": "queued",
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "request": request.model_dump(),
    }
    
    background_tasks.add_task(run_discovery_pipeline, job_id, request)
    
    return {
        "success": True,
        "job_id": job_id,
        "status": "pending",
        "message": "Discovery pipeline started",
    }


@app.get("/api/discovery/{job_id}")
async def get_discovery_status(job_id: str):
    """Get status of a discovery job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


# ============================================================================
# DTI Prediction
# ============================================================================
# Import the predictor
from bioflow.api.dti_predictor import get_dti_predictor, DeepPurposePredictor

# Global predictor instance (lazy loaded)
_dti_predictor: Optional[DeepPurposePredictor] = None

def get_predictor() -> DeepPurposePredictor:
    """Get or create the DTI predictor instance."""
    global _dti_predictor
    if _dti_predictor is None:
        _dti_predictor = get_dti_predictor()
    return _dti_predictor


@app.post("/api/predict")
async def predict_dti(request: PredictRequest):
    """
    Predict drug-target interaction.
    Uses DeepPurpose under the hood.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(request.drug_smiles, request.target_sequence)
        
        return {
            "success": True,
            "prediction": {
                "drug_smiles": result.drug_smiles,
                "target_sequence": result.target_sequence,
                "binding_affinity": result.binding_affinity,
                "confidence": result.confidence,
                "interaction_probability": min(result.confidence + 0.05, 1.0),
            },
            "metadata": {
                "model": result.model_name,
                "timestamp": datetime.utcnow().isoformat(),
                **result.metadata,
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Data Management
# ============================================================================
@app.post("/api/ingest")
async def ingest_data(request: IngestRequest):
    """Ingest data into vector database."""
    try:
        # TODO: Integrate with Qdrant via bioflow.qdrant_manager
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        
        return {
            "success": True,
            "id": doc_id,
            "modality": request.modality,
            "message": "Data ingested successfully",
        }
        
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/molecules")
async def list_molecules(limit: int = 20, offset: int = 0):
    """List molecules in the database."""
    # TODO: Query from Qdrant
    mock_molecules = [
        {"id": "mol_001", "smiles": "CCO", "name": "Ethanol", "mw": 46.07},
        {"id": "mol_002", "smiles": "CC(=O)O", "name": "Acetic Acid", "mw": 60.05},
        {"id": "mol_003", "smiles": "c1ccccc1", "name": "Benzene", "mw": 78.11},
    ]
    return {
        "molecules": mock_molecules,
        "total": len(mock_molecules),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/proteins")
async def list_proteins(limit: int = 20, offset: int = 0):
    """List proteins in the database."""
    # TODO: Query from Qdrant
    mock_proteins = [
        {"id": "prot_001", "uniprot_id": "P00533", "name": "EGFR", "length": 1210},
        {"id": "prot_002", "uniprot_id": "P04637", "name": "p53", "length": 393},
        {"id": "prot_003", "uniprot_id": "P38398", "name": "BRCA1", "length": 1863},
    ]
    return {
        "proteins": mock_proteins,
        "total": len(mock_proteins),
        "limit": limit,
        "offset": offset,
    }


# ============================================================================
# Explorer (Embeddings)
# ============================================================================
@app.get("/api/explorer/embeddings")
async def get_embeddings(dataset: str = "default", method: str = "umap"):
    """Get 2D projections of embeddings for visualization."""
    import random
    
    # TODO: Get actual embeddings from Qdrant and project
    # Generate mock UMAP-like data
    random.seed(42)
    
    points = []
    for i in range(100):
        cluster = i % 4
        cx, cy = [(2, 3), (-2, -1), (4, -2), (-1, 4)][cluster]
        points.append({
            "id": f"mol_{i:03d}",
            "x": cx + random.gauss(0, 0.8),
            "y": cy + random.gauss(0, 0.8),
            "cluster": cluster,
            "label": f"Molecule {i}",
        })
    
    return {
        "points": points,
        "method": method,
        "dataset": dataset,
        "n_clusters": 4,
    }


# ============================================================================
# Run with: uvicorn bioflow.api.server:app --reload --port 8000
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
