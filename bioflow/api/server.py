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
# Service Imports
# ============================================================================
from bioflow.api.model_service import get_model_service, ModelService
from bioflow.api.qdrant_service import get_qdrant_service, QdrantService

# DeepPurpose router
try:
    from bioflow.api.deeppurpose_api import router as deeppurpose_router
    HAS_DEEPPURPOSE_API = True
except ImportError as e:
    logger.warning(f"DeepPurpose API not available: {e}")
    HAS_DEEPPURPOSE_API = False

# ============================================================================
# In-Memory Job Store (replace with Redis/DB in production)
# ============================================================================
JOBS: Dict[str, Dict[str, Any]] = {}

# ============================================================================
# Global Services (initialized in lifespan)
# ============================================================================
model_service: Optional[ModelService] = None
qdrant_service: Optional[QdrantService] = None


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


class EnhancedSearchRequest(BaseModel):
    """Request for enhanced search with MMR and filters."""
    query: str = Field(..., description="Search query (text, SMILES, or protein sequence)")
    modality: str = Field(default="text", description="Query modality: text, molecule, protein")
    collection: Optional[str] = Field(default=None, description="Target collection")
    top_k: int = Field(default=20, ge=1, le=100)
    use_mmr: bool = Field(default=True, description="Apply MMR diversification")
    lambda_param: float = Field(default=0.7, ge=0.0, le=1.0, description="MMR lambda (1=relevance, 0=diversity)")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")


class HybridSearchRequest(BaseModel):
    """Request for hybrid vector + keyword search."""
    query: str = Field(..., description="Vector search query")
    keywords: List[str] = Field(..., description="Keywords to match")
    modality: str = Field(default="text")
    collection: Optional[str] = None
    top_k: int = Field(default=20, ge=1, le=100)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)


# ============================================================================
# Lifespan (startup/shutdown)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    global model_service, qdrant_service
    
    logger.info("🚀 BioFlow API starting up...")
    
    # Initialize services
    model_service = get_model_service(lazy_load=True)
    qdrant_service = get_qdrant_service(model_service=model_service)
    
    logger.info("✅ Services initialized")
    yield
    
    logger.info("🛑 BioFlow API shutting down...")
    model_service = None
    qdrant_service = None


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

# Include DeepPurpose router if available
if HAS_DEEPPURPOSE_API:
    app.include_router(deeppurpose_router)
    logger.info("✅ DeepPurpose API router included")


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
        
        # Step 1: Encode query
        JOBS[job_id]["progress"] = 25
        JOBS[job_id]["current_step"] = "encode"
        
        if model_service:
            # Detect modality and encode
            query = request.query
            if query.startswith("M") and len(query) > 20 and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in query[:20]):
                encoding = model_service.encode_protein(query)
                query_modality = "protein"
            elif any(c in query for c in "[]()=#@"):  # SMILES-like
                encoding = model_service.encode_molecule(query)
                query_modality = "molecule"
            else:
                encoding = model_service.encode_text(query)
                query_modality = "text"
            
            logger.info(f"Encoded query as {query_modality}")
        
        # Step 2: Search
        JOBS[job_id]["progress"] = 50
        JOBS[job_id]["current_step"] = "search"
        
        candidates = []
        if qdrant_service:
            # Real vector search
            search_results = qdrant_service.search(
                query=request.query,
                modality=query_modality if 'query_modality' in dir() else "text",
                limit=request.limit
            )
            for r in search_results:
                candidates.append({
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "modality": r.modality,
                    "metadata": r.metadata
                })
        
        # No fallback - if no results, return empty list
        # The user must ingest data first
        
        # Step 3: Predict properties
        JOBS[job_id]["progress"] = 75
        JOBS[job_id]["current_step"] = "predict"
        
        # Enrich candidates with property predictions
        if model_service:
            for cand in candidates:
                smiles = cand.get("smiles") or cand.get("content", "")
                if smiles and any(c in smiles for c in "[]()=#@CNO"):
                    try:
                        logp_result = model_service.predict_property(smiles, "logP")
                        mw_result = model_service.predict_property(smiles, "MW")
                        cand["logp"] = round(logp_result.value, 2)
                        cand["mw"] = round(mw_result.value, 2)
                    except Exception as e:
                        logger.warning(f"Property prediction failed: {e}")
        
        # Step 4: Results
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["current_step"] = "complete"
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = {
            "candidates": candidates,
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
# Enhanced Search (Phase 2)
# ============================================================================
_enhanced_search_service = None

def get_enhanced_search_service():
    """Get or create enhanced search service."""
    global _enhanced_search_service
    if _enhanced_search_service is None:
        from bioflow.search.enhanced_search import EnhancedSearchService
        encoder = model_service.get_obm_encoder()
        _enhanced_search_service = EnhancedSearchService(
            qdrant_service=qdrant_service,
            obm_encoder=encoder,
        )
    return _enhanced_search_service


@app.post("/api/search")
async def enhanced_search(request: EnhancedSearchRequest):
    """
    Enhanced semantic search with MMR diversification and evidence linking.
    
    Features:
    - Maximal Marginal Relevance (MMR) for diverse results
    - Evidence links to source databases (PubMed, UniProt, ChEMBL)
    - Citations and source tracking
    - Filtered search by modality, source, etc.
    """
    try:
        if not qdrant_service:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        search_service = get_enhanced_search_service()
        response = search_service.search(
            query=request.query,
            modality=request.modality,
            collection=request.collection,
            top_k=request.top_k,
            use_mmr=request.use_mmr,
            lambda_param=request.lambda_param,
            filters=request.filters,
        )
        
        return response.to_dict()
        
    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/hybrid")
async def hybrid_search(request: HybridSearchRequest):
    """
    Hybrid search combining vector similarity with keyword matching.
    
    Useful when you want results that are both semantically similar
    AND contain specific keywords.
    """
    try:
        if not qdrant_service:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        search_service = get_enhanced_search_service()
        response = search_service.hybrid_search(
            query=request.query,
            keywords=request.keywords,
            modality=request.modality,
            collection=request.collection,
            top_k=request.top_k,
            vector_weight=request.vector_weight,
        )
        
        return response.to_dict()
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Data Management
# ============================================================================
@app.post("/api/ingest")
async def ingest_data(request: IngestRequest):
    """Ingest data into vector database."""
    try:
        if not qdrant_service:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        result = qdrant_service.ingest(
            content=request.content,
            modality=request.modality,
            metadata=request.metadata
        )
        return {
            "success": result.success,
            "id": result.id,
            "collection": result.collection,
            "modality": request.modality,
            "message": result.message,
        }
        
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/molecules")
async def list_molecules(limit: int = 20, offset: int = 0):
    """List molecules in the database."""
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")
    
    try:
        results = qdrant_service.search(
            query="molecule",
            modality="text",
            collection="molecules",
            limit=limit
        )
        molecules = []
        for r in results:
            molecules.append({
                "id": r.id,
                "smiles": r.content,
                "name": r.metadata.get("name", "Unknown"),
                "mw": r.metadata.get("mw", 0),
            })
        return {
            "molecules": molecules,
            "total": len(molecules),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Qdrant query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/proteins")
async def list_proteins(limit: int = 20, offset: int = 0):
    """List proteins in the database."""
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")
    
    try:
        results = qdrant_service.search(
            query="protein kinase receptor",
            modality="text",
            collection="proteins",
            limit=limit
        )
        proteins = []
        for r in results:
            proteins.append({
                "id": r.id,
                "sequence": r.content[:50] + "..." if len(r.content) > 50 else r.content,
                "uniprot_id": r.metadata.get("uniprot_id", ""),
                "name": r.metadata.get("name", "Unknown"),
                "length": len(r.content),
            })
        return {
            "proteins": proteins,
            "total": len(proteins),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Qdrant query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Explorer (Embeddings)
# ============================================================================
@app.get("/api/explorer/embeddings")
async def get_embeddings(dataset: str = "default", method: str = "umap"):
    """Get 2D projections of embeddings for visualization."""
    import numpy as np
    
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")
    
    points = []
    
    try:
        # Get molecules and proteins from Qdrant
        mol_results = qdrant_service.search("", modality="text", collection="molecules", limit=50)
        prot_results = qdrant_service.search("", modality="text", collection="proteins", limit=50)
        
        all_results = mol_results + prot_results
        
        # Simple 2D projection using content hash for deterministic positions
        # (In production, use proper UMAP/t-SNE)
        for i, r in enumerate(all_results):
            np.random.seed(hash(r.content) % 2**32)
            cluster = 0 if r.modality == "molecule" else 1
            cx, cy = [(2, 3), (-2, -1)][cluster]
            points.append({
                "id": r.id,
                "x": float(cx + np.random.randn() * 0.8),
                "y": float(cy + np.random.randn() * 0.8),
                "cluster": cluster,
                "label": r.metadata.get("name", r.content[:20]),
                "modality": r.modality,
            })
    except Exception as e:
        logger.error(f"Failed to get embeddings from Qdrant: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "points": points,
        "method": method,
        "dataset": dataset,
        "n_clusters": len(set(p["cluster"] for p in points)) if points else 0,
    }


# ============================================================================
# Additional API Endpoints
# ============================================================================
@app.post("/api/encode")
async def encode_content(content: str, modality: str = "auto"):
    """Encode content to embedding vector."""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    try:
        # Auto-detect modality
        if modality == "auto":
            if content.startswith("M") and len(content) > 20:
                modality = "protein"
            elif any(c in content for c in "[]()=#@"):
                modality = "molecule"
            else:
                modality = "text"
        
        if modality == "molecule":
            result = model_service.encode_molecule(content)
        elif modality == "protein":
            result = model_service.encode_protein(content)
        else:
            result = model_service.encode_text(content)
        
        return {
            "success": True,
            "embedding": result.vector[:10] + ["..."],  # Truncate for display
            "dimension": len(result.vector),
            "modality": result.modality,
            "model": result.model_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/similarity")
async def compute_similarity(query: str, candidates: str, modality: str = "molecule"):
    """Compute similarity between query and comma-separated candidates."""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    try:
        candidate_list = [c.strip() for c in candidates.split(",")]
        results = model_service.compute_similarity(query, candidate_list, modality)
        return {
            "success": True,
            "query": query,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collections")
async def list_collections():
    """List all vector collections."""
    if not qdrant_service:
        return {"collections": [], "message": "Qdrant not available"}
    
    collections = qdrant_service.list_collections()
    stats = [qdrant_service.get_collection_stats(c) for c in collections]
    return {
        "collections": stats,
        "total": len(collections),
    }


# ============================================================================
# Agent Pipeline Endpoints (Phase 3)
# ============================================================================

class GenerateRequest(BaseModel):
    """Request for molecule generation."""
    prompt: str = Field(..., description="Text description of desired molecule")
    mode: str = Field("text", description="Generation mode: text, mutate, scaffold")
    smiles: Optional[str] = Field(None, description="Seed SMILES for mutate/scaffold mode")
    num_samples: int = Field(5, ge=1, le=50, description="Number of molecules to generate")


class ValidateRequest(BaseModel):
    """Request for molecule validation."""
    smiles: List[str] = Field(..., description="List of SMILES to validate")
    check_lipinski: bool = Field(True, description="Check Lipinski Rule of 5")
    check_admet: bool = Field(True, description="Check ADMET properties")
    check_alerts: bool = Field(True, description="Check structural alerts")


class RankRequest(BaseModel):
    """Request for candidate ranking."""
    candidates: List[Dict[str, Any]] = Field(..., description="Candidates with scores")
    weights: Optional[Dict[str, float]] = Field(None, description="Custom score weights")
    top_k: Optional[int] = Field(None, description="Return top K candidates")


class WorkflowRequest(BaseModel):
    """Request for full discovery workflow."""
    query: str = Field(..., description="Text description of desired molecule")
    num_candidates: int = Field(10, ge=1, le=50, description="Number of candidates to generate")
    top_k: int = Field(5, ge=1, le=20, description="Number of top candidates to return")


# Agent instances (lazy initialized)
_generator_agent = None
_validator_agent = None
_ranker_agent = None


def get_generator_agent():
    """Get or create generator agent."""
    global _generator_agent
    if _generator_agent is None:
        from bioflow.agents import GeneratorAgent
        _generator_agent = GeneratorAgent()
    return _generator_agent


def get_validator_agent():
    """Get or create validator agent."""
    global _validator_agent
    if _validator_agent is None:
        from bioflow.agents import ValidatorAgent
        _validator_agent = ValidatorAgent()
    return _validator_agent


def get_ranker_agent():
    """Get or create ranker agent."""
    global _ranker_agent
    if _ranker_agent is None:
        from bioflow.agents import RankerAgent
        _ranker_agent = RankerAgent()
    return _ranker_agent


@app.post("/api/agents/generate")
async def agent_generate(request: GenerateRequest):
    """
    Generate molecules from text prompt or seed SMILES.
    
    Modes:
    - text: Generate from natural language description
    - mutate: Create variants of a seed molecule
    - scaffold: Generate around a core scaffold
    """
    try:
        agent = get_generator_agent()
        
        if request.mode == "text":
            input_data = request.prompt
        else:
            input_data = {
                "mode": request.mode,
                "prompt": request.prompt,
                "smiles": request.smiles,
                "num_samples": request.num_samples,
            }
        
        result = agent.process(input_data)
        
        return {
            "success": result.success,
            "molecules": result.content,
            "metadata": result.metadata,
        }
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/validate")
async def agent_validate(request: ValidateRequest):
    """
    Validate molecules for ADMET and drug-likeness properties.
    
    Returns validation scores, property values, and structural alerts.
    """
    try:
        agent = get_validator_agent()
        agent.check_lipinski = request.check_lipinski
        agent.check_admet = request.check_admet
        agent.check_alerts = request.check_alerts
        
        result = agent.process(request.smiles)
        
        return {
            "success": result.success,
            "validations": result.content,
            "summary": result.metadata,
        }
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/rank")
async def agent_rank(request: RankRequest):
    """
    Rank candidates based on multiple criteria.
    
    Combines validation scores, confidence, and other metrics.
    """
    try:
        agent = get_ranker_agent()
        
        input_data = {
            "candidates": request.candidates,
            "top_k": request.top_k,
        }
        if request.weights:
            input_data["weights"] = request.weights
        
        result = agent.process(input_data)
        
        return {
            "success": result.success,
            "ranked": result.content,
            "metadata": result.metadata,
        }
    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/workflow")
async def run_discovery_workflow(request: WorkflowRequest):
    """
    Run full discovery workflow: Generate → Validate → Rank.
    
    Returns top candidates with all validation and ranking metadata.
    """
    try:
        from bioflow.agents import DiscoveryWorkflow
        
        workflow = DiscoveryWorkflow(
            num_candidates=request.num_candidates,
            top_k=request.top_k,
        )
        
        result = workflow.run(request.query)
        top_candidates = workflow.get_top_candidates(result)
        
        return {
            "success": result.status.value == "completed",
            "status": result.status.value,
            "steps_completed": result.steps_completed,
            "total_steps": result.total_steps,
            "execution_time_ms": result.execution_time_ms,
            "top_candidates": top_candidates,
            "all_outputs": result.outputs,
            "errors": result.errors,
        }
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run with: uvicorn bioflow.api.server:app --reload --port 8000
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
