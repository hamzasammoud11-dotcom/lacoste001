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
import json
import time
import json
import time
import base64
import requests
import asyncio
import platform
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

# Fix Windows asyncio issues - must be done before any FastAPI imports
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

# Data directory for resolving relative paths
DATA_DIR = os.path.join(ROOT_DIR, "data")

def convert_image_path_to_base64(image_value: Any) -> Optional[str]:
    r"""
    AGGRESSIVE MODE: Convert a local file path to a base64 data URL.
    
    Browsers block local file paths (e.g., C:\Users...), so we MUST convert
    all local images to base64 data URLs for them to display in the UI.
    
    Logic:
    1. If it starts with 'http', return as-is (browser can load it)
    2. If already a data URL, return as-is
    3. Otherwise, treat as local file path:
       - Normalize Windows paths (\ -> /)
       - Try absolute path first
       - Try relative to data/ folder
       - Try just the filename in data/images/
    4. If file not found, return None (don't send broken paths)
    
    Returns the image as a data URL if successful, or None if conversion fails.
    """
    if not image_value:
        return None
    
    if not isinstance(image_value, str):
        return None
    
    image_value = image_value.strip()
    
    # HTTP/HTTPS URL - return as-is (browser can load it)
    if image_value.startswith('http://') or image_value.startswith('https://'):
        return image_value
    
    # Already a data URL - return as-is
    if image_value.startswith('data:'):
        return image_value
    
    # --- AGGRESSIVE LOCAL FILE HANDLING ---
    
    # Normalize path separators (Windows -> Unix style for consistency)
    normalized_path = image_value.replace('\\', '/')
    
    # List of paths to try, in order of priority
    paths_to_try = []
    
    # 1. Try the path as-is (might be absolute)
    paths_to_try.append(image_value)
    paths_to_try.append(normalized_path)
    
    # 2. If it looks like an absolute Windows path, try it
    if len(image_value) > 2 and image_value[1] == ':':
        paths_to_try.append(image_value)
    
    # 3. Try relative to project root
    paths_to_try.append(os.path.join(ROOT_DIR, normalized_path))
    
    # 4. Try relative to data/ folder
    paths_to_try.append(os.path.join(DATA_DIR, normalized_path))
    
    # 5. Try in data/images/ folder
    paths_to_try.append(os.path.join(DATA_DIR, "images", normalized_path))
    
    # 6. Try just the filename in various locations
    filename = os.path.basename(normalized_path)
    paths_to_try.append(os.path.join(DATA_DIR, filename))
    paths_to_try.append(os.path.join(DATA_DIR, "images", filename))
    paths_to_try.append(os.path.join(ROOT_DIR, filename))
    
    # 7. Handle paths that start with 'data/' or './data/'
    if normalized_path.startswith('data/') or normalized_path.startswith('./data/'):
        clean_path = normalized_path.lstrip('./').lstrip('data/')
        paths_to_try.append(os.path.join(DATA_DIR, clean_path))
        paths_to_try.append(os.path.join(DATA_DIR, "images", clean_path))
    
    # Try each path
    for path in paths_to_try:
        if not path:
            continue
        try:
            if os.path.isfile(path):
                with open(path, 'rb') as f:
                    image_bytes = f.read()
                
                # Detect MIME type from extension
                ext = os.path.splitext(path)[1].lower()
                mime_types = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.bmp': 'image/bmp',
                    '.tiff': 'image/tiff',
                    '.tif': 'image/tiff',
                    '.webp': 'image/webp',
                    '.svg': 'image/svg+xml',
                }
                mime_type = mime_types.get(ext, 'image/png')
                
                base64_str = base64.b64encode(image_bytes).decode('utf-8')
                logger.debug(f"Successfully converted image to base64: {path}")
                return f"data:{mime_type};base64,{base64_str}"
        except Exception as e:
            logger.debug(f"Could not read image from {path}: {e}")
            continue
    
    # File not found anywhere - return None (don't send broken path to browser)
    logger.warning(f"Image file not found, tried multiple paths for: {image_value}")
    return None


def ensure_image_is_displayable(metadata: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Ensure the 'image' field in metadata is a displayable format (data URL or HTTP URL).
    
    CRITICAL: Browsers cannot display local file paths like C:\Users\...
    This function converts ALL local image paths to base64 data URLs.
    
    Checks both 'image' and 'image_path' fields in metadata.
    """
    if not metadata:
        return metadata
    
    # Create a copy to avoid modifying original
    metadata = dict(metadata)
    
    # Check 'image' field
    image_value = metadata.get('image')
    if image_value:
        converted = convert_image_path_to_base64(image_value)
        if converted:
            metadata['image'] = converted
        else:
            # Remove broken path - better to show nothing than broken image
            metadata['image'] = None
    
    # Also check 'image_path' field (some records use this)
    image_path = metadata.get('image_path')
    if image_path:
        converted = convert_image_path_to_base64(image_path)
        if converted:
            # Store as 'image' for consistency
            metadata['image'] = converted
            metadata['image_path'] = None  # Clear the local path
        else:
            metadata['image_path'] = None
    
    # Check 'thumbnail' field too
    thumbnail = metadata.get('thumbnail')
    if thumbnail:
        converted = convert_image_path_to_base64(thumbnail)
        if converted:
            metadata['thumbnail'] = converted
        else:
            metadata['thumbnail'] = None
    
    return metadata

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
    modality: str = Field(default="smiles", description="smiles | protein | text | image")
    metadata: Optional[Dict[str, Any]] = None


class ImageIngestRequest(BaseModel):
    """Request to ingest a biological image."""
    image: str = Field(..., description="Image file path, URL, or base64 encoded string")
    image_type: str = Field(default="other", description="microscopy | gel | spectra | xray | other")
    experiment_id: Optional[str] = Field(default=None, description="Experiment identifier")
    description: Optional[str] = Field(default="", description="Image description")
    caption: Optional[str] = Field(default="", description="Image caption")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    collection: Optional[str] = Field(default="bioflow_memory", description="Target collection")


class BatchImageIngestRequest(BaseModel):
    """Request to batch ingest multiple images."""
    images: List[ImageIngestRequest] = Field(..., description="List of images to ingest")
    collection: Optional[str] = Field(default="bioflow_memory", description="Target collection")


class IngestSourceRequest(BaseModel):
    """Request to ingest from a specific source."""
    query: str
    limit: int = Field(default=100, ge=1, le=10000)
    batch_size: Optional[int] = Field(default=None, ge=1, le=1000)
    rate_limit: Optional[float] = Field(default=None, ge=0.0)
    collection: Optional[str] = Field(default="bioflow_memory")
    sync: bool = Field(default=False, description="Run synchronously (may block)")
    # PubMed-specific
    email: Optional[str] = None
    api_key: Optional[str] = None
    # ChEMBL-specific
    search_mode: Optional[str] = Field(default=None, description="target | molecule")


class BatchIngestRequest(BaseModel):
    """Request to ingest multiple items at once."""
    items: List[IngestRequest] = Field(..., description="List of items to ingest")
    parallel: bool = Field(default=True, description="Process items in parallel")
    batch_size: int = Field(default=10, ge=1, le=100, description="Batch size for parallel processing")


class IngestAllRequest(BaseModel):
    """Request to ingest from all sources."""
    query: str
    pubmed_limit: int = Field(default=100, ge=0, le=10000)
    uniprot_limit: int = Field(default=50, ge=0, le=10000)
    chembl_limit: int = Field(default=30, ge=0, le=10000)
    batch_size: Optional[int] = Field(default=None, ge=1, le=1000)
    rate_limit: Optional[float] = Field(default=None, ge=0.0)
    collection: Optional[str] = Field(default="bioflow_memory")
    skip_pubmed: bool = False
    skip_uniprot: bool = False
    skip_chembl: bool = False
    sync: bool = Field(default=False, description="Run synchronously (may block)")
    # PubMed-specific
    email: Optional[str] = None
    api_key: Optional[str] = None
    # ChEMBL-specific
    search_mode: Optional[str] = Field(default=None, description="target | molecule")


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
    modality: str = Field(default="auto", description="Query modality: auto, text, molecule, protein")
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


@app.get("/api/health/metrics")
async def health_metrics():
    """Detailed health metrics for Qdrant and model readiness."""
    metrics = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "qdrant": {"available": False, "collections": []},
        "models": {"available": False, "device": None, "obm_loaded": False},
    }

    if qdrant_service:
        try:
            collections = qdrant_service.list_collections()
            metrics["qdrant"]["available"] = True
            metrics["qdrant"]["collections"] = collections
            try:
                client = qdrant_service._get_client()
                collection_stats = {}
                for name in collections:
                    try:
                        info = client.get_collection(name)
                        collection_stats[name] = {
                            "vectors_count": getattr(info, "vectors_count", None),
                            "points_count": getattr(info, "points_count", None),
                        }
                    except Exception:
                        collection_stats[name] = {}
                metrics["qdrant"]["stats"] = collection_stats
            except Exception:
                metrics["qdrant"]["stats"] = {}
        except Exception:
            metrics["qdrant"]["available"] = False

    if model_service:
        try:
            metrics["models"]["available"] = True
            metrics["models"]["device"] = getattr(model_service, "device", None)
            try:
                _ = model_service.get_obm_encoder()
                metrics["models"]["obm_loaded"] = True
            except Exception:
                metrics["models"]["obm_loaded"] = False
        except Exception:
            metrics["models"]["available"] = False

    return metrics


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


def _fallback_dti_prediction(drug_smiles: str, target_sequence: str) -> Dict[str, Any]:
    """Fallback prediction when DeepPurpose is unavailable."""
    seed = abs(hash(drug_smiles + target_sequence)) % 1000
    affinity = round(0.1 + (seed % 100) / 100.0, 4)
    confidence = 0.2
    return {
        "drug_smiles": drug_smiles,
        "target_sequence": target_sequence,
        "binding_affinity": affinity,
        "confidence": confidence,
        "interaction_probability": min(confidence + 0.05, 1.0),
    }


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
    except ImportError:
        fallback = _fallback_dti_prediction(request.drug_smiles, request.target_sequence)
        return {
            "success": True,
            "prediction": fallback,
            "metadata": {
                "model": "fallback",
                "timestamp": datetime.utcnow().isoformat(),
                "note": "DeepPurpose not installed; using fallback prediction.",
            },
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
        if not model_service:
            raise HTTPException(status_code=503, detail="Model service not available")
        encoder = model_service.get_obm_encoder()
        _enhanced_search_service = EnhancedSearchService(
            qdrant_service=qdrant_service,
            obm_encoder=encoder,
        )
    return _enhanced_search_service


def _log_event(event: str, request_id: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        **fields,
    }
    try:
        logger.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.info(f"{event} {payload}")


@app.post("/api/search")
async def enhanced_search(request: dict = None):
    """
    Enhanced semantic search with MMR diversification and evidence linking.
    
    Features:
    - Maximal Marginal Relevance (MMR) for diverse results
    - Evidence links to source databases (PubMed, UniProt, ChEMBL)
    - Citations and source tracking
    - Filtered search by modality, source, etc.
    
    Accepts both old format (type, limit) and new format (modality, top_k).
    """
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        if not qdrant_service:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        # Handle both old and new request formats
        query = request.get("query", "")
        modality = request.get("modality") or request.get("type", "text")
        top_k = request.get("top_k") or request.get("limit", 20)
        collection = request.get("collection")
        use_mmr = request.get("use_mmr", True)
        lambda_param = request.get("lambda_param", 0.7)
        include_images = request.get("include_images", False)
        filters = request.get("filters") or {}
        dataset = request.get("dataset")  # Optional dataset filter (davis, kiba)
        
        # Add dataset filter if specified
        if dataset:
            filters["source"] = dataset.lower()
        
        # Map old type names to new modality names
        type_to_modality = {
            "drug": "molecule",
            "target": "protein",
            "text": "text",
        }
        modality = type_to_modality.get(modality, modality)
        
        # Check if any collections exist first
        try:
            existing_collections = qdrant_service.list_collections()
            if not existing_collections:
                # No data ingested yet - return empty results
                return {
                    "results": [],
                    "query": query,
                    "modality": modality,
                    "total_found": 0,
                    "returned": 0,
                    "diversity_score": None,
                    "filters_applied": {},
                    "search_time_ms": 0,
                    "message": "No data ingested yet. Please ingest data first."
                }
        except Exception as e:
            logger.warning(f"Failed to list collections: {e}")
            # Try to proceed anyway
        
        search_service = get_enhanced_search_service()
        response = search_service.search(
            query=query,
            modality=modality,
            collection=collection,
            top_k=int(top_k),
            use_mmr=use_mmr,
            lambda_param=lambda_param,
            filters=filters,
            include_images=include_images,
        )
        
        payload = response.to_dict()
        
        # Ensure image paths are converted to base64 for UI display
        for result in payload.get("results", []):
            if result.get("modality") == "image" and result.get("metadata"):
                result["metadata"] = ensure_image_is_displayable(result["metadata"])
        
        _log_event(
            "search",
            request_id,
            query=query[:200] if query else "",
            top_k=top_k,
            use_mmr=use_mmr,
            returned=payload.get("returned"),
            total_found=payload.get("total_found"),
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        return payload
        
    except Exception as e:
        _log_event(
            "search_error",
            request_id,
            error=str(e),
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
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


class ImageSearchRequest(BaseModel):
    """Request for image similarity search."""
    image: str = Field(..., description="Image file path, URL, or base64 encoded string")
    image_type: str = Field(default="other", description="Type of image: microscopy, gel, spectra, xray, other")
    collection: Optional[str] = Field(default=None, description="Target collection")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    use_mmr: bool = Field(default=True, description="Apply MMR diversification")
    lambda_param: float = Field(default=0.7, ge=0.0, le=1.0, description="MMR lambda")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")


@app.post("/api/search/image")
async def search_image(request: ImageSearchRequest):
    """
    Image similarity search.

    Find similar biological images (microscopy, gels, spectra).
    Supports query-by-image and cross-modal search.
    """
    request_id = uuid.uuid4().hex[:12]
    logger.info(f"[{request_id}] Image search request: image_type={request.image_type}, top_k={request.top_k}")
    try:
        if not qdrant_service:
            raise HTTPException(status_code=503, detail="Qdrant service not available")

        search_service = get_enhanced_search_service()
        logger.info(f"[{request_id}] Calling search service with modality=image")
        
        # Use 'search' method (not 'enhanced_search' which doesn't exist)
        response = search_service.search(
            query=request.image,
            modality="image",
            collection=request.collection,
            top_k=request.top_k,
            use_mmr=request.use_mmr,
            lambda_param=request.lambda_param,
            filters=request.filters or {}
        )

        logger.info(f"[{request_id}] Image search returned {response.returned} results")
        
        payload = response.to_dict()
        
        # Ensure image paths are converted to base64 for UI display
        for result in payload.get("results", []):
            if result.get("metadata"):
                result["metadata"] = ensure_image_is_displayable(result["metadata"])
        
        return payload

    except Exception as e:
        logger.error(f"[{request_id}] Image search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Use Case 4: Navigate Neighbors & Design Variants
# ============================================================================

class NeighborRequest(BaseModel):
    """Request for guided neighbor exploration."""
    item_id: str = Field(..., description="ID of the item to find neighbors for")
    collection: Optional[str] = Field(default=None, description="Collection containing the item")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of neighbors")
    include_cross_modal: bool = Field(default=True, description="Include results from other modalities")
    diversity: float = Field(default=0.3, ge=0.0, le=1.0, description="Diversity factor (0=similar, 1=diverse)")


class DesignVariantRequest(BaseModel):
    """Request for design variant suggestions."""
    reference: str = Field(..., description="Reference content (SMILES, sequence, or text)")
    modality: str = Field(default="auto", description="Reference modality")
    num_variants: int = Field(default=5, ge=1, le=20, description="Number of variants")
    diversity: float = Field(default=0.5, ge=0.0, le=1.0, description="Diversity (0=close, 1=diverse)")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Design constraints")


class ExperimentSearchRequest(BaseModel):
    """Request for experiment-focused search (Use Case 4)."""
    query: str = Field(..., description="Search query")
    experiment_type: Optional[str] = Field(default=None, description="binding_assay, activity_assay, admet, phenotypic")
    outcome: Optional[str] = Field(default=None, description="success, failure, partial")
    target: Optional[str] = Field(default=None, description="Target name")
    quality_min: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum quality score")
    top_k: int = Field(default=20, ge=1, le=100)


@app.post("/api/neighbors")
async def find_neighbors(request: NeighborRequest):
    """
    Navigate neighbors - guided exploration for Use Case 4.
    
    Given an item, find semantically similar items across modalities
    with controlled diversity for exploration.
    """
    request_id = uuid.uuid4().hex[:12]
    logger.info(f"[{request_id}] Neighbor search: item_id={request.item_id}, top_k={request.top_k}, diversity={request.diversity}")
    try:
        if not qdrant_service:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        # Get the source item's vector
        client = qdrant_service._get_client()
        collections = [request.collection] if request.collection else qdrant_service.list_collections()
        logger.info(f"[{request_id}] Searching in collections: {collections}")
        
        source_vector = None
        source_item = None
        source_collection = None
        
        for coll in collections:
            try:
                points = client.retrieve(
                    collection_name=coll,
                    ids=[request.item_id],
                    with_payload=True,
                    with_vectors=True,
                )
                if points:
                    source_item = points[0]
                    source_vector = source_item.vector
                    source_collection = coll
                    break
            except Exception:
                continue
        
        if not source_vector:
            raise HTTPException(status_code=404, detail=f"Item {request.item_id} not found")
        
        logger.info(f"[{request_id}] Found source item in collection {source_collection}, vector dim={len(source_vector)}")
        
        # Search for neighbors using the item's vector
        search_service = get_enhanced_search_service()
        
        # Adjust lambda based on diversity preference (inverse)
        lambda_param = 1.0 - request.diversity
        
        # Perform search
        all_neighbors = []
        search_collections = collections if request.include_cross_modal else [source_collection]
        
        for coll in search_collections:
            try:
                results = client.search(
                    collection_name=coll,
                    query_vector=source_vector,
                    limit=request.top_k * 2,  # Get more for diversity filtering
                    with_payload=True,
                    with_vectors=True,
                )
                
                for r in results:
                    if str(r.id) == request.item_id:
                        continue  # Skip self
                    
                    all_neighbors.append({
                        "id": str(r.id),
                        "score": r.score,
                        "content": r.payload.get("content", ""),
                        "modality": r.payload.get("modality", "unknown"),
                        "collection": coll,
                        "metadata": r.payload,
                        "vector": r.vector,
                    })
            except Exception as e:
                logger.warning(f"Neighbor search in {coll} failed: {e}")
        
        # Apply MMR for diversity
        logger.info(f"[{request_id}] Found {len(all_neighbors)} neighbors, applying MMR if needed (top_k={request.top_k})")
        if len(all_neighbors) > request.top_k:
            from bioflow.search.mmr import mmr_rerank
            
            logger.info(f"[{request_id}] Applying MMR reranking with lambda={lambda_param}")
            neighbor_embeddings = [n["vector"] for n in all_neighbors]
            neighbor_results = [
                {
                    "id": n["id"],
                    "score": n["score"],
                    "content": n["content"],
                    "modality": n["modality"],
                    "metadata": n["metadata"],
                }
                for n in all_neighbors
            ]
            
            reranked = mmr_rerank(
                results=neighbor_results,
                query_embedding=source_vector,
                lambda_param=lambda_param,
                top_k=request.top_k,
                embeddings=neighbor_embeddings,
            )
            
            # Convert MMRResult back to neighbor format
            all_neighbors = [
                {
                    "id": r.id,
                    "score": r.original_score,
                    "content": r.content,
                    "modality": r.modality,
                    "collection": next((n["collection"] for n in all_neighbors if n["id"] == r.id), ""),
                    "metadata": r.metadata,
                    "vector": r.embedding,
                }
                for r in reranked
            ]
        else:
            all_neighbors = all_neighbors[:request.top_k]
        
        # Remove vectors from response and ensure images are displayable
        for n in all_neighbors:
            n.pop("vector", None)
            # Convert local image paths to base64 for UI display
            if n.get("modality") == "image" and n.get("metadata"):
                n["metadata"] = ensure_image_is_displayable(n["metadata"])
        
        # Group by modality for faceted navigation
        facets = {}
        for n in all_neighbors:
            mod = n.get("modality", "unknown")
            facets[mod] = facets.get(mod, 0) + 1
        
        return {
            "source_id": request.item_id,
            "source_modality": source_item.payload.get("modality", "unknown"),
            "neighbors": all_neighbors,
            "facets": facets,
            "total_found": len(all_neighbors),
            "diversity_applied": request.diversity,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Neighbor search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/design/variants")
async def suggest_design_variants(request: DesignVariantRequest):
    """
    Design assistance - propose close but diverse variants.
    
    For Use Case 4: Given a reference (molecule, sequence, experiment description),
    find similar items that offer diverse design alternatives with justifications.
    """
    request_id = uuid.uuid4().hex[:12]
    try:
        if not qdrant_service or not model_service:
            raise HTTPException(status_code=503, detail="Services not available")
        
        search_service = get_enhanced_search_service()
        
        # Encode the reference
        modality = request.modality
        if modality == "auto":
            # Auto-detect
            ref = request.reference.strip()
            if ref.isupper() and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in ref[:20]) and len(ref) > 20:
                modality = "protein"
            elif any(c in ref for c in "[]()=#@") or (len(ref) < 100 and not " " in ref):
                modality = "molecule"
            else:
                modality = "text"
        
        # Search with high diversity (low lambda)
        lambda_param = 1.0 - request.diversity
        
        response = search_service.search(
            query=request.reference,
            modality=modality,
            top_k=request.num_variants * 3,  # Get more for filtering
            use_mmr=True,
            lambda_param=lambda_param,
            filters=request.constraints or {},
        )
        
        # Build variant suggestions with justifications
        variants = []
        for i, result in enumerate(response.results[:request.num_variants]):
            # Generate justification based on similarity and differences
            justification = _generate_variant_justification(
                reference=request.reference,
                variant=result,
                modality=modality,
                rank=i + 1,
            )
            
            variants.append({
                "rank": i + 1,
                "id": result.id,
                "content": result.content,
                "modality": result.modality,
                "similarity_score": result.score,
                "diversity_score": result.diversity_penalty or 0.0,
                "justification": justification,
                "evidence_links": [
                    {"source": l.source, "identifier": l.identifier, "url": l.url}
                    for l in result.evidence_links
                ],
                "metadata": result.metadata,
            })
        
        return {
            "reference": request.reference[:200],
            "reference_modality": modality,
            "variants": variants,
            "num_returned": len(variants),
            "diversity_setting": request.diversity,
            "constraints_applied": request.constraints or {},
        }
        
    except Exception as e:
        logger.error(f"Design variant suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiments/search")
async def search_experiments(request: ExperimentSearchRequest):
    """
    Search experimental results with outcome-based filtering.
    
    For Use Case 4: Find experiments with specific outcomes, targets, or types.
    Supports filtering by success/failure labels and quality scores.
    """
    try:
        if not qdrant_service:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        search_service = get_enhanced_search_service()
        
        # Build filters
        filters = {
            "modality": "experiment",
        }
        if request.experiment_type:
            filters["experiment_type"] = request.experiment_type
        if request.outcome:
            filters["outcome"] = request.outcome
        if request.target:
            filters["target"] = request.target
        
        response = search_service.search(
            query=request.query,
            modality="text",  # Experiments are encoded as text
            top_k=request.top_k,
            use_mmr=True,
            lambda_param=0.7,
            filters=filters,
        )
        
        # Post-filter by quality if specified
        results = response.results
        if request.quality_min is not None:
            results = [
                r for r in results 
                if r.metadata.get("quality_score", 0) >= request.quality_min
            ]
        
        # Enrich with experiment-specific info
        enriched = []
        for r in results:
            enriched.append({
                "id": r.id,
                "score": r.score,
                "experiment_id": r.metadata.get("experiment_id"),
                "title": r.metadata.get("title"),
                "experiment_type": r.metadata.get("experiment_type"),
                "outcome": r.metadata.get("outcome"),
                "quality_score": r.metadata.get("quality_score"),
                "measurements": r.metadata.get("measurements", []),
                "conditions": r.metadata.get("conditions", {}),
                "target": r.metadata.get("target"),
                "molecule": r.metadata.get("molecule"),
                "description": r.metadata.get("description"),
                "evidence_links": [l.to_dict() if hasattr(l, 'to_dict') else l.__dict__ for l in r.evidence_links],
            })
        
        return {
            "query": request.query,
            "experiments": enriched,
            "total_found": len(enriched),
            "filters_applied": {
                "experiment_type": request.experiment_type,
                "outcome": request.outcome,
                "target": request.target,
                "quality_min": request.quality_min,
            },
        }
        
    except Exception as e:
        logger.error(f"Experiment search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_variant_justification(
    reference: str,
    variant,
    modality: str,
    rank: int,
) -> str:
    """Generate a justification for why this variant is suggested."""
    score = variant.score if hasattr(variant, 'score') else 0.0
    content = variant.content if hasattr(variant, 'content') else ""
    metadata = variant.metadata if hasattr(variant, 'metadata') else {}
    
    justifications = []
    
    # Similarity-based justification
    if score > 0.9:
        justifications.append(f"Highly similar (score: {score:.2f}) to reference")
    elif score > 0.7:
        justifications.append(f"Moderately similar (score: {score:.2f})")
    else:
        justifications.append(f"Diverse alternative (score: {score:.2f})")
    
    # Source-based justification
    source = metadata.get("source", "")
    if source == "pubmed":
        justifications.append("Supported by published literature")
    elif source == "chembl":
        justifications.append("Has experimental activity data")
    elif source == "uniprot":
        justifications.append("Annotated protein with known function")
    elif source == "experiment":
        outcome = metadata.get("outcome", "")
        if outcome == "success":
            justifications.append("From successful experiment")
        elif outcome == "failure":
            justifications.append("Negative control reference")
    
    # Modality-specific justification
    if modality == "molecule":
        if "target" in metadata:
            justifications.append(f"Known to bind {metadata['target']}")
    elif modality == "protein":
        if "organism" in metadata:
            justifications.append(f"From {metadata['organism']}")
    
    # Evidence links
    evidence = variant.evidence_links if hasattr(variant, 'evidence_links') else []
    if evidence:
        justifications.append(f"Traceable to {len(evidence)} source(s)")
    
    return "; ".join(justifications) if justifications else f"Ranked #{rank} by relevance"
async def ingest_data(request: IngestRequest):
    """Ingest data into vector database."""
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        if not qdrant_service:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        result = qdrant_service.ingest(
            content=request.content,
            modality=request.modality,
            metadata=request.metadata
        )
        _log_event(
            "ingest_single",
            request_id,
            modality=request.modality,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        return {
            "success": result.success,
            "id": result.id,
            "collection": result.collection,
            "modality": request.modality,
            "message": result.message,
        }
        
    except Exception as e:
        _log_event(
            "ingest_single_error",
            request_id,
            error=str(e),
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/batch")
async def ingest_batch(request: BatchIngestRequest):
    """
    Batch ingest multiple items for improved performance.
    
    Processes items in parallel (if parallel=True) to significantly
    improve ingestion speed compared to sequential single-item ingestion.
    """
    import asyncio
    import concurrent.futures
    
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")
    
    items = request.items
    if not items:
        return {"success": True, "ingested": 0, "failed": 0, "rate_per_sec": 0}
    
    results = {"ingested": 0, "failed": 0, "ids": [], "errors": []}
    
    def ingest_one(item: IngestRequest):
        """Ingest a single item (for parallel execution)."""
        try:
            result = qdrant_service.ingest(
                content=item.content,
                modality=item.modality,
                metadata=item.metadata
            )
            return {"success": result.success, "id": result.id, "error": None}
        except Exception as e:
            return {"success": False, "id": None, "error": str(e)}
    
    try:
        if request.parallel:
            # Process in batches using thread pool
            batch_size = min(request.batch_size, len(items))
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [executor.submit(ingest_one, item) for item in items]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res["success"]:
                        results["ingested"] += 1
                        if res["id"]:
                            results["ids"].append(res["id"])
                    else:
                        results["failed"] += 1
                        if res["error"]:
                            results["errors"].append(res["error"])
        else:
            # Sequential processing
            for item in items:
                res = ingest_one(item)
                if res["success"]:
                    results["ingested"] += 1
                    if res["id"]:
                        results["ids"].append(res["id"])
                else:
                    results["failed"] += 1
                    if res["error"]:
                        results["errors"].append(res["error"])
        
        duration_s = time.perf_counter() - start
        rate = results["ingested"] / duration_s if duration_s > 0 else 0
        
        _log_event(
            "ingest_batch",
            request_id,
            count=len(items),
            ingested=results["ingested"],
            failed=results["failed"],
            rate_per_sec=round(rate, 2),
            duration_ms=round(duration_s * 1000, 2),
        )
        
        return {
            "success": results["failed"] == 0,
            "ingested": results["ingested"],
            "failed": results["failed"],
            "ids": results["ids"][:50],  # Limit response size
            "rate_per_sec": round(rate, 2),
            "duration_ms": round(duration_s * 1000, 2),
            "errors": results["errors"][:10] if results["errors"] else None,
        }
        
    except Exception as e:
        _log_event(
            "ingest_batch_error",
            request_id,
            error=str(e),
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        logger.error(f"Batch ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _list_items_by_modality(modality: str, limit: int, offset: int):
    """List items across all collections by modality."""
    if not qdrant_service:
        return [], 0

    collections = []
    try:
        collections = qdrant_service.list_collections()
    except Exception:
        collections = ["molecules", "proteins", "bioflow_memory"]

    items: List[Any] = []
    target = offset + limit
    for coll in collections:
        if len(items) >= target:
            break
        try:
            items.extend(
                qdrant_service.list_items(
                    collection=coll,
                    limit=target,
                    offset=0,
                    filter_modality=modality,
                )
            )
        except Exception:
            continue

    total = len(items)
    return items[offset:offset + limit], total


def _find_point_by_id(point_id: str):
    """Find a point by ID across collections."""
    if not qdrant_service:
        return None, None
    client = qdrant_service._get_client()
    try:
        collections = qdrant_service.list_collections()
    except Exception:
        collections = ["molecules", "proteins", "bioflow_memory"]

    for coll in collections:
        try:
            points = client.retrieve(
                collection_name=coll,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            if points:
                return coll, points[0]
        except Exception:
            continue
    return None, None


@app.get("/api/molecules")
async def list_molecules(limit: int = 20, offset: int = 0):
    """List molecules in the database."""
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")
    
    try:
        results, total = _list_items_by_modality("molecule", limit, offset)
        molecules = []
        for r in results:
            metadata = r.metadata or {}
            # Ensure smiles is always a string
            smiles = r.content
            if not isinstance(smiles, str):
                smiles = str(smiles) if smiles else ""
            # Also check metadata for smiles
            if not smiles:
                smiles = metadata.get("smiles", "") or metadata.get("SMILES", "") or ""
            molecules.append({
                "id": r.id,
                "smiles": smiles,
                "name": metadata.get("name", metadata.get("title", "Unknown")),
                "pubchemCid": metadata.get("pubchem_cid", metadata.get("pubchemCid", metadata.get("cid", 0))),
                "description": metadata.get("description", metadata.get("title", "")),
                "mw": metadata.get("mw", metadata.get("molecular_weight", 0)),
            })
        return {
            "molecules": molecules,
            "total": total,
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
        results, total = _list_items_by_modality("protein", limit, offset)
        proteins = []
        for r in results:
            metadata = r.metadata or {}
            pdb_ids = metadata.get("pdb_ids", []) or []
            if isinstance(pdb_ids, str):
                pdb_ids = [pdb_ids]
            pdb_id = metadata.get("pdb_id") or (pdb_ids[0] if pdb_ids else "")
            name = metadata.get("name") or metadata.get("protein_name") or metadata.get("entry_name") or "Unknown"
            proteins.append({
                "id": r.id,
                "sequence": r.content[:50] + "..." if len(r.content) > 50 else r.content,
                "uniprot_id": metadata.get("uniprot_id", metadata.get("accession", "")),
                "name": name,
                "pdbId": pdb_id,
                "description": metadata.get("function", metadata.get("description", "")),
                "length": len(r.content),
            })
        return {
            "proteins": proteins,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Qdrant query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/molecules/{molecule_id}")
async def get_molecule(molecule_id: str):
    """Get molecule details by ID."""
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")

    _coll, point = _find_point_by_id(molecule_id)
    if point is None:
        raise HTTPException(status_code=404, detail="Molecule not found")

    payload = point.payload or {}
    smiles = payload.get("smiles") or payload.get("content", "")
    return {
        "id": str(point.id),
        "name": payload.get("name", payload.get("title", "Unknown")),
        "smiles": smiles,
        "pubchemCid": payload.get("pubchem_cid", payload.get("pubchemCid", payload.get("cid", 0))),
        "description": payload.get("description", payload.get("title", "")),
    }


@app.get("/api/molecules/{molecule_id}/sdf")
async def get_molecule_sdf(molecule_id: str):
    """Get molecule 3D structure as SDF."""
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")

    _coll, point = _find_point_by_id(molecule_id)
    if point is None:
        raise HTTPException(status_code=404, detail="Molecule not found")

    payload = point.payload or {}
    smiles = payload.get("smiles") or payload.get("content", "")
    if not smiles:
        raise HTTPException(status_code=404, detail="Molecule SMILES not available")

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception:
        raise HTTPException(status_code=503, detail="RDKit is required for 3D structures")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES")

    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        pass
    sdf = Chem.MolToMolBlock(mol)
    return PlainTextResponse(sdf, media_type="chemical/x-mdl-sdfile")


@app.get("/api/proteins/{protein_id}")
async def get_protein(protein_id: str):
    """Get protein details by ID."""
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")

    _coll, point = _find_point_by_id(protein_id)
    if point is None:
        raise HTTPException(status_code=404, detail="Protein not found")

    payload = point.payload or {}
    pdb_ids = payload.get("pdb_ids", []) or []
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]
    pdb_id = payload.get("pdb_id") or (pdb_ids[0] if pdb_ids else "")
    name = payload.get("name") or payload.get("protein_name") or payload.get("entry_name") or "Unknown"

    return {
        "id": str(point.id),
        "pdbId": pdb_id,
        "name": name,
        "description": payload.get("function", payload.get("description", "")),
    }


@app.get("/api/proteins/{protein_id}/pdb")
async def get_protein_pdb(protein_id: str):
    """Get protein structure as PDB text."""
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")

    _coll, point = _find_point_by_id(protein_id)
    if point is None:
        raise HTTPException(status_code=404, detail="Protein not found")

    payload = point.payload or {}
    pdb_ids = payload.get("pdb_ids", []) or []
    if isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids]
    pdb_id = payload.get("pdb_id") or (pdb_ids[0] if pdb_ids else "")
    if not pdb_id:
        raise HTTPException(status_code=404, detail="No PDB ID available for this protein")

    try:
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        resp = requests.get(pdb_url, timeout=20)
        if resp.status_code != 200:
            raise HTTPException(status_code=404, detail="PDB file not found")
        return PlainTextResponse(resp.text, media_type="chemical/x-pdb")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Explorer (Embeddings)
# ============================================================================
@app.get("/api/explorer/embeddings")
async def get_embeddings(
    dataset: str = "default",
    method: str = "pca",
    query: Optional[str] = None,
    modality: str = "auto",
    limit: int = 100,
):
    """Get 3D projections of embeddings for visualization."""
    import numpy as np

    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")

    def _detect_modality(q: str) -> str:
        if not q:
            return "text"
        seq = q.replace("\n", "").replace(" ", "")
        if len(seq) >= 25 and all(c.upper() in "ACDEFGHIKLMNPQRSTVWYBXZJUO" for c in seq[:25]):
            return "protein"
        if any(c in q for c in "[]()=#@") and any(c.isalpha() for c in q):
            return "molecule"
        return "text"

    def _cluster_for_modality(m: str) -> int:
        if m == "molecule":
            return 0
        if m == "protein":
            return 1
        return 2

    points = []
    vectors = []

    try:
        if query:
            effective_modality = modality if modality != "auto" else _detect_modality(query)
            search_results = qdrant_service.search(
                query=query,
                modality=effective_modality,
                limit=min(limit, 500),
                with_vectors=True,
            )
            for r in search_results:
                if r.vector is None:
                    continue
                vectors.append(r.vector)
                points.append({
                    "id": r.id,
                    "label": r.metadata.get("name", r.content[:40]),
                    "content": r.content,
                    "modality": r.modality,
                    "source": r.metadata.get("source", "unknown"),
                    "score": r.score,
                    "cluster": _cluster_for_modality(r.modality),
                    "metadata": r.metadata,
                })
        else:
            client = qdrant_service._get_client()
            collections = []
            try:
                collections = qdrant_service.list_collections()
            except Exception:
                collections = ["molecules", "proteins", "bioflow_memory"]

            per_collection = max(10, int(limit / max(1, len(collections))))
            for coll in collections:
                try:
                    res, _ = client.scroll(
                        collection_name=coll,
                        limit=per_collection,
                        with_payload=True,
                        with_vectors=True,
                    )
                    for p in res:
                        vec = p.vector
                        if isinstance(vec, dict):
                            vec = list(vec.values())[0] if vec else None
                        if vec is None:
                            continue
                        payload = p.payload or {}
                        modality_val = payload.get("modality", "text")
                        if modality_val == "smiles":
                            modality_val = "molecule"
                        vectors.append(vec)
                        points.append({
                            "id": str(p.id),
                            "label": payload.get("name", payload.get("content", "")[:40]),
                            "content": payload.get("content", ""),
                            "modality": modality_val,
                            "source": payload.get("source", "unknown"),
                            "score": payload.get("score", 0),
                            "cluster": _cluster_for_modality(modality_val),
                            "metadata": payload,
                        })
                except Exception as e:
                    logger.warning(f"Failed to scroll collection {coll}: {e}")
    except Exception as e:
        logger.error(f"Failed to get embeddings from Qdrant: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # If we have vectors, compute projection (fallback to deterministic positions on failure)
    coords = []
    method_used = method.lower()
    if vectors and len(vectors) >= 2:
        try:
            if method_used not in ("pca", "umap", "tsne"):
                method_used = "pca"
            arr = np.array(vectors, dtype=float)

            if method_used == "umap":
                try:
                    import umap  # type: ignore
                    reducer = umap.UMAP(n_components=3, random_state=42)
                    coords = reducer.fit_transform(arr).tolist()
                except Exception as e:
                    logger.warning(f"UMAP unavailable, falling back to PCA: {e}")
                    method_used = "pca"

            if method_used == "tsne":
                from sklearn.manifold import TSNE
                perplexity = min(30, max(5, int(len(arr) / 3)))
                tsne = TSNE(n_components=3, init="random", learning_rate="auto", perplexity=perplexity)
                coords = tsne.fit_transform(arr).tolist()

            if method_used == "pca":
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                coords = pca.fit_transform(arr).tolist()
        except Exception as e:
            logger.warning(f"Projection failed, using fallback: {e}")
            method_used = "fallback"

    if not coords:
        coords = []
        for p in points:
            np.random.seed(hash(p["id"]) % 2**32)
            cx, cy, cz = [(2, 3, 1), (-2, -1, -1), (1, -3, 0)][p["cluster"] % 3]
            coords.append([
                float(cx + np.random.randn() * 0.6),
                float(cy + np.random.randn() * 0.6),
                float(cz + np.random.randn() * 0.6),
            ])
        method_used = "fallback"

    for p, c in zip(points, coords):
        p["x"], p["y"], p["z"] = float(c[0]), float(c[1]), float(c[2])

    avg_score = float(np.mean([p.get("score", 0) for p in points])) if points else 0.0

    return {
        "points": points,
        "method": method_used,
        "dataset": dataset,
        "n_clusters": len(set(p["cluster"] for p in points)) if points else 0,
        "avg_score": avg_score,
    }


# ============================================================================
# Source Ingestion (Phase 3)
# ============================================================================
def _resolve_batch_size(requested: Optional[int]) -> int:
    if requested is not None:
        return requested
    env_val = os.getenv("INGEST_BATCH_SIZE")
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass
    return 50


def _resolve_rate_limit(source: str, requested: Optional[float]) -> float:
    if requested is not None:
        return requested
    env_key = f"{source.upper()}_RATE_LIMIT"
    env_val = os.getenv(env_key)
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    defaults = {"pubmed": 0.4, "uniprot": 0.2, "chembl": 0.3}
    return defaults.get(source, 0.3)


def _execute_source_ingestion(source: str, request: IngestSourceRequest):
    if not model_service or not qdrant_service:
        raise HTTPException(status_code=503, detail="Services not available")

    encoder = model_service.get_obm_encoder()
    batch_size = _resolve_batch_size(request.batch_size)
    rate_limit = _resolve_rate_limit(source, request.rate_limit)
    collection = request.collection or "bioflow_memory"

    if source == "pubmed":
        from bioflow.ingestion.pubmed_ingestor import PubMedIngestor
        ingestor = PubMedIngestor(
            qdrant_service=qdrant_service,
            obm_encoder=encoder,
            collection=collection,
            batch_size=batch_size,
            rate_limit=rate_limit,
            email=request.email or os.getenv("NCBI_EMAIL", "bioflow@example.com"),
            api_key=request.api_key or os.getenv("NCBI_API_KEY"),
        )
        return ingestor.ingest(request.query, request.limit)

    if source == "uniprot":
        from bioflow.ingestion.uniprot_ingestor import UniProtIngestor
        ingestor = UniProtIngestor(
            qdrant_service=qdrant_service,
            obm_encoder=encoder,
            collection=collection,
            batch_size=batch_size,
            rate_limit=rate_limit,
        )
        return ingestor.ingest(request.query, request.limit)

    if source == "chembl":
        from bioflow.ingestion.chembl_ingestor import ChEMBLIngestor
        ingestor = ChEMBLIngestor(
            qdrant_service=qdrant_service,
            obm_encoder=encoder,
            collection=collection,
            batch_size=batch_size,
            rate_limit=rate_limit,
            search_mode=request.search_mode or os.getenv("CHEMBL_SEARCH_MODE", "target"),
        )
        return ingestor.ingest(request.query, request.limit)

    raise HTTPException(status_code=400, detail=f"Unknown source: {source}")


def _start_ingestion_job(source: str, payload: Dict[str, Any], background_tasks: BackgroundTasks):
    job_id = f"ing_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().isoformat()
    JOBS[job_id] = {
        "job_id": job_id,
        "type": "ingestion",
        "source": source,
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "request": payload,
    }

    background_tasks.add_task(_run_ingestion_job, job_id, source, payload)
    return job_id


def _execute_all_ingestion(request: IngestAllRequest):
    results = {}
    if not request.skip_pubmed:
        results["pubmed"] = _execute_source_ingestion(
            "pubmed",
            IngestSourceRequest(
                query=request.query,
                limit=request.pubmed_limit,
                batch_size=request.batch_size,
                rate_limit=request.rate_limit,
                collection=request.collection,
                email=request.email,
                api_key=request.api_key,
            ),
        )
    if not request.skip_uniprot:
        results["uniprot"] = _execute_source_ingestion(
            "uniprot",
            IngestSourceRequest(
                query=request.query,
                limit=request.uniprot_limit,
                batch_size=request.batch_size,
                rate_limit=request.rate_limit,
                collection=request.collection,
            ),
        )
    if not request.skip_chembl:
        results["chembl"] = _execute_source_ingestion(
            "chembl",
            IngestSourceRequest(
                query=request.query,
                limit=request.chembl_limit,
                batch_size=request.batch_size,
                rate_limit=request.rate_limit,
                collection=request.collection,
                search_mode=request.search_mode,
            ),
        )
    return results


def _run_ingestion_job(job_id: str, source: str, payload: Dict[str, Any]) -> None:
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()
    try:
        if source == "all":
            results = _execute_all_ingestion(IngestAllRequest(**payload))
            JOBS[job_id]["result"] = {k: v.to_dict() for k, v in results.items()}
        else:
            req = IngestSourceRequest(**payload)
            result = _execute_source_ingestion(source, req)
            JOBS[job_id]["result"] = result.to_dict()

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat()
        logger.error(f"Ingestion job failed: {e}")


@app.post("/api/ingest/pubmed")
async def ingest_pubmed(request: IngestSourceRequest, background_tasks: BackgroundTasks):
    if request.sync:
        result = _execute_source_ingestion("pubmed", request)
        return {"success": True, "result": result.to_dict()}
    job_id = _start_ingestion_job("pubmed", request.model_dump(), background_tasks)
    return {"success": True, "job_id": job_id, "status": "pending"}


@app.post("/api/ingest/uniprot")
async def ingest_uniprot(request: IngestSourceRequest, background_tasks: BackgroundTasks):
    if request.sync:
        result = _execute_source_ingestion("uniprot", request)
        return {"success": True, "result": result.to_dict()}
    job_id = _start_ingestion_job("uniprot", request.model_dump(), background_tasks)
    return {"success": True, "job_id": job_id, "status": "pending"}


@app.post("/api/ingest/chembl")
async def ingest_chembl(request: IngestSourceRequest, background_tasks: BackgroundTasks):
    if request.sync:
        result = _execute_source_ingestion("chembl", request)
        return {"success": True, "result": result.to_dict()}
    job_id = _start_ingestion_job("chembl", request.model_dump(), background_tasks)
    return {"success": True, "job_id": job_id, "status": "pending"}


@app.post("/api/ingest/all")
async def ingest_all(request: IngestAllRequest, background_tasks: BackgroundTasks):
    if request.sync:
        results = _execute_all_ingestion(request)
        return {"success": True, "result": {k: v.to_dict() for k, v in results.items()}}
    job_id = _start_ingestion_job("all", request.model_dump(), background_tasks)
    return {"success": True, "job_id": job_id, "status": "pending"}


@app.post("/api/ingest/image")
async def ingest_image(request: ImageIngestRequest):
    """Ingest a single biological image."""
    try:
        from bioflow.ingestion.image_ingestor import ImageIngestor

        # Initialize ingestor
        ingestor = ImageIngestor(
            qdrant_service=qdrant_service,
            obm_encoder=model_service.get_obm_encoder(),
            collection=request.collection
        )

        # Prepare image data
        image_data = {
            "image": request.image,
            "image_type": request.image_type,
            "experiment_id": request.experiment_id or "",
            "description": request.description,
            "caption": request.caption,
            "metadata": request.metadata or {}
        }

        # Ingest single image
        result = ingestor.batch_ingest([image_data], collection=request.collection)

        return {
            "success": True,
            "result": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Image ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/image/batch")
async def ingest_images_batch(request: BatchImageIngestRequest):
    """Batch ingest multiple biological images."""
    try:
        from bioflow.ingestion.image_ingestor import ImageIngestor

        # Initialize ingestor
        ingestor = ImageIngestor(
            qdrant_service=qdrant_service,
            obm_encoder=model_service.get_obm_encoder(),
            collection=request.collection
        )

        # Prepare image data
        images_data = []
        for img in request.images:
            images_data.append({
                "image": img.image,
                "image_type": img.image_type,
                "experiment_id": img.experiment_id or "",
                "description": img.description,
                "caption": img.caption,
                "metadata": img.metadata or {}
            })

        # Batch ingest
        result = ingestor.batch_ingest(images_data, collection=request.collection)

        return {
            "success": True,
            "result": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Batch image ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Use Case 4: Experiment Ingestion
# ============================================================================

class ExperimentIngestRequest(BaseModel):
    """Request to ingest experimental data."""
    experiment_id: Optional[str] = Field(default=None, description="Unique experiment identifier")
    title: str = Field(..., description="Experiment title")
    type: str = Field(default="other", description="Experiment type: binding_assay, activity_assay, admet, phenotypic")
    measurements: List[Dict[str, Any]] = Field(default_factory=list, description="Measurements [{name, value, unit}]")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Experimental conditions")
    outcome: str = Field(default="unknown", description="success, failure, partial, inconclusive")
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Quality score")
    description: str = Field(default="", description="Experiment description")
    protocol: Optional[str] = Field(default=None, description="Protocol used")
    molecule: Optional[str] = Field(default=None, description="SMILES if applicable")
    target: Optional[str] = Field(default=None, description="Target name/sequence")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    collection: Optional[str] = Field(default="bioflow_memory", description="Target collection")


class BatchExperimentIngestRequest(BaseModel):
    """Request to batch ingest experiments."""
    experiments: List[ExperimentIngestRequest] = Field(..., description="List of experiments")
    collection: Optional[str] = Field(default="bioflow_memory", description="Target collection")


@app.post("/api/ingest/experiment")
async def ingest_experiment(request: ExperimentIngestRequest):
    """
    Ingest a single experimental result.
    
    Supports Use Case 4: Measurements, conditions, outcomes.
    """
    try:
        from bioflow.ingestion.experiment_ingestor import ExperimentIngestor
        
        ingestor = ExperimentIngestor(
            qdrant_service=qdrant_service,
            obm_encoder=model_service.get_obm_encoder(),
            collection=request.collection
        )
        
        experiment_data = {
            "experiment_id": request.experiment_id,
            "title": request.title,
            "type": request.type,
            "measurements": request.measurements,
            "conditions": request.conditions,
            "outcome": request.outcome,
            "quality_score": request.quality_score,
            "description": request.description,
            "protocol": request.protocol,
            "molecule": request.molecule,
            "target": request.target,
            "metadata": request.metadata or {},
        }
        
        result = ingestor.ingest_experiments([experiment_data], collection=request.collection)
        
        return {
            "success": result.total_indexed > 0,
            "experiment_id": request.experiment_id,
            "result": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Experiment ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/experiment/batch")
async def ingest_experiments_batch(request: BatchExperimentIngestRequest):
    """
    Batch ingest multiple experiments.
    
    Supports Use Case 4: Bulk experiment data loading.
    """
    try:
        from bioflow.ingestion.experiment_ingestor import ExperimentIngestor
        
        ingestor = ExperimentIngestor(
            qdrant_service=qdrant_service,
            obm_encoder=model_service.get_obm_encoder(),
            collection=request.collection
        )
        
        experiments_data = []
        for exp in request.experiments:
            experiments_data.append({
                "experiment_id": exp.experiment_id,
                "title": exp.title,
                "type": exp.type,
                "measurements": exp.measurements,
                "conditions": exp.conditions,
                "outcome": exp.outcome,
                "quality_score": exp.quality_score,
                "description": exp.description,
                "protocol": exp.protocol,
                "molecule": exp.molecule,
                "target": exp.target,
                "metadata": exp.metadata or {},
            })
        
        result = ingestor.ingest_experiments(experiments_data, collection=request.collection)
        
        return {
            "success": result.total_indexed > 0,
            "total": len(experiments_data),
            "result": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Batch experiment ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ingest/jobs/{job_id}")
async def get_ingest_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


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


@app.get("/api/stats")
async def get_stats():
    """Get system statistics for the data page."""
    total_vectors = 0
    collections_info = []
    molecules_count = 0
    proteins_count = 0
    
    if qdrant_service:
        try:
            collections = qdrant_service.list_collections()
            for coll in collections:
                try:
                    stats = qdrant_service.get_collection_stats(coll)
                    points = stats.get("points_count", 0)
                    total_vectors += points
                    collections_info.append(stats)
                    
                    # Track by collection type
                    if "molecule" in coll.lower():
                        molecules_count += points
                    elif "protein" in coll.lower():
                        proteins_count += points
                except Exception as e:
                    logger.warning(f"Failed to get stats for {coll}: {e}")
        except Exception as e:
            logger.warning(f"Failed to list collections: {e}")
    
    # Format datasets for frontend
    datasets = []
    for coll_stat in collections_info:
        coll_name = coll_stat.get("name", "unknown")
        points = coll_stat.get("points_count", 0)
        datasets.append({
            "name": coll_name,
            "type": "molecule" if "molecule" in coll_name.lower() else ("protein" if "protein" in coll_name.lower() else "mixed"),
            "count": f"{points:,}",
            "size": f"{(points * 3072) / 1024 / 1024:.1f} MB",  # Estimate based on vector size
            "updated": "Recently",
        })
    
    return {
        "datasets": datasets,
        "stats": {
            "datasets": len(collections_info),
            "molecules": f"{molecules_count:,}",
            "proteins": f"{proteins_count:,}",
            "storage": f"{(total_vectors * 3072) / 1024 / 1024:.1f} MB",
        },
        # Also include legacy fields for compatibility
        "total_vectors": total_vectors,
        "collections": collections_info,
        "model_status": "loaded" if model_service else "not_loaded",
        "qdrant_status": "connected" if qdrant_service else "disconnected",
    }


@app.get("/api/points")
async def get_points(limit: int = 500, view: str = "combined"):
    """Get points for visualization using pre-computed PCA from bio_discovery."""
    import numpy as np
    
    if not qdrant_service:
        return _get_mock_points(limit)
    
    points = []
    try:
        # Get available collections
        collections = qdrant_service.list_collections()
        if not collections:
            return _get_mock_points(limit)
        
        main_collection = collections[0]
        logger.info(f"Getting points from collection: {main_collection}")
        
        # Get Qdrant client directly to scroll through data
        client = qdrant_service._get_client()
        
        # Scroll through collection to get points with PCA data
        scroll_result = client.scroll(
            collection_name=main_collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        
        records = scroll_result[0]
        
        for i, r in enumerate(records):
            payload = r.payload
            
            # Get PCA coordinates based on view type
            pca_key = f"pca_{view}" if view in ("drug", "target", "combined") else "pca_combined"
            pca_coords = payload.get(pca_key) or payload.get("pca_combined") or [0, 0]
            
            # Use first 2 dimensions of PCA for x, y
            x = float(pca_coords[0]) if len(pca_coords) > 0 else 0
            y = float(pca_coords[1]) if len(pca_coords) > 1 else 0
            
            # Determine cluster based on affinity class
            affinity = payload.get("affinity_class", "unknown")
            cluster = 0 if affinity == "high" else (1 if affinity == "medium" else 2)
            
            # Get label from SMILES
            smiles = payload.get("smiles", "")
            label = smiles[:30] + "..." if len(smiles) > 30 else smiles
            
            points.append({
                "id": str(r.id),
                "x": x * 10,  # Scale for visualization
                "y": y * 10,
                "cluster": cluster,
                "label": label,
                "modality": "molecule",
                "affinity_class": affinity,
                "score": payload.get("label_true", 0),
            })
        
        logger.info(f"Loaded {len(points)} points from {main_collection}")
        
    except Exception as e:
        logger.error(f"Failed to get points from Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return _get_mock_points(limit)
    except Exception as e:
        logger.warning(f"Failed to get points from Qdrant: {e}")
        return _get_mock_points(limit)
    
    return {
        "points": points,
        "total": len(points),
        "view": view,
    }


def _get_mock_points(limit: int):
    """Generate mock points for visualization when Qdrant unavailable."""
    import numpy as np
    points = []
    n_molecules = min(limit // 2, 50)
    n_proteins = min(limit // 2, 50)
    
    # Mock molecules
    for i in range(n_molecules):
        np.random.seed(i)
        points.append({
            "id": f"mol-{i}",
            "x": float(2 + np.random.randn() * 0.8),
            "y": float(3 + np.random.randn() * 0.8),
            "cluster": 0,
            "label": f"Molecule-{i}",
            "modality": "molecule",
        })
    
    # Mock proteins
    for i in range(n_proteins):
        np.random.seed(i + 1000)
        points.append({
            "id": f"prot-{i}",
            "x": float(-2 + np.random.randn() * 0.8),
            "y": float(-1 + np.random.randn() * 0.8),
            "cluster": 1,
            "label": f"Protein-{i}",
            "modality": "protein",
        })
    
    return {
        "points": points,
        "total": len(points),
        "view": "mock",
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
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        from bioflow.agents import DiscoveryWorkflow
        
        workflow = DiscoveryWorkflow(
            num_candidates=request.num_candidates,
            top_k=request.top_k,
        )
        
        result = workflow.run(request.query)
        top_candidates = workflow.get_top_candidates(result)
        
        payload = {
            "success": result.status.value == "completed",
            "status": result.status.value,
            "steps_completed": result.steps_completed,
            "total_steps": result.total_steps,
            "execution_time_ms": result.execution_time_ms,
            "top_candidates": top_candidates,
            "candidates": top_candidates,
            "all_outputs": result.outputs,
            "errors": result.errors,
        }
        _log_event(
            "workflow",
            request_id,
            status=payload.get("status"),
            steps_completed=payload.get("steps_completed"),
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        return payload
    except Exception as e:
        _log_event(
            "workflow_error",
            request_id,
            error=str(e),
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        logger.error(f"Workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run with: uvicorn bioflow.api.server:app --reload --port 8000
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
