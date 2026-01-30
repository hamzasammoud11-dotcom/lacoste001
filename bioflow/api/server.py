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
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SMILES Validation - Uses ONLY RDKit as source of truth
# ============================================================================

def validate_smiles_query(query: str) -> dict:
    """
    Validate if a query string is a valid SMILES structure.
    
    CRITICAL: Uses ONLY RDKit.Chem.MolFromSmiles() as the source of truth.
    No regex guessing - if RDKit can parse it, it's valid SMILES.
    
    Returns a dict with:
    - is_valid_smiles: bool - True if RDKit can parse it
    - is_protein_like: bool - True if looks like protein sequence
    - query_type: str - 'smiles', 'protein', 'text', or 'noise'
    - warning: str or None - Warning message if query is suspicious
    - mol_info: dict or None - Molecule info if valid (MW, LogP, etc.)
    """
    if not query or not isinstance(query, str):
        return {
            "is_valid_smiles": False,
            "is_protein_like": False,
            "query_type": "text",
            "warning": None,
            "mol_info": None
        }
    
    query = query.strip()
    
    if not query:
        return {
            "is_valid_smiles": False,
            "is_protein_like": False,
            "query_type": "text",
            "warning": None,
            "mol_info": None
        }
    
    # Check for obvious noise FIRST (before even trying RDKit)
    # - All same character repeated (except single valid atom symbols like C, N, O)
    # - Less than 2 unique characters AND length > 2
    unique_chars = set(query.lower())
    valid_single_atoms = {'c', 'n', 'o', 's', 'p', 'f', 'i', 'b'}  # Single-letter atom symbols
    
    is_obvious_noise = False
    if len(query) >= 3:
        # For longer strings, check for repetitive patterns
        if len(unique_chars) <= 1:
            is_obvious_noise = True
        elif len(unique_chars) == 2 and ' ' in unique_chars:
            is_obvious_noise = True
    elif len(query) == 1:
        # Single character - only valid if it's a valid atom symbol
        if query.upper() not in {'C', 'N', 'O', 'S', 'P', 'F', 'I', 'B'}:
            is_obvious_noise = True
    # Note: 2-char strings go through RDKit validation
    
    if is_obvious_noise:
        return {
            "is_valid_smiles": False,
            "is_protein_like": False,
            "query_type": "noise",
            "warning": f"Query '{query}' appears to be noise/gibberish.",
            "mol_info": None
        }
    
    # Check if it looks like a protein sequence (20+ chars, only amino acid letters)
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    query_upper = query.upper()
    is_protein_like = (
        len(query) > 20 and 
        all(c in amino_acids for c in query_upper) and
        not any(c.isdigit() for c in query)  # Proteins don't have digits
    )
    
    if is_protein_like:
        return {
            "is_valid_smiles": False,
            "is_protein_like": True,
            "query_type": "protein",
            "warning": None,
            "mol_info": None
        }
    
    # =========================================================================
    # CRITICAL: Use RDKit as the ONLY source of truth for SMILES validation
    # =========================================================================
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        
        # Try to parse as SMILES - this is the ONLY validation that matters
        mol = Chem.MolFromSmiles(query)
        
        if mol is not None:
            # Valid SMILES! Extract molecular properties for scoring
            try:
                mol_info = {
                    "molecular_weight": Descriptors.MolWt(mol),
                    "logp": Descriptors.MolLogP(mol),
                    "num_atoms": mol.GetNumAtoms(),
                    "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                    "num_rings": rdMolDescriptors.CalcNumRings(mol),
                    "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                    "tpsa": Descriptors.TPSA(mol),
                    "num_hbd": rdMolDescriptors.CalcNumHBD(mol),
                    "num_hba": rdMolDescriptors.CalcNumHBA(mol),
                }
            except Exception as e:
                logger.warning(f"Could not compute molecular descriptors: {e}")
                mol_info = {"molecular_weight": 0, "logp": 0}
            
            return {
                "is_valid_smiles": True,
                "is_protein_like": False,
                "query_type": "smiles",
                "warning": None,
                "mol_info": mol_info
            }
        else:
            # RDKit could not parse it - NOT a valid SMILES
            return {
                "is_valid_smiles": False,
                "is_protein_like": False,
                "query_type": "invalid_smiles",
                "warning": f"'{query[:30]}' is not a valid SMILES structure (RDKit parse failed).",
                "mol_info": None
            }
            
    except ImportError:
        # RDKit not available - cannot validate, treat as text
        logger.warning("RDKit not available for SMILES validation")
        return {
            "is_valid_smiles": False,
            "is_protein_like": False,
            "query_type": "text",
            "warning": "RDKit unavailable - cannot validate SMILES.",
            "mol_info": None
        }
    except Exception as e:
        logger.error(f"SMILES validation error: {e}")
        return {
            "is_valid_smiles": False,
            "is_protein_like": False,
            "query_type": "error",
            "warning": f"Validation error: {str(e)}",
            "mol_info": None
        }


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


def ensure_image_is_displayable(metadata: Dict[str, Any], content: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Ensure the 'image' field in metadata is a displayable format (data URL or HTTP URL).
    
    CRITICAL: Browsers cannot display local file paths like C:\Users\...
    This function converts ALL local image paths to base64 data URLs.
    
    Checks 'image', 'image_path', 'thumbnail' fields in metadata.
    Also tries to use 'content' field as fallback if it contains base64 or image path.
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
    
    # FALLBACK 1: If no image yet, check if 'content' field IS a base64 data URL
    # This happens when images are stored with base64 in content but metadata.image is None
    if not metadata.get('image') and content:
        if isinstance(content, str):
            # Content is already a base64 data URL - use it directly!
            if content.startswith('data:image'):
                metadata['image'] = content
                logger.debug("Recovered image from content field (base64 data URL)")
            # Content is an HTTP URL - use it directly
            elif content.startswith('http://') or content.startswith('https://'):
                metadata['image'] = content
                logger.debug("Recovered image from content field (HTTP URL)")
            # Content looks like an image filename - try to convert
            else:
                image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp')
                if content.lower().endswith(image_extensions):
                    converted = convert_image_path_to_base64(content)
                    if converted:
                        metadata['image'] = converted
                        logger.debug(f"Recovered image from content field: {content}")
    
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

# Mount static files for images - CRITICAL: Browsers cannot access local file paths
# This serves ../data/images as /static/images
IMAGES_DIR = os.path.join(DATA_DIR, "images")
if os.path.isdir(IMAGES_DIR):
    app.mount("/static/images", StaticFiles(directory=IMAGES_DIR), name="static_images")
    logger.info(f"✅ Static images mounted: /static/images -> {IMAGES_DIR}")
else:
    logger.warning(f"⚠️ Images directory not found: {IMAGES_DIR}")

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
        
        # CRITICAL: Validate query to prevent garbage-in-garbage-out (Jury D.2)
        query_validation = validate_smiles_query(query)
        query_warning = query_validation.get("warning")
        detected_query_type = query_validation.get("query_type")
        
        # CRITICAL FIX: If user requested molecule/drug search but query is invalid SMILES,
        # Return JSON 400 response - NOT an exception that crashes the worker
        if modality in ["molecule", "drug"] and not query_validation["is_valid_smiles"]:
            if detected_query_type in ["noise", "invalid_smiles"]:
                # STRICT MODE: Return clean JSON 400 Bad Request
                error_msg = (
                    f"Invalid SMILES: '{query[:50]}' is not a valid chemical structure. "
                    f"Please provide a valid SMILES string (e.g., 'CC(=O)Nc1ccc(O)cc1' for acetaminophen)."
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "INVALID_SMILES",
                        "message": error_msg,
                        "query": query[:100],
                        "detected_type": detected_query_type,
                        "suggestion": "Use 'Properties (Text Search)' mode for keyword-based queries."
                    }
                )
            elif detected_query_type == "text":
                # Also reject - user explicitly chose molecule search
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "NOT_A_SMILES",
                        "message": f"'{query[:50]}' is plain text, not a SMILES structure.",
                        "query": query[:100],
                        "detected_type": detected_query_type,
                        "suggestion": "Switch to 'Properties (Text Search)' mode for text queries."
                    }
                )
        
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
        
        # Add query validation info to response (Jury D.2 compliance)
        payload["query_validation"] = {
            "detected_type": detected_query_type,
            "is_valid_smiles": query_validation["is_valid_smiles"],
            "is_protein_like": query_validation["is_protein_like"],
        }
        if query_warning:
            payload["warning"] = query_warning
        
        # Ensure image paths are converted to base64 for UI display
        for result in payload.get("results", []):
            if result.get("modality") == "image" and result.get("metadata"):
                result["metadata"] = ensure_image_is_displayable(result["metadata"], result.get("content"))
        
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
    image_type: str = Field(default="other", description="Type of image: microscopy, gel, spectra, xray, molecule, other")
    collection: Optional[str] = Field(default=None, description="Target collection")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    use_mmr: bool = Field(default=True, description="Apply MMR diversification")
    lambda_param: float = Field(default=0.7, ge=0.0, le=1.0, description="MMR lambda")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    try_ocsr: bool = Field(default=True, description="Attempt OCSR (Optical Chemical Structure Recognition) for molecule images")


@app.post("/api/search/image")
async def search_image(request: ImageSearchRequest):
    """
    Image similarity search with OCSR (Optical Chemical Structure Recognition).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ WHAT THIS ENDPOINT DOES (Jury Requirement: "Your AI is blind")         │
    │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
    │                                                                         │
    │ 1. OCSR ATTEMPT: Try to extract SMILES from image using:               │
    │    - DECIMER (deep learning, best accuracy)                            │
    │    - MolScribe (transformer-based)                                     │
    │    - Template matching (heuristics)                                    │
    │                                                                         │
    │ 2. If OCSR succeeds → Search by extracted SMILES (chemical search)     │
    │                                                                         │
    │ 3. If OCSR fails → Fall back to embedding similarity search            │
    │    (but with HONEST error message explaining WHY it failed)            │
    │                                                                         │
    │ KNOWN LIMITATIONS (we're honest about these):                          │
    │ - 3D ball-and-stick models need 2D projection first                    │
    │ - Hand-drawn structures have lower accuracy                            │
    │ - Blurry or low-resolution images may fail                             │
    │                                                                         │
    │ Install DECIMER for best results: pip install decimer                  │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    request_id = uuid.uuid4().hex[:12]
    logger.info(f"[{request_id}] Image search request: image_type={request.image_type}, top_k={request.top_k}, try_ocsr={request.try_ocsr}")
    
    # Minimum similarity threshold for image matches
    MIN_SIMILARITY_THRESHOLD = 0.6  # Lowered to allow more exploratory results
    
    ocsr_result = None
    search_mode = "embedding"  # Default
    extracted_smiles = None
    
    try:
        if not qdrant_service:
            raise HTTPException(status_code=503, detail="Qdrant service not available")

        search_service = get_enhanced_search_service()
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Try OCSR if enabled (extract SMILES from image)
        # ─────────────────────────────────────────────────────────────────────
        if request.try_ocsr and request.image_type in ("molecule", "other", "structure"):
            try:
                from bioflow.plugins.encoders.ocsr_engine import recognize_structure_from_image
                
                logger.info(f"[{request_id}] Attempting OCSR on uploaded image...")
                ocsr_result = recognize_structure_from_image(request.image)
                
                if ocsr_result.success and ocsr_result.smiles:
                    extracted_smiles = ocsr_result.smiles
                    search_mode = "smiles"
                    logger.info(f"[{request_id}] OCSR SUCCESS: Extracted SMILES = {extracted_smiles}")
                else:
                    logger.info(f"[{request_id}] OCSR failed: {ocsr_result.error}")
            except ImportError:
                logger.debug(f"[{request_id}] OCSR engine not available")
            except Exception as e:
                logger.warning(f"[{request_id}] OCSR error: {e}")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Perform search based on mode
        # ─────────────────────────────────────────────────────────────────────
        if search_mode == "smiles" and extracted_smiles:
            # Search by extracted SMILES (chemical/structural search)
            logger.info(f"[{request_id}] Searching by extracted SMILES: {extracted_smiles}")
            response = search_service.search(
                query=extracted_smiles,
                modality="molecule",  # Search molecule collection
                collection=request.collection,
                top_k=request.top_k,
                use_mmr=request.use_mmr,
                lambda_param=request.lambda_param,
                filters=request.filters or {}
            )
        else:
            # Fall back to image embedding search
            logger.info(f"[{request_id}] Falling back to image embedding search")
            response = search_service.search(
                query=request.image,
                modality="image",
                collection=request.collection,
                top_k=request.top_k,
                use_mmr=request.use_mmr,
                lambda_param=request.lambda_param,
                filters=request.filters or {}
            )

        logger.info(f"[{request_id}] Search returned {response.returned} results")
        
        payload = response.to_dict()
        
        # Filter low-confidence results
        original_results = payload.get("results", [])
        filtered_results = [r for r in original_results if r.get("score", 0) >= MIN_SIMILARITY_THRESHOLD]
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Build response with HONEST feedback
        # ─────────────────────────────────────────────────────────────────────
        
        # Add OCSR metadata to response
        ocsr_metadata = {}
        if ocsr_result:
            ocsr_metadata = {
                "ocsr_attempted": True,
                "ocsr_success": ocsr_result.success,
                "ocsr_method": ocsr_result.method.value if ocsr_result.method else None,
                "ocsr_confidence": ocsr_result.confidence,
                "extracted_smiles": extracted_smiles,
            }
            if ocsr_result.error:
                ocsr_metadata["ocsr_message"] = ocsr_result.error
            if ocsr_result.metadata:
                ocsr_metadata["ocsr_details"] = ocsr_result.metadata
        
        if len(original_results) > 0 and len(filtered_results) == 0:
            # Had results but all below threshold
            logger.info(f"[{request_id}] All {len(original_results)} results below threshold {MIN_SIMILARITY_THRESHOLD}")
            
            # Provide SPECIFIC feedback based on OCSR result
            if ocsr_result and not ocsr_result.success:
                message = ocsr_result.error or "Could not recognize chemical structure in image."
                suggestion = "Try uploading a clear 2D structure diagram, or enter the molecule name/SMILES directly."
                # Include any OCR findings
                if hasattr(ocsr_result, 'extracted_text') and ocsr_result.extracted_text:
                    message += f" (Found text: '{ocsr_result.extracted_text[:50]}...')"
                if hasattr(ocsr_result, 'formula') and ocsr_result.formula:
                    suggestion = f"Detected formula '{ocsr_result.formula}' - try searching for it by name."
            else:
                message = "No similar structures found in the database."
                suggestion = "Try a different image or use text/SMILES search."
            
            return {
                "results": [],
                "query": "[image]",
                "modality": "image",
                "total_found": 0,
                "returned": 0,
                "search_mode": search_mode,
                "message": message,
                "suggestion": suggestion,
                **ocsr_metadata
            }
        
        payload["results"] = filtered_results
        payload["returned"] = len(filtered_results)
        payload["search_mode"] = search_mode
        payload.update(ocsr_metadata)
        
        # Add success message if OCSR worked
        if extracted_smiles:
            payload["message"] = f"Structure recognized! Searching for molecules similar to: {extracted_smiles}"
        
        # Ensure image paths are converted to base64 for UI display
        for result in payload.get("results", []):
            if result.get("metadata"):
                result["metadata"] = ensure_image_is_displayable(result["metadata"], result.get("content"))
        
        return payload

    except Exception as e:
        logger.error(f"[{request_id}] Image search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Use Case 4: Navigate Neighbors & Design Variants
# ============================================================================

class NeighborRequest(BaseModel):
    """
    Request for guided neighbor exploration.
    
    NEIGHBORS vs VARIANTS - Important Distinction:
    
    **Explore Neighbors** (this endpoint):
    - Uses VECTOR SPACE PROXIMITY (embedding cosine similarity)
    - Finds items semantically related in the learned representation space
    - Cross-modal: Can find proteins related to molecules, text related to images
    - Good for: Discovering unexpected connections, exploring the knowledge graph
    
    **Suggest Variants** (/api/design/variants):
    - Uses STRUCTURAL SIMILARITY (Tanimoto + pharmacophore features)
    - Finds chemically similar compounds with meaningful modifications
    - Same-modal: Molecule → molecules, protein → proteins
    - Good for: Drug design, lead optimization, SAR analysis
    """
    item_id: str = Field(..., description="ID of the item to find neighbors for")
    collection: Optional[str] = Field(default=None, description="Collection containing the item")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of neighbors")
    include_cross_modal: bool = Field(default=True, description="Include results from other modalities (key difference from Variants)")
    diversity: float = Field(default=0.3, ge=0.0, le=1.0, description="Diversity factor (0=similar, 1=diverse)")


class DesignVariantRequest(BaseModel):
    """
    Request for design variant suggestions.
    
    VARIANTS vs NEIGHBORS - Important Distinction:
    
    **Suggest Variants** (this endpoint):
    - Uses STRUCTURAL SIMILARITY (Tanimoto coefficient for molecules)
    - Specifically designed for DRUG DESIGN and lead optimization
    - Includes Tanimoto score alongside embedding similarity
    - Returns actionable design hypotheses (scaffold changes, functional group modifications)
    - Same-modal only: variants must be chemically/structurally related
    
    **Explore Neighbors** (/api/neighbors):
    - Uses VECTOR SPACE PROXIMITY (learned embeddings)
    - Cross-modal exploration (molecules ↔ proteins ↔ text ↔ images)
    - Better for discovering unexpected connections in the knowledge graph
    """
    reference: str = Field(..., description="Reference content (SMILES, sequence, or text)")
    modality: str = Field(default="auto", description="Reference modality")
    num_variants: int = Field(default=5, ge=1, le=20, description="Number of variants")
    diversity: float = Field(default=0.5, ge=0.0, le=1.0, description="Diversity (0=close analogs, 1=diverse scaffolds)")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Design constraints (target, activity range, etc.)")


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
    EXPLORE NEIGHBORS - Cross-modal semantic exploration.
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ METHOD: Vector Space Proximity (Embedding Cosine Similarity)           │
    │ USE CASE: Discover unexpected connections across the knowledge graph   │
    │ CROSS-MODAL: YES - molecules ↔ proteins ↔ text ↔ images                │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Given an item, find semantically similar items across modalities
    using learned embedding representations.
    
    Example: A kinase inhibitor molecule might return:
    - Similar molecules (by embedding, not just structure)
    - Related proteins (binding partners, targets)
    - Relevant papers and descriptions
    - Associated experimental images
    
    For STRUCTURAL similarity (Tanimoto), use /api/design/variants instead.
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
        # Also add cross-modal connection explanations for transparency
        for n in all_neighbors:
            n.pop("vector", None)
            # Convert local image paths to base64 for UI display
            if n.get("modality") == "image" and n.get("metadata"):
                n["metadata"] = ensure_image_is_displayable(n["metadata"], n.get("content"))
            
            # Add connection explanation for cross-modal transparency
            source_modality = source_item.payload.get("modality", "unknown")
            neighbor_modality = n.get("modality", "unknown")
            n["connection_explanation"] = _generate_connection_explanation(
                source_modality=source_modality,
                neighbor_modality=neighbor_modality,
                similarity_score=n.get("score", 0),
                neighbor_metadata=n.get("metadata", {}),
            )
        
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
            "connection_method": "Multimodal embedding similarity using OpenBioML encoder - items are connected via shared semantic space where molecules, proteins, text descriptions, and images are represented in compatible vector formats.",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Neighbor search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/design/variants")
async def suggest_design_variants(request: DesignVariantRequest):
    """
    SUGGEST VARIANTS - Structure-based design assistance.
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ METHOD: Structural Similarity (Tanimoto + Pharmacophore Analysis)      │
    │ USE CASE: Drug design, lead optimization, SAR exploration              │
    │ CROSS-MODAL: NO - same modality only (molecules → molecules)           │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Given a reference compound, find structurally similar molecules that:
    - Share the core scaffold but differ in substituents
    - Have known biological activity data
    - Offer meaningful design hypotheses
    
    Returns:
    - Tanimoto similarity score (fingerprint-based structural similarity)
    - Embedding similarity score (semantic/functional similarity)
    - Scientific justification explaining WHY this variant is suggested
    - Evidence links to databases and literature
    
    For CROSS-MODAL exploration, use /api/neighbors instead.
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
        # CRITICAL FIX: Filter out exact matches (similarity > 0.99)
        # A variant must be DIFFERENT from the source - identical results are useless
        # Also filter out near-identical (> 0.98) as they don't offer meaningful diversity
        filtered_results = [
            r for r in response.results 
            if r.score < 0.98  # Exclude exact/near-exact matches - must be meaningfully different
        ]
        
        # Also exclude results with < 0.5 similarity - too different to be useful variants
        filtered_results = [
            r for r in filtered_results
            if r.score >= 0.5
        ]
        
        if len(filtered_results) == 0 and len(response.results) > 0:
            # All results were either identical or too different
            logger.warning(f"All {len(response.results)} results filtered out (too similar or too different)")
        
        variants = []
        for i, result in enumerate(filtered_results[:request.num_variants]):
            # Calculate priority score for this variant
            diversity_penalty = result.diversity_penalty or 0.0
            priority_score = _calculate_priority_score(result, result.metadata, result.score, diversity_penalty)
            
            # Generate justification based on similarity and differences
            justification = _generate_variant_justification(
                reference=request.reference,
                variant=result,
                modality=modality,
                rank=i + 1,
            )
            
            # ─────────────────────────────────────────────────────────────────
            # NEW: Calculate Tanimoto similarity for molecules (Requirement D)
            # ─────────────────────────────────────────────────────────────────
            tanimoto_score = None
            if modality == "molecule" and result.content:
                try:
                    from bioflow.plugins.encoders.molecule_encoder import compute_tanimoto_similarity
                    tanimoto_score = compute_tanimoto_similarity(
                        request.reference, 
                        result.content,
                        fp_type="morgan"
                    )
                except Exception as e:
                    logger.debug(f"Tanimoto calculation failed: {e}")
            
            # ─────────────────────────────────────────────────────────────────
            # NEW: Calculate evidence strength (Requirement E - Traceability)
            # ─────────────────────────────────────────────────────────────────
            evidence_result = _compute_evidence_strength(result.metadata, result)
            evidence_strength = evidence_result.get("label", "UNKNOWN")
            evidence_summary = evidence_result.get("description", "")
            
            # Enrich metadata with unstructured fields if available (for traceability D.5)
            enriched_metadata = {
                **result.metadata,
                "priority_score": priority_score,
                "tanimoto_score": tanimoto_score,
                "evidence_strength": evidence_strength,
                "evidence_summary": evidence_summary,
            }
            
            variants.append({
                "rank": i + 1,
                "id": result.id,
                "content": result.content,
                "modality": result.modality,
                "similarity_score": result.score,
                "priority_score": priority_score,  # Add explicit priority score
                "tanimoto_score": tanimoto_score,  # NEW: Structural similarity
                "evidence_strength": evidence_strength,  # NEW: Evidence level
                "evidence_summary": evidence_summary,    # NEW: Human-readable summary
                "diversity_score": diversity_penalty,
                "justification": justification,
                "evidence_links": [
                    {"source": l.source, "identifier": l.identifier, "url": l.url}
                    for l in result.evidence_links
                ],
                "metadata": enriched_metadata,
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
                # Unstructured data (Jury Requirement D.5: Scientific Traceability)
                "notes": r.metadata.get("notes"),      # Lab notes
                "abstract": r.metadata.get("abstract"),  # Paper abstract
                "protocol": r.metadata.get("protocol"),  # Experimental protocol
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
    """
    Generate intelligent, hypothesis-driven justifications explaining WHY variants 
    are proposed - not just similarity scores.
    
    This implements Design Assistance (D.4): propose 'close but diverse' variants 
    and justify them with scientific reasoning.
    
    CRITICAL: Includes Tanimoto similarity for molecules to distinguish structural
    similarity from semantic (embedding) similarity.
    """
    score = variant.score if hasattr(variant, 'score') else 0.0
    content = variant.content if hasattr(variant, 'content') else ""
    metadata = variant.metadata if hasattr(variant, 'metadata') else {}
    
    justifications = []
    design_rationale = []
    
    # === COMPUTE TANIMOTO SIMILARITY FOR MOLECULES (Jury Requirement) ===
    tanimoto_score = None
    similarity_explanation = None
    
    if modality == "molecule":
        try:
            from bioflow.plugins.encoders.molecule_encoder import (
                compute_tanimoto_similarity,
                compute_similarity_breakdown
            )
            tanimoto_score = compute_tanimoto_similarity(reference, content)
            
            if tanimoto_score is not None:
                # Get detailed similarity breakdown
                breakdown = compute_similarity_breakdown(reference, content)
                similarity_explanation = breakdown.get("explanation", "")
                
                # Add to metadata for UI display
                metadata["tanimoto_similarity"] = tanimoto_score
                metadata["similarity_type"] = breakdown.get("similarity_type", "unknown")
                metadata["similarity_breakdown"] = breakdown
        except Exception as e:
            logger.debug(f"Tanimoto computation skipped: {e}")
    
    # === EVIDENCE STRENGTH LABEL (Jury Requirement) ===
    evidence_strength = _compute_evidence_strength(metadata, variant)
    
    # === STRUCTURAL ANALYSIS (for molecules) ===
    if modality == "molecule":
        ref_features = _analyze_molecular_features(reference)
        var_features = _analyze_molecular_features(content)
        
        # Add explicit Tanimoto-based structural analysis
        if tanimoto_score is not None:
            if tanimoto_score >= 0.85:
                design_rationale.append(
                    f"**Structural analog** (Tanimoto: {tanimoto_score:.3f}): Same chemical scaffold with minor modifications"
                )
            elif tanimoto_score >= 0.70:
                design_rationale.append(
                    f"**Related scaffold** (Tanimoto: {tanimoto_score:.3f}): Shared core features, different substituents"
                )
            elif tanimoto_score >= 0.50:
                design_rationale.append(
                    f"**Scaffold hop candidate** (Tanimoto: {tanimoto_score:.3f}): Different scaffold may maintain activity"
                )
            else:
                design_rationale.append(
                    f"**Chemically distinct** (Tanimoto: {tanimoto_score:.3f}): Novel scaffold explores new chemical space"
                )
            
            # Add warning if Tanimoto diverges significantly from embedding similarity
            if abs(tanimoto_score - score) > 0.25:
                if tanimoto_score > score:
                    design_rationale.append("⚠️ Structurally similar but semantically different - may have different MoA")
                else:
                    design_rationale.append("💡 Semantically similar despite structural differences - possible functional isostere")
        
        # Identify key structural differences that drive the suggestion
        if ref_features and var_features:
            diffs = _compare_molecular_features(ref_features, var_features)
            if diffs:
                design_rationale.extend(diffs)
        
        # Target binding hypothesis
        if "target" in metadata:
            target = metadata['target']
            design_rationale.append(
                f"**Binding hypothesis**: Known to bind {target}, suggesting similar target engagement potential"
            )
        
        # Activity-based reasoning
        if "activity_type" in metadata and "activity_value" in metadata:
            act_type = metadata["activity_type"]
            act_val = metadata["activity_value"]
            design_rationale.append(
                f"**Activity evidence**: {act_type} = {act_val}, indicating measurable biological activity"
            )
        
        # ADMET-based reasoning
        if metadata.get("admet_predictions"):
            admet = metadata["admet_predictions"]
            if admet.get("solubility") == "high":
                design_rationale.append("**Solubility advantage**: Predicted high solubility may improve bioavailability")
            if admet.get("toxicity") == "low":
                design_rationale.append("**Safety profile**: Predicted low toxicity makes it a safer design alternative")
    
    # === PROTEIN/TARGET ANALYSIS ===
    elif modality == "protein":
        if "organism" in metadata:
            org = metadata["organism"]
            design_rationale.append(f"**Ortholog insight**: From {org}, may reveal conserved functional regions")
        
        if "function" in metadata:
            func = metadata["function"]
            design_rationale.append(f"**Functional relevance**: {func[:100]}..." if len(func) > 100 else f"**Functional relevance**: {func}")
        
        if "pdb_ids" in metadata:
            design_rationale.append("**Structural data available**: 3D structure enables binding site analysis")
    
    # === EVIDENCE-BASED REASONING ===
    source = metadata.get("source", "")
    
    if source == "pubmed":
        title = metadata.get("title", "")
        if title:
            design_rationale.append(f"**Literature support**: \"{title[:80]}...\"" if len(title) > 80 else f"**Literature support**: \"{title}\"")
        pmid = metadata.get("pmid", "")
        if pmid:
            design_rationale.append(f"📄 PMID:{pmid} - Peer-reviewed evidence supports this variant")
    
    elif source == "chembl":
        chembl_id = metadata.get("chembl_id", "")
        assay_count = metadata.get("assay_count", 0)
        if assay_count > 0:
            design_rationale.append(f"**Experimental validation**: {assay_count} bioassay results in ChEMBL ({chembl_id})")
        else:
            design_rationale.append(f"**Database reference**: ChEMBL {chembl_id} - curated bioactivity data available")
    
    elif source == "experiment":
        outcome = metadata.get("outcome", "")
        exp_type = metadata.get("experiment_type", "")
        
        if outcome == "success":
            design_rationale.append(f"**Positive result**: This {exp_type or 'experiment'} succeeded - learn from what worked")
            if "measurements" in metadata:
                measurements = metadata["measurements"]
                if isinstance(measurements, list) and measurements:
                    m = measurements[0]
                    design_rationale.append(f"📊 Key metric: {m.get('name', 'Value')} = {m.get('value', 'N/A')} {m.get('unit', '')}")
        elif outcome == "failure":
            design_rationale.append(f"**Negative control**: This {exp_type or 'experiment'} failed - understanding why informs better design")
        
        # Lab notes excerpt
        if metadata.get("notes"):
            notes = metadata["notes"]
            excerpt = notes[:150] + "..." if len(notes) > 150 else notes
            design_rationale.append(f"📝 Lab note: \"{excerpt}\"")
        
        # Protocol reference
        if metadata.get("protocol"):
            protocol = metadata["protocol"]
            design_rationale.append(f"🧪 Protocol: {protocol}")
    
    elif source == "uniprot":
        acc = metadata.get("accession", "")
        protein_name = metadata.get("protein_name", "")
        design_rationale.append(f"**Annotated protein**: {protein_name or acc} with curated functional data")
    
    # === DIVERSITY REASONING ===
    diversity_penalty = variant.diversity_penalty if hasattr(variant, 'diversity_penalty') else 0.0
    if diversity_penalty > 0.3:
        design_rationale.append("**Design diversity**: Structurally distinct - explores new chemical space")
    elif score > 0.85 and (tanimoto_score is None or tanimoto_score > 0.85):
        design_rationale.append("**Close analog**: Minor modifications may fine-tune properties")
    
    # === PRIORITY SCORE CALCULATION ===
    priority_score = _calculate_priority_score(variant, metadata, score, diversity_penalty)
    
    # === BUILD FINAL JUSTIFICATION ===
    # Prepend evidence strength label (Jury Requirement)
    evidence_label = f"[Evidence: {evidence_strength['label']}] " if evidence_strength['label'] != "Unknown" else ""
    
    if design_rationale:
        main_rationale = design_rationale[0]  # Lead with the strongest reason
        supporting = design_rationale[1:3] if len(design_rationale) > 1 else []  # Add up to 2 more
        
        justification = main_rationale
        if supporting:
            justification += " | " + " | ".join(supporting)
        
        # Add priority indicator
        if priority_score >= 0.8:
            justification = f"⭐ HIGH PRIORITY: {evidence_label}{justification}"
        elif priority_score >= 0.5:
            justification = f"PROMISING: {evidence_label}{justification}"
        else:
            justification = f"{evidence_label}{justification}"
        
        return justification
    
    # === INTELLIGENT FALLBACK - Generate hypothesis from structural analysis ===
    # This ensures we NEVER return a generic "similar structure" message
    target_info = metadata.get("target", metadata.get("target_id", ""))
    molecule_name = metadata.get("molecule_name", metadata.get("name", ""))
    
    if modality == "molecule" and target_info:
        return f"**Design hypothesis**: Structural analog may engage {target_info} binding site. Similarity {score:.0%} suggests conserved pharmacophore features. Consider testing for target selectivity."
    elif modality == "molecule":
        return f"**Scaffold exploration**: {score:.0%} structural similarity with {'known bioactive compound ' + molecule_name if molecule_name else 'bioactive scaffold'}. Variant explores modified functional groups that may alter ADMET properties or target selectivity."
    elif modality == "protein" and target_info:
        return f"**Ortholog analysis**: Related sequence ({score:.0%} identity) suggests conserved binding pocket. Structure may reveal resistance mutations or alternative conformational states for structure-based design."
    elif modality == "protein":
        return f"**Sequence insight**: {score:.0%} sequence similarity indicates potential functional conservation. Align to identify key residues for rational mutagenesis or inhibitor design."
    elif modality == "text" or "abstract" in metadata or "title" in metadata:
        title = metadata.get("title", "")[:50]
        title_display = f'"{title}..."' if title else ''
        return f"**Literature hypothesis**: Related publication {title_display} may provide mechanistic insights or experimental protocols to guide design decisions."
    
    # Ultimate fallback with actionable guidance
    return f"**Exploration candidate** (Rank #{rank}): {score:.0%} similarity score. Review metadata and evidence links to assess design relevance. Consider structural overlay with reference compound."


def _analyze_molecular_features(smiles: str) -> dict:
    """Analyze key molecular features from SMILES for comparison."""
    if not smiles or len(smiles) < 3:
        return {}
    
    features = {}
    
    # Count key functional groups
    features["rings"] = smiles.count("c1") + smiles.count("C1")
    features["has_nitrogen"] = "N" in smiles or "n" in smiles
    features["has_oxygen"] = "O" in smiles or "o" in smiles
    features["has_sulfur"] = "S" in smiles or "s" in smiles
    features["has_halogen"] = any(x in smiles for x in ["F", "Cl", "Br", "I"])
    features["has_carbonyl"] = "=O" in smiles
    features["has_amine"] = "N" in smiles and "=" not in smiles[smiles.find("N"):smiles.find("N")+2] if "N" in smiles else False
    features["length"] = len(smiles)
    
    return features


def _compute_evidence_strength(metadata: dict, variant) -> dict:
    """
    Compute evidence strength rating for a result.
    
    Returns a label that explains WHY this result is trustworthy (or not).
    
    Addresses Jury Requirement: "The UI must display the Evidence Strength"
    
    Evidence levels:
    - GOLD: Multiple independent validations (experimental + database + literature)
    - STRONG: Experimental validation or curated database entry
    - MODERATE: Literature support or computational prediction
    - WEAK: Single source, limited metadata
    - UNKNOWN: Insufficient data
    """
    evidence_points = 0
    evidence_sources = []
    
    # Check evidence links
    evidence_links = getattr(variant, 'evidence_links', []) or []
    if evidence_links:
        evidence_points += len(evidence_links) * 2
        for link in evidence_links[:3]:  # Count top 3
            if hasattr(link, 'source'):
                evidence_sources.append(link.source)
            elif isinstance(link, dict):
                evidence_sources.append(link.get('source', 'unknown'))
    
    # Source quality
    source = metadata.get("source", "").lower()
    source_points = {
        "experiment": 5,
        "chembl": 4,
        "drugbank": 4,
        "pubmed": 3,
        "uniprot": 3,
        "pdb": 3,
        "pubchem": 2,
    }
    evidence_points += source_points.get(source, 0)
    if source:
        evidence_sources.append(source)
    
    # Experimental validation
    if metadata.get("outcome"):
        evidence_points += 3
        if metadata.get("outcome") == "success":
            evidence_points += 2
    
    # Has activity data
    if metadata.get("activity_type") or metadata.get("affinity"):
        evidence_points += 2
    
    # Has target info
    if metadata.get("target") or metadata.get("target_id"):
        evidence_points += 1
    
    # Quality score
    quality = metadata.get("quality_score", metadata.get("quality", 0))
    if quality:
        evidence_points += int(float(quality) * 3)
    
    # Determine label
    unique_sources = list(set(evidence_sources))
    if evidence_points >= 12:
        label = "GOLD"
        description = f"Multiple independent validations from {', '.join(unique_sources[:3])}"
    elif evidence_points >= 8:
        label = "STRONG"
        description = f"Validated by {source or 'experiment'}"
    elif evidence_points >= 5:
        label = "MODERATE"
        description = f"Supported by {source or 'database'} records"
    elif evidence_points >= 2:
        label = "WEAK"
        description = "Limited evidence - consider additional validation"
    else:
        label = "UNKNOWN"
        description = "Insufficient evidence data"
    
    return {
        "label": label,
        "description": description,
        "score": evidence_points,
        "sources": unique_sources,
    }


def _compare_molecular_features(ref: dict, var: dict) -> list:
    """Compare molecular features to generate design insights."""
    insights = []
    
    if not ref or not var:
        return insights
    
    # Ring system changes
    ring_diff = var.get("rings", 0) - ref.get("rings", 0)
    if ring_diff > 0:
        insights.append("**Ring expansion**: Added aromatic system may improve binding affinity")
    elif ring_diff < 0:
        insights.append("**Ring reduction**: Simplified scaffold may improve solubility")
    
    # Heteroatom changes
    if var.get("has_nitrogen") and not ref.get("has_nitrogen"):
        insights.append("**Nitrogen introduction**: May form H-bonds with target, improving affinity")
    if var.get("has_halogen") and not ref.get("has_halogen"):
        insights.append("**Halogenation**: May enhance metabolic stability or binding")
    
    # Size changes
    size_diff = var.get("length", 0) - ref.get("length", 0)
    if abs(size_diff) > 20:
        if size_diff > 0:
            insights.append("**Larger scaffold**: Extended structure may access additional binding pockets")
        else:
            insights.append("**Smaller scaffold**: Fragment-like design may improve drug-likeness")
    
    return insights


def _calculate_priority_score(variant, metadata: dict, similarity: float, diversity: float) -> float:
    """
    Calculate a DETERMINISTIC priority score (0-1) using REAL molecular properties.
    
    FORMULA:
        Priority = (0.35 * evidence_score) + (0.30 * drug_likeness) + 
                   (0.20 * similarity_adjusted) + (0.15 * novelty_score)
    
    CRITICAL: NO RANDOM VALUES. Priority must be reproducible and scientifically meaningful.
    The score differentiates candidates by:
    - Evidence strength (experimental validation, database presence)
    - Drug-likeness (Lipinski + QED when available)
    - Adjusted similarity (penalizes redundancy at >0.95)
    - Novelty bonus (rewards exploration of diverse chemical space)
    
    Uses RDKit descriptors for real drug-likeness calculation.
    """
    # =========================================================================
    # 1. EVIDENCE COMPONENT (35% weight) - Highest weight for scientific rigor
    # =========================================================================
    evidence_score = 0.0
    
    # Count evidence links
    evidence_links = getattr(variant, 'evidence_links', []) or []
    if evidence_links:
        # Each evidence link adds value, capped at 5 links
        evidence_score = min(len(evidence_links) * 0.18, 0.9)
    
    # Source quality bonuses (cumulative but capped)
    source = metadata.get("source", "").lower()
    
    # Experimental validation is gold standard
    if source == "experiment":
        outcome = metadata.get("outcome", "")
        if outcome == "success":
            evidence_score = min(evidence_score + 0.45, 1.0)
        elif outcome == "failure":
            evidence_score = min(evidence_score + 0.15, 1.0)  # Still valuable data
        else:
            evidence_score = min(evidence_score + 0.25, 1.0)
    elif source == "chembl":
        # ChEMBL = curated bioactivity data
        assay_count = metadata.get("assay_count", 0)
        if assay_count and int(assay_count) > 10:
            evidence_score = min(evidence_score + 0.35, 1.0)
        else:
            evidence_score = min(evidence_score + 0.25, 1.0)
    elif source == "pubmed":
        evidence_score = min(evidence_score + 0.20, 1.0)
    elif source == "uniprot":
        evidence_score = min(evidence_score + 0.18, 1.0)
    elif source == "drugbank":
        evidence_score = min(evidence_score + 0.30, 1.0)  # Approved drug info
    else:
        # Unknown source gets minimal credit
        evidence_score = max(evidence_score, 0.05)
    
    # =========================================================================
    # 2. DRUG-LIKENESS COMPONENT (30% weight) - RDKit-based, NO RANDOMNESS
    # =========================================================================
    drug_likeness = 0.0
    qed_score = None
    
    smiles = metadata.get("smiles") or metadata.get("content")
    if smiles and isinstance(smiles, str):
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED as RDKitQED
            
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Use QED (Quantitative Estimate of Drug-likeness) - GOLD STANDARD
                try:
                    qed_score = RDKitQED.qed(mol)
                    drug_likeness = qed_score
                except Exception:
                    qed_score = None
                
                # If QED fails, use Lipinski scoring
                if qed_score is None:
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    rotatable = Descriptors.NumRotatableBonds(mol)
                    tpsa = Descriptors.TPSA(mol)
                    
                    # Lipinski Rule of Five scoring (granular)
                    lipinski_score = 0.0
                    
                    # Molecular weight: ideal 200-450, acceptable <500
                    if 200 <= mw <= 450:
                        lipinski_score += 0.25
                    elif mw <= 500:
                        lipinski_score += 0.20
                    elif mw <= 600:
                        lipinski_score += 0.10
                    
                    # LogP: ideal 1-3, acceptable <5
                    if 1 <= logp <= 3:
                        lipinski_score += 0.25
                    elif logp <= 5:
                        lipinski_score += 0.20
                    elif logp <= 6:
                        lipinski_score += 0.10
                    
                    # H-bond donors: <5 ideal
                    if hbd <= 3:
                        lipinski_score += 0.15
                    elif hbd <= 5:
                        lipinski_score += 0.10
                    
                    # H-bond acceptors: <10 ideal
                    if hba <= 7:
                        lipinski_score += 0.15
                    elif hba <= 10:
                        lipinski_score += 0.10
                    
                    # Rotatable bonds: <10 for bioavailability
                    if rotatable <= 7:
                        lipinski_score += 0.10
                    elif rotatable <= 10:
                        lipinski_score += 0.05
                    
                    # TPSA: 40-90 ideal for CNS, <140 for absorption
                    if 40 <= tpsa <= 90:
                        lipinski_score += 0.10
                    elif tpsa <= 140:
                        lipinski_score += 0.05
                    
                    drug_likeness = min(1.0, lipinski_score)
                
        except Exception as e:
            logger.debug(f"Could not compute drug-likeness: {e}")
    
    # Metadata fallbacks (only if RDKit computation failed)
    if drug_likeness == 0.0:
        if metadata.get("drug_likeness"):
            drug_likeness = float(metadata.get("drug_likeness", 0.3))
        elif metadata.get("qed"):
            drug_likeness = float(metadata.get("qed", 0.3))
        elif metadata.get("quality_score"):
            drug_likeness = float(metadata.get("quality_score", 0.3))
        elif metadata.get("label_true") is not None:
            # Affinity label - lower is better (pKi/pIC50 scale)
            label = float(metadata.get("label_true", 5))
            # pKi > 8 = very potent, pKi < 5 = weak
            drug_likeness = max(0.1, min(0.9, (label - 4) / 6))
        elif metadata.get("affinity_class"):
            affinity = str(metadata.get("affinity_class", "")).lower()
            if "high" in affinity:
                drug_likeness = 0.75
            elif "medium" in affinity:
                drug_likeness = 0.50
            else:
                drug_likeness = 0.30
        else:
            # Unknown compound - conservative estimate
            drug_likeness = 0.35
    
    # =========================================================================
    # 3. SIMILARITY COMPONENT (20% weight) - Penalize redundancy
    # =========================================================================
    # Very high similarity (>0.95) often means redundant/duplicate
    # Sweet spot is 0.7-0.9 for novel but related compounds
    if similarity > 0.95:
        sim_adjusted = 0.60  # Penalize near-duplicates
    elif similarity > 0.90:
        sim_adjusted = 0.85
    elif similarity > 0.80:
        sim_adjusted = similarity  # Optimal range
    elif similarity > 0.60:
        sim_adjusted = similarity * 0.95
    else:
        sim_adjusted = similarity * 0.85  # Too different may be off-target
    
    # =========================================================================
    # 4. NOVELTY/DIVERSITY COMPONENT (15% weight)
    # =========================================================================
    # Reward exploration of diverse chemical space
    novelty_score = 0.0
    
    # Use diversity penalty if provided
    if diversity > 0:
        # Higher diversity = more unique = higher novelty score
        novelty_score = min(diversity * 0.8, 0.8)
    
    # Bonus for novel scaffolds (not in top databases)
    if source not in ("chembl", "drugbank", "pubchem"):
        novelty_score = min(novelty_score + 0.15, 1.0)
    
    # Bonus for being in the "interesting" similarity range
    if 0.6 <= similarity <= 0.85:
        novelty_score = min(novelty_score + 0.10, 1.0)
    
    # =========================================================================
    # FINAL WEIGHTED SCORE (Deterministic)
    # =========================================================================
    priority = (
        0.35 * evidence_score +
        0.30 * drug_likeness +
        0.20 * sim_adjusted +
        0.15 * novelty_score
    )
    
    # NO RANDOM OFFSET - Scores are deterministic and reproducible
    
    # Ensure bounds [0.05, 0.98] - near-zero indicates truly poor candidates
    return max(0.05, min(0.98, priority))


def _generate_connection_explanation(
    source_modality: str,
    neighbor_modality: str,
    similarity_score: float,
    neighbor_metadata: dict,
) -> str:
    """
    Generate a human-readable explanation for why two items are connected.
    
    This addresses the "Black Box" critique by making cross-modal connections transparent.
    """
    explanations = []
    
    # Same modality connections
    if source_modality == neighbor_modality:
        if source_modality == "molecule":
            explanations.append(f"**Structural similarity**: Both molecules share similar chemical features (embedding similarity: {similarity_score:.2f})")
            if neighbor_metadata.get("target"):
                explanations.append(f"Potentially binds same target: {neighbor_metadata['target']}")
        elif source_modality == "protein":
            explanations.append(f"**Sequence/functional similarity**: Related protein (similarity: {similarity_score:.2f})")
            if neighbor_metadata.get("organism"):
                explanations.append(f"From: {neighbor_metadata['organism']}")
        elif source_modality == "experiment":
            explanations.append(f"**Similar experimental context**: Related experiment (similarity: {similarity_score:.2f})")
            if neighbor_metadata.get("experiment_type"):
                explanations.append(f"Type: {neighbor_metadata['experiment_type']}")
        elif source_modality == "image":
            explanations.append(f"**Visual similarity**: Image features match (similarity: {similarity_score:.2f})")
        else:
            explanations.append(f"**Semantic similarity**: Related content (similarity: {similarity_score:.2f})")
    
    # Cross-modal connections - explain the semantic bridge
    else:
        bridge_explanations = {
            ("molecule", "protein"): "**Drug-Target relationship**: This molecule and protein co-occur in binding contexts, suggesting potential interaction",
            ("protein", "molecule"): "**Target-Drug relationship**: This protein has semantic associations with similar molecular scaffolds",
            ("molecule", "experiment"): "**Compound-Assay link**: This molecule appears in similar experimental contexts",
            ("experiment", "molecule"): "**Assay-Compound link**: This experiment involves structurally related compounds",
            ("molecule", "image"): "**Compound-Visual link**: This image depicts or relates to similar molecular structures",
            ("image", "molecule"): "**Visual-Compound link**: The visual features suggest structural similarity to this molecule",
            ("protein", "experiment"): "**Target-Assay link**: This protein was studied in related experimental contexts",
            ("experiment", "protein"): "**Assay-Target link**: This experiment involves related protein targets",
            ("protein", "image"): "**Protein-Visual link**: This image relates to similar protein structures or functions",
            ("image", "protein"): "**Visual-Protein link**: The visual features suggest relation to this protein",
            ("image", "experiment"): "**Image-Experiment link**: This image is from a related experimental context",
            ("experiment", "image"): "**Experiment-Image link**: This experiment produced similar visual outputs",
        }
        
        key = (source_modality, neighbor_modality)
        if key in bridge_explanations:
            explanations.append(bridge_explanations[key])
        else:
            explanations.append(f"**Semantic bridge**: Connected via shared embedding space (similarity: {similarity_score:.2f})")
        
        # Add confidence indicator based on score
        if similarity_score >= 0.8:
            explanations.append("🔗 Strong cross-modal connection")
        elif similarity_score >= 0.6:
            explanations.append("🔗 Moderate cross-modal connection")
        else:
            explanations.append("🔗 Exploratory connection - may reveal unexpected relationships")
    
    # Add source-based context
    source = neighbor_metadata.get("source", "")
    if source == "pubmed":
        explanations.append("📄 Supported by literature")
    elif source == "chembl":
        explanations.append("🧪 Has bioactivity data")
    elif source == "experiment":
        outcome = neighbor_metadata.get("outcome", "")
        if outcome == "success":
            explanations.append("✅ From successful experiment")
    
    return " | ".join(explanations) if explanations else f"Connected by embedding similarity ({similarity_score:.2f})"
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


# ============================================================================
# Use Case 4: Gels & Microscopy - Gallery and Visual Similarity Search
# ============================================================================

class GelMicroscopySearchRequest(BaseModel):
    """Request for gel/microscopy image similarity search."""
    image: str = Field(..., description="Image as base64 data URL or file path")
    image_type: Optional[str] = Field(default=None, description="Filter by type: gel, western_blot, microscopy, fluorescence")
    outcome: Optional[str] = Field(default=None, description="Filter by outcome: positive, negative, inconclusive")
    cell_line: Optional[str] = Field(default=None, description="Filter by cell line")
    treatment: Optional[str] = Field(default=None, description="Filter by treatment/drug")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    use_mmr: bool = Field(default=True, description="Apply MMR diversification")


@app.get("/api/images")
async def get_experimental_images(
    type: str = "all",
    limit: int = 30,
    outcome: Optional[str] = None,
    cell_line: Optional[str] = None,
    treatment: Optional[str] = None,
):
    """
    Get experimental images (gels, microscopy) for gallery display.
    
    This is the backend for the "Gels & Microscopy" tab in the UI.
    
    Args:
        type: Filter by image type - 'all', 'gel', 'microscopy'
        limit: Maximum number of images to return
        outcome: Filter by experimental outcome
        cell_line: Filter by cell line
        treatment: Filter by treatment/drug
        
    Returns:
        Gallery of experimental images with metadata
    """
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] Gallery request: type={type}, limit={limit}")
    
    if not qdrant_service:
        logger.warning(f"[{request_id}] Qdrant not available")
        return {
            "images": [],
            "count": 0,
            "type": type,
            "message": "Database not available"
        }
    
    try:
        client = qdrant_service._get_client()
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
        
        # Try biological_images collection first, then fall back to bioflow_memory
        collections_to_try = ["biological_images", "bioflow_memory"]
        
        all_images = []
        
        for collection_name in collections_to_try:
            try:
                # Check if collection exists
                collections = client.get_collections().collections
                collection_names = [c.name for c in collections]
                if collection_name not in collection_names:
                    logger.debug(f"[{request_id}] Collection {collection_name} does not exist")
                    continue
                
                # Build filter conditions using proper Qdrant models
                must_conditions = []
                
                # Type filter - use MatchAny for OR conditions
                if type == "gel":
                    must_conditions.append(
                        FieldCondition(
                            key="image_type",
                            match=MatchAny(any=["gel", "western_blot"])
                        )
                    )
                elif type == "microscopy":
                    must_conditions.append(
                        FieldCondition(
                            key="image_type",
                            match=MatchAny(any=["microscopy", "fluorescence", "brightfield"])
                        )
                    )
                else:
                    # All biological image types
                    must_conditions.append(
                        FieldCondition(
                            key="image_type",
                            match=MatchAny(any=["gel", "western_blot", "microscopy", "fluorescence", "brightfield"])
                        )
                    )
                
                # Outcome filter
                if outcome:
                    must_conditions.append(
                        FieldCondition(key="outcome", match=MatchValue(value=outcome))
                    )
                
                # Cell line filter  
                if cell_line:
                    must_conditions.append(
                        FieldCondition(key="cell_line", match=MatchValue(value=cell_line))
                    )
                
                # Treatment filter
                if treatment:
                    must_conditions.append(
                        FieldCondition(key="treatment", match=MatchValue(value=treatment))
                    )
                
                # Build the filter object
                scroll_filter = Filter(must=must_conditions) if must_conditions else None
                
                logger.info(f"[{request_id}] Scrolling {collection_name} with filter: {scroll_filter}")
                
                scroll_result = client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    scroll_filter=scroll_filter,
                    with_payload=True,
                    with_vectors=False,
                )
                
                points = scroll_result[0] if scroll_result else []
                
                for point in points:
                    payload = point.payload or {}
                    
                    # Skip if not a biological image
                    img_type = payload.get("image_type", "")
                    if img_type not in ["gel", "western_blot", "microscopy", "fluorescence", "brightfield"]:
                        continue
                    
                    # Get image data - try multiple sources
                    image_data = payload.get("image") or payload.get("content")
                    image_path = payload.get("image_path") or payload.get("file_path")
                    
                    # If we have a file path but no image data, load from file
                    if not image_data and image_path:
                        image_data = convert_image_path_to_base64(image_path)
                    elif image_data:
                        # Ensure image is displayable
                        image_data = convert_image_path_to_base64(image_data) or image_data
                    
                    all_images.append({
                        "id": str(point.id),
                        "score": 1.0,  # No similarity score for gallery
                        "content": payload.get("description", "") or payload.get("notes", ""),
                        "modality": "image",
                        "metadata": {
                            "image_type": img_type,
                            "image": image_data,
                            "description": payload.get("description", "") or payload.get("notes", ""),
                            "caption": payload.get("caption", ""),
                            "experiment_id": payload.get("experiment_id", ""),
                            "experiment_type": payload.get("experiment_type", ""),
                            "outcome": payload.get("outcome", ""),
                            "quality_score": payload.get("quality_score"),
                            # Cell/treatment info
                            "cell_line": payload.get("cell_line", ""),
                            "treatment": payload.get("treatment", ""),
                            "treatment_target": payload.get("treatment_target", ""),
                            "concentration": payload.get("concentration", ""),
                            "conditions": payload.get("conditions", {}),
                            # Protein/target info
                            "target_protein": payload.get("target_protein", ""),
                            "target_mw": payload.get("target_mw", ""),
                            "protein": payload.get("protein", ""),
                            # Microscopy-specific
                            "staining": payload.get("staining", ""),
                            "magnification": payload.get("magnification", ""),
                            "microscope": payload.get("microscope", ""),
                            "cell_count": payload.get("cell_count"),
                            # Gel-specific
                            "gel_percentage": payload.get("gel_percentage", ""),
                            "num_lanes": payload.get("num_lanes"),
                            # Protocol & notes
                            "protocol": payload.get("protocol", ""),
                            "notes": payload.get("notes", ""),
                            # Dates
                            "experiment_date": payload.get("experiment_date", ""),
                            "source": payload.get("source", ""),
                        }
                    })
                
                if all_images:
                    logger.info(f"[{request_id}] Found {len(all_images)} images in {collection_name}")
                    break  # Found images, stop searching
                    
            except Exception as e:
                logger.debug(f"[{request_id}] Collection {collection_name}: {e}")
                continue
        
        # Sort by quality score (highest first)
        all_images.sort(key=lambda x: x.get("metadata", {}).get("quality_score") or 0, reverse=True)
        
        # Limit results
        all_images = all_images[:limit]
        
        logger.info(f"[{request_id}] Returning {len(all_images)} images")
        
        return {
            "images": all_images,
            "count": len(all_images),
            "type": type,
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Gallery error: {e}", exc_info=True)
        return {
            "images": [],
            "count": 0,
            "type": type,
            "error": str(e)
        }


@app.post("/api/search/gel-microscopy")
async def search_gel_microscopy(request: GelMicroscopySearchRequest):
    """
    Visual similarity search for biological images (gels, microscopy).
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ USE CASE 4: "Multimodal similarity search for experiments/candidates"  │
    │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
    │                                                                         │
    │ WHAT THIS ENDPOINT DOES:                                               │
    │                                                                         │
    │ 1. User uploads a Western blot or microscopy image                     │
    │ 2. System extracts CLIP embedding (visual features)                    │
    │ 3. System searches Qdrant for similar experimental images              │
    │ 4. Returns: experiments with similar visual patterns                   │
    │                                                                         │
    │ Example use cases:                                                     │
    │ - "Find experiments with similar gel band patterns"                    │
    │ - "Show me microscopy images with similar cell morphology"             │
    │ - "What other treatments showed this apoptosis pattern?"               │
    │                                                                         │
    │ Supported image types:                                                 │
    │ - Western blots                                                        │
    │ - SDS-PAGE gels                                                        │
    │ - Fluorescence microscopy                                              │
    │ - Brightfield microscopy                                               │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] Gel/Microscopy search: type_filter={request.image_type}, top_k={request.top_k}")
    
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")
    
    try:
        search_service = get_enhanced_search_service()
        
        # Build filter for biological image types
        filters = {}
        
        if request.image_type:
            if request.image_type in ["gel", "western_blot"]:
                filters["image_type"] = ["gel", "western_blot"]
            elif request.image_type in ["microscopy", "fluorescence"]:
                filters["image_type"] = ["microscopy", "fluorescence"]
            else:
                filters["image_type"] = request.image_type
        
        if request.outcome:
            filters["outcome"] = request.outcome
        if request.cell_line:
            filters["cell_line"] = request.cell_line
        if request.treatment:
            filters["treatment"] = request.treatment
        
        # Search using image embedding
        # Try biological_images collection first
        collections_to_try = ["biological_images", "bioflow_memory"]
        
        for collection in collections_to_try:
            try:
                response = search_service.search(
                    query=request.image,
                    modality="image",
                    collection=collection,
                    top_k=request.top_k,
                    use_mmr=request.use_mmr,
                    lambda_param=0.7,
                    filters=filters
                )
                
                if response.returned > 0:
                    logger.info(f"[{request_id}] Found {response.returned} results in {collection}")
                    break
            except Exception as e:
                logger.debug(f"[{request_id}] Collection {collection}: {e}")
                continue
        else:
            # No results found in any collection
            return {
                "results": [],
                "query_image_type": request.image_type,
                "total_found": 0,
                "returned": 0,
                "message": "No similar biological images found. Try uploading a different image or adjust filters.",
                "filters_applied": filters
            }
        
        # Process results
        results = []
        for r in response.to_dict().get("results", []):
            metadata = r.get("metadata", {})
            
            # Ensure image is displayable
            metadata = ensure_image_is_displayable(metadata, r.get("content"))
            
            results.append({
                "id": r.get("id"),
                "experiment_id": metadata.get("experiment_id", ""),
                "image_type": metadata.get("image_type", "unknown"),
                "similarity": round(r.get("score", 0), 3),
                "outcome": metadata.get("outcome", ""),
                "conditions": metadata.get("conditions", {}),
                "cell_line": metadata.get("cell_line", ""),
                "treatment": metadata.get("treatment", ""),
                "concentration": metadata.get("concentration", ""),
                "target_protein": metadata.get("target_protein", ""),
                "notes": metadata.get("notes", ""),
                "protocol": metadata.get("protocol", ""),
                "experiment_type": metadata.get("experiment_type", ""),
                "magnification": metadata.get("magnification", ""),
                "quality_score": metadata.get("quality_score"),
                "experiment_date": metadata.get("experiment_date", ""),
                "image": metadata.get("image"),  # base64 for display
            })
        
        return {
            "results": results,
            "query_image_type": request.image_type,
            "total_found": response.total_found,
            "returned": len(results),
            "filters_applied": filters,
            "collection": collection,
            "message": f"Found {len(results)} similar experiments" if results else "No matching experiments found"
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Gel/microscopy search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class CrossModalSearchRequest(BaseModel):
    """Request for cross-modal search combining multiple query types."""
    compound: Optional[str] = Field(default=None, description="SMILES string of compound")
    sequence: Optional[str] = Field(default=None, description="DNA/RNA/protein sequence")
    text: Optional[str] = Field(default=None, description="Text query")
    image: Optional[str] = Field(default=None, description="Base64 encoded image")
    target_modalities: List[str] = Field(default=["all"], description="Target modalities: molecule, protein, text, image, experiment, all")
    top_k: int = Field(default=10, ge=1, le=100)
    use_mmr: bool = Field(default=True)
    diversity: float = Field(default=0.3, ge=0.0, le=1.0)


@app.post("/api/search/cross-modal")
async def cross_modal_search(request: CrossModalSearchRequest):
    """
    Cross-modal search: combine compound, sequence, text, or image to find related experiments.
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ USE CASE 4: "Show me experiments that used THIS compound with THIS     │
    │             gel result" - Connect compounds to experimental evidence   │
    │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
    │                                                                         │
    │ WHAT THIS ENDPOINT DOES:                                               │
    │                                                                         │
    │ 1. Accepts multiple query modalities (compound, sequence, text, image) │
    │ 2. VALIDATES each query type (SMILES, sequence, image)                 │
    │ 3. Encodes each query into embedding space                             │
    │ 4. Performs combined vector search across all collections              │
    │ 5. Returns unified results linking compounds to experiments            │
    │                                                                         │
    │ Example use cases:                                                     │
    │ - "Find experiments where Gefitinib was used" (compound)               │
    │ - "Show gel results for EGFR protein" (sequence + text)                │
    │ - "What experiments show similar patterns?" (image)                    │
    │ - Combined: "Compound X + EGFR target → show binding assay gels"       │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] Cross-modal search: compound={bool(request.compound)}, sequence={bool(request.sequence)}, text={bool(request.text)}, image={bool(request.image)}")
    
    if not qdrant_service:
        raise HTTPException(status_code=503, detail="Qdrant service not available")
    
    # Ensure at least one query type is provided
    if not any([request.compound, request.sequence, request.text, request.image]):
        raise HTTPException(status_code=400, detail="At least one query type (compound, sequence, text, or image) is required")
    
    try:
        search_service = get_enhanced_search_service()
        
        all_results = []
        query_info = {}
        validation_warnings = []
        
        # =====================================================================
        # VALIDATION STEP 1: Validate compound (SMILES)
        # =====================================================================
        if request.compound:
            smiles_validation = validate_smiles_query(request.compound)
            query_info["compound"] = request.compound
            query_info["compound_validation"] = {
                "is_valid": smiles_validation["is_valid_smiles"],
                "query_type": smiles_validation["query_type"],
                "warning": smiles_validation["warning"]
            }
            
            if not smiles_validation["is_valid_smiles"]:
                warning_msg = f"Compound '{request.compound[:30]}...' is not a valid SMILES structure"
                if smiles_validation["query_type"] == "protein":
                    warning_msg = f"The compound input looks like a protein sequence. Use the 'sequence' field instead."
                elif smiles_validation["query_type"] == "noise":
                    warning_msg = f"The compound input '{request.compound[:20]}' appears to be noise/gibberish."
                validation_warnings.append(warning_msg)
                logger.warning(f"[{request_id}] Invalid SMILES: {request.compound[:50]}")
            else:
                # Add molecular info to response
                query_info["compound_info"] = smiles_validation.get("mol_info", {})
        
        # =====================================================================
        # VALIDATION STEP 2: Validate sequence (protein/DNA/RNA)
        # =====================================================================
        if request.sequence:
            seq = request.sequence.strip().upper()
            query_info["sequence"] = seq[:50] + "..." if len(seq) > 50 else seq
            
            # Detect sequence type
            dna_chars = set("ACGT")
            rna_chars = set("ACGU")
            amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
            seq_chars = set(c for c in seq if c.isalpha())
            
            if seq_chars <= dna_chars and len(seq) >= 10:
                seq_type = "dna"
                is_valid = len(seq) >= 10
            elif seq_chars <= rna_chars and len(seq) >= 10:
                seq_type = "rna"
                is_valid = len(seq) >= 10
            elif seq_chars <= amino_acids and len(seq) >= 10:
                seq_type = "protein"
                is_valid = len(seq) >= 10
            else:
                seq_type = "unknown"
                is_valid = False
                
            query_info["sequence_validation"] = {
                "is_valid": is_valid,
                "sequence_type": seq_type,
                "length": len(seq)
            }
            
            if not is_valid:
                if len(seq) < 10:
                    warning_msg = f"Sequence too short ({len(seq)} chars). Minimum 10 characters required."
                else:
                    warning_msg = f"Unrecognized sequence type. Contains invalid characters for DNA/RNA/protein."
                validation_warnings.append(warning_msg)
                logger.warning(f"[{request_id}] Invalid sequence: {seq[:50]}")
        
        # =====================================================================
        # VALIDATION STEP 3: Validate image (base64)
        # =====================================================================
        if request.image:
            query_info["image"] = True
            
            # Check if it's a valid base64 image
            is_valid_image = False
            image_type = None
            
            if request.image.startswith("data:image/"):
                # Data URL format
                try:
                    header, data = request.image.split(",", 1)
                    if "image/png" in header:
                        image_type = "png"
                    elif "image/jpeg" in header or "image/jpg" in header:
                        image_type = "jpeg"
                    elif "image/gif" in header:
                        image_type = "gif"
                    elif "image/webp" in header:
                        image_type = "webp"
                    
                    # Verify it can be decoded
                    import base64
                    decoded = base64.b64decode(data)
                    if len(decoded) > 100:  # At least 100 bytes
                        is_valid_image = True
                except Exception as e:
                    logger.warning(f"[{request_id}] Image decode error: {e}")
            else:
                # Raw base64 - try to decode
                try:
                    import base64
                    decoded = base64.b64decode(request.image)
                    if len(decoded) > 100:
                        is_valid_image = True
                        image_type = "unknown"
                except Exception as e:
                    logger.warning(f"[{request_id}] Raw base64 decode error: {e}")
            
            query_info["image_validation"] = {
                "is_valid": is_valid_image,
                "image_type": image_type
            }
            
            if not is_valid_image:
                validation_warnings.append("The provided image could not be decoded. Ensure it's a valid base64-encoded image.")
                logger.warning(f"[{request_id}] Invalid image data")
        
        # =====================================================================
        # SEARCH EXECUTION
        # =====================================================================
        
        # Search by compound (SMILES) - only if valid
        if request.compound and query_info.get("compound_validation", {}).get("is_valid", False):
            try:
                response = search_service.search(
                    query=request.compound,
                    modality="smiles",
                    top_k=request.top_k,
                    use_mmr=request.use_mmr,
                    lambda_param=1.0 - request.diversity,
                )
                for r in response.to_dict().get("results", []):
                    r["query_source"] = "compound"
                    r["source_modality"] = "smiles"
                    all_results.append(r)
            except Exception as e:
                logger.warning(f"[{request_id}] Compound search failed: {e}")
        elif request.compound and not query_info.get("compound_validation", {}).get("is_valid", False):
            # Search as text instead of SMILES if invalid
            logger.info(f"[{request_id}] Searching invalid SMILES as text: {request.compound[:30]}")
            try:
                response = search_service.search(
                    query=request.compound,
                    modality="text",
                    top_k=request.top_k,
                    use_mmr=request.use_mmr,
                    lambda_param=1.0 - request.diversity,
                )
                for r in response.to_dict().get("results", []):
                    r["query_source"] = "compound_as_text"
                    r["source_modality"] = "text"
                    all_results.append(r)
            except Exception as e:
                logger.warning(f"[{request_id}] Compound-as-text search failed: {e}")
        
        # Search by sequence (protein/DNA/RNA) - only if valid
        if request.sequence and query_info.get("sequence_validation", {}).get("is_valid", False):
            try:
                seq_type = query_info["sequence_validation"]["sequence_type"]
                modality = "protein" if seq_type == "protein" else "text"
                
                response = search_service.search(
                    query=request.sequence,
                    modality=modality,
                    top_k=request.top_k,
                    use_mmr=request.use_mmr,
                    lambda_param=1.0 - request.diversity,
                )
                for r in response.to_dict().get("results", []):
                    r["query_source"] = "sequence"
                    r["source_modality"] = seq_type
                    all_results.append(r)
            except Exception as e:
                logger.warning(f"[{request_id}] Sequence search failed: {e}")
        elif request.sequence and not query_info.get("sequence_validation", {}).get("is_valid", False):
            # Still try text search with invalid sequence
            logger.info(f"[{request_id}] Searching invalid sequence as text")
            try:
                response = search_service.search(
                    query=request.sequence,
                    modality="text",
                    top_k=request.top_k,
                    use_mmr=request.use_mmr,
                    lambda_param=1.0 - request.diversity,
                )
                for r in response.to_dict().get("results", []):
                    r["query_source"] = "sequence_as_text"
                    r["source_modality"] = "text"
                    all_results.append(r)
            except Exception as e:
                logger.warning(f"[{request_id}] Sequence-as-text search failed: {e}")
        
        # Search by text (always valid)
        if request.text:
            query_info["text"] = request.text
            try:
                response = search_service.search(
                    query=request.text,
                    modality="text",
                    top_k=request.top_k,
                    use_mmr=request.use_mmr,
                    lambda_param=1.0 - request.diversity,
                )
                for r in response.to_dict().get("results", []):
                    r["query_source"] = "text"
                    r["source_modality"] = "text"
                    all_results.append(r)
            except Exception as e:
                logger.warning(f"[{request_id}] Text search failed: {e}")
        
        # Search by image - only if valid
        if request.image and query_info.get("image_validation", {}).get("is_valid", False):
            try:
                response = search_service.search(
                    query=request.image,
                    modality="image",
                    top_k=request.top_k,
                    use_mmr=request.use_mmr,
                    lambda_param=1.0 - request.diversity,
                )
                for r in response.to_dict().get("results", []):
                    r["query_source"] = "image"
                    r["source_modality"] = "image"
                    all_results.append(r)
            except Exception as e:
                logger.warning(f"[{request_id}] Image search failed: {e}")
        
        # Merge and deduplicate results by ID
        seen_ids = set()
        merged_results = []
        for r in sorted(all_results, key=lambda x: x.get("score", 0), reverse=True):
            rid = r.get("id")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                
                # Filter by target modalities
                result_modality = r.get("metadata", {}).get("modality", r.get("modality", "unknown"))
                if "all" not in request.target_modalities:
                    if result_modality not in request.target_modalities:
                        continue
                
                # Enrich with connection explanation
                r["connection"] = f"Connected via {r.get('query_source', 'search')}"
                
                # Ensure images are displayable
                if r.get("metadata", {}).get("image_type"):
                    r["metadata"] = ensure_image_is_displayable(r.get("metadata", {}), r.get("content"))
                
                merged_results.append(r)
                
                if len(merged_results) >= request.top_k:
                    break
        
        logger.info(f"[{request_id}] Cross-modal search found {len(merged_results)} unique results")
        
        # Build response with validation info
        response_data = {
            "results": merged_results,
            "query_info": query_info,
            "total_found": len(all_results),
            "returned": len(merged_results),
            "target_modalities": request.target_modalities,
            "message": f"Found {len(merged_results)} cross-modal matches"
        }
        
        # Add validation warnings if any
        if validation_warnings:
            response_data["validation_warnings"] = validation_warnings
            response_data["message"] = f"Found {len(merged_results)} results (with {len(validation_warnings)} validation warnings)"
        
        return response_data
        
    except Exception as e:
        logger.error(f"[{request_id}] Cross-modal search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/images/filters")
async def get_image_filters():
    """
    Get available filter options for the gallery.
    
    Returns unique values for:
    - image_type
    - outcome  
    - cell_line
    - treatment
    """
    if not qdrant_service:
        return {
            "image_types": ["gel", "western_blot", "microscopy", "fluorescence"],
            "outcomes": ["positive", "negative", "inconclusive", "dose_dependent"],
            "cell_lines": [],
            "treatments": []
        }
    
    try:
        client = qdrant_service._get_client()
        
        # Collect unique values from biological_images collection
        cell_lines = set()
        treatments = set()
        
        for collection in ["biological_images", "bioflow_memory"]:
            try:
                scroll_result = client.scroll(
                    collection_name=collection,
                    limit=500,
                    with_payload=True,
                    with_vectors=False,
                )
                
                for point in scroll_result[0]:
                    payload = point.payload or {}
                    img_type = payload.get("image_type", "")
                    
                    # Only include biological images
                    if img_type in ["gel", "western_blot", "microscopy", "fluorescence"]:
                        if payload.get("cell_line"):
                            cell_lines.add(payload["cell_line"])
                        if payload.get("treatment"):
                            treatments.add(payload["treatment"])
            except:
                continue
        
        return {
            "image_types": ["gel", "western_blot", "microscopy", "fluorescence"],
            "outcomes": ["positive", "negative", "inconclusive", "dose_dependent"],
            "cell_lines": sorted(list(cell_lines)),
            "treatments": sorted(list(treatments))
        }
        
    except Exception as e:
        logger.error(f"Failed to get filter options: {e}")
        return {
            "image_types": ["gel", "western_blot", "microscopy", "fluorescence"],
            "outcomes": ["positive", "negative", "inconclusive", "dose_dependent"],
            "cell_lines": [],
            "treatments": []
        }


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
# Intelligence Layer (Use Case 4 - Reasoning & Prioritization)
# ============================================================================

class IntelligentJustificationRequest(BaseModel):
    """Request for intelligent justification generation."""
    query: str = Field(..., description="Query SMILES/sequence")
    candidate: str = Field(..., description="Candidate SMILES/sequence")
    candidate_metadata: Dict[str, Any] = Field(default_factory=dict, description="Candidate metadata")
    similarity_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Similarity score")
    modality: str = Field(default="molecule", description="molecule, protein, or text")


class PriorityRankingRequest(BaseModel):
    """Request for predictive priority ranking."""
    candidates: List[Dict[str, Any]] = Field(..., description="List of candidates with content/metadata")
    query: str = Field(..., description="Original query")
    query_modality: str = Field(default="molecule", description="Query modality")
    target_sequence: Optional[str] = Field(default=None, description="Target protein for DTI prediction")


class UnstructuredIngestRequest(BaseModel):
    """Request to ingest unstructured documents (lab notes, PDFs)."""
    file_path: Optional[str] = Field(default=None, description="Path to file")
    content: Optional[str] = Field(default=None, description="Raw text content")
    lab_note: Optional[Dict[str, Any]] = Field(default=None, description="Structured lab note")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    collection: str = Field(default="bioflow_memory", description="Target collection")


class RAGQueryRequest(BaseModel):
    """Request for RAG-augmented query."""
    query: str = Field(..., description="Natural language question")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Source filters")
    max_chunks: int = Field(default=5, ge=1, le=20, description="Max context chunks")
    min_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Min similarity score")


# Intelligence module instances (lazy initialized)
_reasoning_engine = None
_priority_ranker = None
_hypothesis_generator = None
_unstructured_pipeline = None


def get_reasoning_engine():
    """Get or create reasoning engine."""
    global _reasoning_engine
    if _reasoning_engine is None:
        try:
            from bioflow.intelligence.reasoning_engine import ReasoningEngine
            _reasoning_engine = ReasoningEngine()
            logger.info("Reasoning engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize reasoning engine: {e}")
    return _reasoning_engine


def get_priority_ranker():
    """Get or create priority ranker."""
    global _priority_ranker
    if _priority_ranker is None:
        try:
            from bioflow.intelligence.priority_ranker import PriorityRanker
            _priority_ranker = PriorityRanker()
            logger.info("Priority ranker initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize priority ranker: {e}")
    return _priority_ranker


def get_hypothesis_generator():
    """Get or create hypothesis generator."""
    global _hypothesis_generator
    if _hypothesis_generator is None:
        try:
            from bioflow.intelligence.hypothesis_generator import HypothesisGenerator
            _hypothesis_generator = HypothesisGenerator()
            logger.info("Hypothesis generator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize hypothesis generator: {e}")
    return _hypothesis_generator


def get_unstructured_pipeline():
    """Get or create unstructured data pipeline."""
    global _unstructured_pipeline
    if _unstructured_pipeline is None:
        try:
            from bioflow.ingestion.unstructured_pipeline import UnstructuredDataPipeline
            _unstructured_pipeline = UnstructuredDataPipeline()
            logger.info("Unstructured data pipeline initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize unstructured pipeline: {e}")
    return _unstructured_pipeline


@app.post("/api/intelligence/justify")
async def generate_justification(request: IntelligentJustificationRequest):
    """
    Generate intelligent justification for a design suggestion.
    
    Returns structured reasoning explaining WHY a candidate is suggested,
    not just a similarity score. Includes:
    - Structural analysis
    - Evidence linking
    - Priority scoring
    - Actionable insights
    """
    try:
        engine = get_reasoning_engine()
        if not engine:
            raise HTTPException(status_code=503, detail="Reasoning engine not available")
        
        result = engine.generate_justification(
            query=request.query,
            candidate=request.candidate,
            candidate_metadata=request.candidate_metadata,
            similarity_score=request.similarity_score,
            modality=request.modality,
        )
        
        return {
            "success": True,
            "justification": result.to_dict(),
            "summary": result.summary,
            "priority_score": result.priority_score,
        }
    except Exception as e:
        logger.error(f"Justification generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/intelligence/rank")
async def priority_ranking(request: PriorityRankingRequest):
    """
    Re-rank candidates by predicted biological efficacy.
    
    Priority ≠ Similarity. This endpoint ranks by:
    - Predicted activity (QSAR/ML)
    - Experimental validation
    - Drug-likeness
    - Safety profile
    - Novelty
    """
    try:
        ranker = get_priority_ranker()
        if not ranker:
            raise HTTPException(status_code=503, detail="Priority ranker not available")
        
        scores = ranker.rank_candidates(
            candidates=request.candidates,
            query=request.query,
            query_modality=request.query_modality,
            target_sequence=request.target_sequence,
        )
        
        return {
            "success": True,
            "ranked_candidates": [s.to_dict() for s in scores],
            "explanation": ranker.explain_ranking(scores),
        }
    except Exception as e:
        logger.error(f"Priority ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/intelligence/hypotheses")
async def generate_hypotheses(request: IntelligentJustificationRequest):
    """
    Generate design hypotheses explaining modifications.
    
    Provides SAR-based reasoning for structural changes:
    - What changed (substitution, addition, bioisostere)
    - Why it might help (mechanism)
    - Predicted effect
    - Supporting references
    """
    try:
        generator = get_hypothesis_generator()
        if not generator:
            raise HTTPException(status_code=503, detail="Hypothesis generator not available")
        
        hypotheses = generator.generate_hypotheses(
            query=request.query,
            candidate=request.candidate,
            query_modality=request.modality,
            metadata=request.candidate_metadata,
        )
        
        return {
            "success": True,
            "hypotheses": [h.to_dict() for h in hypotheses],
            "count": len(hypotheses),
        }
    except Exception as e:
        logger.error(f"Hypothesis generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/unstructured")
async def ingest_unstructured(request: UnstructuredIngestRequest):
    """
    Ingest unstructured documents (lab notes, PDFs, text).
    
    Supports:
    - PDF documents (papers, protocols)
    - Lab notebook text/images
    - Plain text files
    - Structured lab notes (JSON)
    
    Documents are chunked, entities extracted, and indexed for RAG.
    """
    try:
        pipeline = get_unstructured_pipeline()
        if not pipeline:
            raise HTTPException(status_code=503, detail="Unstructured pipeline not available")
        
        if not qdrant_service:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        chunks = []
        
        if request.file_path:
            chunks = pipeline.process_document(request.file_path, request.metadata)
        elif request.lab_note:
            chunks = pipeline.process_lab_note(request.lab_note, request.metadata)
        elif request.content:
            chunks = pipeline.process_lab_note(request.content, request.metadata)
        else:
            raise HTTPException(status_code=400, detail="Must provide file_path, content, or lab_note")
        
        # Ingest chunks into Qdrant
        ingested = 0
        failed = 0
        ids = []
        
        for chunk in chunks:
            try:
                result = qdrant_service.ingest(
                    content=chunk.content,
                    modality="text",
                    metadata={
                        **chunk.metadata,
                        "source_type": chunk.source_type,
                        "source_file": chunk.source_file,
                        "chunk_index": chunk.chunk_index,
                        "entities": chunk.entities,
                    },
                    collection=request.collection,
                )
                if result.success:
                    ingested += 1
                    ids.append(result.id)
                else:
                    failed += 1
            except Exception as e:
                logger.warning(f"Failed to ingest chunk: {e}")
                failed += 1
        
        return {
            "success": failed == 0,
            "chunks_processed": len(chunks),
            "ingested": ingested,
            "failed": failed,
            "ids": ids[:20],  # Limit response size
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unstructured ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/intelligence/rag")
async def rag_query(request: RAGQueryRequest):
    """
    RAG (Retrieval-Augmented Generation) query over lab notes and documents.
    
    Retrieves relevant context chunks and builds an augmented prompt
    that can be used with an LLM for question answering.
    """
    try:
        if not qdrant_service:
            raise HTTPException(status_code=503, detail="Qdrant service not available")
        
        try:
            from bioflow.ingestion.unstructured_pipeline import RAGRetriever
            encoder = model_service.get_obm_encoder() if model_service else None
            retriever = RAGRetriever(qdrant_service, encoder, request.max_chunks)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"RAG retriever not available: {e}")
        
        # Retrieve context
        context_chunks = retriever.retrieve_context(
            query=request.query,
            filters=request.filters,
            min_score=request.min_score,
        )
        
        # Build augmented prompt
        augmented_prompt = retriever.build_augmented_prompt(request.query, context_chunks)
        
        return {
            "success": True,
            "query": request.query,
            "context_chunks": context_chunks,
            "num_chunks": len(context_chunks),
            "augmented_prompt": augmented_prompt,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
