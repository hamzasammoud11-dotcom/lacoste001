"""
BioFlow Qdrant Service
=======================
Vector database service for semantic search and data storage.
Wraps Qdrant client with BioFlow-specific operations.

NO FALLBACKS - Qdrant must be available or operations will fail explicitly.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)


class CollectionType(str, Enum):
    """Standard collections in BioFlow."""
    MOLECULES = "molecules"
    PROTEINS = "proteins"
    LITERATURE = "literature"
    MIXED = "bioflow_memory"


@dataclass
class SearchResult:
    """Result from vector search."""
    id: str
    score: float
    content: str
    modality: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestResult:
    """Result from data ingestion."""
    id: str
    collection: str
    success: bool
    message: str = ""


class QdrantServiceError(Exception):
    """Raised when a Qdrant operation fails."""
    pass


class DependencyError(QdrantServiceError):
    """Raised when a required dependency is not available."""
    pass


class QdrantService:
    """
    Vector database service using Qdrant.
    
    NO FALLBACKS - Qdrant client must be available or operations will fail.
    
    Provides:
    - Semantic search across molecules, proteins, and text
    - Data ingestion with automatic embedding
    - Collection management
    
    Example:
        >>> from bioflow.api.model_service import get_model_service
        >>> 
        >>> model_service = get_model_service()
        >>> qdrant = QdrantService(model_service=model_service)
        >>> 
        >>> # Ingest a molecule
        >>> result = qdrant.ingest("CCO", "molecule", {"name": "Ethanol"})
        >>> 
        >>> # Search
        >>> results = qdrant.search("alcohol", limit=5)
    """
    
    def __init__(
        self,
        model_service=None,
        url: str = None,
        path: str = None,
        vector_dim: int = 768
    ):
        """
        Initialize QdrantService.
        
        Args:
            model_service: ModelService for embeddings
            url: Qdrant server URL (e.g., http://localhost:6333)
            path: Path for local Qdrant storage
            vector_dim: Dimension of embedding vectors
            
        Raises:
            DependencyError: If qdrant-client is not installed
        """
        self.model_service = model_service
        self.url = url or os.getenv("QDRANT_URL")
        self.path = path or os.getenv("QDRANT_PATH", "./qdrant_data")
        self.vector_dim = vector_dim
        
        self._client = None
        self._initialized_collections: set = set()
        
        # Validate qdrant-client is available
        self._validate_dependencies()
        
        logger.info(f"QdrantService initialized")
    
    def _validate_dependencies(self):
        """Validate that qdrant-client is available."""
        try:
            from qdrant_client import QdrantClient
            logger.info("qdrant-client available")
        except ImportError:
            raise DependencyError(
                "qdrant-client is required but not installed. "
                "Install with: pip install qdrant-client"
            )
    
    def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is not None:
            return self._client
        
        try:
            from qdrant_client import QdrantClient
            
            if self.url:
                self._client = QdrantClient(url=self.url)
                logger.info(f"Connected to Qdrant at {self.url}")
            else:
                self._client = QdrantClient(path=self.path)
                logger.info(f"Using local Qdrant at {self.path}")
            
            return self._client
        except Exception as e:
            raise QdrantServiceError(f"Failed to create Qdrant client: {e}")
    
    def _ensure_collection(self, collection: str):
        """Ensure collection exists."""
        if collection in self._initialized_collections:
            return
        
        client = self._get_client()
        
        try:
            from qdrant_client.models import VectorParams, Distance
            
            collections = client.get_collections().collections
            exists = any(c.name == collection for c in collections)
            
            if not exists:
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {collection}")
        except Exception as e:
            raise QdrantServiceError(f"Failed to ensure collection {collection}: {e}")
        
        self._initialized_collections.add(collection)
    
    def _get_embedding(self, content: str, modality: str) -> List[float]:
        """Get embedding for content."""
        if self.model_service is None:
            from bioflow.api.model_service import get_model_service
            self.model_service = get_model_service()
        
        if modality == "molecule" or modality == "smiles":
            result = self.model_service.encode_molecule(content)
        elif modality == "protein":
            result = self.model_service.encode_protein(content)
        else:
            result = self.model_service.encode_text(content)
        
        return result.vector
    
    # =========================================================================
    # Ingestion
    # =========================================================================
    
    def ingest(
        self,
        content: str,
        modality: str,
        metadata: Dict[str, Any] = None,
        collection: str = None,
        id: str = None
    ) -> IngestResult:
        """
        Ingest content into vector database.
        
        Args:
            content: SMILES, protein sequence, or text
            modality: Type of content (molecule, protein, text)
            metadata: Additional metadata
            collection: Target collection (auto-selected if None)
            id: Custom ID (generated if None)
            
        Returns:
            IngestResult with status
            
        Raises:
            QdrantServiceError: If ingestion fails
        """
        # Select collection
        if collection is None:
            if modality in ("molecule", "smiles"):
                collection = CollectionType.MOLECULES.value
            elif modality == "protein":
                collection = CollectionType.PROTEINS.value
            else:
                collection = CollectionType.MIXED.value
        
        self._ensure_collection(collection)
        
        # Generate ID
        point_id = id or str(uuid.uuid4())
        
        # Get embedding - will raise if fails
        vector = self._get_embedding(content, modality)
        
        # Prepare payload
        payload = {
            "content": content,
            "modality": modality,
            **(metadata or {})
        }
        
        # Insert into Qdrant
        client = self._get_client()
        
        try:
            from qdrant_client.models import PointStruct
            client.upsert(
                collection_name=collection,
                points=[PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )]
            )
            
            return IngestResult(
                id=point_id,
                collection=collection,
                success=True,
                message=f"Ingested into {collection}"
            )
        except Exception as e:
            raise QdrantServiceError(f"Ingestion failed: {e}")
    
    def ingest_batch(
        self,
        items: List[Dict[str, Any]],
        collection: str = None
    ) -> List[IngestResult]:
        """
        Ingest multiple items.
        
        Args:
            items: List of dicts with content, modality, metadata
            collection: Target collection
            
        Returns:
            List of IngestResult
        """
        results = []
        for item in items:
            result = self.ingest(
                content=item.get("content", ""),
                modality=item.get("modality", "text"),
                metadata=item.get("metadata"),
                collection=collection,
                id=item.get("id")
            )
            results.append(result)
        return results
    
    # =========================================================================
    # Search
    # =========================================================================
    
    def search(
        self,
        query: str,
        modality: str = "text",
        collection: str = None,
        limit: int = 10,
        filter_modality: str = None
    ) -> List[SearchResult]:
        """
        Semantic search across vectors.
        
        Args:
            query: Search query (SMILES, sequence, or text)
            modality: Type of query
            collection: Collection to search (all if None)
            limit: Maximum results
            filter_modality: Filter results by modality
            
        Returns:
            List of SearchResult
            
        Raises:
            QdrantServiceError: If search fails
        """
        # Get query embedding
        query_vector = self._get_embedding(query, modality)
        
        # Determine collections to search
        if collection:
            collections = [collection]
        else:
            collections = [c.value for c in CollectionType]
        
        client = self._get_client()
        all_results = []
        
        for coll in collections:
            try:
                if coll not in self._initialized_collections:
                    continue
                
                filter_conditions = None
                if filter_modality:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    filter_conditions = Filter(
                        must=[FieldCondition(
                            key="modality",
                            match=MatchValue(value=filter_modality)
                        )]
                    )
                
                # Use query_points for newer qdrant-client versions
                results = client.query_points(
                    collection_name=coll,
                    query=query_vector,
                    limit=limit,
                    query_filter=filter_conditions
                ).points
                
                for r in results:
                    all_results.append(SearchResult(
                        id=str(r.id),
                        score=r.score,
                        content=r.payload.get("content", ""),
                        modality=r.payload.get("modality", "unknown"),
                        metadata=r.payload
                    ))
            except Exception as e:
                raise QdrantServiceError(f"Search in {coll} failed: {e}")
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:limit]
    
    # =========================================================================
    # Collection Management
    # =========================================================================
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        client = self._get_client()
        
        try:
            collections = client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            raise QdrantServiceError(f"Failed to list collections: {e}")
    
    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        client = self._get_client()
        
        try:
            info = client.get_collection(collection)
            # Convert to dict to safely access fields (Pydantic v2 raises on missing attrs)
            info_dict = info.model_dump() if hasattr(info, 'model_dump') else info.dict()
            
            # Modern Qdrant uses indexed_vectors_count and points_count
            vectors_count = info_dict.get('indexed_vectors_count', 0) or info_dict.get('points_count', 0)
            points_count = info_dict.get('points_count', 0) or vectors_count
            status = info_dict.get('status', 'unknown')
            
            return {
                "name": collection,
                "vectors_count": vectors_count or 0,
                "points_count": points_count or 0,
                "status": status
            }
        except Exception as e:
            raise QdrantServiceError(f"Failed to get stats for {collection}: {e}")
    
    def delete_collection(self, collection: str) -> bool:
        """Delete a collection."""
        client = self._get_client()
        
        try:
            client.delete_collection(collection)
            self._initialized_collections.discard(collection)
            return True
        except Exception as e:
            raise QdrantServiceError(f"Failed to delete {collection}: {e}")


# ============================================================================
# Singleton Instance
# ============================================================================
_qdrant_service: Optional[QdrantService] = None


def get_qdrant_service(
    model_service=None,
    url: str = None,
    path: str = None,
    reset: bool = False
) -> QdrantService:
    """
    Get or create the global QdrantService instance.
    
    Args:
        model_service: ModelService for embeddings
        url: Qdrant server URL
        path: Local storage path
        reset: Force create new instance
        
    Returns:
        QdrantService singleton
        
    Raises:
        DependencyError: If qdrant-client is not installed
    """
    global _qdrant_service
    if _qdrant_service is None or reset:
        _qdrant_service = QdrantService(
            model_service=model_service,
            url=url,
            path=path
        )
    return _qdrant_service
