"""
Qdrant Retriever - Vector Database Integration
================================================

Implements BioRetriever interface for Qdrant vector database.
Provides semantic search and ingestion for the BioFlow platform.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import uuid

from bioflow.core import BioRetriever, BioEncoder, Modality, RetrievalResult

logger = logging.getLogger(__name__)

# Lazy import
_qdrant_client = None


def _load_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                PointStruct,
                VectorParams,
                Distance,
                Filter,
                FieldCondition,
                MatchValue,
            )
            _qdrant_client = {
                "QdrantClient": QdrantClient,
                "PointStruct": PointStruct,
                "VectorParams": VectorParams,
                "Distance": Distance,
                "Filter": Filter,
                "FieldCondition": FieldCondition,
                "MatchValue": MatchValue,
            }
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )
    return _qdrant_client


class QdrantRetriever(BioRetriever):
    """
    Vector database retriever using Qdrant.
    
    Supports:
    - Semantic search with embedding vectors
    - Payload filtering (by modality, species, etc.)
    - Batch ingestion of data
    
    Example:
        >>> from bioflow.plugins.obm_encoder import OBMEncoder
        >>> 
        >>> encoder = OBMEncoder()
        >>> retriever = QdrantRetriever(encoder=encoder, collection="molecules")
        >>> 
        >>> # Ingest data
        >>> retriever.ingest("CCO", Modality.SMILES, {"name": "Ethanol"})
        >>> 
        >>> # Search
        >>> results = retriever.search("alcohol compounds", limit=5)
    """
    
    def __init__(
        self,
        encoder: BioEncoder,
        collection: str = "bioflow_memory",
        url: str = None,
        path: str = None,
        distance: str = "cosine"
    ):
        """
        Initialize QdrantRetriever.
        
        Args:
            encoder: BioEncoder instance for vectorization
            collection: Default collection name
            url: Qdrant server URL (for remote)
            path: Local path for persistent storage
            distance: Distance metric (cosine, euclid, dot)
        """
        qdrant = _load_qdrant()
        
        self.encoder = encoder
        self.collection = collection
        self.distance = distance
        
        # Initialize client
        if url:
            self.client = qdrant["QdrantClient"](url=url)
            logger.info(f"Connected to Qdrant server at {url}")
        elif path:
            self.client = qdrant["QdrantClient"](path=path)
            logger.info(f"Using local Qdrant at {path}")
        else:
            self.client = qdrant["QdrantClient"](":memory:")
            logger.info("Using in-memory Qdrant (data will be lost on exit)")
        
        # Create collection if not exists
        self._ensure_collection()
    
    def _ensure_collection(self, name: str = None):
        """Ensure collection exists."""
        qdrant = _load_qdrant()
        name = name or self.collection
        
        collections = [c.name for c in self.client.get_collections().collections]
        
        if name not in collections:
            distance_map = {
                "cosine": qdrant["Distance"].COSINE,
                "euclid": qdrant["Distance"].EUCLID,
                "dot": qdrant["Distance"].DOT,
            }
            
            self.client.create_collection(
                collection_name=name,
                vectors_config=qdrant["VectorParams"](
                    size=self.encoder.dimension,
                    distance=distance_map.get(self.distance, qdrant["Distance"].COSINE)
                )
            )
            logger.info(f"Created collection: {name} (dim={self.encoder.dimension})")
    
    def search(
        self,
        query: Union[List[float], str],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection: str = None,
        modality: Modality = Modality.TEXT,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Search for similar items.
        
        Args:
            query: Query vector or raw content to encode
            limit: Maximum results
            filters: Payload filters (e.g., {"species": "human"})
            collection: Collection to search (uses default if None)
            modality: Modality of query (if string)
            
        Returns:
            List of RetrievalResult sorted by similarity
        """
        qdrant = _load_qdrant()
        collection = collection or self.collection
        
        # Encode query if string
        if isinstance(query, str):
            result = self.encoder.encode(query, modality)
            query_vector = result.vector
        else:
            query_vector = query
        
        # Build filter
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    qdrant["FieldCondition"](
                        key=key,
                        match=qdrant["MatchValue"](value=value)
                    )
                )
            qdrant_filter = qdrant["Filter"](must=conditions)
        
        # Search (use search() for qdrant-client v1.x compatibility)
        try:
            # Use search() which exists in all versions
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter
            )
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            results = []
        
        # Convert to RetrievalResult with safe modality mapping
        def _safe_modality(payload: dict) -> Modality:
            raw = payload.get("modality")
            if isinstance(raw, Modality):
                return raw
            if not isinstance(raw, str):
                return Modality.TEXT
            norm = raw.strip().lower()
            # Map legacy/synonym values
            synonym_map = {"molecule": "smiles", "drug": "smiles"}
            if norm in synonym_map:
                norm = synonym_map[norm]
            try:
                return Modality(norm)
            except ValueError:
                return Modality.TEXT
        
        return [
            RetrievalResult(
                id=str(r.id),
                score=r.score,
                content=r.payload.get("content", ""),
                modality=_safe_modality(r.payload),
                payload=r.payload
            )
            for r in results
        ]
    
    def ingest(
        self,
        content: Any,
        modality: Modality,
        payload: Optional[Dict[str, Any]] = None,
        collection: str = None,
        id: str = None
    ) -> str:
        """
        Ingest content into the vector database.
        
        Args:
            content: Raw content to encode
            modality: Type of content
            payload: Additional metadata
            collection: Target collection
            id: Custom ID (auto-generated if None)
            
        Returns:
            ID of inserted item
        """
        qdrant = _load_qdrant()
        collection = collection or self.collection
        
        # Encode content
        result = self.encoder.encode(content, modality)
        
        # Generate ID
        point_id = id or str(uuid.uuid4())
        
        # Build payload
        full_payload = {
            "content": content,
            "modality": modality.value,
            **(payload or {})
        }
        
        # Insert
        self.client.upsert(
            collection_name=collection,
            points=[
                qdrant["PointStruct"](
                    id=point_id,
                    vector=result.vector,
                    payload=full_payload
                )
            ]
        )
        
        logger.debug(f"Ingested {modality.value}: {point_id}")
        return point_id
    
    def batch_ingest(
        self,
        items: List[Dict[str, Any]],
        collection: str = None
    ) -> List[str]:
        """
        Batch ingest multiple items.
        
        Args:
            items: List of {"content": ..., "modality": ..., "payload": ...}
            collection: Target collection
            
        Returns:
            List of inserted IDs
        """
        qdrant = _load_qdrant()
        collection = collection or self.collection
        
        points = []
        ids = []
        
        for item in items:
            content = item["content"]
            modality = Modality(item.get("modality", "text"))
            payload = item.get("payload", {})
            
            result = self.encoder.encode(content, modality)
            point_id = str(uuid.uuid4())
            
            points.append(
                qdrant["PointStruct"](
                    id=point_id,
                    vector=result.vector,
                    payload={"content": content, "modality": modality.value, **payload}
                )
            )
            ids.append(point_id)
        
        self.client.upsert(collection_name=collection, points=points)
        logger.info(f"Batch ingested {len(ids)} items to {collection}")
        
        return ids
    
    def count(self, collection: str = None) -> int:
        """Get count of items in collection."""
        collection = collection or self.collection
        return self.client.count(collection_name=collection).count
    
    def delete_collection(self, collection: str = None):
        """Delete a collection."""
        collection = collection or self.collection
        self.client.delete_collection(collection_name=collection)
        logger.info(f"Deleted collection: {collection}")
