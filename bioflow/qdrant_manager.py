"""
Qdrant Manager - Vector Database Integration
==============================================

This module provides high-level management for Qdrant collections,
including ingestion, search, and retrieval operations for BioFlow.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        PointStruct, 
        VectorParams, 
        Distance,
        Filter,
        FieldCondition,
        MatchValue,
        SearchRequest,
        UpdateStatus
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from bioflow.obm_wrapper import OBMWrapper, EmbeddingResult, ModalityType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for search results."""
    id: str
    score: float
    content: str
    modality: str
    payload: Dict[str, Any] = field(default_factory=dict)


class QdrantManager:
    """
    High-level manager for Qdrant vector database operations.
    
    Provides methods for:
    - Collection management (create, delete, info)
    - Data ingestion with automatic embedding
    - Cross-modal similarity search
    - Filtered retrieval
    """
    
    def __init__(
        self,
        obm: OBMWrapper,
        qdrant_url: Optional[str] = None,
        qdrant_path: Optional[str] = None,
        default_collection: str = "bioflow_memory"
    ):
        """
        Initialize QdrantManager.
        
        Args:
            obm: Initialized OBMWrapper instance.
            qdrant_url: URL for remote Qdrant server.
            qdrant_path: Path for local persistent storage.
            default_collection: Default collection name.
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required. Install with: pip install qdrant-client")
        
        self.obm = obm
        self.default_collection = default_collection
        
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
            logger.info(f"Connected to Qdrant server at {qdrant_url}")
        elif qdrant_path:
            self.client = QdrantClient(path=qdrant_path)
            logger.info(f"Using local Qdrant at {qdrant_path}")
        else:
            self.client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant (data will be lost on exit)")
    
    def create_collection(
        self, 
        name: Optional[str] = None,
        recreate: bool = False
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            name: Collection name (uses default if None).
            recreate: If True, deletes existing collection first.
            
        Returns:
            True if created successfully.
        """
        name = name or self.default_collection
        
        if recreate:
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.obm.vector_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection '{name}' with dim={self.obm.vector_dim}")
            return True
        except Exception as e:
            logger.warning(f"Collection might exist: {e}")
            return False
    
    def collection_exists(self, name: str = None) -> bool:
        """Check if collection exists."""
        name = name or self.default_collection
        try:
            collections = self.client.get_collections().collections
            return any(c.name == name for c in collections)
        except Exception:
            return False
    
    def get_collection_info(self, name: str = None) -> Dict[str, Any]:
        """Get collection statistics."""
        name = name or self.default_collection
        try:
            info = self.client.get_collection(name)
            # Handle different qdrant-client versions
            points_count = getattr(info, 'points_count', None) or getattr(info, 'vectors_count', 0)
            status = getattr(info.status, 'value', 'unknown') if hasattr(info, 'status') and info.status else 'unknown'
            
            # Try to get vector size from config
            vector_size = self.obm.vector_dim
            if hasattr(info, 'config') and info.config:
                if hasattr(info.config, 'params') and hasattr(info.config.params, 'vectors'):
                    vectors_config = info.config.params.vectors
                    if hasattr(vectors_config, 'size'):
                        vector_size = vectors_config.size
                    elif isinstance(vectors_config, dict) and '' in vectors_config:
                        vector_size = vectors_config[''].size
            
            return {
                "name": name,
                "points_count": points_count,
                "status": status,
                "vector_size": vector_size
            }
        except Exception as e:
            return {"error": str(e)}
    
    def ingest(
        self,
        items: List[Dict[str, Any]],
        collection: str = None,
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Ingest multiple items with automatic embedding.
        
        Args:
            items: List of dicts with 'content', 'modality', and optional metadata.
            collection: Target collection name.
            batch_size: Number of items per batch.
            
        Returns:
            Statistics dict with success/failure counts.
        """
        collection = collection or self.default_collection
        
        if not self.collection_exists(collection):
            self.create_collection(collection)
        
        stats = {"success": 0, "failed": 0, "skipped": 0}
        points = []
        
        for item in items:
            content = item.get("content")
            modality = item.get("modality", item.get("type", "text"))
            
            if not content:
                stats["skipped"] += 1
                continue
            
            try:
                embedding = self.obm.encode(content, modality)
                
                payload = {k: v for k, v in item.items() if k != "content"}
                payload["content"] = content
                payload["modality"] = modality
                payload["content_hash"] = embedding.content_hash
                
                point_id = item.get("id", str(uuid.uuid4()))
                
                points.append(PointStruct(
                    id=point_id if isinstance(point_id, int) else hash(point_id) % (10**8),
                    vector=embedding.vector.tolist(),
                    payload=payload
                ))
                stats["success"] += 1
                
                # Batch upload
                if len(points) >= batch_size:
                    self.client.upsert(collection_name=collection, points=points)
                    points = []
                    
            except Exception as e:
                logger.error(f"Failed to embed: {e}")
                stats["failed"] += 1
        
        # Upload remaining
        if points:
            self.client.upsert(collection_name=collection, points=points)
        
        logger.info(f"Ingestion complete: {stats}")
        return stats
    
    def search(
        self,
        query: str,
        query_modality: str = "text",
        collection: str = None,
        limit: int = 10,
        filter_modality: str = None,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """
        Search for similar items.
        
        Args:
            query: Query content (text, SMILES, or protein sequence).
            query_modality: Modality of the query.
            collection: Collection to search.
            limit: Maximum results to return.
            filter_modality: Only return results of this modality.
            filters: Additional payload filters.
            
        Returns:
            List of SearchResult objects.
        """
        collection = collection or self.default_collection
        
        # Encode query
        embedding = self.obm.encode(query, query_modality)
        
        # Build filter
        qdrant_filter = None
        conditions = []
        
        if filter_modality:
            conditions.append(
                FieldCondition(key="modality", match=MatchValue(value=filter_modality))
            )
        
        if filters:
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        
        if conditions:
            qdrant_filter = Filter(must=conditions)
        
        # Search using query_points (new API)
        results = self.client.query_points(
            collection_name=collection,
            query=embedding.vector.tolist(),
            limit=limit,
            query_filter=qdrant_filter
        )
        
        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                content=r.payload.get("content", "") if r.payload else "",
                modality=r.payload.get("modality", "unknown") if r.payload else "unknown",
                payload=r.payload or {}
            )
            for r in results.points
        ]
    
    def cross_modal_search(
        self,
        query: str,
        query_modality: str,
        target_modality: str,
        collection: str = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search across modalities (e.g., text query → molecule results).
        
        Args:
            query: Query content.
            query_modality: Modality of query ('text', 'smiles', 'protein').
            target_modality: Modality of desired results.
            collection: Collection to search.
            limit: Maximum results.
            
        Returns:
            List of SearchResult objects from target modality.
        """
        return self.search(
            query=query,
            query_modality=query_modality,
            collection=collection,
            limit=limit,
            filter_modality=target_modality
        )
    
    def get_neighbors_diversity(
        self,
        query: str,
        query_modality: str,
        collection: str = None,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze diversity of top-k neighbors.
        
        Returns statistics about the embedding neighborhood:
        - Mean/std of similarity scores
        - Modality distribution
        - Diversity score
        """
        results = self.search(query, query_modality, collection, limit=k)
        
        if not results:
            return {"error": "No results found"}
        
        scores = [r.score for r in results]
        modalities = [r.modality for r in results]
        
        # Modality distribution
        modality_counts = {}
        for m in modalities:
            modality_counts[m] = modality_counts.get(m, 0) + 1
        
        # Diversity score (1 - variance of normalized scores)
        import numpy as np
        scores_arr = np.array(scores)
        diversity = 1.0 - np.std(scores_arr) if len(scores_arr) > 1 else 0.0
        
        return {
            "k": k,
            "mean_similarity": float(np.mean(scores_arr)),
            "std_similarity": float(np.std(scores_arr)),
            "min_similarity": float(np.min(scores_arr)),
            "max_similarity": float(np.max(scores_arr)),
            "modality_distribution": modality_counts,
            "diversity_score": float(diversity)
        }
