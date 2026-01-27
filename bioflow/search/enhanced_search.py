"""
Enhanced Search Service
========================

Combines vector search with:
- MMR diversification
- Evidence linking
- Advanced filtering (modality, source, date, organism)
- Hybrid search (vector + keyword)

Usage:
    from bioflow.search import EnhancedSearchService
    
    service = EnhancedSearchService(qdrant_service, obm_encoder)
    results = service.search(
        query="EGFR inhibitor",
        modality="text",
        use_mmr=True,
        lambda_param=0.7,
        filters={"source": "pubmed", "year_min": 2020}
    )
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from bioflow.search.mmr import MMRReranker, MMRResult
from bioflow.search.evidence import EvidenceLinker, EnrichedResult, EvidenceLink

logger = logging.getLogger(__name__)


@dataclass
class SearchFilters:
    """Advanced search filters."""
    modality: Optional[str] = None  # text, molecule, protein
    source: Optional[str] = None  # pubmed, uniprot, chembl
    sources: Optional[List[str]] = None  # Multiple sources
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    organism: Optional[str] = None
    organism_id: Optional[int] = None
    has_structure: Optional[bool] = None
    keywords: Optional[List[str]] = None  # For hybrid search


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with all metadata."""
    id: str
    score: float
    mmr_score: Optional[float]
    diversity_penalty: Optional[float]
    content: str
    modality: str
    metadata: Dict[str, Any]
    evidence_links: List[EvidenceLink]
    source_type: str
    citation: Optional[str]
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "score": self.score,
            "mmr_score": self.mmr_score,
            "diversity_penalty": self.diversity_penalty,
            "content": self.content,
            "modality": self.modality,
            "metadata": self.metadata,
            "evidence_links": [
                {
                    "source": l.source,
                    "identifier": l.identifier,
                    "url": l.url,
                    "label": l.label,
                }
                for l in self.evidence_links
            ],
            "source_type": self.source_type,
            "citation": self.citation,
            "rank": self.rank,
        }


@dataclass
class SearchResponse:
    """Complete search response with metadata."""
    results: List[EnhancedSearchResult]
    query: str
    modality: str
    total_found: int
    returned: int
    diversity_score: Optional[float]
    filters_applied: Dict[str, Any]
    search_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query,
            "modality": self.modality,
            "total_found": self.total_found,
            "returned": self.returned,
            "diversity_score": self.diversity_score,
            "filters_applied": self.filters_applied,
            "search_time_ms": self.search_time_ms,
        }


class EnhancedSearchService:
    """
    Enhanced search with MMR, evidence linking, and filters.
    """
    
    def __init__(
        self,
        qdrant_service,
        obm_encoder,
        default_lambda: float = 0.7,
        default_top_k: int = 20,
    ):
        """
        Initialize enhanced search service.
        
        Args:
            qdrant_service: QdrantService instance
            obm_encoder: OBMEncoder instance
            default_lambda: Default MMR lambda (relevance vs diversity)
            default_top_k: Default number of results
        """
        self.qdrant = qdrant_service
        self.encoder = obm_encoder
        self.mmr_reranker = MMRReranker(lambda_param=default_lambda)
        self.evidence_linker = EvidenceLinker()
        self.default_top_k = default_top_k
    
    def search(
        self,
        query: str,
        modality: str = "text",
        collection: str = None,
        top_k: int = None,
        use_mmr: bool = True,
        lambda_param: float = None,
        filters: Union[SearchFilters, Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> SearchResponse:
        """
        Execute enhanced search with all features.
        
        Args:
            query: Search query (text, SMILES, or protein sequence)
            modality: Query modality (text, molecule, protein)
            collection: Target collection (None = search all)
            top_k: Number of results to return
            use_mmr: Apply MMR diversification
            lambda_param: Override default lambda for MMR
            filters: Search filters
            include_embeddings: Include embeddings in response
            
        Returns:
            SearchResponse with enhanced results
        """
        import time
        start_time = time.time()
        
        top_k = top_k or self.default_top_k
        
        # Parse filters
        if isinstance(filters, dict):
            filters = SearchFilters(**filters)
        elif filters is None:
            filters = SearchFilters()
        
        # Get query embedding
        from bioflow.core.base import Modality
        modality_enum = self._get_modality_enum(modality)
        query_result = self.encoder.encode(query, modality_enum)
        query_embedding = query_result.vector
        
        # Execute vector search
        # Fetch more results if using MMR (for better diversity)
        fetch_limit = top_k * 3 if use_mmr else top_k
        
        raw_results = self._execute_search(
            query_embedding=query_embedding,
            collection=collection,
            limit=fetch_limit,
            filters=filters,
        )
        
        total_found = len(raw_results)
        
        # Apply MMR if requested
        if use_mmr and len(raw_results) > 1:
            # Get embeddings for MMR
            embeddings = self._get_result_embeddings(raw_results) if include_embeddings or use_mmr else None
            
            mmr_results = self.mmr_reranker.rerank(
                results=raw_results,
                query_embedding=query_embedding,
                embeddings=embeddings,
                top_k=top_k,
            )
            
            # Calculate diversity score
            diversity_score = self.mmr_reranker.compute_diversity_score(mmr_results)
            
            # Convert to enhanced results
            enhanced_results = self._mmr_to_enhanced(mmr_results, include_embeddings)
        else:
            diversity_score = None
            enhanced_results = self._raw_to_enhanced(raw_results[:top_k])
        
        # Apply evidence linking
        enhanced_results = self._add_evidence_links(enhanced_results)
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=enhanced_results,
            query=query,
            modality=modality,
            total_found=total_found,
            returned=len(enhanced_results),
            diversity_score=diversity_score,
            filters_applied=self._filters_to_dict(filters),
            search_time_ms=search_time_ms,
        )
    
    def hybrid_search(
        self,
        query: str,
        keywords: List[str],
        modality: str = "text",
        collection: str = None,
        top_k: int = None,
        vector_weight: float = 0.7,
    ) -> SearchResponse:
        """
        Hybrid search combining vector similarity with keyword matching.
        
        Args:
            query: Vector search query
            keywords: Keywords for text matching
            modality: Query modality
            collection: Target collection
            top_k: Number of results
            vector_weight: Weight for vector score (0-1), keyword = 1 - vector_weight
            
        Returns:
            SearchResponse with hybrid-scored results
        """
        top_k = top_k or self.default_top_k
        
        # Execute vector search
        filters = SearchFilters(keywords=keywords)
        response = self.search(
            query=query,
            modality=modality,
            collection=collection,
            top_k=top_k * 2,  # Fetch more for filtering
            use_mmr=False,
            filters=filters,
        )
        
        # Score boost for keyword matches
        keyword_weight = 1.0 - vector_weight
        enhanced_results = []
        
        for result in response.results:
            content = result.content.lower()
            metadata_str = str(result.metadata).lower()
            
            # Count keyword matches
            keyword_matches = sum(
                1 for kw in keywords
                if kw.lower() in content or kw.lower() in metadata_str
            )
            keyword_score = keyword_matches / len(keywords) if keywords else 0
            
            # Hybrid score
            hybrid_score = (
                vector_weight * result.score +
                keyword_weight * keyword_score
            )
            
            # Update score
            result.score = hybrid_score
            enhanced_results.append(result)
        
        # Re-sort by hybrid score
        enhanced_results.sort(key=lambda x: x.score, reverse=True)
        enhanced_results = enhanced_results[:top_k]
        
        # Update ranks
        for i, r in enumerate(enhanced_results):
            r.rank = i + 1
        
        response.results = enhanced_results
        response.returned = len(enhanced_results)
        
        return response
    
    def _execute_search(
        self,
        query_embedding: List[float],
        collection: str,
        limit: int,
        filters: SearchFilters,
    ) -> List[Dict[str, Any]]:
        """Execute search against Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        
        # Build filter conditions
        conditions = []
        
        if filters.modality:
            conditions.append(FieldCondition(
                key="modality",
                match=MatchValue(value=filters.modality)
            ))
        
        if filters.source:
            conditions.append(FieldCondition(
                key="source",
                match=MatchValue(value=filters.source)
            ))
        
        if filters.sources:
            # Multiple sources - need OR logic
            # Qdrant supports this via should conditions
            pass  # TODO: Implement OR logic
        
        if filters.organism:
            conditions.append(FieldCondition(
                key="organism",
                match=MatchValue(value=filters.organism)
            ))
        
        # Build filter
        query_filter = Filter(must=conditions) if conditions else None
        
        # Get client and search
        client = self.qdrant._get_client()
        
        # Determine collections
        if collection:
            collections = [collection]
        else:
            collections = self.qdrant.list_collections()
        
        all_results = []
        
        for coll in collections:
            try:
                results = client.query_points(
                    collection_name=coll,
                    query=query_embedding,
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True,
                    with_vectors=True,  # Need vectors for MMR
                ).points
                
                for r in results:
                    all_results.append({
                        'id': str(r.id),
                        'score': r.score,
                        'content': r.payload.get('content', ''),
                        'modality': r.payload.get('modality', 'unknown'),
                        'metadata': r.payload,
                        'vector': r.vector,
                    })
            except Exception as e:
                logger.warning(f"Search in {coll} failed: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:limit]
    
    def _get_result_embeddings(self, results: List[Dict]) -> List[List[float]]:
        """Extract embeddings from results."""
        embeddings = []
        for r in results:
            vec = r.get('vector')
            if vec is not None:
                if isinstance(vec, dict):
                    # Named vectors - get the default one
                    vec = list(vec.values())[0] if vec else []
                embeddings.append(vec)
            else:
                # Missing vector - use zeros (will have low similarity)
                embeddings.append([0.0] * 768)
        return embeddings
    
    def _mmr_to_enhanced(
        self,
        mmr_results: List[MMRResult],
        include_embeddings: bool = False,
    ) -> List[EnhancedSearchResult]:
        """Convert MMR results to enhanced results."""
        enhanced = []
        for i, r in enumerate(mmr_results):
            enhanced.append(EnhancedSearchResult(
                id=r.id,
                score=r.original_score,
                mmr_score=r.mmr_score,
                diversity_penalty=r.diversity_penalty,
                content=r.content,
                modality=r.modality,
                metadata=r.metadata,
                evidence_links=[],  # Added later
                source_type=r.metadata.get('source', 'unknown'),
                citation=None,  # Added later
                rank=i + 1,
            ))
        return enhanced
    
    def _raw_to_enhanced(self, results: List[Dict]) -> List[EnhancedSearchResult]:
        """Convert raw results to enhanced results."""
        enhanced = []
        for i, r in enumerate(results):
            enhanced.append(EnhancedSearchResult(
                id=r.get('id', ''),
                score=r.get('score', 0),
                mmr_score=None,
                diversity_penalty=None,
                content=r.get('content', ''),
                modality=r.get('modality', 'unknown'),
                metadata=r.get('metadata', {}),
                evidence_links=[],
                source_type=r.get('metadata', {}).get('source', 'unknown'),
                citation=None,
                rank=i + 1,
            ))
        return enhanced
    
    def _add_evidence_links(
        self,
        results: List[EnhancedSearchResult],
    ) -> List[EnhancedSearchResult]:
        """Add evidence links to results."""
        for r in results:
            enriched = self.evidence_linker.enrich({
                'id': r.id,
                'score': r.score,
                'content': r.content,
                'modality': r.modality,
                'metadata': r.metadata,
            })
            r.evidence_links = enriched.evidence_links
            r.citation = enriched.citation
        return results
    
    def _get_modality_enum(self, modality: str):
        """Convert string modality to enum."""
        from bioflow.core.base import Modality
        mapping = {
            "text": Modality.TEXT,
            "molecule": Modality.SMILES,
            "smiles": Modality.SMILES,
            "protein": Modality.PROTEIN,
        }
        return mapping.get(modality.lower(), Modality.TEXT)
    
    def _filters_to_dict(self, filters: SearchFilters) -> Dict[str, Any]:
        """Convert filters to dictionary."""
        return {
            k: v for k, v in {
                "modality": filters.modality,
                "source": filters.source,
                "sources": filters.sources,
                "year_min": filters.year_min,
                "year_max": filters.year_max,
                "organism": filters.organism,
                "keywords": filters.keywords,
            }.items() if v is not None
        }
