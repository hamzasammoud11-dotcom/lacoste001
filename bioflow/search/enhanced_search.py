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
import re
import threading
from collections import OrderedDict

from bioflow.search.mmr import MMRReranker, MMRResult
from bioflow.search.mmr import mmr_rerank
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
            # UI expects `source`; keep `source_type` for backward compatibility.
            "source": self.source_type,
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
        query_cache_max_size: int = 256,
        query_cache_ttl_s: float = 300.0,
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

        # Simple in-memory cache for query embeddings (reduces repeated encoding latency).
        self._query_cache_max_size = int(query_cache_max_size)
        self._query_cache_ttl_s = float(query_cache_ttl_s)
        self._query_cache: "OrderedDict[tuple[str, str], tuple[float, List[float]]]" = OrderedDict()
        self._query_cache_lock = threading.Lock()
    
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
        
        # Normalize / auto-detect modality for encoding.
        requested_modality = (modality or "text").lower()
        if requested_modality == "auto":
            requested_modality = self._detect_modality(query)

        # Get query embedding (cached)
        from bioflow.core.base import Modality
        modality_enum = self._get_modality_enum(requested_modality)
        query_embedding = self._get_query_embedding_cached(query, modality_enum)
        query_dim = len(query_embedding)
        
        # Execute vector search
        # Fetch more results if using MMR (for better diversity)
        fetch_limit = top_k * 3 if use_mmr else top_k
        need_vectors = bool(use_mmr or include_embeddings)
        
        raw_results = self._execute_search(
            query_embedding=query_embedding,
            collection=collection,
            limit=fetch_limit,
            filters=filters,
            with_vectors=need_vectors,
        )

        # Post-filters that are difficult to express robustly in Qdrant filters.
        raw_results = self._apply_post_filters(raw_results, filters)
        
        total_found = len(raw_results)
        
        # Apply MMR if requested
        if use_mmr and len(raw_results) > 1:
            # Get embeddings for MMR
            embeddings = (
                self._get_result_embeddings(raw_results, expected_dim=query_dim)
                if include_embeddings or use_mmr
                else None
            )

            effective_lambda = (
                float(lambda_param)
                if lambda_param is not None
                else float(self.mmr_reranker.lambda_param)
            )
            
            mmr_results = mmr_rerank(
                results=raw_results,
                query_embedding=query_embedding,
                lambda_param=effective_lambda,
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
            modality=requested_modality,
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
        with_vectors: bool,
    ) -> List[Dict[str, Any]]:
        """Execute search against Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Build filter conditions
        must_conditions = []
        
        if filters.modality:
            requested = str(filters.modality).lower()
            if requested in ("molecule", "smiles"):
                # Historical payloads may use "smiles". Treat both as molecule.
                must_conditions.append(Filter(should=[
                    FieldCondition(key="modality", match=MatchValue(value="molecule")),
                    FieldCondition(key="modality", match=MatchValue(value="smiles")),
                ]))
            else:
                must_conditions.append(FieldCondition(
                    key="modality",
                    match=MatchValue(value=requested)
                ))
        
        if filters.source:
            must_conditions.append(FieldCondition(
                key="source",
                match=MatchValue(value=filters.source)
            ))
        
        if filters.sources:
            must_conditions.append(Filter(should=[
                FieldCondition(key="source", match=MatchValue(value=s))
                for s in filters.sources
            ]))
        
        if filters.organism:
            must_conditions.append(FieldCondition(
                key="organism",
                match=MatchValue(value=filters.organism)
            ))

        if filters.organism_id is not None:
            must_conditions.append(FieldCondition(
                key="organism_id",
                match=MatchValue(value=filters.organism_id)
            ))
        
        # Build filter
        query_filter = None
        if must_conditions:
            query_filter = Filter(must=must_conditions)
        
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
                    with_vectors=with_vectors,
                ).points
                
                for r in results:
                    payload_modality = r.payload.get('modality', 'unknown')
                    # Normalize legacy modality value for UI consistency.
                    normalized_modality = "molecule" if payload_modality == "smiles" else payload_modality
                    all_results.append({
                        'id': str(r.id),
                        'score': r.score,
                        'content': r.payload.get('content', ''),
                        'modality': normalized_modality,
                        'metadata': r.payload,
                        'vector': r.vector if with_vectors else None,
                    })
            except Exception as e:
                logger.warning(f"Search in {coll} failed: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:limit]
    
    def _get_result_embeddings(
        self,
        results: List[Dict],
        expected_dim: int,
    ) -> List[List[float]]:
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
                embeddings.append([0.0] * expected_dim)
        return embeddings
    
    def _extract_source(self, metadata: Dict[str, Any]) -> str:
        """
        Extract source from metadata with fallback chain.
        
        Tries multiple field names and patterns to ensure traceability.
        """
        if not metadata:
            return "unknown"
        
        # Direct source fields (priority order)
        for field in ["source", "database", "origin", "db", "data_source"]:
            val = metadata.get(field)
            if val and isinstance(val, str):
                return val.lower()
        
        # Check for known identifiers that imply source
        if metadata.get("pmid") or metadata.get("pubmed_id"):
            return "pubmed"
        if metadata.get("uniprot_id") or metadata.get("accession"):
            return "uniprot"
        if metadata.get("chembl_id"):
            return "chembl"
        if metadata.get("drugbank_id"):
            return "drugbank"
        if metadata.get("pdb_id"):
            return "pdb"
        
        # Check for URL patterns
        url = metadata.get("url", "")
        if "pubmed" in url.lower() or "ncbi.nlm.nih.gov" in url.lower():
            return "pubmed"
        if "uniprot" in url.lower():
            return "uniprot"
        if "chembl" in url.lower():
            return "chembl"
        
        # Check modality hints
        modality = metadata.get("modality", "")
        if modality in ("protein", "sequence"):
            return "protein_db"
        if modality in ("molecule", "smiles", "compound"):
            return "molecule_db"
        
        return "unknown"
    
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
                source_type=self._extract_source(r.metadata),
                citation=None,  # Added later
                rank=i + 1,
            ))
        return enhanced
    
    def _raw_to_enhanced(self, results: List[Dict]) -> List[EnhancedSearchResult]:
        """Convert raw results to enhanced results."""
        enhanced = []
        for i, r in enumerate(results):
            metadata = r.get('metadata', {})
            enhanced.append(EnhancedSearchResult(
                id=r.get('id', ''),
                score=r.get('score', 0),
                mmr_score=None,
                diversity_penalty=None,
                content=r.get('content', ''),
                modality=r.get('modality', 'unknown'),
                metadata=metadata,
                evidence_links=[],
                source_type=self._extract_source(metadata),
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

    def _get_query_embedding_cached(self, query: str, modality_enum) -> List[float]:
        import time

        key = (getattr(modality_enum, "value", str(modality_enum)), str(query))
        now = time.time()

        with self._query_cache_lock:
            cached = self._query_cache.get(key)
            if cached is not None:
                ts, vec = cached
                if (now - ts) <= self._query_cache_ttl_s:
                    self._query_cache.move_to_end(key)
                    return vec
                self._query_cache.pop(key, None)

        query_result = self.encoder.encode(query, modality_enum)
        vec = query_result.vector.tolist() if hasattr(query_result.vector, "tolist") else list(query_result.vector)

        with self._query_cache_lock:
            self._query_cache[key] = (now, vec)
            self._query_cache.move_to_end(key)
            while len(self._query_cache) > self._query_cache_max_size:
                self._query_cache.popitem(last=False)

        return vec

    def _detect_modality(self, query: str) -> str:
        """
        Heuristically detect modality from raw query.

        Notes:
        - "protein": long sequences of amino-acid letters
        - "molecule": SMILES-like strings (bond symbols, brackets, digits, etc.)
        - otherwise: "text"
        """
        q = (query or "").strip()
        if not q:
            return "text"

        # Protein FASTA / sequence heuristic (AAs + length)
        seq = q.replace("\n", "").replace(" ", "")
        if len(seq) >= 25 and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]+", seq, flags=re.IGNORECASE):
            return "protein"

        # SMILES heuristic: contains typical SMILES tokens and at least one letter/digit.
        if re.search(r"[\[\]=#@\\/()%0-9]", q) and re.search(r"[A-Za-z]", q):
            return "molecule"

        return "text"

    def _apply_post_filters(
        self,
        results: List[Dict[str, Any]],
        filters: SearchFilters,
    ) -> List[Dict[str, Any]]:
        """Apply filters that are not reliably expressed in Qdrant payload filters."""
        if not results:
            return results

        year_min = filters.year_min
        year_max = filters.year_max
        keywords = filters.keywords or []

        def extract_year(payload: Dict[str, Any]) -> Optional[int]:
            for key in ("year", "publication_year", "pub_year", "date"):
                v = payload.get(key)
                if v is None:
                    continue
                if isinstance(v, int):
                    return v
                if isinstance(v, str):
                    m = re.search(r"(19\d{2}|20\d{2})", v)
                    if m:
                        try:
                            return int(m.group(1))
                        except Exception:
                            return None
            return None

        filtered: List[Dict[str, Any]] = []
        for r in results:
            payload = r.get("metadata", {}) or {}

            if year_min is not None or year_max is not None:
                y = extract_year(payload)
                if y is None:
                    continue
                if year_min is not None and y < year_min:
                    continue
                if year_max is not None and y > year_max:
                    continue

            if keywords:
                hay = (r.get("content", "") or "").lower() + " " + str(payload).lower()
                if not all(str(kw).lower() in hay for kw in keywords):
                    continue

            filtered.append(r)

        return filtered
    
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
