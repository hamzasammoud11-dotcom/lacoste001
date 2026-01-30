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
    """Advanced search filters for multimodal biological discovery."""
    modality: Optional[str] = None  # text, molecule, protein, image, experiment
    source: Optional[str] = None  # pubmed, uniprot, chembl, experiment
    sources: Optional[List[str]] = None  # Multiple sources
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    organism: Optional[str] = None
    organism_id: Optional[int] = None
    has_structure: Optional[bool] = None
    keywords: Optional[List[str]] = None  # For hybrid search
    # Experiment-specific filters (Use Case 4)
    experiment_type: Optional[str] = None  # binding_assay, activity_assay, admet, phenotypic
    outcome: Optional[str] = None  # success, failure, partial
    quality_min: Optional[float] = None  # Minimum quality score (0-1)
    target: Optional[str] = None  # Target name for experiments


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
        # Calculate priority score based on multiple factors
        priority_score = self._calculate_priority_score()
        
        # Generate justification explaining WHY this result is recommended (D.4)
        justification = self._generate_justification(priority_score)
        
        return {
            "id": self.id,
            "score": self.score,
            "mmr_score": self.mmr_score,
            "diversity_penalty": self.diversity_penalty,
            "content": self.content,
            "modality": self.modality,
            "metadata": {
                **self.metadata,
                "priority_score": priority_score,  # Add priority score to metadata
                "justification": justification,    # Add justification to metadata (D.4)
            },
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
    
    def _generate_justification(self, priority_score: float) -> str:
        """
        Generate a human-readable justification explaining WHY this result is recommended.
        
        Addresses Jury D.4: Design Assistance & Justification
        The UI must show the "Why" for each result without requiring a modal click.
        """
        parts = []
        
        # Priority indicator
        if priority_score >= 0.8:
            parts.append("⭐ HIGH PRIORITY")
        elif priority_score >= 0.6:
            parts.append("💡 PROMISING")
        
        # Evidence-based justification
        source = self.metadata.get("source", self.source_type or "").lower()
        
        if source == "experiment" or self.metadata.get("experiment_type"):
            outcome = self.metadata.get("outcome", "")
            exp_type = self.metadata.get("experiment_type", "assay")
            if outcome == "success":
                parts.append(f"Experimental {exp_type} succeeded - validated result")
            elif outcome == "failure":
                parts.append(f"Experimental {exp_type} failed - learn what didn't work")
            elif outcome:
                parts.append(f"Experimental {exp_type}: {outcome}")
            
            # Add quality info if available
            quality = self.metadata.get("quality_score")
            if quality and float(quality) >= 0.8:
                parts.append(f"High quality data ({float(quality):.0%})")
        
        elif source == "pubmed":
            title = self.metadata.get("title", "")
            if title:
                parts.append(f"Literature: \"{title[:60]}...\"" if len(title) > 60 else f"Literature: \"{title}\"")
            pmid = self.metadata.get("pmid", "")
            if pmid:
                parts.append(f"📄 PMID:{pmid}")
        
        elif source == "chembl":
            chembl_id = self.metadata.get("chembl_id", "")
            assay_count = self.metadata.get("assay_count", 0)
            if assay_count:
                parts.append(f"Validated in {assay_count} bioassays")
            if chembl_id:
                parts.append(f"ChEMBL: {chembl_id}")
            
            # Activity data
            if self.metadata.get("activity_type") and self.metadata.get("activity_value"):
                parts.append(f"{self.metadata['activity_type']}: {self.metadata['activity_value']}")
        
        elif source == "uniprot":
            protein_name = self.metadata.get("protein_name", self.metadata.get("name", ""))
            if protein_name:
                parts.append(f"Protein: {protein_name}")
            organism = self.metadata.get("organism", "")
            if organism:
                parts.append(f"Organism: {organism}")
        
        # Target binding info
        if self.metadata.get("target"):
            parts.append(f"Target: {self.metadata['target']}")
        
        # Affinity class
        affinity = self.metadata.get("affinity_class", "")
        if affinity == "high":
            parts.append("High binding affinity")
        elif affinity == "medium":
            parts.append("Moderate binding affinity")
        
        # Similarity context
        if self.score >= 0.9:
            parts.append("Structurally very similar")
        elif self.score >= 0.7:
            parts.append("Structurally related")
        elif self.score >= 0.5:
            parts.append("Structurally distinct - explores chemical space")
        
        # Default if nothing specific found
        if not parts:
            if self.modality in ("molecule", "smiles"):
                parts.append(f"Similar molecular structure (score: {self.score:.2f})")
            elif self.modality == "protein":
                parts.append(f"Related protein sequence (score: {self.score:.2f})")
            elif self.modality == "image":
                parts.append(f"Visually similar (score: {self.score:.2f})")
            else:
                parts.append(f"Semantic match (score: {self.score:.2f})")
        
        return " | ".join(parts)
    
    def _calculate_priority_score(self) -> float:
        """
        Calculate a DETERMINISTIC priority score (0-1) based on multiple factors.
        
        Priority ≠ Similarity. A high-priority result is one that:
        - Has strong evidence backing (papers, experiments, databases)
        - Shows experimental validation
        - Has complete metadata and conditions
        - Balances relevance with information quality
        
        NO RANDOM VALUES - Score is reproducible.
        """
        # Use a component-based approach with fixed weights
        evidence_component = 0.0
        source_component = 0.0
        outcome_component = 0.0
        quality_component = 0.0
        completeness_component = 0.0
        similarity_component = 0.0
        
        # =========================================================================
        # 1. EVIDENCE STRENGTH (30% weight)
        # =========================================================================
        if self.evidence_links:
            # More sources = higher confidence, but with diminishing returns
            n_links = len(self.evidence_links)
            if n_links >= 5:
                evidence_component = 1.0
            elif n_links >= 3:
                evidence_component = 0.85
            elif n_links >= 2:
                evidence_component = 0.70
            elif n_links >= 1:
                evidence_component = 0.50
        
        # =========================================================================
        # 2. SOURCE QUALITY (25% weight)
        # =========================================================================
        source = self.metadata.get("source", self.source_type or "").lower()
        source_scores = {
            "experiment": 1.00,    # Highest - experimental data
            "chembl": 0.90,        # Curated bioactivity database
            "drugbank": 0.88,      # Approved drug info
            "pubmed": 0.80,        # Literature support
            "uniprot": 0.75,       # Protein annotation
            "pdb": 0.75,           # Structural data
            "pubchem": 0.65,       # Chemical database
        }
        source_component = source_scores.get(source, 0.30)  # Unknown sources get baseline
        
        # =========================================================================
        # 3. EXPERIMENTAL OUTCOME (20% weight)
        # =========================================================================
        outcome = self.metadata.get("outcome", "")
        if outcome == "success":
            outcome_component = 1.0
        elif outcome == "partial":
            outcome_component = 0.6
        elif outcome == "failure":
            outcome_component = 0.25  # Still valuable (negative data)
        elif outcome:  # Any other outcome text
            outcome_component = 0.35
        else:
            outcome_component = 0.0  # No outcome info
        
        # =========================================================================
        # 4. QUALITY METRICS (10% weight)
        # =========================================================================
        quality = self.metadata.get("quality_score", self.metadata.get("quality", 0))
        if quality:
            quality_component = float(quality)
        else:
            # Infer quality from data richness
            has_smiles = bool(self.metadata.get("smiles") or self.metadata.get("content"))
            has_target = bool(self.metadata.get("target"))
            has_activity = bool(self.metadata.get("activity_type") or self.metadata.get("affinity"))
            quality_component = 0.3 * int(has_smiles) + 0.35 * int(has_target) + 0.35 * int(has_activity)
        
        # =========================================================================
        # 5. METADATA COMPLETENESS (5% weight)
        # =========================================================================
        completeness_fields = ["conditions", "measurements", "protocol", "notes", "target", "organism"]
        complete_count = sum(1 for f in completeness_fields if self.metadata.get(f))
        completeness_component = complete_count / len(completeness_fields)
        
        # =========================================================================
        # 6. SIMILARITY RELEVANCE (10% weight)
        # =========================================================================
        # Optimal range: 0.65-0.90 (novel but related)
        # Too similar (>0.95) = likely redundant
        # Too different (<0.5) = may be irrelevant
        if 0.65 <= self.score <= 0.90:
            similarity_component = 1.0  # Sweet spot
        elif self.score > 0.90:
            similarity_component = 0.7  # Penalize near-duplicates
        elif self.score > 0.50:
            similarity_component = 0.8 * self.score + 0.2
        else:
            similarity_component = self.score  # Low relevance
        
        # =========================================================================
        # FINAL WEIGHTED SCORE (Deterministic)
        # =========================================================================
        priority = (
            0.30 * evidence_component +
            0.25 * source_component +
            0.20 * outcome_component +
            0.10 * quality_component +
            0.05 * completeness_component +
            0.10 * similarity_component
        )
        
        # Ensure we return a bounded value
        return min(max(priority, 0.0), 1.0)


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
        include_images: bool = False,
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
            include_images: Also include relevant images in results
            
        Returns:
            SearchResponse with enhanced results
        """
        import time
        start_time = time.time()
        
        logger.info(f"EnhancedSearchService.search: query='{query[:100] if len(query) > 100 else query}...', modality={modality}, include_images={include_images}")
        
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
            include_images=include_images,
        )
        
        logger.info(f"Initial search returned {len(raw_results)} results")

        # If include_images is True, perform a dedicated image search to ensure they are represented
        # often image scores are on a different scale or lower than exact molecule matches
        if include_images:
            logger.info(f"Performing dedicated image search for include_images=True")
            # Create image-specific filters (preserving other filters like source/year)
            image_filters = SearchFilters(**{
                k: v for k, v in filters.__dict__.items() 
                if k != 'modality' and v is not None
            })
            image_filters.modality = "image"
            
            image_results = self._execute_search(
                query_embedding=query_embedding,
                collection=collection,
                limit=max(5, top_k // 2),  # Ensure we get at least some images
                filters=image_filters,
                with_vectors=need_vectors,
                include_images=False  # Avoid efficient OR logic recursion
            )
            
            logger.info(f"Dedicated image search returned {len(image_results)} image results")
            
            # Merge results (deduplicate by id)
            existing_ids = {r['id'] for r in raw_results}
            for img in image_results:
                if img['id'] not in existing_ids:
                    raw_results.append(img)
            
            # If using MMR, the diversity reranker should pick them up.
            # If NOT using MMR, we might need to interleave manually if we want to force them.
            if not use_mmr:
                # Interleave if strictly ranking by score would bury them
                # Sort everything by score first
                raw_results.sort(key=lambda x: x['score'], reverse=True)

        # Post-filters that are difficult to express robustly in Qdrant filters.
        raw_results = self._apply_post_filters(raw_results, filters)
        
        total_found = len(raw_results)
        
        # Apply MMR if requested
        # Logic update: Force-include images if requested, as they often have lower scores than exact molecule matches
        # and would be dropped by standard MMR or top-k truncation.
        
        forced_images = []
        mmr_pool = raw_results
        
        if include_images:
            images = [r for r in raw_results if r.get('modality') == 'image']
            others = [r for r in raw_results if r.get('modality') != 'image']
            
            if images:
                # Force top 3 images or 30% of top_k, whichever is larger, but constrained by availability and total top_k
                target_img_cnt = min(len(images), max(3, int(top_k * 0.3)))
                target_img_cnt = min(target_img_cnt, top_k)
                
                forced_images = images[:target_img_cnt]
                mmr_pool = others # We only run MMR/truncation on the non-images
                
                # Reduce the quota for the pool
                top_k = max(0, top_k - len(forced_images))
            else:
                top_k = top_k # No images found, proceed as normal
        
        if use_mmr and len(mmr_pool) > 0:
            # Get embeddings for MMR
            embeddings = (
                self._get_result_embeddings(mmr_pool, expected_dim=query_dim)
                if include_embeddings or use_mmr
                else None
            )

            effective_lambda = (
                float(lambda_param)
                if lambda_param is not None
                else float(self.mmr_reranker.lambda_param)
            )
            
            mmr_results = mmr_rerank(
                results=mmr_pool,
                query_embedding=query_embedding,
                lambda_param=effective_lambda,
                embeddings=embeddings,
                top_k=top_k,
            )
            
            # Inject forced images as MMRResults
            for img in forced_images:
                # Construct MMRResult manually for the forced images
                mmr_img = MMRResult(
                    id=str(img['id']),
                    original_score=float(img['score']),
                    mmr_score=float(img['score']), # Dummy value
                    diversity_penalty=0.0,
                    content=img.get('content', ''),
                    modality=img.get('modality', 'image'),
                    metadata=img.get('metadata', {}),
                    embedding=img.get('vector')
                )
                mmr_results.append(mmr_img)
            
            # Calculate diversity score
            diversity_score = self.mmr_reranker.compute_diversity_score(mmr_results)
            
            # Convert to enhanced results
            enhanced_results = self._mmr_to_enhanced(mmr_results, include_embeddings)
        else:
            # Manual truncation without MMR
            final_pool = mmr_pool[:top_k]
            combined = forced_images + final_pool
            
            diversity_score = None
            enhanced_results = self._raw_to_enhanced(combined)
        
        # Sort by original score for display
        enhanced_results.sort(key=lambda x: x.score, reverse=True)
        # Update ranks after sorting
        for i, r in enumerate(enhanced_results):
            r.rank = i + 1
        
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
        include_images: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute search against Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Build filter conditions
        must_conditions = []
        
        if filters.modality:
            requested = str(filters.modality).lower()
            modality_conditions = []
            
            # Handle molecule/smiles alias
            if requested in ("molecule", "smiles"):
                modality_conditions.append(FieldCondition(key="modality", match=MatchValue(value="molecule")))
                modality_conditions.append(FieldCondition(key="modality", match=MatchValue(value="smiles")))
            else:
                modality_conditions.append(FieldCondition(key="modality", match=MatchValue(value=requested)))
            
            # If include_images is requested, add it to the allowed modalities
            if include_images:
                modality_conditions.append(FieldCondition(key="modality", match=MatchValue(value="image")))
            
            # Combine into a SHOULD filter if we have multiple options
            if len(modality_conditions) > 1:
                must_conditions.append(Filter(should=modality_conditions))
            else:
                must_conditions.append(modality_conditions[0])
        
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
        
        # Experiment-specific filters (Use Case 4)
        if filters.experiment_type:
            must_conditions.append(FieldCondition(
                key="experiment_type",
                match=MatchValue(value=filters.experiment_type)
            ))
        
        if filters.outcome:
            must_conditions.append(FieldCondition(
                key="outcome",
                match=MatchValue(value=filters.outcome)
            ))
        
        if filters.target:
            must_conditions.append(FieldCondition(
                key="target",
                match=MatchValue(value=filters.target)
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
                # Use search() for qdrant-client v1.x compatibility
                results = client.search(
                    collection_name=coll,
                    query_vector=query_embedding,
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
                
                for r in results:
                    payload_modality = r.payload.get('modality', 'unknown')
                    # Normalize legacy modality value for UI consistency.
                    normalized_modality = "molecule" if payload_modality == "smiles" else payload_modality
                    # Support both old schema (content) and bio_discovery schema (smiles/target_seq)
                    content = r.payload.get('content', '')
                    if not content:
                        content = r.payload.get('smiles', '') or r.payload.get('target_seq', '')
                    all_results.append({
                        'id': str(r.id),
                        'score': r.score,
                        'content': content,
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
            "image": Modality.IMAGE,
            "experiment": Modality.TEXT,  # Experiments use text encoder
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
