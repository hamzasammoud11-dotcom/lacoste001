"""
BioFlow Search Module
======================

Provides enhanced search capabilities:
- MMR (Maximal Marginal Relevance) diversification
- Evidence linking with source tracking
- Advanced filtering by modality, source, date
- Hybrid search (vector + keyword)
"""

from bioflow.search.mmr import MMRReranker, mmr_rerank
from bioflow.search.enhanced_search import EnhancedSearchService
from bioflow.search.evidence import EvidenceLinker

__all__ = [
    "MMRReranker",
    "mmr_rerank",
    "EnhancedSearchService",
    "EvidenceLinker",
]
