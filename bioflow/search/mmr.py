"""
Maximal Marginal Relevance (MMR) Reranking
============================================

Implements MMR for search result diversification.
MMR balances relevance with diversity to avoid redundant results.

Formula:
    MMR = λ * Sim(q, d) - (1 - λ) * max(Sim(d, d_selected))

Where:
    - λ (lambda): Trade-off between relevance and diversity (0-1)
    - Sim(q, d): Similarity between query and document
    - Sim(d, d_selected): Maximum similarity to already selected documents

Usage:
    from bioflow.search.mmr import mmr_rerank
    
    diverse_results = mmr_rerank(
        results=search_results,
        query_embedding=query_vec,
        lambda_param=0.7,  # Higher = more relevance, Lower = more diversity
        top_k=10
    )
"""

import logging
import numpy as np
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MMRResult:
    """Result with MMR score."""
    id: str
    original_score: float
    mmr_score: float
    diversity_penalty: float
    content: str
    modality: str
    metadata: dict
    embedding: Optional[List[float]] = None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_similarity_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    n = len(embeddings)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                matrix[i, j] = sim
                matrix[j, i] = sim
    
    return matrix


def mmr_rerank(
    results: List[dict],
    query_embedding: Union[List[float], np.ndarray],
    lambda_param: float = 0.7,
    top_k: int = 10,
    embeddings: Optional[List[List[float]]] = None,
) -> List[MMRResult]:
    """
    Rerank search results using Maximal Marginal Relevance.
    
    Args:
        results: List of search results with 'id', 'score', 'content', 'modality', 'metadata'
        query_embedding: Query vector for relevance computation
        lambda_param: Balance between relevance (1.0) and diversity (0.0)
        top_k: Number of results to return
        embeddings: Optional pre-computed embeddings for each result
        
    Returns:
        List of MMRResult with diversified ordering
    """
    if not results:
        return []
    
    n = len(results)
    if n <= 1:
        return [MMRResult(
            id=results[0].get('id', ''),
            original_score=results[0].get('score', 0),
            mmr_score=results[0].get('score', 0),
            diversity_penalty=0.0,
            content=results[0].get('content', ''),
            modality=results[0].get('modality', 'unknown'),
            metadata=results[0].get('metadata', {}),
            embedding=embeddings[0] if embeddings else None
        )]
    
    # Convert to numpy
    query_vec = np.array(query_embedding)
    
    # Get or compute embeddings
    if embeddings:
        doc_embeddings = [np.array(e) for e in embeddings]
    else:
        # If no embeddings provided, use original scores as relevance
        # This is a fallback - ideally embeddings should be provided
        logger.warning("No embeddings provided for MMR - using original scores only")
        doc_embeddings = None
    
    # Compute relevance scores (similarity to query)
    if doc_embeddings:
        relevance_scores = [cosine_similarity(query_vec, e) for e in doc_embeddings]
    else:
        # Normalize original scores to [0, 1]
        max_score = max(r.get('score', 0) for r in results)
        if max_score > 0:
            relevance_scores = [r.get('score', 0) / max_score for r in results]
        else:
            relevance_scores = [0.5] * n
    
    # Compute pairwise similarity matrix
    if doc_embeddings:
        sim_matrix = compute_similarity_matrix(doc_embeddings)
    else:
        # Identity matrix as fallback (no diversity information)
        sim_matrix = np.eye(n)
    
    # MMR selection
    selected_indices = []
    remaining_indices = list(range(n))
    mmr_results = []
    
    for _ in range(min(top_k, n)):
        if not remaining_indices:
            break
        
        best_idx = None
        best_mmr = float('-inf')
        best_diversity_penalty = 0.0
        
        for idx in remaining_indices:
            # Relevance component
            relevance = relevance_scores[idx]
            
            # Diversity component (max similarity to already selected)
            if selected_indices:
                max_sim_to_selected = max(sim_matrix[idx, s] for s in selected_indices)
            else:
                max_sim_to_selected = 0.0
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx
                best_diversity_penalty = max_sim_to_selected
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            result = results[best_idx]
            mmr_results.append(MMRResult(
                id=result.get('id', ''),
                original_score=result.get('score', 0),
                mmr_score=best_mmr,
                diversity_penalty=best_diversity_penalty,
                content=result.get('content', ''),
                modality=result.get('modality', 'unknown'),
                metadata=result.get('metadata', {}),
                embedding=doc_embeddings[best_idx].tolist() if doc_embeddings else None
            ))
    
    return mmr_results


class MMRReranker:
    """
    MMR Reranker with configurable parameters.
    
    Example:
        reranker = MMRReranker(lambda_param=0.6)
        diverse_results = reranker.rerank(results, query_embedding)
    """
    
    def __init__(
        self,
        lambda_param: float = 0.7,
        top_k: int = 10,
        min_diversity: float = 0.0,
    ):
        """
        Initialize MMR reranker.
        
        Args:
            lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
            top_k: Default number of results to return
            min_diversity: Minimum diversity threshold (0-1)
        """
        self.lambda_param = lambda_param
        self.top_k = top_k
        self.min_diversity = min_diversity
    
    def rerank(
        self,
        results: List[dict],
        query_embedding: Union[List[float], np.ndarray],
        embeddings: Optional[List[List[float]]] = None,
        top_k: Optional[int] = None,
    ) -> List[MMRResult]:
        """
        Rerank results using MMR.
        
        Args:
            results: Search results
            query_embedding: Query vector
            embeddings: Document embeddings (optional)
            top_k: Override default top_k
            
        Returns:
            Diversified results
        """
        return mmr_rerank(
            results=results,
            query_embedding=query_embedding,
            lambda_param=self.lambda_param,
            top_k=top_k or self.top_k,
            embeddings=embeddings,
        )
    
    def compute_diversity_score(self, results: List[MMRResult]) -> float:
        """
        Compute average diversity score for a result set.
        
        Returns:
            Diversity score (0-1), higher = more diverse
        """
        if not results or len(results) < 2:
            return 1.0
        
        # Average pairwise diversity = 1 - average similarity
        embeddings = [r.embedding for r in results if r.embedding]
        if len(embeddings) < 2:
            return 1.0
        
        total_sim = 0.0
        count = 0
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                total_sim += cosine_similarity(
                    np.array(embeddings[i]),
                    np.array(embeddings[j])
                )
                count += 1
        
        avg_sim = total_sim / count if count > 0 else 0
        return 1.0 - avg_sim
