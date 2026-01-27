"""
Ranker Agent
============

Score-based ranking with feedback loop integration.

Features:
- Multi-criteria ranking
- Feedback-based re-ranking
- Score normalization and weighting
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from bioflow.agents.base import BaseAgent, AgentMessage, AgentContext, AgentType

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    """A ranked candidate with scores."""
    smiles: str
    rank: int
    final_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "smiles": self.smiles,
            "rank": self.rank,
            "final_score": self.final_score,
            "component_scores": self.component_scores,
            "metadata": self.metadata,
        }


class ScoringStrategy(Enum):
    """How to combine multiple scores."""
    WEIGHTED_SUM = "weighted_sum"
    WEIGHTED_PRODUCT = "weighted_product"
    MIN_SCORE = "min_score"
    PARETO = "pareto"


class RankerAgent(BaseAgent):
    """
    Agent for ranking molecules based on multiple criteria.
    
    Combines scores from:
    - Validation results (ADMET, toxicity)
    - Generation confidence
    - Similarity to query
    - Custom criteria
    
    Supports feedback loops for iterative refinement.
    
    Example:
        >>> agent = RankerAgent()
        >>> candidates = [
        ...     {"smiles": "CCO", "validation_score": 0.8, "similarity": 0.9},
        ...     {"smiles": "CC", "validation_score": 0.9, "similarity": 0.7},
        ... ]
        >>> result = agent.process(candidates)
        >>> print(result.content[0]["rank"])  # 1
    """
    
    # Default weights for different score components
    DEFAULT_WEIGHTS = {
        "validation_score": 0.3,
        "similarity": 0.25,
        "confidence": 0.2,
        "qed": 0.15,
        "novelty": 0.1,
    }
    
    def __init__(
        self,
        name: str = "CandidateRanker",
        strategy: ScoringStrategy = ScoringStrategy.WEIGHTED_SUM,
        weights: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ranker agent.
        
        Args:
            name: Agent name
            strategy: How to combine scores
            weights: Custom score weights
            config: Additional configuration
        """
        super().__init__(name, AgentType.RANKER, config)
        self.strategy = strategy
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._feedback_history: List[Dict[str, Any]] = []
    
    def process(
        self,
        input_data: Union[List[Dict[str, Any]], Dict[str, Any]],
        context: Optional[AgentContext] = None,
    ) -> AgentMessage:
        """
        Rank candidates based on scores.
        
        Args:
            input_data: Either:
                - List[dict]: Candidates with score fields
                - dict: {"candidates": list, "weights": dict, "top_k": int}
            context: Optional shared context
            
        Returns:
            AgentMessage with ranked candidates
        """
        if not self._initialized:
            self.initialize()
        
        # Parse input
        if isinstance(input_data, list):
            candidates = input_data
            weights = self.weights
            top_k = None
        else:
            candidates = input_data.get("candidates", [])
            weights = input_data.get("weights", self.weights)
            top_k = input_data.get("top_k")
        
        if not candidates:
            return AgentMessage(
                sender=self.name,
                content=[],
                metadata={"error": "No candidates to rank"},
                success=False,
            )
        
        try:
            # Calculate final scores
            scored = self._calculate_scores(candidates, weights)
            
            # Sort by final score
            scored.sort(key=lambda x: x.final_score, reverse=True)
            
            # Assign ranks
            for i, candidate in enumerate(scored):
                candidate.rank = i + 1
            
            # Apply top_k limit
            if top_k is not None:
                scored = scored[:top_k]
            
            return AgentMessage(
                sender=self.name,
                content=[c.to_dict() for c in scored],
                metadata={
                    "total_candidates": len(candidates),
                    "returned": len(scored),
                    "strategy": self.strategy.value,
                    "weights_used": weights,
                    "top_score": scored[0].final_score if scored else 0,
                },
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return AgentMessage(
                sender=self.name,
                content=[],
                metadata={"error": str(e)},
                success=False,
                error=str(e),
            )
    
    def _calculate_scores(
        self,
        candidates: List[Dict[str, Any]],
        weights: Dict[str, float],
    ) -> List[RankedCandidate]:
        """Calculate final scores for all candidates."""
        results = []
        
        for cand in candidates:
            smiles = cand.get("smiles", "")
            component_scores = {}
            
            # Extract known score components
            for key in weights.keys():
                if key in cand:
                    value = cand[key]
                    # Normalize to 0-1 if needed
                    if isinstance(value, (int, float)):
                        component_scores[key] = min(1.0, max(0.0, float(value)))
            
            # Calculate final score based on strategy
            final_score = self._combine_scores(component_scores, weights)
            
            results.append(RankedCandidate(
                smiles=smiles,
                rank=0,  # Will be assigned after sorting
                final_score=round(final_score, 4),
                component_scores=component_scores,
                metadata={k: v for k, v in cand.items() if k not in weights},
            ))
        
        return results
    
    def _combine_scores(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        """Combine component scores into final score."""
        if not scores:
            return 0.0
        
        if self.strategy == ScoringStrategy.WEIGHTED_SUM:
            total_weight = sum(weights.get(k, 0) for k in scores.keys())
            if total_weight == 0:
                return sum(scores.values()) / len(scores)
            
            weighted_sum = sum(
                scores[k] * weights.get(k, 0)
                for k in scores.keys()
            )
            return weighted_sum / total_weight
        
        elif self.strategy == ScoringStrategy.WEIGHTED_PRODUCT:
            product = 1.0
            for key, score in scores.items():
                weight = weights.get(key, 1.0)
                product *= score ** weight
            return product ** (1.0 / len(scores))
        
        elif self.strategy == ScoringStrategy.MIN_SCORE:
            return min(scores.values())
        
        elif self.strategy == ScoringStrategy.PARETO:
            # Pareto ranking - count domination
            # For single candidate, just average
            return sum(scores.values()) / len(scores)
        
        return 0.0
    
    def apply_feedback(
        self,
        feedback: Dict[str, Any],
    ) -> None:
        """
        Apply feedback to adjust weights.
        
        Args:
            feedback: {
                "smiles": str,
                "action": "promote" | "demote" | "exclude",
                "reason": str,
                "score_adjustments": {component: delta}
            }
        """
        self._feedback_history.append(feedback)
        
        action = feedback.get("action")
        adjustments = feedback.get("score_adjustments", {})
        
        if action == "promote":
            # Increase weight of components that contributed to this candidate
            for component, delta in adjustments.items():
                if component in self.weights:
                    self.weights[component] = min(1.0, self.weights[component] + abs(delta))
        
        elif action == "demote":
            # Decrease weight of components that contributed to this candidate
            for component, delta in adjustments.items():
                if component in self.weights:
                    self.weights[component] = max(0.0, self.weights[component] - abs(delta))
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        logger.info(f"Applied feedback: {action}, new weights: {self.weights}")
    
    def rerank_with_feedback(
        self,
        candidates: List[Dict[str, Any]],
        feedback_list: List[Dict[str, Any]],
    ) -> AgentMessage:
        """
        Re-rank candidates after applying multiple feedback items.
        
        Args:
            candidates: Original candidates
            feedback_list: List of feedback items
            
        Returns:
            AgentMessage with re-ranked candidates
        """
        # Apply all feedback
        for feedback in feedback_list:
            self.apply_feedback(feedback)
        
        # Re-rank with updated weights
        return self.process(candidates)


class FeedbackLoop:
    """
    Manages iterative refinement of rankings.
    
    Example:
        >>> loop = FeedbackLoop(ranker)
        >>> candidates = [...]
        >>> ranked = loop.initial_rank(candidates)
        >>> # User provides feedback
        >>> refined = loop.refine({"smiles": "CCO", "action": "promote"})
    """
    
    def __init__(self, ranker: RankerAgent):
        self.ranker = ranker
        self.candidates: List[Dict[str, Any]] = []
        self.iterations: int = 0
        self.history: List[AgentMessage] = []
    
    def initial_rank(
        self,
        candidates: List[Dict[str, Any]],
    ) -> AgentMessage:
        """Initial ranking of candidates."""
        self.candidates = candidates
        self.iterations = 0
        result = self.ranker.process(candidates)
        self.history.append(result)
        return result
    
    def refine(
        self,
        feedback: Dict[str, Any],
    ) -> AgentMessage:
        """
        Refine ranking based on feedback.
        
        Args:
            feedback: Feedback on a specific candidate
            
        Returns:
            Updated ranking
        """
        self.ranker.apply_feedback(feedback)
        self.iterations += 1
        result = self.ranker.process(self.candidates)
        self.history.append(result)
        
        logger.info(f"Refinement iteration {self.iterations}")
        return result
    
    def get_convergence_score(self) -> float:
        """
        Calculate how stable the ranking has become.
        
        Higher score = more stable = converged.
        """
        if len(self.history) < 2:
            return 0.0
        
        # Compare last two rankings
        last = self.history[-1].content
        prev = self.history[-2].content
        
        if not last or not prev:
            return 0.0
        
        # Count rank changes
        last_order = [c["smiles"] for c in last]
        prev_order = [c["smiles"] for c in prev]
        
        matches = sum(1 for i, s in enumerate(last_order) 
                     if i < len(prev_order) and prev_order[i] == s)
        
        return matches / max(len(last_order), len(prev_order))
