"""
BioFlow Intelligence Layer
===========================

Provides intelligent reasoning, justification, and design assistance 
beyond simple similarity search.

This module addresses the "Intelligence Layer" requirement from Use Case 4:
- Design hypothesis generation with scientific rationale
- Prioritization based on predicted biological activity (not just similarity)
- Evidence-based reasoning linking suggestions to experiments
- LLM-style explanations for cross-modal connections

Components:
- ReasoningEngine: Generate natural language justifications
- PriorityRanker: Re-rank by predicted efficacy
- HypothesisGenerator: Propose design rationale
- EvidenceAnalyzer: Link suggestions to experimental evidence
"""

from bioflow.intelligence.reasoning_engine import (
    ReasoningEngine,
    DesignHypothesis,
    JustificationResult,
    EvidenceChain,
)
from bioflow.intelligence.priority_ranker import (
    PriorityRanker,
    PriorityScore,
    RankingCriteria,
)
from bioflow.intelligence.hypothesis_generator import (
    HypothesisGenerator,
    StructuralHypothesis,
    FunctionalHypothesis,
)

__all__ = [
    "ReasoningEngine",
    "DesignHypothesis",
    "JustificationResult",
    "EvidenceChain",
    "PriorityRanker",
    "PriorityScore",
    "RankingCriteria",
    "HypothesisGenerator",
    "StructuralHypothesis",
    "FunctionalHypothesis",
]
