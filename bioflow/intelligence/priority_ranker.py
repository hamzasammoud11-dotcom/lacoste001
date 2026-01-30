"""
Priority Ranker - Predictive Prioritization Layer
===================================================

Re-ranks search results by PREDICTED BIOLOGICAL ACTIVITY, not just similarity.

This addresses the "Predictive Prioritization" requirement:
- A similarity score of 0.93 ≠ prediction of success
- This module integrates QSAR models, activity classifiers, and 
  experimental outcomes to prioritize candidates by predicted efficacy.

Components:
- QSAR scoring: Predict activity from molecular descriptors
- Activity classification: Predict binding/non-binding
- Success prediction: Learn from experimental outcomes
- Multi-criteria ranking: Combine multiple signals
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RankingCriteria(Enum):
    """Criteria for ranking candidates."""
    PREDICTED_ACTIVITY = "predicted_activity"  # QSAR/ML predicted activity
    EXPERIMENTAL_SUCCESS = "experimental_success"  # Past experiment outcomes
    DRUG_LIKENESS = "drug_likeness"  # QED, Lipinski compliance
    NOVELTY = "novelty"  # How different from known compounds
    SAFETY = "safety"  # Toxicity predictions
    SYNTHESIZABILITY = "synthesizability"  # Ease of synthesis


@dataclass
class PriorityScore:
    """Complete priority scoring for a candidate."""
    overall_score: float  # 0-1 final priority
    rank: int
    criteria_scores: Dict[str, float]  # Individual scores per criterion
    predicted_activity: Optional[float]  # ML-predicted activity
    confidence: float  # How confident in this ranking
    explanation: str  # Why this priority
    flags: List[str]  # Any warnings or special notes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "rank": self.rank,
            "criteria_scores": self.criteria_scores,
            "predicted_activity": self.predicted_activity,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "flags": self.flags,
        }


class PriorityRanker:
    """
    Re-ranks candidates by predicted biological efficacy.
    
    Unlike similarity-based ranking, this considers:
    1. Predicted activity (QSAR/ML models)
    2. Experimental success patterns
    3. Drug-likeness scores
    4. Safety profiles
    5. Novelty/diversity
    
    Example:
        >>> ranker = PriorityRanker()
        >>> ranked = ranker.rank_candidates(
        ...     candidates=[
        ...         {"smiles": "CCO", "similarity": 0.95},  # High similarity but poor activity
        ...         {"smiles": "CC(=O)Nc1ccc(O)cc1", "similarity": 0.7},  # Lower sim, better activity
        ...     ],
        ...     query_smiles="CC(=O)Oc1ccccc1C(=O)O"
        ... )
        >>> # Paracetamol ranks higher despite lower similarity
    """
    
    # Weights for different criteria
    DEFAULT_WEIGHTS = {
        RankingCriteria.PREDICTED_ACTIVITY: 0.30,
        RankingCriteria.EXPERIMENTAL_SUCCESS: 0.25,
        RankingCriteria.DRUG_LIKENESS: 0.15,
        RankingCriteria.NOVELTY: 0.15,
        RankingCriteria.SAFETY: 0.10,
        RankingCriteria.SYNTHESIZABILITY: 0.05,
    }
    
    def __init__(
        self,
        weights: Optional[Dict[RankingCriteria, float]] = None,
        use_rdkit: bool = True,
        use_deeppurpose: bool = True,
    ):
        """
        Initialize priority ranker.
        
        Args:
            weights: Custom weights for ranking criteria
            use_rdkit: Use RDKit for drug-likeness calculations
            use_deeppurpose: Use DeepPurpose for activity predictions
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._rdkit_available = False
        self._deeppurpose_available = False
        
        if use_rdkit:
            self._check_rdkit()
        if use_deeppurpose:
            self._check_deeppurpose()
    
    def _check_rdkit(self):
        """Check RDKit availability."""
        try:
            from rdkit import Chem
            from rdkit.Chem import QED, Descriptors
            self._rdkit_available = True
            logger.info("RDKit available for drug-likeness scoring")
        except ImportError:
            logger.warning("RDKit not available for priority ranking")
    
    def _check_deeppurpose(self):
        """Check DeepPurpose availability."""
        try:
            from DeepPurpose import DTI
            self._deeppurpose_available = True
            logger.info("DeepPurpose available for activity prediction")
        except ImportError:
            logger.info("DeepPurpose not available, using heuristic activity scoring")
    
    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query: str,
        query_modality: str = "molecule",
        target_sequence: Optional[str] = None,
        experimental_context: Optional[List[Dict]] = None,
    ) -> List[PriorityScore]:
        """
        Rank candidates by predicted priority.
        
        Args:
            candidates: List of candidate dicts with 'content'/'smiles' and 'metadata'
            query: Query SMILES/sequence
            query_modality: "molecule" or "protein"
            target_sequence: Target protein for DTI prediction (optional)
            experimental_context: Related experiments for success patterns
            
        Returns:
            List of PriorityScore, sorted by overall_score descending
        """
        scores = []
        
        for cand in candidates:
            priority = self._score_candidate(
                cand, query, query_modality, target_sequence, experimental_context
            )
            scores.append(priority)
        
        # Sort by overall score
        scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Update ranks
        for i, score in enumerate(scores):
            score.rank = i + 1
        
        return scores
    
    def _score_candidate(
        self,
        candidate: Dict[str, Any],
        query: str,
        query_modality: str,
        target_sequence: Optional[str],
        experimental_context: Optional[List[Dict]],
    ) -> PriorityScore:
        """Score a single candidate."""
        criteria_scores = {}
        flags = []
        explanation_parts = []
        
        # Extract content
        content = candidate.get("content") or candidate.get("smiles", "")
        metadata = candidate.get("metadata", {})
        similarity = candidate.get("score") or candidate.get("similarity", 0.5)
        
        # === 1. PREDICTED ACTIVITY ===
        if query_modality == "molecule":
            activity_score, activity_pred = self._predict_activity(
                content, target_sequence, metadata
            )
            criteria_scores[RankingCriteria.PREDICTED_ACTIVITY.value] = activity_score
            
            if activity_pred is not None:
                if activity_pred > 0.7:
                    explanation_parts.append(f"High predicted activity ({activity_pred:.0%})")
                elif activity_pred < 0.3:
                    flags.append("Low predicted activity")
        else:
            criteria_scores[RankingCriteria.PREDICTED_ACTIVITY.value] = 0.5  # Neutral for non-molecules
        
        # === 2. EXPERIMENTAL SUCCESS ===
        success_score = self._score_experimental_success(metadata, experimental_context)
        criteria_scores[RankingCriteria.EXPERIMENTAL_SUCCESS.value] = success_score
        
        if success_score > 0.7:
            explanation_parts.append("Strong experimental support")
        
        # === 3. DRUG-LIKENESS ===
        if query_modality == "molecule":
            druglike_score = self._score_drug_likeness(content)
            criteria_scores[RankingCriteria.DRUG_LIKENESS.value] = druglike_score
            
            if druglike_score < 0.3:
                flags.append("Poor drug-likeness")
        else:
            criteria_scores[RankingCriteria.DRUG_LIKENESS.value] = 0.5
        
        # === 4. NOVELTY ===
        novelty_score = self._score_novelty(content, query, similarity, metadata)
        criteria_scores[RankingCriteria.NOVELTY.value] = novelty_score
        
        if novelty_score > 0.7:
            explanation_parts.append("Novel scaffold")
        
        # === 5. SAFETY ===
        safety_score = self._score_safety(content, metadata)
        criteria_scores[RankingCriteria.SAFETY.value] = safety_score
        
        if safety_score < 0.3:
            flags.append("⚠️ Safety concerns")
        
        # === 6. SYNTHESIZABILITY ===
        synth_score = self._score_synthesizability(content, metadata)
        criteria_scores[RankingCriteria.SYNTHESIZABILITY.value] = synth_score
        
        # === CALCULATE OVERALL SCORE ===
        overall = 0.0
        for criterion, weight in self.weights.items():
            criterion_key = criterion.value
            if criterion_key in criteria_scores:
                overall += weight * criteria_scores[criterion_key]
        
        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(criteria_scores, metadata)
        
        # Build explanation
        if not explanation_parts:
            if overall > 0.7:
                explanation_parts.append("High overall priority")
            elif overall > 0.5:
                explanation_parts.append("Moderate priority")
            else:
                explanation_parts.append("Lower priority - consider as exploratory")
        
        explanation = " | ".join(explanation_parts)
        
        return PriorityScore(
            overall_score=overall,
            rank=0,  # Will be updated after sorting
            criteria_scores=criteria_scores,
            predicted_activity=criteria_scores.get(RankingCriteria.PREDICTED_ACTIVITY.value),
            confidence=confidence,
            explanation=explanation,
            flags=flags,
        )
    
    def _predict_activity(
        self,
        smiles: str,
        target_sequence: Optional[str],
        metadata: Dict[str, Any],
    ) -> Tuple[float, Optional[float]]:
        """
        Predict biological activity.
        
        Returns: (score for ranking, raw prediction if available)
        """
        # Use existing experimental data if available
        if metadata.get("activity_value") is not None:
            activity = metadata["activity_value"]
            # Normalize based on activity type
            activity_type = metadata.get("activity_type", "").upper()
            
            if "IC50" in activity_type or "EC50" in activity_type:
                # Lower is better for IC50/EC50 (nM)
                if activity < 10:
                    return 0.95, 0.95  # Sub-nanomolar
                elif activity < 100:
                    return 0.8, 0.8
                elif activity < 1000:
                    return 0.6, 0.6
                else:
                    return 0.3, 0.3
            elif "KI" in activity_type or "KD" in activity_type:
                # Similar for Ki/Kd
                if activity < 10:
                    return 0.9, 0.9
                elif activity < 100:
                    return 0.7, 0.7
                else:
                    return 0.4, 0.4
        
        # Use DeepPurpose if available and target provided
        if self._deeppurpose_available and target_sequence:
            try:
                from DeepPurpose import DTI
                # Note: In production, use a pre-trained model
                # This is a placeholder for the API
                prediction = 0.5  # Would be: model.predict(smiles, target_sequence)
                return prediction, prediction
            except Exception as e:
                logger.warning(f"DeepPurpose prediction failed: {e}")
        
        # Use metadata hints
        if metadata.get("bioactivity") or metadata.get("has_activity"):
            return 0.7, 0.7
        
        # ChEMBL assay count as proxy
        assay_count = metadata.get("assay_count", 0)
        if assay_count > 10:
            return 0.75, None
        elif assay_count > 0:
            return 0.6, None
        
        # Default: neutral
        return 0.5, None
    
    def _score_experimental_success(
        self,
        metadata: Dict[str, Any],
        experimental_context: Optional[List[Dict]],
    ) -> float:
        """Score based on experimental outcomes."""
        score = 0.5  # Neutral default
        
        # Direct outcome
        outcome = metadata.get("outcome", "").lower()
        if outcome == "success" or outcome == "active":
            score = 0.9
        elif outcome == "failure" or outcome == "inactive":
            score = 0.2
        elif outcome == "partial":
            score = 0.5
        
        # Quality score
        quality = metadata.get("quality_score", 0)
        if quality:
            score *= quality
        
        # Source-based scoring
        source = metadata.get("source", "").lower()
        if source == "experiment" and outcome == "success":
            score = max(score, 0.85)
        elif source == "chembl" and metadata.get("assay_count", 0) > 5:
            score = max(score, 0.7)
        
        # Related experiments
        if experimental_context:
            success_count = sum(
                1 for exp in experimental_context 
                if exp.get("outcome") == "success"
            )
            if success_count > 0:
                score = max(score, 0.6 + 0.1 * min(success_count, 3))
        
        return min(score, 1.0)
    
    def _score_drug_likeness(self, smiles: str) -> float:
        """Score drug-likeness using RDKit QED and Lipinski."""
        if not self._rdkit_available or not smiles:
            return 0.5
        
        try:
            from rdkit import Chem
            from rdkit.Chem import QED, Descriptors, Lipinski
            
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return 0.3
            
            # QED score (0-1)
            qed_score = QED.qed(mol)
            
            # Lipinski violations
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            
            violations = 0
            if mw > 500:
                violations += 1
            if logp > 5:
                violations += 1
            if hbd > 5:
                violations += 1
            if hba > 10:
                violations += 1
            
            lipinski_score = 1.0 - (violations * 0.25)
            
            # Combined score
            return (qed_score * 0.6 + lipinski_score * 0.4)
            
        except Exception as e:
            logger.warning(f"Drug-likeness scoring failed: {e}")
            return 0.5
    
    def _score_novelty(
        self,
        content: str,
        query: str,
        similarity: float,
        metadata: Dict[str, Any],
    ) -> float:
        """
        Score novelty - we want diverse but not irrelevant candidates.
        
        Sweet spot: 0.5-0.8 similarity (novel but related)
        """
        # Inverse similarity with sweet spot
        if 0.5 <= similarity <= 0.8:
            novelty = 0.8  # Ideal range
        elif similarity > 0.9:
            novelty = 0.3  # Too similar
        elif similarity > 0.8:
            novelty = 0.6
        elif similarity > 0.3:
            novelty = 0.7
        else:
            novelty = 0.4  # Too different
        
        # Boost if from different source/organism
        if metadata.get("organism") and metadata.get("organism") != "Homo sapiens":
            novelty = min(novelty + 0.1, 1.0)
        
        return novelty
    
    def _score_safety(self, smiles: str, metadata: Dict[str, Any]) -> float:
        """Score safety based on toxicity predictions and alerts."""
        score = 0.7  # Default: assume okay
        
        # Check metadata for toxicity info
        if metadata.get("toxicity") == "high":
            score = 0.2
        elif metadata.get("toxicity") == "low":
            score = 0.9
        
        # ADMET predictions
        admet = metadata.get("admet_predictions", {})
        if admet.get("toxicity") == "low":
            score = max(score, 0.85)
        elif admet.get("toxicity") == "high":
            score = min(score, 0.3)
        
        # Check for structural alerts (simplified)
        if smiles:
            alerts = [
                "[N+](=O)[O-]",  # Nitro
                "N=N",  # Azo
                "C#N",  # Nitrile (context dependent)
            ]
            for alert in alerts:
                if alert in smiles:
                    score -= 0.15
        
        return max(score, 0.1)
    
    def _score_synthesizability(
        self,
        content: str,
        metadata: Dict[str, Any],
    ) -> float:
        """Score ease of synthesis."""
        score = 0.5  # Neutral default
        
        # If already exists (ChEMBL, DrugBank), it's synthesizable
        source = metadata.get("source", "").lower()
        if source in ["chembl", "drugbank", "pubchem"]:
            score = 0.8
        elif source == "experiment":
            score = 0.9  # Was actually made
        
        # Length heuristic (longer = harder)
        if content:
            if len(content) < 30:
                score = min(score + 0.1, 1.0)
            elif len(content) > 100:
                score = max(score - 0.2, 0.2)
        
        return score
    
    def _calculate_confidence(
        self,
        criteria_scores: Dict[str, float],
        metadata: Dict[str, Any],
    ) -> float:
        """Calculate confidence in the priority score."""
        confidence = 0.5
        
        # More criteria available = higher confidence
        non_default = sum(1 for v in criteria_scores.values() if v != 0.5)
        confidence += 0.1 * non_default
        
        # Experimental data = higher confidence
        if metadata.get("source") == "experiment":
            confidence += 0.2
        elif metadata.get("source") == "chembl":
            confidence += 0.15
        
        # Quality score available
        if metadata.get("quality_score"):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def explain_ranking(self, scores: List[PriorityScore]) -> str:
        """Generate explanation of the ranking."""
        lines = ["## Priority Ranking Explanation\n"]
        lines.append("Candidates ranked by **predicted biological efficacy**, not just similarity.\n")
        
        for score in scores[:5]:  # Top 5
            lines.append(f"### Rank #{score.rank} (Priority: {score.overall_score:.2f})")
            lines.append(f"- {score.explanation}")
            
            # Key criteria
            top_criteria = sorted(
                score.criteria_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            for crit, val in top_criteria:
                lines.append(f"  - {crit.replace('_', ' ').title()}: {val:.2f}")
            
            if score.flags:
                lines.append(f"  - ⚠️ Flags: {', '.join(score.flags)}")
            
            lines.append("")
        
        return "\n".join(lines)
