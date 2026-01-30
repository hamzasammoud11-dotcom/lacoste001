"""
Reasoning Engine - Intelligence Layer for BioFlow
==================================================

Generates intelligent, hypothesis-driven justifications explaining WHY 
variants/candidates are proposed, not just similarity scores.

This addresses the core "Intelligence Layer" requirement:
- Analyzes chemical/biological properties
- Compares structural features  
- Generates natural language explanations
- Links suggestions to scientific evidence

Unlike simple "similarity = 0.93" scores, this produces explanations like:
"Suggested Variant B because the added hydroxyl group improves solubility 
matching your target profile, and ChEMBL data shows 3 successful binding 
assays with the same scaffold."

Architecture:
1. Feature Extraction: Parse molecular/sequence features
2. Comparative Analysis: Compare query vs candidate features
3. Evidence Retrieval: Find supporting experimental data
4. Hypothesis Generation: Formulate design rationale
5. Justification Synthesis: Natural language explanation
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class JustificationType(Enum):
    """Types of design justifications."""
    STRUCTURAL = "structural"      # Based on molecular structure
    FUNCTIONAL = "functional"      # Based on biological function
    EXPERIMENTAL = "experimental"  # Based on experimental evidence
    LITERATURE = "literature"      # Based on publications
    COMPUTATIONAL = "computational"  # Based on predictions


@dataclass
class EvidenceChain:
    """Chain of evidence supporting a design suggestion."""
    source: str           # pubmed, chembl, experiment, etc.
    source_id: str        # PMID, ChEMBL ID, experiment ID
    claim: str            # What this evidence supports
    strength: float       # 0-1 confidence in this evidence
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "claim": self.claim,
            "strength": self.strength,
            "url": self.url,
        }


@dataclass
class DesignHypothesis:
    """A hypothesis explaining why a design variant might work."""
    hypothesis_type: JustificationType
    title: str            # Short title: "Improved Solubility"
    rationale: str        # Full explanation
    confidence: float     # 0-1 how confident we are
    supporting_evidence: List[EvidenceChain] = field(default_factory=list)
    structural_changes: List[str] = field(default_factory=list)
    predicted_effects: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.hypothesis_type.value,
            "title": self.title,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.supporting_evidence],
            "structural_changes": self.structural_changes,
            "predicted_effects": self.predicted_effects,
        }


@dataclass
class JustificationResult:
    """Complete justification for a design suggestion."""
    summary: str                 # One-line summary
    detailed: str               # Full explanation
    hypotheses: List[DesignHypothesis]
    priority_score: float       # 0-1 overall priority
    priority_factors: Dict[str, float]  # Breakdown of priority
    evidence_strength: float    # 0-1 how well-supported
    actionable_insights: List[str]  # What to do next
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "detailed": self.detailed,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "priority_score": self.priority_score,
            "priority_factors": self.priority_factors,
            "evidence_strength": self.evidence_strength,
            "actionable_insights": self.actionable_insights,
        }


class ReasoningEngine:
    """
    Generates intelligent justifications for design suggestions.
    
    This is the core "Intelligence Layer" that transforms raw similarity
    search results into actionable, evidence-backed recommendations with
    natural language explanations.
    
    Example:
        >>> engine = ReasoningEngine()
        >>> result = engine.generate_justification(
        ...     query_smiles="CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        ...     candidate_smiles="CC(=O)Nc1ccc(O)cc1",  # Paracetamol
        ...     candidate_metadata={"target": "COX", "source": "chembl"},
        ...     similarity_score=0.72
        ... )
        >>> print(result.summary)
        "⭐ HIGH PRIORITY: Retains acetyl group for COX binding while replacing 
        carboxylic acid with phenol - may reduce GI irritation. 3 supporting 
        ChEMBL assays confirm anti-inflammatory activity."
    """
    
    # Functional group patterns for structural analysis
    FUNCTIONAL_GROUPS = {
        "hydroxyl": (r"[OH]", "Hydroxyl group (-OH) improves water solubility and can form H-bonds with target"),
        "carboxylic_acid": (r"C\(=O\)O", "Carboxylic acid enables ionic interactions but may limit membrane permeability"),
        "amine": (r"[NH2]|[NH]", "Amine groups can form salt bridges and improve target binding"),
        "carbonyl": (r"C=O", "Carbonyl enables H-bond acceptance and metabolic oxidation"),
        "aromatic": (r"c1.*c1|C1.*C1", "Aromatic rings contribute to hydrophobic binding and π-stacking"),
        "halogen": (r"\[F\]|\[Cl\]|\[Br\]|\[I\]|F|Cl|Br|I", "Halogen improves metabolic stability and lipophilicity"),
        "nitro": (r"\[N\+\].*\[O-\]", "Nitro group - potential toxicity concern (structural alert)"),
        "sulfonamide": (r"S\(=O\)\(=O\)N", "Sulfonamide enables strong H-bonding to target"),
        "ether": (r"COC", "Ether linkage improves solubility while maintaining lipophilicity"),
        "amide": (r"C\(=O\)N", "Amide bond provides metabolic stability and H-bonding"),
    }
    
    # Structure-Activity Relationship (SAR) templates
    SAR_TEMPLATES = {
        "add_hydroxyl": {
            "effect": "Improved solubility",
            "mechanism": "Hydroxyl groups increase water solubility through H-bonding with water molecules",
            "tradeoff": "May reduce membrane permeability slightly",
        },
        "add_halogen": {
            "effect": "Enhanced metabolic stability",
            "mechanism": "Halogens block metabolic hot spots, reducing Phase I metabolism",
            "tradeoff": "Increases molecular weight and lipophilicity",
        },
        "add_methyl": {
            "effect": "Improved potency (magic methyl effect)",
            "mechanism": "Methyl groups can fill hydrophobic pockets and improve binding",
            "tradeoff": "May alter selectivity profile",
        },
        "ring_expansion": {
            "effect": "Increased rigidity and selectivity",
            "mechanism": "Additional rings reduce conformational flexibility, improving target selectivity",
            "tradeoff": "Increases molecular weight, may reduce solubility",
        },
        "ring_contraction": {
            "effect": "Improved drug-likeness",
            "mechanism": "Smaller scaffolds improve oral bioavailability",
            "tradeoff": "May reduce binding affinity",
        },
        "bioisosteric_replacement": {
            "effect": "Maintained activity with improved properties",
            "mechanism": "Bioisosteres mimic electronic and steric properties while altering ADMET",
            "tradeoff": "Activity may vary - requires experimental validation",
        },
    }

    def __init__(self, use_rdkit: bool = True):
        """Initialize reasoning engine."""
        self.use_rdkit = use_rdkit
        self._rdkit_available = False
        self._check_rdkit()
    
    def _check_rdkit(self):
        """Check if RDKit is available for detailed structural analysis."""
        if self.use_rdkit:
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors, AllChem, Lipinski
                self._rdkit_available = True
                logger.info("RDKit available for structural reasoning")
            except ImportError:
                logger.warning("RDKit not available, using pattern-based reasoning")
    
    def generate_justification(
        self,
        query: str,
        candidate: str,
        candidate_metadata: Dict[str, Any],
        similarity_score: float,
        modality: str = "molecule",
        experimental_data: Optional[List[Dict]] = None,
    ) -> JustificationResult:
        """
        Generate a complete justification for why a candidate is suggested.
        
        This is the main entry point that orchestrates all reasoning components.
        
        Args:
            query: Query SMILES/sequence
            candidate: Candidate SMILES/sequence
            candidate_metadata: Metadata about the candidate
            similarity_score: Vector similarity score
            modality: "molecule", "protein", or "text"
            experimental_data: Optional experimental results
            
        Returns:
            JustificationResult with full explanation
        """
        hypotheses = []
        evidence_chains = []
        priority_factors = {}
        actionable_insights = []
        
        # === 1. STRUCTURAL ANALYSIS ===
        if modality == "molecule":
            structural_hypothesis = self._analyze_molecular_structure(
                query, candidate, similarity_score
            )
            if structural_hypothesis:
                hypotheses.append(structural_hypothesis)
        elif modality == "protein":
            sequence_hypothesis = self._analyze_protein_sequence(
                query, candidate, candidate_metadata, similarity_score
            )
            if sequence_hypothesis:
                hypotheses.append(sequence_hypothesis)
        
        # === 2. EVIDENCE ANALYSIS ===
        evidence_hypothesis = self._analyze_evidence(
            candidate_metadata, experimental_data
        )
        if evidence_hypothesis:
            hypotheses.append(evidence_hypothesis)
            evidence_chains.extend(evidence_hypothesis.supporting_evidence)
        
        # === 3. LITERATURE ANALYSIS ===
        literature_hypothesis = self._analyze_literature(candidate_metadata)
        if literature_hypothesis:
            hypotheses.append(literature_hypothesis)
            evidence_chains.extend(literature_hypothesis.supporting_evidence)
        
        # === 4. PRIORITY SCORING ===
        priority_score, priority_factors = self._calculate_priority(
            hypotheses, evidence_chains, candidate_metadata, similarity_score
        )
        
        # === 5. ACTIONABLE INSIGHTS ===
        actionable_insights = self._generate_actionable_insights(
            hypotheses, candidate_metadata, modality
        )
        
        # === 6. SYNTHESIS ===
        summary, detailed = self._synthesize_justification(
            hypotheses, priority_score, evidence_chains, modality
        )
        
        # Calculate evidence strength
        evidence_strength = sum(e.strength for e in evidence_chains) / max(len(evidence_chains), 1)
        
        return JustificationResult(
            summary=summary,
            detailed=detailed,
            hypotheses=hypotheses,
            priority_score=priority_score,
            priority_factors=priority_factors,
            evidence_strength=evidence_strength,
            actionable_insights=actionable_insights,
        )
    
    def _analyze_molecular_structure(
        self,
        query_smiles: str,
        candidate_smiles: str,
        similarity: float,
    ) -> Optional[DesignHypothesis]:
        """Analyze structural differences between query and candidate molecules."""
        changes = []
        predicted_effects = {}
        rationale_parts = []
        
        # Extract functional groups
        query_groups = self._extract_functional_groups(query_smiles)
        candidate_groups = self._extract_functional_groups(candidate_smiles)
        
        # Find differences
        added_groups = set(candidate_groups.keys()) - set(query_groups.keys())
        removed_groups = set(query_groups.keys()) - set(candidate_groups.keys())
        
        # Generate SAR insights for added groups
        for group in added_groups:
            _, description = self.FUNCTIONAL_GROUPS.get(group, ("", "Unknown functional group"))
            changes.append(f"Added {group.replace('_', ' ')}")
            rationale_parts.append(description)
            
            # Predict effect based on known SAR
            if group == "hydroxyl":
                predicted_effects["solubility"] = "increased"
                predicted_effects["permeability"] = "potentially decreased"
            elif group == "halogen":
                predicted_effects["metabolic_stability"] = "increased"
                predicted_effects["lipophilicity"] = "increased"
            elif group == "amine":
                predicted_effects["target_binding"] = "potentially improved"
                predicted_effects["basicity"] = "increased"
        
        # Generate SAR insights for removed groups
        for group in removed_groups:
            changes.append(f"Removed {group.replace('_', ' ')}")
            if group == "nitro":
                predicted_effects["toxicity_risk"] = "reduced"
                rationale_parts.append("Removal of nitro group reduces potential toxicity")
            elif group == "carboxylic_acid":
                predicted_effects["permeability"] = "potentially increased"
                rationale_parts.append("Removal of carboxylic acid may improve membrane permeability")
        
        # Analyze size/complexity changes
        size_diff = len(candidate_smiles) - len(query_smiles)
        if size_diff > 20:
            changes.append("Scaffold expansion")
            predicted_effects["selectivity"] = "potentially improved"
            rationale_parts.append("Larger scaffold may access additional binding pockets")
        elif size_diff < -20:
            changes.append("Scaffold simplification")
            predicted_effects["drug_likeness"] = "improved"
            rationale_parts.append("Simpler scaffold may improve oral bioavailability")
        
        # RDKit-based detailed analysis if available
        if self._rdkit_available:
            rdkit_insights = self._rdkit_structural_analysis(query_smiles, candidate_smiles)
            if rdkit_insights:
                changes.extend(rdkit_insights.get("changes", []))
                rationale_parts.extend(rdkit_insights.get("rationale", []))
                predicted_effects.update(rdkit_insights.get("effects", {}))
        
        if not changes:
            return None
        
        # Calculate confidence based on number of insights
        confidence = min(0.3 + 0.15 * len(changes), 0.9)
        
        # Build rationale
        rationale = " ".join(rationale_parts) if rationale_parts else \
            f"Structural analog with {similarity:.0%} similarity - modifications may alter ADMET profile"
        
        return DesignHypothesis(
            hypothesis_type=JustificationType.STRUCTURAL,
            title="Structural Modification",
            rationale=rationale,
            confidence=confidence,
            structural_changes=changes,
            predicted_effects=predicted_effects,
        )
    
    def _analyze_protein_sequence(
        self,
        query_seq: str,
        candidate_seq: str,
        metadata: Dict[str, Any],
        similarity: float,
    ) -> Optional[DesignHypothesis]:
        """Analyze protein sequence relationships."""
        rationale_parts = []
        changes = []
        predicted_effects = {}
        
        # Check organism
        organism = metadata.get("organism", "")
        if organism:
            if organism.lower() != "homo sapiens":
                rationale_parts.append(f"Ortholog from {organism} may reveal evolutionary constraints")
                changes.append(f"Cross-species comparison ({organism})")
                predicted_effects["conservation_insight"] = "functional regions likely conserved"
            else:
                rationale_parts.append("Human protein with potentially similar function")
        
        # Check function
        function = metadata.get("function", "")
        if function:
            # Truncate if too long
            func_short = function[:200] + "..." if len(function) > 200 else function
            rationale_parts.append(f"Functional annotation: {func_short}")
            predicted_effects["functional_relevance"] = "aligned with query function"
        
        # Check for PDB structure
        pdb_ids = metadata.get("pdb_ids", [])
        if pdb_ids:
            changes.append("3D structure available")
            rationale_parts.append(f"Structural data ({len(pdb_ids)} PDB entries) enables binding site analysis")
            predicted_effects["structure_based_design"] = "feasible"
        
        # Check similarity range
        if similarity > 0.9:
            rationale_parts.append("Very high sequence identity suggests conserved binding site")
        elif similarity > 0.7:
            rationale_parts.append("High similarity with potential variation in selectivity-determining regions")
        elif similarity > 0.5:
            rationale_parts.append("Moderate similarity - may reveal alternative conformational states")
        
        if not rationale_parts:
            return None
        
        return DesignHypothesis(
            hypothesis_type=JustificationType.FUNCTIONAL,
            title="Protein Sequence Analysis",
            rationale=" ".join(rationale_parts),
            confidence=min(0.4 + similarity * 0.4, 0.9),
            structural_changes=changes,
            predicted_effects=predicted_effects,
        )
    
    def _analyze_evidence(
        self,
        metadata: Dict[str, Any],
        experimental_data: Optional[List[Dict]] = None,
    ) -> Optional[DesignHypothesis]:
        """Analyze experimental evidence supporting the candidate."""
        evidence_chains = []
        rationale_parts = []
        
        source = metadata.get("source", "").lower()
        
        # ChEMBL evidence
        if source == "chembl":
            chembl_id = metadata.get("chembl_id", metadata.get("source_id", ""))
            assay_count = metadata.get("assay_count", 0)
            activity_type = metadata.get("activity_type", "")
            activity_value = metadata.get("activity_value", None)
            
            if assay_count > 0:
                evidence_chains.append(EvidenceChain(
                    source="chembl",
                    source_id=chembl_id,
                    claim=f"{assay_count} bioassay results validate biological activity",
                    strength=min(0.3 + 0.1 * assay_count, 0.9),
                    url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/"
                ))
                rationale_parts.append(f"Experimentally validated: {assay_count} ChEMBL assays")
            
            if activity_type and activity_value is not None:
                evidence_chains.append(EvidenceChain(
                    source="chembl",
                    source_id=chembl_id,
                    claim=f"Measured {activity_type} = {activity_value}",
                    strength=0.8,
                ))
                rationale_parts.append(f"Quantitative activity: {activity_type} = {activity_value}")
        
        # Experiment evidence
        elif source == "experiment":
            outcome = metadata.get("outcome", "")
            exp_type = metadata.get("experiment_type", "")
            measurements = metadata.get("measurements", [])
            quality = metadata.get("quality_score", 0)
            exp_id = metadata.get("experiment_id", "")
            
            if outcome == "success":
                evidence_chains.append(EvidenceChain(
                    source="experiment",
                    source_id=exp_id,
                    claim=f"Successful {exp_type} - learn from what worked",
                    strength=0.85 * quality if quality else 0.7,
                ))
                rationale_parts.append(f"✅ Successful experiment: {exp_type}")
                
                # Add measurement details
                if measurements:
                    for m in measurements[:2]:  # Top 2 measurements
                        m_name = m.get("name", "Value")
                        m_val = m.get("value", "N/A")
                        m_unit = m.get("unit", "")
                        rationale_parts.append(f"📊 {m_name}: {m_val} {m_unit}")
            
            elif outcome == "failure":
                evidence_chains.append(EvidenceChain(
                    source="experiment",
                    source_id=exp_id,
                    claim=f"Failed {exp_type} - understanding why informs better design",
                    strength=0.5,
                ))
                rationale_parts.append(f"⚠️ Failed experiment provides negative control")
            
            # Lab notes
            notes = metadata.get("notes", "")
            if notes:
                excerpt = notes[:150] + "..." if len(notes) > 150 else notes
                rationale_parts.append(f"📝 Lab note insight: \"{excerpt}\"")
            
            # Protocol
            protocol = metadata.get("protocol", "")
            if protocol:
                rationale_parts.append(f"🧪 Protocol available for reproduction")
        
        # Include additional experimental data if provided
        if experimental_data:
            for exp in experimental_data[:3]:  # Top 3 experiments
                exp_id = exp.get("experiment_id", "unknown")
                outcome = exp.get("outcome", "")
                if outcome == "success":
                    evidence_chains.append(EvidenceChain(
                        source="experiment",
                        source_id=exp_id,
                        claim=f"Related successful experiment: {exp.get('title', 'N/A')}",
                        strength=0.7,
                    ))
        
        if not evidence_chains:
            return None
        
        # Calculate overall confidence
        avg_strength = sum(e.strength for e in evidence_chains) / len(evidence_chains)
        
        return DesignHypothesis(
            hypothesis_type=JustificationType.EXPERIMENTAL,
            title="Experimental Validation",
            rationale=" | ".join(rationale_parts),
            confidence=avg_strength,
            supporting_evidence=evidence_chains,
        )
    
    def _analyze_literature(
        self,
        metadata: Dict[str, Any],
    ) -> Optional[DesignHypothesis]:
        """Analyze literature evidence."""
        evidence_chains = []
        rationale_parts = []
        
        source = metadata.get("source", "").lower()
        
        if source == "pubmed":
            pmid = metadata.get("pmid", metadata.get("source_id", ""))
            title = metadata.get("title", "")
            abstract = metadata.get("abstract", "")
            doi = metadata.get("doi", "")
            
            if title:
                evidence_chains.append(EvidenceChain(
                    source="pubmed",
                    source_id=pmid,
                    claim=f"Literature support: {title[:100]}...",
                    strength=0.7,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                ))
                rationale_parts.append(f"📄 Literature: \"{title[:80]}...\"")
            
            if abstract:
                # Extract key findings from abstract
                key_phrases = self._extract_key_findings(abstract)
                if key_phrases:
                    rationale_parts.append(f"Key finding: {key_phrases[0]}")
            
            if doi:
                evidence_chains.append(EvidenceChain(
                    source="doi",
                    source_id=doi,
                    claim="Peer-reviewed publication",
                    strength=0.75,
                    url=f"https://doi.org/{doi}"
                ))
        
        if not evidence_chains:
            return None
        
        avg_strength = sum(e.strength for e in evidence_chains) / len(evidence_chains)
        
        return DesignHypothesis(
            hypothesis_type=JustificationType.LITERATURE,
            title="Literature Support",
            rationale=" | ".join(rationale_parts),
            confidence=avg_strength,
            supporting_evidence=evidence_chains,
        )
    
    def _calculate_priority(
        self,
        hypotheses: List[DesignHypothesis],
        evidence: List[EvidenceChain],
        metadata: Dict[str, Any],
        similarity: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate priority score (different from similarity!).
        
        Priority is based on:
        - Evidence strength (highest weight)
        - Hypothesis confidence
        - Experimental validation
        - Actionability
        """
        factors = {}
        
        # Evidence factor (30%)
        if evidence:
            evidence_score = sum(e.strength for e in evidence) / len(evidence)
            factors["evidence_strength"] = evidence_score * 0.3
        else:
            factors["evidence_strength"] = 0.0
        
        # Hypothesis confidence (25%)
        if hypotheses:
            hyp_score = sum(h.confidence for h in hypotheses) / len(hypotheses)
            factors["hypothesis_confidence"] = hyp_score * 0.25
        else:
            factors["hypothesis_confidence"] = 0.0
        
        # Experimental validation (25%)
        source = metadata.get("source", "").lower()
        outcome = metadata.get("outcome", "")
        if source == "experiment" and outcome == "success":
            factors["experimental_validation"] = 0.25
        elif source == "chembl" and metadata.get("assay_count", 0) > 0:
            factors["experimental_validation"] = 0.2
        elif source == "pubmed":
            factors["experimental_validation"] = 0.15
        else:
            factors["experimental_validation"] = 0.05
        
        # Optimal similarity range (20%) - not too similar, not too different
        if 0.6 <= similarity <= 0.85:
            factors["similarity_sweet_spot"] = 0.2
        elif similarity > 0.85:
            factors["similarity_sweet_spot"] = 0.1  # Too similar = redundant
        elif similarity > 0.4:
            factors["similarity_sweet_spot"] = 0.15
        else:
            factors["similarity_sweet_spot"] = 0.05  # Too different
        
        priority_score = sum(factors.values())
        
        return priority_score, factors
    
    def _generate_actionable_insights(
        self,
        hypotheses: List[DesignHypothesis],
        metadata: Dict[str, Any],
        modality: str,
    ) -> List[str]:
        """Generate actionable next steps based on analysis."""
        insights = []
        
        # Structural insights
        for h in hypotheses:
            if h.hypothesis_type == JustificationType.STRUCTURAL:
                if h.predicted_effects.get("solubility") == "increased":
                    insights.append("🧪 Validate solubility improvement with HPLC assay")
                if h.predicted_effects.get("metabolic_stability") == "increased":
                    insights.append("🧪 Test metabolic stability in liver microsomes")
                if h.predicted_effects.get("toxicity_risk") == "reduced":
                    insights.append("✅ Reduced toxicity risk - consider advancing to in vivo")
            
            if h.hypothesis_type == JustificationType.FUNCTIONAL:
                if h.predicted_effects.get("structure_based_design") == "feasible":
                    insights.append("🔬 Use PDB structure for docking studies")
        
        # Evidence-based insights
        if metadata.get("source") == "experiment":
            if metadata.get("protocol"):
                insights.append("📋 Protocol available - can reproduce conditions")
            if metadata.get("notes"):
                insights.append("📝 Review lab notes for optimization hints")
        
        if metadata.get("source") == "chembl":
            insights.append("📊 Cross-reference ChEMBL for selectivity data")
        
        # Default insights based on modality
        if not insights:
            if modality == "molecule":
                insights.append("🔄 Consider synthesizing and testing binding affinity")
            elif modality == "protein":
                insights.append("🧬 Align sequences to identify key residues")
        
        return insights[:5]  # Max 5 insights
    
    def _synthesize_justification(
        self,
        hypotheses: List[DesignHypothesis],
        priority_score: float,
        evidence: List[EvidenceChain],
        modality: str,
    ) -> Tuple[str, str]:
        """Synthesize final summary and detailed justification."""
        
        # Build summary
        summary_parts = []
        
        # Priority indicator
        if priority_score >= 0.7:
            summary_parts.append("⭐ HIGH PRIORITY:")
        elif priority_score >= 0.5:
            summary_parts.append("PROMISING:")
        else:
            summary_parts.append("Exploratory:")
        
        # Main hypothesis
        if hypotheses:
            main_hyp = max(hypotheses, key=lambda h: h.confidence)
            summary_parts.append(main_hyp.title)
            
            # Add key rationale
            rationale_short = main_hyp.rationale[:150] + "..." if len(main_hyp.rationale) > 150 else main_hyp.rationale
            summary_parts.append(f"- {rationale_short}")
        
        # Evidence count
        if evidence:
            summary_parts.append(f"[{len(evidence)} supporting evidence sources]")
        
        summary = " ".join(summary_parts)
        
        # Build detailed explanation
        detailed_parts = []
        
        for i, h in enumerate(hypotheses, 1):
            detailed_parts.append(f"\n**{i}. {h.title}** (confidence: {h.confidence:.0%})")
            detailed_parts.append(h.rationale)
            
            if h.structural_changes:
                detailed_parts.append(f"  Changes: {', '.join(h.structural_changes)}")
            
            if h.predicted_effects:
                effects = [f"{k}: {v}" for k, v in h.predicted_effects.items()]
                detailed_parts.append(f"  Predicted effects: {'; '.join(effects)}")
            
            if h.supporting_evidence:
                for e in h.supporting_evidence:
                    detailed_parts.append(f"  📎 {e.source.upper()}: {e.claim}")
        
        detailed = "\n".join(detailed_parts)
        
        return summary, detailed
    
    def _extract_functional_groups(self, smiles: str) -> Dict[str, bool]:
        """Extract functional groups from SMILES using patterns."""
        found = {}
        for group_name, (pattern, _) in self.FUNCTIONAL_GROUPS.items():
            if re.search(pattern, smiles, re.IGNORECASE):
                found[group_name] = True
        return found
    
    def _extract_key_findings(self, abstract: str) -> List[str]:
        """Extract key findings from abstract text."""
        findings = []
        
        # Look for conclusion markers
        conclusion_patterns = [
            r"(?:conclude|found|demonstrate|show|reveal)[sd]?\s+that\s+([^.]+)",
            r"(?:results?\s+)?indicate[sd]?\s+that\s+([^.]+)",
            r"(?:significant(?:ly)?|effective(?:ly)?)\s+([^.]+)",
        ]
        
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            findings.extend(matches[:1])  # Take first match per pattern
        
        return findings[:3]  # Max 3 findings
    
    def _rdkit_structural_analysis(
        self,
        query_smiles: str,
        candidate_smiles: str,
    ) -> Optional[Dict[str, Any]]:
        """Perform detailed RDKit-based structural analysis."""
        if not self._rdkit_available:
            return None
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski, AllChem
            from rdkit import DataStructs
            
            query_mol = Chem.MolFromSmiles(query_smiles)
            cand_mol = Chem.MolFromSmiles(candidate_smiles)
            
            if not query_mol or not cand_mol:
                return None
            
            changes = []
            rationale = []
            effects = {}
            
            # Compare molecular properties
            query_mw = Descriptors.MolWt(query_mol)
            cand_mw = Descriptors.MolWt(cand_mol)
            mw_diff = cand_mw - query_mw
            
            if abs(mw_diff) > 50:
                if mw_diff > 0:
                    changes.append(f"Increased MW (+{mw_diff:.0f} Da)")
                    rationale.append("Larger molecule may have improved binding but reduced permeability")
                else:
                    changes.append(f"Decreased MW ({mw_diff:.0f} Da)")
                    rationale.append("Smaller molecule may have better oral bioavailability")
            
            # Compare LogP
            query_logp = Descriptors.MolLogP(query_mol)
            cand_logp = Descriptors.MolLogP(cand_mol)
            logp_diff = cand_logp - query_logp
            
            if abs(logp_diff) > 1:
                if logp_diff > 0:
                    changes.append(f"Increased lipophilicity (ΔLogP = +{logp_diff:.1f})")
                    effects["lipophilicity"] = "increased"
                    rationale.append("Higher LogP may improve membrane permeability but reduce solubility")
                else:
                    changes.append(f"Decreased lipophilicity (ΔLogP = {logp_diff:.1f})")
                    effects["lipophilicity"] = "decreased"
                    rationale.append("Lower LogP may improve aqueous solubility")
            
            # Compare H-bond donors/acceptors
            query_hbd = Lipinski.NumHDonors(query_mol)
            cand_hbd = Lipinski.NumHDonors(cand_mol)
            query_hba = Lipinski.NumHAcceptors(query_mol)
            cand_hba = Lipinski.NumHAcceptors(cand_mol)
            
            if cand_hbd > query_hbd:
                changes.append(f"Added H-bond donors (+{cand_hbd - query_hbd})")
                rationale.append("Additional H-bond donors may improve target binding")
            if cand_hba > query_hba:
                changes.append(f"Added H-bond acceptors (+{cand_hba - query_hba})")
                rationale.append("Additional H-bond acceptors may improve solubility")
            
            # Tanimoto similarity for structural diversity context
            fp1 = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, 2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2, 2048)
            tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
            
            if tanimoto < 0.5:
                changes.append("Structurally diverse scaffold")
                rationale.append("Distinct scaffold explores new chemical space")
                effects["novelty"] = "high"
            elif tanimoto > 0.8:
                changes.append("Close structural analog")
                rationale.append("Minor modifications may fine-tune properties")
                effects["novelty"] = "low"
            
            return {
                "changes": changes,
                "rationale": rationale,
                "effects": effects,
            }
            
        except Exception as e:
            logger.warning(f"RDKit analysis failed: {e}")
            return None
