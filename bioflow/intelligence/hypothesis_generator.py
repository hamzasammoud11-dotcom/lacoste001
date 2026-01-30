"""
Hypothesis Generator - Design Intelligence
============================================

Generates scientific design hypotheses explaining WHY modifications
might improve a compound or protein.

This addresses the "Design Assistance" requirement (D.4):
- Propose 'close but diverse' variants
- Justify with scientific reasoning
- Link to Structure-Activity Relationships (SAR)

Examples of hypotheses:
- "Adding a fluorine at the ortho position may improve metabolic stability"
- "The L858R mutation creates a larger binding pocket favoring this scaffold"
- "Hydroxyl→methyl replacement increases LogP for better membrane permeability"
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModificationType(Enum):
    """Types of molecular modifications."""
    SUBSTITUTION = "substitution"
    ADDITION = "addition"
    DELETION = "deletion"
    RING_MODIFICATION = "ring_modification"
    BIOISOSTERE = "bioisostere"
    SCAFFOLD_HOP = "scaffold_hop"


@dataclass
class StructuralHypothesis:
    """Hypothesis based on structural modifications."""
    modification_type: ModificationType
    description: str        # What changed
    rationale: str         # Why this might help
    predicted_effect: str  # Expected outcome
    confidence: float      # 0-1
    sar_basis: Optional[str] = None  # SAR knowledge supporting this
    references: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.modification_type.value,
            "description": self.description,
            "rationale": self.rationale,
            "predicted_effect": self.predicted_effect,
            "confidence": self.confidence,
            "sar_basis": self.sar_basis,
            "references": self.references,
        }


@dataclass
class FunctionalHypothesis:
    """Hypothesis based on functional/biological reasoning."""
    function_type: str      # binding, selectivity, admet, etc.
    description: str
    mechanism: str          # How this works biologically
    experimental_support: List[str]  # Supporting experiments
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_type": self.function_type,
            "description": self.description,
            "mechanism": self.mechanism,
            "experimental_support": self.experimental_support,
            "confidence": self.confidence,
        }


class HypothesisGenerator:
    """
    Generates scientific design hypotheses for molecular modifications.
    
    Uses:
    - SAR (Structure-Activity Relationship) knowledge
    - Bioisostere databases
    - Medicinal chemistry heuristics
    - Experimental outcome patterns
    
    Example:
        >>> gen = HypothesisGenerator()
        >>> hypotheses = gen.generate_hypotheses(
        ...     query="CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        ...     candidate="CC(=O)Oc1ccc(F)cc1C(=O)O",  # Fluoro-aspirin
        ... )
        >>> print(hypotheses[0].rationale)
        "Fluorine substitution at para position may block CYP metabolism..."
    """
    
    # Common bioisosteric replacements
    BIOISOSTERES = {
        "carboxylic_acid": [
            ("tetrazole", "Maintains acidity while improving permeability"),
            ("sulfonamide", "Provides H-bonding with different geometry"),
            ("phosphonic_acid", "Stronger acid, higher metabolic stability"),
        ],
        "hydroxyl": [
            ("thiol", "Maintains H-bonding, different redox properties"),
            ("amino", "Basic instead of neutral, forms salts"),
            ("fluorine", "Similar size, removes H-bonding"),
        ],
        "amide": [
            ("ester", "Similar geometry, different metabolic fate"),
            ("sulfonamide", "Bioisostere with different H-bonding"),
            ("reverse_amide", "Inverted H-bonding pattern"),
        ],
        "phenyl": [
            ("pyridine", "Adds H-bond acceptor, changes basicity"),
            ("thienyl", "Similar size, different electronics"),
            ("cyclopropyl", "Reduces aromaticity, improves metabolic stability"),
        ],
    }
    
    # SAR rules from medicinal chemistry
    SAR_RULES = {
        "ortho_fluorine": {
            "effect": "Metabolic stability",
            "mechanism": "Blocks CYP450 hydroxylation at adjacent positions",
            "applicability": "Phenyl rings prone to oxidation",
        },
        "para_halogen": {
            "effect": "Potency enhancement",
            "mechanism": "Halogen bonding with target carbonyl groups",
            "applicability": "Binding pockets with accessible carbonyls",
        },
        "magic_methyl": {
            "effect": "Potency boost (2-10x)",
            "mechanism": "Fills hydrophobic micro-pocket, favorable entropy",
            "applicability": "Targets with small hydrophobic pockets",
        },
        "rigidification": {
            "effect": "Selectivity improvement",
            "mechanism": "Reduces conformational flexibility, favors bioactive conformation",
            "applicability": "Flexible molecules with known bioactive conformation",
        },
        "solubility_tail": {
            "effect": "Improved solubility",
            "mechanism": "Polyethylene glycol or morpholine groups increase hydrophilicity",
            "applicability": "Compounds with poor aqueous solubility",
        },
    }
    
    def __init__(self, use_rdkit: bool = True):
        """Initialize hypothesis generator."""
        self._rdkit_available = False
        if use_rdkit:
            self._check_rdkit()
    
    def _check_rdkit(self):
        """Check RDKit availability."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, rdFMCS
            self._rdkit_available = True
        except ImportError:
            logger.warning("RDKit not available, using pattern-based hypothesis generation")
    
    def generate_hypotheses(
        self,
        query: str,
        candidate: str,
        query_modality: str = "molecule",
        metadata: Optional[Dict[str, Any]] = None,
        max_hypotheses: int = 5,
    ) -> List[StructuralHypothesis]:
        """
        Generate design hypotheses comparing query and candidate.
        
        Args:
            query: Query SMILES/sequence
            candidate: Candidate SMILES/sequence
            query_modality: "molecule" or "protein"
            metadata: Additional context
            max_hypotheses: Maximum hypotheses to generate
            
        Returns:
            List of StructuralHypothesis
        """
        hypotheses = []
        metadata = metadata or {}
        
        if query_modality == "molecule":
            # Structural comparison
            hypotheses.extend(self._generate_molecular_hypotheses(query, candidate, metadata))
            
            # Bioisostere detection
            hypotheses.extend(self._detect_bioisosteric_replacements(query, candidate))
            
            # SAR-based hypotheses
            hypotheses.extend(self._apply_sar_rules(query, candidate, metadata))
            
        elif query_modality == "protein":
            hypotheses.extend(self._generate_protein_hypotheses(query, candidate, metadata))
        
        # Sort by confidence and return top N
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:max_hypotheses]
    
    def _generate_molecular_hypotheses(
        self,
        query: str,
        candidate: str,
        metadata: Dict[str, Any],
    ) -> List[StructuralHypothesis]:
        """Generate hypotheses based on molecular differences."""
        hypotheses = []
        
        # Size comparison
        size_diff = len(candidate) - len(query)
        if abs(size_diff) > 10:
            if size_diff > 0:
                hypotheses.append(StructuralHypothesis(
                    modification_type=ModificationType.ADDITION,
                    description="Scaffold expansion",
                    rationale="Larger scaffold may access additional binding interactions",
                    predicted_effect="Potentially improved affinity or selectivity",
                    confidence=0.5,
                    sar_basis="Lead optimization often involves systematic scaffold growth",
                ))
            else:
                hypotheses.append(StructuralHypothesis(
                    modification_type=ModificationType.DELETION,
                    description="Scaffold simplification",
                    rationale="Smaller molecules often have better oral bioavailability",
                    predicted_effect="Improved drug-likeness and ADMET",
                    confidence=0.5,
                    sar_basis="Fragment-based approaches start with small, efficient binders",
                ))
        
        # Halogen detection
        query_halogens = sum(1 for h in ['F', 'Cl', 'Br', 'I'] if h in query)
        cand_halogens = sum(1 for h in ['F', 'Cl', 'Br', 'I'] if h in candidate)
        
        if cand_halogens > query_halogens:
            added_halogen = None
            for h in ['F', 'Cl', 'Br', 'I']:
                if candidate.count(h) > query.count(h):
                    added_halogen = h
                    break
            
            halogen_effects = {
                'F': "Fluorine is small, electronegative, and blocks metabolic oxidation",
                'Cl': "Chlorine adds lipophilicity and can form halogen bonds",
                'Br': "Bromine provides strong halogen bonding capability",
                'I': "Iodine is a synthetic handle for further modifications",
            }
            
            hypotheses.append(StructuralHypothesis(
                modification_type=ModificationType.SUBSTITUTION,
                description=f"Halogenation ({added_halogen})",
                rationale=halogen_effects.get(added_halogen, "Halogen introduction"),
                predicted_effect="Improved metabolic stability and/or binding affinity",
                confidence=0.7,
                sar_basis="Halogenation is a common medicinal chemistry strategy",
            ))
        
        # Nitrogen detection (amines, heterocycles)
        query_n = query.count('N') + query.count('n')
        cand_n = candidate.count('N') + candidate.count('n')
        
        if cand_n > query_n:
            hypotheses.append(StructuralHypothesis(
                modification_type=ModificationType.ADDITION,
                description="Nitrogen introduction",
                rationale="Nitrogen atoms provide H-bond donors/acceptors and affect pKa",
                predicted_effect="May improve target binding and aqueous solubility",
                confidence=0.6,
                sar_basis="Aza-analogs often show improved properties",
            ))
        
        # Oxygen detection (ethers, alcohols)
        query_o = query.count('O') + query.count('o')
        cand_o = candidate.count('O') + candidate.count('o')
        
        if cand_o > query_o:
            hypotheses.append(StructuralHypothesis(
                modification_type=ModificationType.ADDITION,
                description="Oxygen-containing group added",
                rationale="Oxygen groups improve water solubility via H-bonding",
                predicted_effect="Improved solubility, potentially reduced LogP",
                confidence=0.6,
                sar_basis="Hydroxyl and ether groups commonly improve ADMET",
            ))
        
        # Target-based hypothesis if available
        target = metadata.get("target", "")
        if target:
            hypotheses.append(StructuralHypothesis(
                modification_type=ModificationType.SUBSTITUTION,
                description=f"Optimized for {target}",
                rationale=f"This scaffold shows binding to {target} in reference data",
                predicted_effect=f"Maintained or improved {target} binding",
                confidence=0.65,
                sar_basis=f"Established SAR for {target} binding",
            ))
        
        return hypotheses
    
    def _detect_bioisosteric_replacements(
        self,
        query: str,
        candidate: str,
    ) -> List[StructuralHypothesis]:
        """Detect bioisosteric replacements between molecules."""
        hypotheses = []
        
        # Check for common bioisosteric patterns
        bioisostere_patterns = [
            ("C(=O)O", "c1nnn[nH]1", "Carboxylic acid → Tetrazole"),
            ("C(=O)O", "S(=O)(=O)N", "Carboxylic acid → Sulfonamide"),
            ("O", "S", "Oxygen → Sulfur (thio analog)"),
            ("N", "O", "Nitrogen → Oxygen"),
            ("c1ccccc1", "c1ccncc1", "Phenyl → Pyridine"),
            ("c1ccccc1", "c1ccsc1", "Phenyl → Thienyl"),
            ("C(=O)N", "C(=O)O", "Amide → Ester"),
        ]
        
        for query_pattern, cand_pattern, description in bioisostere_patterns:
            if query_pattern in query and cand_pattern in candidate:
                # Find corresponding bioisostere rationale
                for group, replacements in self.BIOISOSTERES.items():
                    for replacement, rationale in replacements:
                        if replacement in description.lower():
                            hypotheses.append(StructuralHypothesis(
                                modification_type=ModificationType.BIOISOSTERE,
                                description=description,
                                rationale=rationale,
                                predicted_effect="Maintained binding with altered properties",
                                confidence=0.75,
                                sar_basis=f"Classical bioisosteric replacement: {group}",
                            ))
                            break
        
        return hypotheses
    
    def _apply_sar_rules(
        self,
        query: str,
        candidate: str,
        metadata: Dict[str, Any],
    ) -> List[StructuralHypothesis]:
        """Apply SAR rules to explain modifications."""
        hypotheses = []
        
        # Check for fluorine at aromatic positions
        if ('c' in query or 'C1' in query) and 'F' in candidate and 'F' not in query:
            rule = self.SAR_RULES["ortho_fluorine"]
            hypotheses.append(StructuralHypothesis(
                modification_type=ModificationType.SUBSTITUTION,
                description="Aromatic fluorination",
                rationale=rule["mechanism"],
                predicted_effect=rule["effect"],
                confidence=0.7,
                sar_basis=f"Applicable when: {rule['applicability']}",
            ))
        
        # Check for methyl addition (magic methyl)
        query_methyl = query.count('C') - query.count('c')  # Rough heuristic
        cand_methyl = candidate.count('C') - candidate.count('c')
        
        if cand_methyl > query_methyl and len(candidate) - len(query) < 5:
            rule = self.SAR_RULES["magic_methyl"]
            hypotheses.append(StructuralHypothesis(
                modification_type=ModificationType.ADDITION,
                description="Methyl group addition (magic methyl effect)",
                rationale=rule["mechanism"],
                predicted_effect=rule["effect"],
                confidence=0.65,
                sar_basis=f"Applicable when: {rule['applicability']}",
            ))
        
        # Check for rigidification (ring formation)
        query_rings = query.count('1') + query.count('2')  # Ring closures
        cand_rings = candidate.count('1') + candidate.count('2')
        
        if cand_rings > query_rings:
            rule = self.SAR_RULES["rigidification"]
            hypotheses.append(StructuralHypothesis(
                modification_type=ModificationType.RING_MODIFICATION,
                description="Ring addition (rigidification)",
                rationale=rule["mechanism"],
                predicted_effect=rule["effect"],
                confidence=0.6,
                sar_basis=f"Applicable when: {rule['applicability']}",
            ))
        
        return hypotheses
    
    def _generate_protein_hypotheses(
        self,
        query: str,
        candidate: str,
        metadata: Dict[str, Any],
    ) -> List[StructuralHypothesis]:
        """Generate hypotheses for protein sequences."""
        hypotheses = []
        
        # Organism-based hypothesis
        organism = metadata.get("organism", "")
        if organism and organism.lower() != "homo sapiens":
            hypotheses.append(StructuralHypothesis(
                modification_type=ModificationType.SUBSTITUTION,
                description=f"Ortholog from {organism}",
                rationale="Cross-species comparison reveals evolutionary constraints on function",
                predicted_effect="Conserved residues likely critical for activity",
                confidence=0.65,
                sar_basis="Ortholog analysis in drug discovery",
            ))
        
        # Function-based hypothesis
        function = metadata.get("function", "")
        if function:
            hypotheses.append(StructuralHypothesis(
                modification_type=ModificationType.SUBSTITUTION,
                description="Functionally related protein",
                rationale=f"Similar function ({function[:100]}...) suggests conserved binding site",
                predicted_effect="May reveal alternative drug targets or selectivity determinants",
                confidence=0.6,
            ))
        
        # Structure availability
        pdb_ids = metadata.get("pdb_ids", [])
        if pdb_ids:
            hypotheses.append(StructuralHypothesis(
                modification_type=ModificationType.SUBSTITUTION,
                description="Structure available for comparison",
                rationale="3D structure enables binding site analysis and docking",
                predicted_effect="Structure-based design feasible",
                confidence=0.7,
                references=[f"PDB: {pid}" for pid in pdb_ids[:3]],
            ))
        
        return hypotheses
    
    def generate_functional_hypotheses(
        self,
        candidate: Dict[str, Any],
        experimental_data: Optional[List[Dict]] = None,
    ) -> List[FunctionalHypothesis]:
        """
        Generate functional/biological hypotheses from experimental patterns.
        
        This learns from experimental outcomes to suggest WHY something might work.
        """
        hypotheses = []
        metadata = candidate.get("metadata", {})
        
        # Target-based hypothesis
        target = metadata.get("target", "")
        if target:
            hypotheses.append(FunctionalHypothesis(
                function_type="binding",
                description=f"Target engagement with {target}",
                mechanism=f"Structural features enable binding to {target} active site",
                experimental_support=[],
                confidence=0.6,
            ))
        
        # Activity-based hypothesis
        outcome = metadata.get("outcome", "")
        if outcome == "success":
            hypotheses.append(FunctionalHypothesis(
                function_type="activity",
                description="Validated biological activity",
                mechanism="Experimental success indicates functional engagement",
                experimental_support=[metadata.get("experiment_id", "")],
                confidence=0.8,
            ))
        
        # Learn from experimental context
        if experimental_data:
            successful = [e for e in experimental_data if e.get("outcome") == "success"]
            if len(successful) >= 2:
                hypotheses.append(FunctionalHypothesis(
                    function_type="pattern",
                    description="Pattern from successful experiments",
                    mechanism=f"Similar conditions succeeded in {len(successful)} experiments",
                    experimental_support=[e.get("experiment_id", "") for e in successful[:3]],
                    confidence=0.7,
                ))
        
        return hypotheses
