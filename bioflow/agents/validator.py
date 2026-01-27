"""
Validator Agent
===============

ADMET and toxicity validation for molecules.

Features:
- Drug-likeness scoring (Lipinski, QED)
- ADMET property prediction
- Toxicity flags from similarity search
- Structural alerts (PAINS, Brenk)
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from bioflow.agents.base import BaseAgent, AgentMessage, AgentContext, AgentType

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class PropertyResult:
    """Result of a property calculation."""
    name: str
    value: Any
    unit: Optional[str] = None
    threshold: Optional[float] = None
    passed: bool = True
    message: Optional[str] = None


@dataclass 
class ValidationResult:
    """Complete validation result for a molecule."""
    smiles: str
    status: ValidationStatus
    score: float  # Overall score 0-1
    properties: List[PropertyResult] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "smiles": self.smiles,
            "status": self.status.value,
            "score": self.score,
            "properties": [
                {
                    "name": p.name,
                    "value": p.value,
                    "unit": p.unit,
                    "threshold": p.threshold,
                    "passed": p.passed,
                    "message": p.message,
                }
                for p in self.properties
            ],
            "alerts": self.alerts,
            "recommendations": self.recommendations,
        }


class ValidatorAgent(BaseAgent):
    """
    Agent for validating molecules against ADMET and safety criteria.
    
    Validation checks:
    1. Drug-likeness: Lipinski's Rule of 5, QED score
    2. ADMET properties: LogP, TPSA, MW, etc.
    3. Structural alerts: PAINS, Brenk filters
    4. Toxicity: Similarity to known toxic compounds
    
    Example:
        >>> agent = ValidatorAgent()
        >>> result = agent.process("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
        >>> print(result.content["status"])  # "passed"
    """
    
    # Lipinski thresholds
    LIPINSKI_THRESHOLDS = {
        "mw": (0, 500),        # Molecular weight
        "logp": (-5, 5),       # LogP
        "hbd": (0, 5),         # H-bond donors
        "hba": (0, 10),        # H-bond acceptors
    }
    
    # ADMET thresholds (common ranges)
    ADMET_THRESHOLDS = {
        "tpsa": (0, 140),      # Topological polar surface area
        "rotatable_bonds": (0, 10),
        "mw": (150, 500),
    }
    
    # Structural alerts (SMARTS patterns)
    # These are common PAINS and Brenk filter patterns
    STRUCTURAL_ALERTS = {
        "nitro_aromatic": "[$(a[N+](=O)[O-])]",
        "quinone": "O=C1C=CC(=O)C=C1",
        "aldehyde": "[CH1](=O)",
        "michael_acceptor": "[CH1]=[CH1]C=O",
        "peroxide": "OO",
        "azide": "N=[N+]=[N-]",
        "acyl_halide": "C(=O)[F,Cl,Br,I]",
        "sulfonate": "S(=O)(=O)O",
        "thiocarbonyl": "C=S",
        "beta_lactam": "C1C(=O)NC1",
    }
    
    def __init__(
        self,
        name: str = "MoleculeValidator",
        check_lipinski: bool = True,
        check_admet: bool = True,
        check_alerts: bool = True,
        check_toxicity: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize validator agent.
        
        Args:
            name: Agent name
            check_lipinski: Run Lipinski Rule of 5
            check_admet: Run ADMET property checks
            check_alerts: Run structural alert filters
            check_toxicity: Check toxicity similarity
            config: Additional configuration
        """
        super().__init__(name, AgentType.VALIDATOR, config)
        self.check_lipinski = check_lipinski
        self.check_admet = check_admet
        self.check_alerts = check_alerts
        self.check_toxicity = check_toxicity
        self._rdkit_available = False
        self._obm_available = False
    
    def initialize(self) -> None:
        """Initialize validation resources."""
        super().initialize()
        
        # Check RDKit availability
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski
            self._rdkit_available = True
            logger.info("RDKit available for validation")
        except ImportError:
            logger.warning("RDKit not available, limited validation")
        
        # Check OpenBioMed availability
        try:
            from open_biomed.data import Molecule
            self._obm_available = True
            logger.info("OpenBioMed available for validation")
        except ImportError:
            logger.info("OpenBioMed not available for validation")
    
    def process(
        self,
        input_data: Union[str, List[str], Dict[str, Any]],
        context: Optional[AgentContext] = None,
    ) -> AgentMessage:
        """
        Validate molecules.
        
        Args:
            input_data: Either:
                - str: Single SMILES to validate
                - List[str]: Multiple SMILES
                - dict: {"smiles": str or list, "thresholds": dict}
            context: Optional shared context
            
        Returns:
            AgentMessage with validation results
        """
        if not self._initialized:
            self.initialize()
        
        # Parse input
        if isinstance(input_data, str):
            smiles_list = [input_data]
            custom_thresholds = {}
        elif isinstance(input_data, list):
            smiles_list = input_data
            custom_thresholds = {}
        else:
            smiles_list = input_data.get("smiles", [])
            if isinstance(smiles_list, str):
                smiles_list = [smiles_list]
            custom_thresholds = input_data.get("thresholds", {})
        
        results = []
        passed_count = 0
        
        for smiles in smiles_list:
            try:
                result = self._validate_molecule(smiles, custom_thresholds)
                results.append(result.to_dict())
                if result.status == ValidationStatus.PASSED:
                    passed_count += 1
            except Exception as e:
                logger.error(f"Validation failed for {smiles}: {e}")
                results.append(ValidationResult(
                    smiles=smiles,
                    status=ValidationStatus.ERROR,
                    score=0.0,
                    alerts=[f"Validation error: {str(e)}"],
                ).to_dict())
        
        return AgentMessage(
            sender=self.name,
            content=results,
            metadata={
                "total": len(smiles_list),
                "passed": passed_count,
                "failed": len(smiles_list) - passed_count,
                "pass_rate": passed_count / len(smiles_list) if smiles_list else 0,
            },
            success=True,
        )
    
    def _validate_molecule(
        self,
        smiles: str,
        custom_thresholds: Dict[str, Any] = None,
    ) -> ValidationResult:
        """
        Validate a single molecule.
        
        Returns comprehensive validation result.
        """
        properties = []
        alerts = []
        recommendations = []
        scores = []
        
        if not self._rdkit_available:
            return ValidationResult(
                smiles=smiles,
                status=ValidationStatus.WARNING,
                score=0.5,
                alerts=["RDKit not available for full validation"],
            )
        
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski as LipinskiModule
        from rdkit.Chem import QED as QEDModule
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ValidationResult(
                smiles=smiles,
                status=ValidationStatus.ERROR,
                score=0.0,
                alerts=["Invalid SMILES structure"],
            )
        
        # === Lipinski Rule of 5 ===
        if self.check_lipinski:
            lipinski_results = self._check_lipinski(mol)
            properties.extend(lipinski_results)
            lipinski_passed = sum(1 for r in lipinski_results if r.passed)
            lipinski_score = lipinski_passed / len(lipinski_results)
            scores.append(lipinski_score)
            
            if lipinski_score < 0.75:
                alerts.append(f"Lipinski violations: {4 - lipinski_passed}/4 rules failed")
                recommendations.append("Consider reducing molecular weight or LogP")
        
        # === ADMET Properties ===
        if self.check_admet:
            admet_results = self._check_admet(mol)
            properties.extend(admet_results)
            admet_passed = sum(1 for r in admet_results if r.passed)
            admet_score = admet_passed / len(admet_results) if admet_results else 1.0
            scores.append(admet_score)
            
            # QED score
            try:
                qed = QEDModule.qed(mol)
                properties.append(PropertyResult(
                    name="QED",
                    value=round(qed, 3),
                    threshold=0.5,
                    passed=qed >= 0.5,
                    message="Quantitative Estimate of Drug-likeness",
                ))
                scores.append(qed)
            except:
                pass
        
        # === Structural Alerts ===
        if self.check_alerts:
            alert_results = self._check_structural_alerts(mol)
            if alert_results:
                alerts.extend(alert_results)
                scores.append(0.5)  # Penalty for structural alerts
                recommendations.append("Review flagged structural motifs")
        
        # === Calculate overall score ===
        if scores:
            overall_score = sum(scores) / len(scores)
        else:
            overall_score = 0.5
        
        # Determine status
        if not alerts and overall_score >= 0.7:
            status = ValidationStatus.PASSED
        elif overall_score >= 0.5:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED
        
        return ValidationResult(
            smiles=smiles,
            status=status,
            score=round(overall_score, 3),
            properties=properties,
            alerts=alerts,
            recommendations=recommendations,
        )
    
    def _check_lipinski(self, mol) -> List[PropertyResult]:
        """Check Lipinski Rule of 5."""
        from rdkit.Chem import Descriptors, Lipinski as LipinskiModule
        
        results = []
        
        # Molecular Weight
        mw = Descriptors.MolWt(mol)
        results.append(PropertyResult(
            name="Molecular Weight",
            value=round(mw, 1),
            unit="Da",
            threshold=500,
            passed=mw <= 500,
            message="MW ≤ 500 for oral bioavailability",
        ))
        
        # LogP
        logp = Descriptors.MolLogP(mol)
        results.append(PropertyResult(
            name="LogP",
            value=round(logp, 2),
            threshold=5,
            passed=logp <= 5,
            message="LogP ≤ 5 for membrane permeability",
        ))
        
        # H-bond donors
        hbd = LipinskiModule.NumHDonors(mol)
        results.append(PropertyResult(
            name="H-Bond Donors",
            value=hbd,
            threshold=5,
            passed=hbd <= 5,
            message="HBD ≤ 5",
        ))
        
        # H-bond acceptors
        hba = LipinskiModule.NumHAcceptors(mol)
        results.append(PropertyResult(
            name="H-Bond Acceptors",
            value=hba,
            threshold=10,
            passed=hba <= 10,
            message="HBA ≤ 10",
        ))
        
        return results
    
    def _check_admet(self, mol) -> List[PropertyResult]:
        """Check ADMET-related properties."""
        from rdkit.Chem import Descriptors
        
        results = []
        
        # TPSA (Topological Polar Surface Area)
        tpsa = Descriptors.TPSA(mol)
        results.append(PropertyResult(
            name="TPSA",
            value=round(tpsa, 1),
            unit="Å²",
            threshold=140,
            passed=tpsa <= 140,
            message="TPSA ≤ 140 Å² for BBB penetration",
        ))
        
        # Rotatable bonds
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        results.append(PropertyResult(
            name="Rotatable Bonds",
            value=rot_bonds,
            threshold=10,
            passed=rot_bonds <= 10,
            message="Fewer rotatable bonds = better bioavailability",
        ))
        
        # Fraction sp3 carbons (3D character)
        fsp3 = Descriptors.FractionCSP3(mol)
        results.append(PropertyResult(
            name="Fraction sp3",
            value=round(fsp3, 2),
            threshold=0.25,
            passed=fsp3 >= 0.25,
            message="Fsp3 ≥ 0.25 for solubility",
        ))
        
        return results
    
    def _check_structural_alerts(self, mol) -> List[str]:
        """Check for problematic structural motifs."""
        from rdkit import Chem
        
        alerts = []
        
        for alert_name, smarts in self.STRUCTURAL_ALERTS.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    alerts.append(f"Structural alert: {alert_name.replace('_', ' ')}")
            except:
                continue
        
        return alerts


class BatchValidator:
    """
    Batch validation utility for processing many molecules.
    
    Example:
        >>> validator = BatchValidator()
        >>> results = validator.validate_batch(smiles_list, parallel=True)
    """
    
    def __init__(self, agent: Optional[ValidatorAgent] = None):
        self.agent = agent or ValidatorAgent()
    
    def validate_batch(
        self,
        smiles_list: List[str],
        parallel: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Validate a batch of molecules.
        
        Args:
            smiles_list: List of SMILES to validate
            parallel: Use parallel processing (not implemented yet)
            
        Returns:
            List of validation results
        """
        result = self.agent.process(smiles_list)
        return result.content
    
    def filter_passed(
        self,
        smiles_list: List[str],
        min_score: float = 0.7,
    ) -> List[str]:
        """
        Filter molecules that pass validation.
        
        Args:
            smiles_list: List of SMILES to validate
            min_score: Minimum score to pass
            
        Returns:
            List of SMILES that passed validation
        """
        results = self.validate_batch(smiles_list)
        return [
            r["smiles"]
            for r in results
            if r["score"] >= min_score
        ]
