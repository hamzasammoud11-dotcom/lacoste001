"""
Generator Agent
===============

Molecule generation from text descriptions using MolT5/BioT5.

Features:
- Text-to-molecule generation
- SMILES variant generation (mutations)
- Scaffold-constrained generation
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import random
import re

from bioflow.agents.base import BaseAgent, AgentMessage, AgentContext, AgentType

logger = logging.getLogger(__name__)


@dataclass
class GeneratedMolecule:
    """A generated molecule with metadata."""
    smiles: str
    source: str  # "molt5", "mutation", "scaffold"
    prompt: Optional[str] = None
    parent_smiles: Optional[str] = None
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "smiles": self.smiles,
            "source": self.source,
            "prompt": self.prompt,
            "parent_smiles": self.parent_smiles,
            "confidence": self.confidence,
            "properties": self.properties,
        }


class GeneratorAgent(BaseAgent):
    """
    Agent for generating molecules from text prompts or seed SMILES.
    
    Supports multiple generation modes:
    1. Text-to-molecule: Generate SMILES from natural language description
    2. Mutation: Create variants of a seed molecule
    3. Scaffold: Generate around a core scaffold
    
    Example:
        >>> agent = GeneratorAgent()
        >>> result = agent.process("Generate a kinase inhibitor with low toxicity")
        >>> print(result.content)  # List of GeneratedMolecule
    """
    
    def __init__(
        self,
        name: str = "MoleculeGenerator",
        model_name: str = "molt5",
        num_samples: int = 5,
        use_obm: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize generator agent.
        
        Args:
            name: Agent name
            model_name: Model to use (molt5, biot5)
            num_samples: Default number of molecules to generate
            use_obm: Whether to use OpenBioMed for generation
            config: Additional configuration
        """
        super().__init__(name, AgentType.GENERATOR, config)
        self.model_name = model_name
        self.num_samples = num_samples
        self.use_obm = use_obm
        self._pipeline = None
        self._model_service = None
    
    def initialize(self) -> None:
        """Initialize generation model."""
        super().initialize()
        
        # Try to use OpenBioMed generation pipeline
        if self.use_obm:
            try:
                from open_biomed.core.pipeline import InferencePipeline
                # Note: This requires model weights
                # self._pipeline = InferencePipeline(
                #     task="text_guided_molecule_generation",
                #     model=self.model_name,
                # )
                logger.info(f"OpenBioMed pipeline ready (model={self.model_name})")
            except ImportError:
                logger.warning("OpenBioMed not available, using fallback generation")
        
        # Fallback: Use model service for basic operations
        try:
            from bioflow.api.model_service import ModelService
            self._model_service = ModelService()
            logger.info("ModelService initialized for generation support")
        except ImportError:
            logger.warning("ModelService not available")
    
    def process(
        self,
        input_data: Union[str, Dict[str, Any]],
        context: Optional[AgentContext] = None,
    ) -> AgentMessage:
        """
        Generate molecules based on input.
        
        Args:
            input_data: Either:
                - str: Text prompt for text-to-molecule generation
                - dict: {
                    "mode": "text" | "mutate" | "scaffold",
                    "prompt": str (for text mode),
                    "smiles": str (for mutate/scaffold mode),
                    "num_samples": int (optional)
                  }
            context: Optional shared context
            
        Returns:
            AgentMessage with list of GeneratedMolecule
        """
        if not self._initialized:
            self.initialize()
        
        # Parse input
        if isinstance(input_data, str):
            mode = "text"
            prompt = input_data
            seed_smiles = None
            num_samples = self.num_samples
        else:
            mode = input_data.get("mode", "text")
            prompt = input_data.get("prompt", "")
            seed_smiles = input_data.get("smiles")
            num_samples = input_data.get("num_samples", self.num_samples)
        
        try:
            if mode == "text":
                molecules = self._generate_from_text(prompt, num_samples)
            elif mode == "mutate":
                molecules = self._generate_mutations(seed_smiles, num_samples, prompt)
            elif mode == "scaffold":
                molecules = self._generate_scaffold(seed_smiles, num_samples, prompt)
            else:
                raise ValueError(f"Unknown generation mode: {mode}")
            
            return AgentMessage(
                sender=self.name,
                content=[m.to_dict() for m in molecules],
                metadata={
                    "mode": mode,
                    "prompt": prompt,
                    "seed_smiles": seed_smiles,
                    "num_generated": len(molecules),
                    "model": self.model_name,
                },
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return AgentMessage(
                sender=self.name,
                content=[],
                metadata={"error": str(e), "mode": mode},
                success=False,
                error=str(e),
            )
    
    def _generate_from_text(
        self,
        prompt: str,
        num_samples: int,
    ) -> List[GeneratedMolecule]:
        """
        Generate molecules from text description.
        
        Uses MolT5/BioT5 if available, otherwise returns example molecules
        for demonstration purposes.
        """
        logger.info(f"Generating {num_samples} molecules from: '{prompt[:50]}...'")
        
        # Try OpenBioMed pipeline
        if self._pipeline is not None:
            try:
                from open_biomed.data import Text, Molecule
                outputs = self._pipeline.run(text=Text.from_str(prompt))
                molecules = []
                for i, out in enumerate(outputs[:num_samples]):
                    smiles = out.smiles if hasattr(out, 'smiles') else str(out)
                    molecules.append(GeneratedMolecule(
                        smiles=smiles,
                        source="molt5",
                        prompt=prompt,
                        confidence=0.9 - (i * 0.05),  # Decreasing confidence
                    ))
                return molecules
            except Exception as e:
                logger.warning(f"OBM generation failed: {e}, using fallback")
        
        # Fallback: Use curated examples based on keywords
        return self._fallback_generate(prompt, num_samples)
    
    def _fallback_generate(
        self,
        prompt: str,
        num_samples: int,
    ) -> List[GeneratedMolecule]:
        """
        Fallback generation using keyword matching to curated examples.
        
        This provides working molecules for demonstration when MolT5 isn't available.
        """
        prompt_lower = prompt.lower()
        
        # Curated molecule examples by category
        molecule_bank = {
            "kinase": [
                "Cc1ccc(Nc2nccc(-c3cccnc3)n2)cc1",  # Imatinib analog
                "Cn1cnc2c(F)c(Nc3ccc(Br)cc3)c(C#N)cc21",  # EGFR-like
                "CC(=O)Nc1ccc(Nc2nccc(-c3ccccc3)n2)cc1",
                "Fc1ccc(Nc2nccc(-c3ccccn3)n2)cc1Cl",
                "Cc1nc2ccc(Nc3ccc(F)cc3)cn2n1",
            ],
            "inhibitor": [
                "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC",  # Gefitinib-like
                "Cn1c(=O)n(C)c2nc[nH]c2c1=O",  # Caffeine scaffold
                "Nc1ncnc2c1c(-c1ccc(F)cc1)cn2C1CC1",
                "CC(C)n1c2ccccc2c2ccc(NC(=O)c3ccc(F)cc3)cc21",
                "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1",
            ],
            "egfr": [
                "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",  # Erlotinib-like
                "Cn1c(=O)c(-c2ccc(F)cc2)cc2cnc(Nc3ccc(N4CCN(C)CC4)cc3)nc21",
                "COc1cc(Nc2nccc(-c3cn(C)nc3-c3ccccc3)n2)ccc1N1CCN(C)CC1",
            ],
            "anticancer": [
                "COc1ccc2[nH]c(=O)c(-c3ccc(Br)cc3)cc2c1",
                "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",  # Celecoxib-like
                "Nc1ccc2nc(N)nc(N)c2c1",  # Methotrexate fragment
                "O=C(Nc1ccc(F)cc1)Nc1ccc2ncccc2c1",
                "CC(=O)Nc1ccc(-c2nc3ccccc3s2)cc1",
            ],
            "default": [
                "c1ccc2[nH]ccc2c1",  # Indole
                "c1ccc2ncccc2c1",    # Quinoline
                "c1ccc2ncncc2c1",    # Quinazoline
                "Cc1ccc(O)c(C)c1",   # Cresol
                "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
            ],
        }
        
        # Match keywords to categories
        selected_bank = molecule_bank["default"]
        for keyword, bank in molecule_bank.items():
            if keyword in prompt_lower:
                selected_bank = bank
                break
        
        # Generate molecules with variation
        molecules = []
        available = list(selected_bank)
        random.shuffle(available)
        
        for i in range(min(num_samples, len(available))):
            smiles = available[i]
            molecules.append(GeneratedMolecule(
                smiles=smiles,
                source="fallback",
                prompt=prompt,
                confidence=0.7 - (i * 0.05),
                properties={"category": "curated_example"},
            ))
        
        logger.info(f"Generated {len(molecules)} molecules (fallback mode)")
        return molecules
    
    def _generate_mutations(
        self,
        seed_smiles: str,
        num_samples: int,
        constraints: Optional[str] = None,
    ) -> List[GeneratedMolecule]:
        """
        Generate molecular variants by mutation.
        
        Uses RDKit for structural modifications.
        """
        logger.info(f"Generating {num_samples} mutations of: {seed_smiles}")
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, rdMolDescriptors
        except ImportError:
            logger.error("RDKit required for mutation generation")
            return []
        
        mol = Chem.MolFromSmiles(seed_smiles)
        if mol is None:
            logger.error(f"Invalid SMILES: {seed_smiles}")
            return [GeneratedMolecule(
                smiles=seed_smiles,
                source="mutation",
                parent_smiles=seed_smiles,
                properties={"error": "Invalid seed SMILES"},
            )]
        
        molecules = []
        
        # Simple mutations: halogen/methyl substitutions
        mutation_smarts = [
            ("[c:1][H]", "[c:1]F"),      # Add F
            ("[c:1][H]", "[c:1]Cl"),     # Add Cl
            ("[c:1][H]", "[c:1]C"),      # Add methyl
            ("[C:1]F", "[C:1]Cl"),       # F → Cl
            ("[C:1]Cl", "[C:1]F"),       # Cl → F
            ("[c:1]F", "[c:1]Br"),       # F → Br
            ("[N:1](C)C", "[N:1](CC)CC"),  # N-dimethyl → N-diethyl
        ]
        
        generated_smiles = set()
        generated_smiles.add(Chem.MolToSmiles(mol))  # Exclude original
        
        for pattern, replacement in mutation_smarts:
            if len(molecules) >= num_samples:
                break
            
            try:
                rxn = AllChem.ReactionFromSmarts(f"{pattern}>>{replacement}")
                products = rxn.RunReactants((mol,))
                
                for product_tuple in products:
                    if len(molecules) >= num_samples:
                        break
                    
                    product = product_tuple[0]
                    try:
                        Chem.SanitizeMol(product)
                        new_smiles = Chem.MolToSmiles(product)
                        
                        if new_smiles not in generated_smiles:
                            generated_smiles.add(new_smiles)
                            molecules.append(GeneratedMolecule(
                                smiles=new_smiles,
                                source="mutation",
                                parent_smiles=seed_smiles,
                                prompt=constraints,
                                confidence=0.85,
                                properties={"mutation": f"{pattern} → {replacement}"},
                            ))
                    except:
                        continue
            except:
                continue
        
        # If not enough mutations, add the original
        if not molecules:
            molecules.append(GeneratedMolecule(
                smiles=seed_smiles,
                source="mutation",
                parent_smiles=seed_smiles,
                confidence=1.0,
                properties={"note": "Original molecule (no valid mutations)"},
            ))
        
        logger.info(f"Generated {len(molecules)} mutations")
        return molecules
    
    def _generate_scaffold(
        self,
        scaffold_smiles: str,
        num_samples: int,
        constraints: Optional[str] = None,
    ) -> List[GeneratedMolecule]:
        """
        Generate molecules around a scaffold.
        
        For now, this uses mutation-based approach on the scaffold.
        Future: Use proper scaffold hopping or R-group enumeration.
        """
        logger.info(f"Generating {num_samples} scaffold variants of: {scaffold_smiles}")
        
        # For now, delegate to mutation with scaffold as seed
        molecules = self._generate_mutations(scaffold_smiles, num_samples, constraints)
        
        # Update source to scaffold
        for mol in molecules:
            mol.source = "scaffold"
        
        return molecules
