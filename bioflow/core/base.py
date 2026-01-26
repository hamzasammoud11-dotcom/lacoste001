"""
BioFlow Core Abstractions
==========================

Defines the fundamental interfaces for all tools in the BioFlow platform.
All encoders, predictors, generators, and retrievers must implement these.

Open-Source Models Supported:
- Text: PubMedBERT, SciBERT, Specter
- Molecules: ChemBERTa, RDKit FP
- Proteins: ESM-2, ProtBERT
- Images: CLIP, BioMedCLIP
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class Modality(Enum):
    """Supported data modalities in BioFlow."""
    TEXT = "text"
    SMILES = "smiles"
    PROTEIN = "protein"
    IMAGE = "image"
    GENOMIC = "genomic"
    STRUCTURE = "structure"


@dataclass
class EmbeddingResult:
    """Result of an encoding operation."""
    vector: List[float]
    modality: Modality
    dimension: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self):
        return len(self.vector)


@dataclass
class PredictionResult:
    """Result of a prediction operation."""
    score: float
    label: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class RetrievalResult:
    """Result of a retrieval/search operation."""
    id: str
    score: float
    content: Any
    modality: Modality
    payload: Dict[str, Any] = field(default_factory=dict)


class BioEncoder(ABC):
    """
    Interface for any tool that converts biological data into vectors.
    
    Implementations:
    - OBMEncoder: Multimodal (text, SMILES, protein)
    - ESM2Encoder: Protein sequences
    - ChemBERTaEncoder: SMILES molecules
    - PubMedBERTEncoder: Biomedical text
    - CLIPEncoder: Images
    
    Example:
        >>> encoder = ESM2Encoder(device="cuda")
        >>> result = encoder.encode("MKTVRQERLKSIVRILERSKEPVSG", Modality.PROTEIN)
        >>> print(len(result.vector))  # 1280
    """
    
    @abstractmethod
    def encode(self, content: Any, modality: Modality) -> EmbeddingResult:
        """
        Encode content into a vector representation.
        
        Args:
            content: Raw input (text, SMILES string, protein sequence, etc.)
            modality: Type of the input data
            
        Returns:
            EmbeddingResult with vector and metadata
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of output vectors."""
        pass
    
    @property
    def supported_modalities(self) -> List[Modality]:
        """Return list of modalities this encoder supports."""
        return [Modality.TEXT]  # Override in subclasses
    
    def batch_encode(self, contents: List[Any], modality: Modality) -> List[EmbeddingResult]:
        """Encode multiple items. Override for optimized batch processing."""
        return [self.encode(c, modality) for c in contents]


class BioPredictor(ABC):
    """
    Interface for tools that predict properties, affinities, or interactions.
    
    Implementations:
    - DeepPurposePredictor: DTI prediction
    - ToxicityPredictor: ADMET properties
    - BindingAffinityPredictor: Kd/Ki estimation
    
    Example:
        >>> predictor = DeepPurposePredictor()
        >>> result = predictor.predict(drug="CCO", target="MKTVRQ...")
        >>> print(result.score)  # 0.85
    """
    
    @abstractmethod
    def predict(self, drug: str, target: str) -> PredictionResult:
        """
        Predict interaction/property between drug and target.
        
        Args:
            drug: SMILES string of drug molecule
            target: Protein sequence or identifier
            
        Returns:
            PredictionResult with score and metadata
        """
        pass
    
    def batch_predict(self, pairs: List[tuple]) -> List[PredictionResult]:
        """Predict for multiple drug-target pairs."""
        return [self.predict(d, t) for d, t in pairs]


class BioGenerator(ABC):
    """
    Interface for tools that generate new biological candidates.
    
    Implementations:
    - MoleculeGenerator: SMILES generation
    - ProteinGenerator: Sequence design
    - VariantGenerator: Mutation suggestions
    
    Example:
        >>> generator = MoleculeGenerator()
        >>> candidates = generator.generate(
        ...     seed="CCO", 
        ...     constraints={"mw_max": 500, "logp_max": 5}
        ... )
    """
    
    @abstractmethod
    def generate(self, seed: Any, constraints: Dict[str, Any]) -> List[Any]:
        """
        Generate new candidates based on seed and constraints.
        
        Args:
            seed: Starting point (molecule, sequence, etc.)
            constraints: Dictionary of constraints (e.g., MW, toxicity)
            
        Returns:
            List of generated candidates
        """
        pass


class BioRetriever(ABC):
    """
    Interface for vector database retrieval operations.
    
    Implementations:
    - QdrantRetriever: Qdrant vector search
    - FAISSRetriever: FAISS similarity search
    
    Example:
        >>> retriever = QdrantRetriever(collection="molecules")
        >>> results = retriever.search(query_vector, limit=10)
    """
    
    @abstractmethod
    def search(
        self, 
        query: Union[List[float], str],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Search for similar items in vector database.
        
        Args:
            query: Query vector or raw content to encode first
            limit: Maximum number of results
            filters: Metadata filters to apply
            
        Returns:
            List of RetrievalResult sorted by similarity
        """
        pass
    
    @abstractmethod
    def ingest(
        self,
        content: Any,
        modality: Modality,
        payload: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ingest content into the vector database.
        
        Args:
            content: Raw content to encode and store
            modality: Type of content
            payload: Additional metadata to store
            
        Returns:
            ID of the inserted item
        """
        pass


class BioTool(ABC):
    """
    General wrapper for miscellaneous tools.
    
    Implementations:
    - RDKitTool: Molecular operations
    - VisualizationTool: Plotting and visualization
    - FilterTool: Candidate filtering
    """
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        pass
    
    @property
    def name(self) -> str:
        """Return tool name."""
        return self.__class__.__name__
