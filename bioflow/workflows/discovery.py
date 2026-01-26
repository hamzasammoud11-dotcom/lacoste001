"""
BioFlow Discovery Pipeline
===========================

High-level API for common discovery workflows.
Connects encoders, retrievers, and predictors into seamless pipelines.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from bioflow.core import (
    Modality,
    ToolRegistry,
    BioFlowOrchestrator,
    WorkflowConfig,
    NodeConfig,
    NodeType,
    RetrievalResult,
)
from bioflow.core.nodes import (
    NodeResult,
    EncodeNode,
    RetrieveNode,
    PredictNode,
    IngestNode,
    FilterNode,
    TraceabilityNode,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Complete result from a discovery pipeline."""
    query: str
    query_modality: str
    candidates: List[Dict[str, Any]]
    predictions: List[Dict[str, Any]]
    top_hits: List[Dict[str, Any]]
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_modality": self.query_modality,
            "num_candidates": len(self.candidates),
            "num_predictions": len(self.predictions),
            "top_hits": self.top_hits[:5],
            "execution_time_ms": self.execution_time_ms,
        }
    
    def __repr__(self):
        return f"DiscoveryResult(hits={len(self.top_hits)}, time={self.execution_time_ms:.0f}ms)"


class DiscoveryPipeline:
    """
    High-level discovery pipeline for drug-target interactions.
    
    Workflow:
    1. Encode query (text/molecule/protein) → vector
    2. Search vector DB for similar compounds/sequences
    3. Predict binding affinity for each candidate
    4. Filter and rank by predicted score
    5. Add evidence links for traceability
    
    Example:
        >>> from bioflow.plugins import OBMEncoder, QdrantRetriever, DeepPurposePredictor
        >>> 
        >>> pipeline = DiscoveryPipeline(
        ...     encoder=OBMEncoder(),
        ...     retriever=QdrantRetriever(...),
        ...     predictor=DeepPurposePredictor()
        ... )
        >>> 
        >>> # Search for drug candidates
        >>> results = pipeline.discover(
        ...     query="EGFR inhibitor with low toxicity",
        ...     target_sequence="MRKH...",
        ...     limit=20
        ... )
    """
    
    def __init__(
        self,
        encoder,
        retriever,
        predictor,
        collection: str = "molecules"
    ):
        """
        Initialize discovery pipeline.
        
        Args:
            encoder: BioEncoder instance (e.g., OBMEncoder)
            retriever: BioRetriever instance (e.g., QdrantRetriever)
            predictor: BioPredictor instance (e.g., DeepPurposePredictor)
            collection: Default collection for retrieval
        """
        self.encoder = encoder
        self.retriever = retriever
        self.predictor = predictor
        self.collection = collection
        
        # Initialize nodes
        self._encode_node = EncodeNode("encode", encoder, auto_detect=True)
        self._retrieve_node = RetrieveNode("retrieve", retriever, collection=collection)
        self._predict_node = PredictNode("predict", predictor)
        self._filter_node = FilterNode("filter", threshold=0.3, top_k=10)
        self._trace_node = TraceabilityNode("trace")
        
        logger.info("DiscoveryPipeline initialized")
    
    def discover(
        self,
        query: str,
        target_sequence: str,
        modality: Modality = None,
        limit: int = 20,
        filters: Dict[str, Any] = None,
        threshold: float = 0.3,
        top_k: int = 10
    ) -> DiscoveryResult:
        """
        Run full discovery pipeline.
        
        Args:
            query: Search query (text, SMILES, or protein)
            target_sequence: Target protein sequence for DTI prediction
            modality: Input modality (auto-detected if None)
            limit: Number of candidates to retrieve
            filters: Metadata filters for retrieval
            threshold: Minimum prediction score
            top_k: Number of top hits to return
            
        Returns:
            DiscoveryResult with ranked candidates
        """
        start_time = datetime.now()
        
        # Detect modality if not provided
        if modality is None:
            if hasattr(self.encoder, 'encode_auto'):
                # Will auto-detect
                modality = Modality.TEXT
            else:
                modality = Modality.TEXT
        
        # Step 1: Encode query
        logger.info(f"Encoding query: {query[:50]}...")
        self._encode_node.modality = modality
        encode_result = self._encode_node.execute(query)
        
        # Step 2: Retrieve candidates
        logger.info(f"Retrieving up to {limit} candidates...")
        self._retrieve_node.limit = limit
        self._retrieve_node.filters = filters or {}
        retrieve_result = self._retrieve_node.execute(
            encode_result.data,
            context={"modality": modality}
        )
        candidates = retrieve_result.data
        
        if not candidates:
            logger.warning("No candidates found")
            return DiscoveryResult(
                query=query,
                query_modality=modality.value,
                candidates=[],
                predictions=[],
                top_hits=[],
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        
        # Step 3: Predict binding
        logger.info(f"Predicting binding for {len(candidates)} candidates...")
        self._predict_node.target_sequence = target_sequence
        self._predict_node.threshold = threshold
        predict_result = self._predict_node.execute(candidates)
        predictions = predict_result.data
        
        # Step 4: Filter and rank
        logger.info("Filtering and ranking...")
        self._filter_node.threshold = threshold
        self._filter_node.top_k = top_k
        filter_result = self._filter_node.execute(predictions)
        top_hits = filter_result.data
        
        # Step 5: Add evidence links
        logger.info("Adding evidence links...")
        trace_result = self._trace_node.execute(top_hits)
        enriched_hits = trace_result.data
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return DiscoveryResult(
            query=query,
            query_modality=modality.value,
            candidates=[{"id": c.id, "content": c.content, "score": c.score} for c in candidates],
            predictions=predictions,
            top_hits=enriched_hits,
            execution_time_ms=execution_time,
            metadata={
                "collection": self.collection,
                "limit": limit,
                "threshold": threshold,
                "target_length": len(target_sequence)
            }
        )
    
    def ingest(
        self,
        data: List[Dict[str, Any]],
        modality: Modality = Modality.SMILES,
        content_field: str = "smiles"
    ) -> List[str]:
        """
        Ingest data into the vector database.
        
        Args:
            data: List of items with content and metadata
            modality: Type of content
            content_field: Field name containing the content
            
        Returns:
            List of ingested IDs
        """
        ingest_node = IngestNode(
            "ingest",
            self.retriever,
            collection=self.collection,
            modality=modality,
            content_field=content_field
        )
        
        result = ingest_node.execute(data)
        logger.info(f"Ingested {len(result.data)} items into {self.collection}")
        
        return result.data
    
    def search(
        self,
        query: str,
        modality: Modality = None,
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        """
        Simple similarity search without prediction.
        
        Args:
            query: Search query
            modality: Input modality
            limit: Maximum results
            filters: Metadata filters
            
        Returns:
            List of similar items
        """
        # Encode
        self._encode_node.modality = modality or Modality.TEXT
        self._encode_node.auto_detect = modality is None
        encode_result = self._encode_node.execute(query)
        
        # Retrieve
        self._retrieve_node.limit = limit
        self._retrieve_node.filters = filters or {}
        retrieve_result = self._retrieve_node.execute(encode_result.data)
        
        return retrieve_result.data


class LiteratureMiningPipeline:
    """
    Pipeline for searching and analyzing scientific literature.
    
    Workflow:
    1. Encode query (text/molecule/protein)
    2. Search literature database
    3. Extract relevant evidence
    4. Rank by relevance and diversity
    """
    
    def __init__(
        self,
        encoder,
        retriever,
        collection: str = "pubmed_abstracts"
    ):
        self.encoder = encoder
        self.retriever = retriever
        self.collection = collection
        
        self._encode_node = EncodeNode("encode", encoder, auto_detect=True)
        self._retrieve_node = RetrieveNode("retrieve", retriever, collection=collection)
        self._trace_node = TraceabilityNode("trace")
    
    def search(
        self,
        query: str,
        modality: Modality = Modality.TEXT,
        limit: int = 20,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search literature for relevant papers.
        
        Args:
            query: Search query
            modality: Query type
            limit: Maximum results
            filters: Filters (e.g., year, species)
            
        Returns:
            List of papers with evidence links
        """
        # Encode query
        self._encode_node.modality = modality
        encode_result = self._encode_node.execute(query)
        
        # Search
        self._retrieve_node.limit = limit
        self._retrieve_node.filters = filters or {}
        retrieve_result = self._retrieve_node.execute(encode_result.data)
        
        # Add evidence links
        trace_result = self._trace_node.execute(retrieve_result.data)
        
        return trace_result.data


class ProteinDesignPipeline:
    """
    Pipeline for protein/antibody design workflows.
    
    Workflow:
    1. Encode seed protein
    2. Find similar sequences in database
    3. Analyze conservation and mutations
    4. Suggest design candidates
    """
    
    def __init__(
        self,
        encoder,
        retriever,
        collection: str = "proteins"
    ):
        self.encoder = encoder
        self.retriever = retriever
        self.collection = collection
    
    def find_homologs(
        self,
        sequence: str,
        limit: int = 50,
        species_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        Find homologous proteins.
        
        Args:
            sequence: Query protein sequence
            limit: Maximum results
            species_filter: Filter by species
            
        Returns:
            List of homologous proteins with metadata
        """
        # Encode
        embedding = self.encoder.encode(sequence, Modality.PROTEIN)
        
        # Build filters
        filters = {}
        if species_filter:
            filters["species"] = species_filter
        
        # Search
        results = self.retriever.search(
            query=embedding.vector,
            limit=limit,
            filters=filters if filters else None,
            collection=self.collection,
            modality=Modality.PROTEIN
        )
        
        return [
            {
                "id": r.id,
                "sequence": r.content,
                "score": r.score,
                **r.payload
            }
            for r in results
        ]
