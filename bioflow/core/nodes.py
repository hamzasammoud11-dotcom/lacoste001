"""
BioFlow Workflow Nodes
=======================

Typed node implementations for the BioFlow orchestrator.
Each node wraps a specific operation in the discovery pipeline.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from bioflow.core import (
    Modality,
    BioEncoder,
    BioPredictor,
    BioRetriever,
    EmbeddingResult,
    PredictionResult,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


@dataclass
class NodeResult:
    """Result from any node execution."""
    node_id: str
    node_type: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __repr__(self):
        return f"NodeResult({self.node_type}: {len(self.data) if hasattr(self.data, '__len__') else 1} items)"


class BaseNode(ABC):
    """Base class for all workflow nodes."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
    
    @property
    @abstractmethod
    def node_type(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, input_data: Any, context: Dict[str, Any] = None) -> NodeResult:
        pass


class EncodeNode(BaseNode):
    """
    Encodes input data into embeddings.
    
    Input: Raw content (text, SMILES, protein sequence)
    Output: EmbeddingResult or list of EmbeddingResults
    """
    
    def __init__(
        self,
        node_id: str,
        encoder: BioEncoder,
        modality: Modality = Modality.TEXT,
        auto_detect: bool = False
    ):
        super().__init__(node_id)
        self.encoder = encoder
        self.modality = modality
        self.auto_detect = auto_detect
    
    @property
    def node_type(self) -> str:
        return "encode"
    
    def execute(self, input_data: Any, context: Dict[str, Any] = None) -> NodeResult:
        """Encode input data."""
        context = context or {}
        
        # Handle batch input
        if isinstance(input_data, list):
            if self.auto_detect and hasattr(self.encoder, 'encode_auto'):
                results = [self.encoder.encode_auto(item) for item in input_data]
            else:
                results = self.encoder.batch_encode(input_data, self.modality)
            data = results
        else:
            if self.auto_detect and hasattr(self.encoder, 'encode_auto'):
                result = self.encoder.encode_auto(input_data)
            else:
                result = self.encoder.encode(input_data, self.modality)
            data = result
        
        return NodeResult(
            node_id=self.node_id,
            node_type=self.node_type,
            data=data,
            metadata={"modality": self.modality.value, "auto_detect": self.auto_detect}
        )


class RetrieveNode(BaseNode):
    """
    Retrieves similar items from vector database.
    
    Input: Query (string or embedding)
    Output: List of RetrievalResults
    """
    
    def __init__(
        self,
        node_id: str,
        retriever: BioRetriever,
        collection: str = None,
        limit: int = 10,
        modality: Modality = Modality.TEXT,
        filters: Dict[str, Any] = None
    ):
        super().__init__(node_id)
        self.retriever = retriever
        self.collection = collection
        self.limit = limit
        self.modality = modality
        self.filters = filters or {}
    
    @property
    def node_type(self) -> str:
        return "retrieve"
    
    def execute(self, input_data: Any, context: Dict[str, Any] = None) -> NodeResult:
        """Retrieve similar items."""
        context = context or {}
        
        # Override from context if provided
        limit = context.get("limit", self.limit)
        filters = {**self.filters, **context.get("filters", {})}
        
        # Handle EmbeddingResult input
        if isinstance(input_data, EmbeddingResult):
            query = input_data.vector
        else:
            query = input_data
        
        results = self.retriever.search(
            query=query,
            limit=limit,
            filters=filters if filters else None,
            collection=self.collection,
            modality=self.modality
        )
        
        return NodeResult(
            node_id=self.node_id,
            node_type=self.node_type,
            data=results,
            metadata={
                "count": len(results),
                "collection": self.collection,
                "filters": filters
            }
        )


class PredictNode(BaseNode):
    """
    Runs predictions on drug-target pairs.
    
    Input: List of candidates (from retrieval) or direct (drug, target) pairs
    Output: List of PredictionResults with scores
    """
    
    def __init__(
        self,
        node_id: str,
        predictor: BioPredictor,
        target_sequence: str = None,
        drug_field: str = "content",
        threshold: float = 0.0
    ):
        super().__init__(node_id)
        self.predictor = predictor
        self.target_sequence = target_sequence
        self.drug_field = drug_field
        self.threshold = threshold
    
    @property
    def node_type(self) -> str:
        return "predict"
    
    def execute(self, input_data: Any, context: Dict[str, Any] = None) -> NodeResult:
        """Run predictions."""
        context = context or {}
        target = context.get("target", self.target_sequence)
        
        if not target:
            raise ValueError("Target sequence is required for prediction")
        
        predictions = []
        
        # Handle different input types
        if isinstance(input_data, list):
            for item in input_data:
                # Extract drug from RetrievalResult or dict
                if isinstance(item, RetrievalResult):
                    drug = item.content
                    source_id = item.id
                elif isinstance(item, dict):
                    drug = item.get(self.drug_field, item.get("smiles", ""))
                    source_id = item.get("id", "unknown")
                else:
                    drug = str(item)
                    source_id = "unknown"
                
                try:
                    result = self.predictor.predict(drug, target)
                    if result.score >= self.threshold:
                        predictions.append({
                            "drug": drug,
                            "source_id": source_id,
                            "prediction": result,
                            "score": result.score
                        })
                except Exception as e:
                    logger.warning(f"Prediction failed for {drug[:20]}...: {e}")
        else:
            result = self.predictor.predict(str(input_data), target)
            predictions.append({
                "drug": str(input_data),
                "prediction": result,
                "score": result.score
            })
        
        # Sort by score
        predictions.sort(key=lambda x: x["score"], reverse=True)
        
        return NodeResult(
            node_id=self.node_id,
            node_type=self.node_type,
            data=predictions,
            metadata={
                "count": len(predictions),
                "threshold": self.threshold,
                "target_length": len(target) if target else 0
            }
        )


class IngestNode(BaseNode):
    """
    Ingests data into the vector database.
    
    Input: List of items to ingest
    Output: List of ingested IDs
    """
    
    def __init__(
        self,
        node_id: str,
        retriever: BioRetriever,
        collection: str = None,
        modality: Modality = Modality.TEXT,
        content_field: str = "content"
    ):
        super().__init__(node_id)
        self.retriever = retriever
        self.collection = collection
        self.modality = modality
        self.content_field = content_field
    
    @property
    def node_type(self) -> str:
        return "ingest"
    
    def execute(self, input_data: Any, context: Dict[str, Any] = None) -> NodeResult:
        """Ingest data into vector DB."""
        context = context or {}
        ids = []
        
        if isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, dict):
                    content = item.get(self.content_field, item.get("smiles", item.get("sequence", "")))
                    payload = {k: v for k, v in item.items() if k != self.content_field}
                else:
                    content = str(item)
                    payload = {}
                
                item_id = self.retriever.ingest(
                    content=content,
                    modality=self.modality,
                    payload=payload,
                    collection=self.collection
                )
                ids.append(item_id)
        else:
            item_id = self.retriever.ingest(
                content=str(input_data),
                modality=self.modality,
                payload=context.get("payload", {}),
                collection=self.collection
            )
            ids.append(item_id)
        
        return NodeResult(
            node_id=self.node_id,
            node_type=self.node_type,
            data=ids,
            metadata={"count": len(ids), "collection": self.collection}
        )


class FilterNode(BaseNode):
    """
    Filters and ranks results.
    
    Input: List of items
    Output: Filtered/ranked list
    """
    
    def __init__(
        self,
        node_id: str,
        score_field: str = "score",
        threshold: float = 0.5,
        top_k: int = None,
        diversity: float = 0.0  # For MMR-style diversification
    ):
        super().__init__(node_id)
        self.score_field = score_field
        self.threshold = threshold
        self.top_k = top_k
        self.diversity = diversity
    
    @property
    def node_type(self) -> str:
        return "filter"
    
    def _get_score(self, item: Any) -> float:
        """Extract score from item."""
        if isinstance(item, dict):
            return item.get(self.score_field, 0)
        elif hasattr(item, self.score_field):
            return getattr(item, self.score_field)
        elif hasattr(item, 'score'):
            return item.score
        return 0
    
    def execute(self, input_data: Any, context: Dict[str, Any] = None) -> NodeResult:
        """Filter and rank results."""
        context = context or {}
        
        if not isinstance(input_data, list):
            input_data = [input_data]
        
        # Filter by threshold
        filtered = [item for item in input_data if self._get_score(item) >= self.threshold]
        
        # Sort by score
        filtered.sort(key=lambda x: self._get_score(x), reverse=True)
        
        # Apply top_k
        if self.top_k:
            filtered = filtered[:self.top_k]
        
        return NodeResult(
            node_id=self.node_id,
            node_type=self.node_type,
            data=filtered,
            metadata={
                "input_count": len(input_data),
                "output_count": len(filtered),
                "threshold": self.threshold
            }
        )


class TraceabilityNode(BaseNode):
    """
    Adds evidence linking and provenance to results.
    
    Input: Results with source IDs
    Output: Results enriched with evidence links
    """
    
    def __init__(
        self,
        node_id: str,
        source_mapping: Dict[str, str] = None  # Maps ID prefixes to URLs
    ):
        super().__init__(node_id)
        self.source_mapping = source_mapping or {
            "PMID": "https://pubmed.ncbi.nlm.nih.gov/{id}",
            "UniProt": "https://www.uniprot.org/uniprot/{id}",
            "ChEMBL": "https://www.ebi.ac.uk/chembl/compound_report_card/{id}",
            "PubChem": "https://pubchem.ncbi.nlm.nih.gov/compound/{id}",
        }
    
    @property
    def node_type(self) -> str:
        return "trace"
    
    def _generate_evidence_link(self, source_id: str, payload: Dict[str, Any]) -> Dict[str, str]:
        """Generate evidence links from source ID and payload."""
        links = {}
        
        # Check for known ID types in payload
        for key, url_template in self.source_mapping.items():
            if key.lower() in payload:
                links[key] = url_template.format(id=payload[key.lower()])
            elif f"{key.lower()}_id" in payload:
                links[key] = url_template.format(id=payload[f"{key.lower()}_id"])
        
        # Check source_id prefix
        for prefix, url_template in self.source_mapping.items():
            if source_id.startswith(prefix):
                id_part = source_id.replace(prefix, "").lstrip("_:-")
                links[prefix] = url_template.format(id=id_part)
        
        return links
    
    def execute(self, input_data: Any, context: Dict[str, Any] = None) -> NodeResult:
        """Add evidence links to results."""
        context = context or {}
        
        if not isinstance(input_data, list):
            input_data = [input_data]
        
        enriched = []
        for item in input_data:
            if isinstance(item, dict):
                source_id = item.get("source_id", item.get("id", ""))
                payload = item.get("payload", item)
                evidence = self._generate_evidence_link(source_id, payload)
                
                enriched_item = {
                    **item,
                    "evidence_links": evidence,
                    "has_evidence": len(evidence) > 0
                }
                enriched.append(enriched_item)
            elif isinstance(item, RetrievalResult):
                evidence = self._generate_evidence_link(item.id, item.payload)
                enriched.append({
                    "id": item.id,
                    "content": item.content,
                    "score": item.score,
                    "modality": item.modality.value,
                    "payload": item.payload,
                    "evidence_links": evidence,
                    "has_evidence": len(evidence) > 0
                })
            else:
                enriched.append(item)
        
        return NodeResult(
            node_id=self.node_id,
            node_type=self.node_type,
            data=enriched,
            metadata={"with_evidence": sum(1 for e in enriched if isinstance(e, dict) and e.get("has_evidence", False))}
        )
