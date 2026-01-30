"""
Experiment Ingestor - Biological Experiment Data Ingestion
============================================================

Ingest experimental results (measurements, conditions, outcomes) into Qdrant.

Supports Use Case 4: Multimodal Biological Design & Discovery Intelligence
- Measurements: numeric values, time series
- Conditions: protocols, parameters
- Outcomes: success/failure labels, quality scores

Example data:
{
    "experiment_id": "EXP001",
    "title": "EGFR binding assay",
    "type": "binding_assay",
    "molecule": "CC(=O)Nc1ccc(O)cc1",  # Related molecule (optional)
    "target": "EGFR",
    "measurements": [
        {"name": "IC50", "value": 0.5, "unit": "nM"},
        {"name": "Ki", "value": 0.3, "unit": "nM"}
    ],
    "conditions": {
        "temperature": "37C",
        "pH": 7.4,
        "buffer": "PBS",
        "incubation_time": "2h"
    },
    "outcome": "success",  # success/failure/partial
    "quality_score": 0.95,
    "description": "High-throughput screening of EGFR inhibitors",
    "protocol": "Standard fluorescence polarization assay",
    "source": "internal_lab",
    "date": "2026-01-15"
}
"""

import logging
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import uuid

from bioflow.ingestion.base_ingestor import BaseIngestor, IngestionResult, DataRecord
from bioflow.core.base import Modality

logger = logging.getLogger(__name__)


@dataclass
class Measurement:
    """A single measurement from an experiment."""
    name: str
    value: float
    unit: str
    error: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRecord:
    """Structured experimental data record."""
    experiment_id: str
    title: str
    experiment_type: str
    measurements: List[Measurement]
    conditions: Dict[str, Any]
    outcome: str  # success, failure, partial, inconclusive
    quality_score: float
    description: str
    protocol: Optional[str] = None
    molecule: Optional[str] = None  # SMILES if applicable
    target: Optional[str] = None  # Target name or sequence
    source: str = "experiment"
    date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperimentIngestor(BaseIngestor):
    """
    Ingestor for biological experimental data.
    
    Supports:
    - Binding assays (IC50, Ki, Kd)
    - Activity assays (EC50, % inhibition)
    - ADMET properties (LogP, solubility)
    - Phenotypic screens (cell viability, morphology)
    - Genetic perturbations (CRISPR, knockdown)
    
    Example:
        >>> ingestor = ExperimentIngestor(qdrant_service, obm_encoder)
        >>> result = ingestor.ingest_experiments([
        ...     {
        ...         "experiment_id": "EXP001",
        ...         "title": "EGFR binding",
        ...         "type": "binding_assay",
        ...         "measurements": [{"name": "IC50", "value": 0.5, "unit": "nM"}],
        ...         "conditions": {"temperature": "37C"},
        ...         "outcome": "success",
        ...         "quality_score": 0.95,
        ...         "description": "EGFR inhibitor screen"
        ...     }
        ... ])
    """
    
    # Experiment type classification
    EXPERIMENT_TYPES = {
        "binding_assay": ["IC50", "Ki", "Kd", "EC50", "binding"],
        "activity_assay": ["% inhibition", "% activity", "fold change"],
        "admet": ["LogP", "solubility", "permeability", "clearance", "half-life"],
        "phenotypic": ["viability", "proliferation", "morphology", "migration"],
        "genetic": ["knockdown", "knockout", "CRISPR", "expression"],
        "structural": ["crystallography", "cryo-EM", "NMR", "mass spec"],
    }
    
    # Outcome labels
    OUTCOMES = ["success", "failure", "partial", "inconclusive", "pending"]
    
    @property
    def source_name(self) -> str:
        return "experiment"
    
    def fetch_data(
        self,
        query: str,
        limit: int
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch experiments from a data source.
        
        For experiments, query can be:
        - Path to JSON file with experiment data
        - Query to filter internal database
        """
        import os
        
        if query.endswith('.json') and os.path.exists(query):
            logger.info(f"Loading experiments from file: {query}")
            with open(query, 'r') as f:
                experiments = json.load(f)
            
            for exp in experiments[:limit]:
                yield exp
        else:
            logger.warning(f"Query '{query}' is not a valid JSON file")
    
    def parse_record(self, raw_data: Dict[str, Any]) -> Optional[DataRecord]:
        """Parse experiment record."""
        try:
            # Generate experiment ID if not provided
            exp_id = raw_data.get("experiment_id")
            if not exp_id:
                # Generate from content hash
                content_hash = hashlib.sha256(
                    json.dumps(raw_data, sort_keys=True).encode()
                ).hexdigest()[:12]
                exp_id = f"exp_{content_hash}"
            
            # Parse measurements
            measurements = []
            for m in raw_data.get("measurements", []):
                measurements.append({
                    "name": m.get("name", "unknown"),
                    "value": m.get("value", 0),
                    "unit": m.get("unit", ""),
                    "error": m.get("error"),
                })
            
            # Build searchable text content
            content_parts = [
                raw_data.get("title", ""),
                raw_data.get("description", ""),
                raw_data.get("protocol", ""),
                raw_data.get("target", ""),
            ]
            
            # Add measurements to content
            for m in measurements:
                content_parts.append(f"{m['name']}: {m['value']} {m['unit']}")
            
            # Add conditions
            conditions = raw_data.get("conditions", {})
            for k, v in conditions.items():
                content_parts.append(f"{k}: {v}")
            
            content = " | ".join(filter(None, content_parts))
            
            # Build metadata
            metadata = {
                "experiment_id": exp_id,
                "title": raw_data.get("title", ""),
                "experiment_type": raw_data.get("type", "unknown"),
                "measurements": measurements,
                "conditions": conditions,
                "outcome": raw_data.get("outcome", "unknown"),
                "outcome_label": self._outcome_to_label(raw_data.get("outcome")),
                "quality_score": raw_data.get("quality_score", 0.0),
                "description": raw_data.get("description", ""),
                "protocol": raw_data.get("protocol"),
                "molecule": raw_data.get("molecule"),
                "target": raw_data.get("target"),
                "source": raw_data.get("source", "experiment"),
                "source_id": f"experiment:{exp_id}",
                "date": raw_data.get("date"),
                "indexed_at": datetime.now().isoformat(),
                **raw_data.get("metadata", {}),
            }
            
            return DataRecord(
                id=exp_id,
                content=content,
                modality="experiment",  # New modality for experiments
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to parse experiment record: {e}")
            return None
    
    def _outcome_to_label(self, outcome: str) -> str:
        """Convert outcome to standardized label."""
        if not outcome:
            return "unknown"
        outcome = outcome.lower().strip()
        if outcome in self.OUTCOMES:
            return outcome
        # Map common aliases
        if outcome in ("positive", "hit", "active", "good"):
            return "success"
        if outcome in ("negative", "miss", "inactive", "bad"):
            return "failure"
        if outcome in ("moderate", "weak", "marginal"):
            return "partial"
        return "inconclusive"
    
    def ingest_experiments(
        self,
        experiments: List[Dict[str, Any]],
        collection: Optional[str] = None
    ) -> IngestionResult:
        """
        Ingest a list of experiments.
        
        Args:
            experiments: List of experiment dictionaries
            collection: Target collection (default: self.collection)
            
        Returns:
            IngestionResult with statistics
        """
        collection = collection or self.collection
        start_time = datetime.now()
        
        logger.info(f"Starting ingestion of {len(experiments)} experiments")
        
        indexed = 0
        failed = 0
        errors = []
        
        for raw_exp in experiments:
            try:
                record = self.parse_record(raw_exp)
                if not record:
                    failed += 1
                    continue
                
                # Encode experiment using text encoder (description-based)
                self._rate_limit_wait()
                
                # Use text modality for encoding experiment descriptions
                modality_enum = self._get_modality_enum("text")
                if not modality_enum:
                    logger.error("Text modality not available")
                    failed += 1
                    continue
                
                embedding_results = self.encoder.encode(
                    record.content,
                    modality=modality_enum
                )
                
                if not isinstance(embedding_results, list):
                    embedding_results = [embedding_results]
                
                if not embedding_results:
                    failed += 1
                    continue
                
                embedding_result = embedding_results[0]
                
                # Check for encoding errors
                if "error" in embedding_result.metadata:
                    logger.error(f"Encoding error: {embedding_result.metadata['error']}")
                    failed += 1
                    errors.append(embedding_result.metadata['error'])
                    continue
                
                # Store in Qdrant
                from qdrant_client.models import PointStruct
                import numpy as np
                
                point = PointStruct(
                    id=record.id,
                    vector=np.array(embedding_result.vector).tolist(),
                    payload={
                        "content": record.content[:1000],  # Limit content size
                        "modality": "experiment",
                        **record.metadata
                    }
                )
                
                self.qdrant._get_client().upsert(
                    collection_name=collection,
                    points=[point]
                )
                
                indexed += 1
                
                if indexed % 20 == 0:
                    logger.info(f"Progress: {indexed} experiments indexed, {failed} failed")
                
            except Exception as e:
                logger.error(f"Failed to ingest experiment: {e}")
                errors.append(str(e))
                failed += 1
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = IngestionResult(
            source=self.source_name,
            total_fetched=len(experiments),
            total_indexed=indexed,
            failed=failed,
            duration_seconds=duration,
            errors=errors[:10]
        )
        
        logger.info(f"✅ Experiment ingestion complete: {indexed} indexed, {failed} failed in {duration:.2f}s")
        
        return result
    
    def ingest_from_file(
        self,
        filepath: str,
        collection: Optional[str] = None
    ) -> IngestionResult:
        """
        Ingest experiments from a JSON file.
        
        Args:
            filepath: Path to JSON file with experiment data
            collection: Target collection
            
        Returns:
            IngestionResult with statistics
        """
        with open(filepath, 'r') as f:
            experiments = json.load(f)
        
        return self.ingest_experiments(experiments, collection)


def create_sample_experiments() -> List[Dict[str, Any]]:
    """Create sample experiments for testing."""
    return [
        {
            "experiment_id": "EXP001",
            "title": "EGFR Kinase Inhibition Assay",
            "type": "binding_assay",
            "molecule": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC",  # Gefitinib-like
            "target": "EGFR",
            "measurements": [
                {"name": "IC50", "value": 0.033, "unit": "µM"},
                {"name": "Ki", "value": 0.025, "unit": "µM"},
            ],
            "conditions": {
                "temperature": "30°C",
                "ATP_concentration": "100 µM",
                "incubation_time": "60 min",
                "buffer": "Tris-HCl pH 7.5",
            },
            "outcome": "success",
            "quality_score": 0.95,
            "description": "High-throughput kinase inhibition screen targeting EGFR. Compound shows potent inhibition with sub-micromolar IC50.",
            "protocol": "Lanthascreen kinase binding assay",
            "source": "internal_hts",
            "date": "2026-01-10",
        },
        {
            "experiment_id": "EXP002",
            "title": "BCL-2 Cell Viability Screen",
            "type": "phenotypic",
            "molecule": "CC1(C)CCC(CN2CCN(c3ccc(C(=O)NS(=O)(=O)c4ccc(NCC5CCOCC5)c([N+](=O)[O-])c4)c(Oc4cnc5[nH]ccc5c4)c3)CC2)=C(c2ccc(Cl)cc2)C1",  # Venetoclax-like
            "target": "BCL-2",
            "measurements": [
                {"name": "EC50", "value": 0.001, "unit": "µM"},
                {"name": "% viability", "value": 15, "unit": "%"},
            ],
            "conditions": {
                "cell_line": "MV-4-11",
                "treatment_time": "72 h",
                "seeding_density": "5000 cells/well",
            },
            "outcome": "success",
            "quality_score": 0.92,
            "description": "BCL-2 selective inhibitor demonstrates potent cytotoxicity in BCL-2 dependent AML cell line.",
            "protocol": "CellTiter-Glo luminescent viability assay",
            "source": "oncology_lab",
            "date": "2026-01-12",
        },
        {
            "experiment_id": "EXP003",
            "title": "ADMET Solubility Assessment",
            "type": "admet",
            "molecule": "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
            "measurements": [
                {"name": "solubility", "value": 14.3, "unit": "mg/mL"},
                {"name": "LogP", "value": 0.46, "unit": ""},
            ],
            "conditions": {
                "pH": 7.4,
                "temperature": "25°C",
                "buffer": "PBS",
            },
            "outcome": "success",
            "quality_score": 0.88,
            "description": "Kinetic solubility assessment in physiological buffer. Compound shows good aqueous solubility.",
            "protocol": "Nephelometry-based kinetic solubility",
            "source": "dmpk_lab",
            "date": "2026-01-08",
        },
        {
            "experiment_id": "EXP004",
            "title": "HTS Primary Screen - Failed Hit",
            "type": "binding_assay",
            "molecule": "c1ccccc1",  # Benzene (inactive control)
            "target": "CDK2",
            "measurements": [
                {"name": "% inhibition", "value": 5, "unit": "%"},
            ],
            "conditions": {
                "concentration": "10 µM",
                "replicate_count": 3,
            },
            "outcome": "failure",
            "quality_score": 0.75,
            "description": "Compound showed no significant inhibition at screening concentration. Classified as inactive.",
            "source": "primary_screen",
            "date": "2026-01-05",
        },
    ]
