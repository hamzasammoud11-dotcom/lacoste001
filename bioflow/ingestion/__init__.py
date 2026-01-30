"""
BioFlow Ingestion Module
=========================

Provides data ingestion pipelines for biological databases:
- PubMed (biomedical literature)
- UniProt (protein sequences)
- ChEMBL (small molecules)
- Images (microscopy, gels, spectra)
- Experiments (measurements, conditions, outcomes)

Usage:
    from bioflow.ingestion import run_full_ingestion
    results = run_full_ingestion("EGFR lung cancer", pubmed_limit=100)
"""

from bioflow.ingestion.pubmed_ingestor import PubMedIngestor
from bioflow.ingestion.uniprot_ingestor import UniProtIngestor
from bioflow.ingestion.chembl_ingestor import ChEMBLIngestor
from bioflow.ingestion.image_ingestor import ImageIngestor
from bioflow.ingestion.experiment_ingestor import ExperimentIngestor
from bioflow.ingestion.base_ingestor import BaseIngestor, IngestionResult
from bioflow.ingestion.ingest_all import run_full_ingestion

__all__ = [
    "PubMedIngestor",
    "UniProtIngestor", 
    "ChEMBLIngestor",
    "ImageIngestor",
    "ExperimentIngestor",
    "BaseIngestor",
    "IngestionResult",
    "run_full_ingestion",
]
