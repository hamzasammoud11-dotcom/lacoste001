"""
BioFlow Workflows
==================

Pre-built pipelines for common discovery tasks.

Pipelines:
- DiscoveryPipeline: Drug discovery with DTI prediction
- LiteratureMiningPipeline: Scientific literature search
- ProteinDesignPipeline: Protein homolog search

Ingestion Utilities:
- load_json_data, load_csv_data
- parse_smiles_file, parse_fasta_file
- generate_sample_* for testing
"""

from bioflow.workflows.discovery import (
    DiscoveryPipeline,
    DiscoveryResult,
    LiteratureMiningPipeline,
    ProteinDesignPipeline,
)

from bioflow.workflows.ingestion import (
    load_json_data,
    load_csv_data,
    parse_smiles_file,
    parse_fasta_file,
    generate_sample_molecules,
    generate_sample_proteins,
    generate_sample_abstracts,
)

__all__ = [
    # Pipelines
    "DiscoveryPipeline",
    "DiscoveryResult",
    "LiteratureMiningPipeline",
    "ProteinDesignPipeline",
    # Ingestion
    "load_json_data",
    "load_csv_data",
    "parse_smiles_file",
    "parse_fasta_file",
    "generate_sample_molecules",
    "generate_sample_proteins",
    "generate_sample_abstracts",
]
