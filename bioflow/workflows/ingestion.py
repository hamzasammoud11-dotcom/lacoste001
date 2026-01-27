"""
Data Ingestion Utilities
=========================

Helpers for ingesting data from common biological sources:
- PubMed abstracts
- UniProt proteins
- ChEMBL molecules
- Custom CSV/JSON files
"""

import logging
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
import json

from bioflow.core import Modality

logger = logging.getLogger(__name__)


def load_json_data(
    path: str,
    content_field: str = "content",
    modality_field: str = None,
    limit: int = None
) -> List[Dict[str, Any]]:
    """
    Load data from JSON file.
    
    Args:
        path: Path to JSON file
        content_field: Field containing main content
        modality_field: Field indicating modality (optional)
        limit: Maximum items to load
        
    Returns:
        List of data items
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = data.get("data", data.get("items", [data]))
    
    if limit:
        data = data[:limit]
    
    logger.info(f"Loaded {len(data)} items from {path}")
    return data


def load_csv_data(
    path: str,
    content_field: str = "content",
    delimiter: str = ",",
    limit: int = None
) -> List[Dict[str, Any]]:
    """
    Load data from CSV file.
    
    Args:
        path: Path to CSV file
        content_field: Column containing main content
        delimiter: CSV delimiter
        limit: Maximum items to load
        
    Returns:
        List of data items as dictionaries
    """
    import csv
    
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            data.append(dict(row))
    
    logger.info(f"Loaded {len(data)} items from {path}")
    return data


def parse_smiles_file(
    path: str,
    name_field: int = 1,
    smiles_field: int = 0,
    has_header: bool = True,
    limit: int = None
) -> List[Dict[str, Any]]:
    """
    Parse SMILES file (commonly .smi format).
    
    Args:
        path: Path to SMILES file
        name_field: Column index for molecule name
        smiles_field: Column index for SMILES string
        has_header: Whether file has header row
        limit: Maximum items to load
        
    Returns:
        List of molecule dictionaries
    """
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if has_header and i == 0:
                continue
            if limit and len(data) >= limit:
                break
            
            parts = line.strip().split()
            if len(parts) >= 2:
                data.append({
                    "smiles": parts[smiles_field],
                    "name": parts[name_field] if len(parts) > name_field else f"mol_{i}",
                    "modality": "smiles"
                })
            elif len(parts) == 1:
                data.append({
                    "smiles": parts[0],
                    "name": f"mol_{i}",
                    "modality": "smiles"
                })
    
    logger.info(f"Loaded {len(data)} molecules from {path}")
    return data


def parse_fasta_file(
    path: str,
    limit: int = None
) -> List[Dict[str, Any]]:
    """
    Parse FASTA file for protein sequences.
    
    Args:
        path: Path to FASTA file
        limit: Maximum sequences to load
        
    Returns:
        List of protein dictionaries
    """
    data = []
    current_header = None
    current_sequence = []
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_header and current_sequence:
                    seq = ''.join(current_sequence)
                    data.append({
                        "sequence": seq,
                        "header": current_header,
                        "uniprot_id": _extract_uniprot_id(current_header),
                        "modality": "protein"
                    })
                    
                    if limit and len(data) >= limit:
                        break
                
                current_header = line[1:]  # Remove >
                current_sequence = []
            else:
                current_sequence.append(line)
        
        # Don't forget last sequence
        if current_header and current_sequence and (not limit or len(data) < limit):
            seq = ''.join(current_sequence)
            data.append({
                "sequence": seq,
                "header": current_header,
                "uniprot_id": _extract_uniprot_id(current_header),
                "modality": "protein"
            })
    
    logger.info(f"Loaded {len(data)} proteins from {path}")
    return data


def _extract_uniprot_id(header: str) -> str:
    """Extract UniProt ID from FASTA header."""
    # Common formats: sp|P12345|NAME or tr|P12345|NAME
    if '|' in header:
        parts = header.split('|')
        if len(parts) >= 2:
            return parts[1]
    # Just take first word
    return header.split()[0]


def generate_sample_molecules() -> List[Dict[str, Any]]:
    """
    Generate sample molecule data for testing.
    
    Returns:
        List of common drug molecules
    """
    return [
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin", "drugbank_id": "DB00945", "modality": "smiles"},
        {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "name": "Caffeine", "pubchem_id": "2519", "modality": "smiles"},
        {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "name": "Ibuprofen", "drugbank_id": "DB01050", "modality": "smiles"},
        {"smiles": "CC(=O)Nc1ccc(cc1)O", "name": "Acetaminophen", "drugbank_id": "DB00316", "modality": "smiles"},
        {"smiles": "CCO", "name": "Ethanol", "pubchem_id": "702", "modality": "smiles"},
        {"smiles": "c1ccccc1", "name": "Benzene", "pubchem_id": "241", "modality": "smiles"},
        {"smiles": "CC(C)NCC(O)c1ccc(O)c(O)c1", "name": "Isoprenaline", "drugbank_id": "DB01064", "modality": "smiles"},
        {"smiles": "Clc1ccc2c(c1)C(=NCC2)c3ccccc3", "name": "Diazepam", "drugbank_id": "DB00829", "modality": "smiles"},
    ]


def generate_sample_proteins() -> List[Dict[str, Any]]:
    """
    Generate sample protein data for testing.
    
    Returns:
        List of common proteins
    """
    return [
        {
            "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "name": "Sample kinase fragment",
            "uniprot_id": "P00533",
            "species": "human",
            "modality": "protein"
        },
        {
            "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            "name": "Hemoglobin alpha",
            "uniprot_id": "P69905",
            "species": "human",
            "modality": "protein"
        },
        {
            "sequence": "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
            "name": "EGFR fragment",
            "uniprot_id": "P00533",
            "species": "human",
            "modality": "protein"
        },
    ]


def generate_sample_abstracts() -> List[Dict[str, Any]]:
    """
    Generate sample PubMed-style abstracts for testing.
    
    Returns:
        List of sample abstracts
    """
    return [
        {
            "content": "EGFR mutations are common in non-small cell lung cancer and predict response to tyrosine kinase inhibitors. Gefitinib and erlotinib have shown significant efficacy in patients with EGFR-mutant tumors.",
            "pmid": "12345678",
            "title": "EGFR inhibitors in lung cancer",
            "year": 2023,
            "modality": "text"
        },
        {
            "content": "Drug-target interaction prediction using deep learning has emerged as a powerful approach for drug discovery. Neural networks can learn complex patterns from molecular structures and protein sequences.",
            "pmid": "23456789",
            "title": "Deep learning for DTI prediction",
            "year": 2024,
            "modality": "text"
        },
        {
            "content": "Aspirin inhibits cyclooxygenase enzymes (COX-1 and COX-2), reducing prostaglandin synthesis. This mechanism underlies its anti-inflammatory and analgesic effects.",
            "pmid": "34567890",
            "title": "Mechanism of aspirin",
            "year": 2022,
            "modality": "text"
        },
    ]
