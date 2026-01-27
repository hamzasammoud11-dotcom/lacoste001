"""
BioFlow Encoders
=================

Open-source encoder implementations for different modalities.

Available Encoders:
- TextEncoder: PubMedBERT / SciBERT for biomedical text
- MoleculeEncoder: ChemBERTa for SMILES molecules
- ProteinEncoder: ESM-2 for protein sequences
"""

from bioflow.plugins.encoders.text_encoder import TextEncoder
from bioflow.plugins.encoders.molecule_encoder import MoleculeEncoder
from bioflow.plugins.encoders.protein_encoder import ProteinEncoder

__all__ = ["TextEncoder", "MoleculeEncoder", "ProteinEncoder"]
