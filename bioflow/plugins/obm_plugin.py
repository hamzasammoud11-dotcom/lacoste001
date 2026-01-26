"""
OBM Plugin - Deprecated
========================

This module is deprecated. Use OBMEncoder from bioflow.plugins.obm_encoder instead.
"""

# Redirect to new implementation
from bioflow.plugins.obm_encoder import OBMEncoder

# Alias for backward compatibility
OBMPlugin = OBMEncoder

__all__ = ["OBMEncoder", "OBMPlugin"]
        if modality == Modality.TEXT:
            return self._encode_text(content)
        elif modality == Modality.SMILES:
            return self._encode_smiles(content)
        elif modality == Modality.PROTEIN:
            return self._encode_protein(content)
        return []

    @property
    def dimension(self) -> int:
        return 4096 # Placeholder for model dimension

    def _encode_text(self, text: str):
        # Placeholder for text encoding using open-source model
        pass

    def _encode_smiles(self, smiles: str):
        # Placeholder for SMILES encoding using open-source model
        pass

    def _encode_protein(self, protein: str):
        # Placeholder for protein encoding using open-source model
        pass

# Auto-register the tool so the orchestrator can find it
ToolRegistry.register_encoder("obm", OBMPlugin())
