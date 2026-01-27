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
