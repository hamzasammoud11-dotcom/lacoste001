"""
OpenBioMed package (vendored)
=============================

This repository vendors an upstream OpenBioMed codebase. For BioFlow, OpenBioMed
is treated as an optional dependency: BioFlow must remain importable even when
some heavy optional dependencies (e.g., `scanpy`) are not installed.

To avoid import-time failures, this package intentionally does not `import *`
from all submodules at import time. Import the specific subpackages you need:

  - `open_biomed.core.*`
  - `open_biomed.data.*`
  - `open_biomed.models.*`
"""

__all__ = []
