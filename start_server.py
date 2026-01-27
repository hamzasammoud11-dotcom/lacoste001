#!/usr/bin/env python3
"""Start the BioFlow API server."""
import sys
import os

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "bioflow.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
