#!/usr/bin/env python
"""Run BioFlow Orchestrator (Langflow) server."""

import subprocess
import os

def main():
    print("Starting BioFlow Orchestrator (Langflow)...")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Prefer langflow executable in .venv if present, else fall back to PATH
    langflow_exe = os.path.join(script_dir, ".venv", "Scripts", "langflow.exe")
    cmd = [langflow_exe, "run", "--host", "0.0.0.0", "--port", "7860"]
    if not os.path.exists(langflow_exe):
        cmd = ["langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
    
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit(
            "Langflow executable not found. Install with: pip install langflow "
            "and ensure it is in your PATH or in .venv\\Scripts."
        )

if __name__ == "__main__":
    main()
