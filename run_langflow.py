#!/usr/bin/env python
"""Run BioFlow Orchestrator (Langflow) server."""

import subprocess
import os

def main():
    print("Starting BioFlow Orchestrator (Langflow)...")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Path to langflow executable in .venv
    langflow_exe = os.path.join(script_dir, ".venv", "Scripts", "langflow.exe")
    
    # Run langflow
    subprocess.run([langflow_exe, "run", "--host", "0.0.0.0", "--port", "7860"])

if __name__ == "__main__":
    main()
