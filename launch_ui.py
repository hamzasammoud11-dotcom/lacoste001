"""
BioFlow UI Launch Script
=========================

Quick launcher for the BioFlow Streamlit application.

Usage:
    python launch_ui.py
    python launch_ui.py --port 8502
    python launch_ui.py --debug
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Launch the BioFlow UI."""
    # Get the app path
    script_dir = Path(__file__).parent
    app_path = script_dir / "bioflow" / "ui" / "app.py"
    
    if not app_path.exists():
        print(f"❌ Error: App not found at {app_path}")
        sys.exit(1)
    
    # Parse arguments
    port = 8501
    debug = False
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--port" and i + 1 < len(sys.argv[1:]):
            port = int(sys.argv[i + 2])
        elif arg == "--debug":
            debug = True
    
    # Build command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ]
    
    if debug:
        cmd.extend(["--logger.level", "debug"])
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   🧬 BioFlow - AI-Powered Drug Discovery Platform        ║
    ║                                                          ║
    ║   Starting server at http://localhost:{port}              ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        subprocess.run(cmd, cwd=str(script_dir))
    except KeyboardInterrupt:
        print("\n\n👋 BioFlow server stopped.")
    except Exception as e:
        print(f"❌ Error launching BioFlow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
