"""
BioFlow UI Launch Script
=========================

Quick launcher for the BioFlow Next.js application.

Usage:
    python launch_ui.py
    python launch_ui.py --port 3001
    python launch_ui.py --debug
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Launch the BioFlow UI."""
    script_dir = Path(__file__).parent
    ui_dir = script_dir / "ui"

    if not (ui_dir / "package.json").exists():
        print(f"❌ Error: Next.js UI not found at {ui_dir}")
        sys.exit(1)
    
    # Parse arguments
    port = 3000
    debug = False
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--port" and i + 1 < len(sys.argv[1:]):
            port = int(sys.argv[i + 2])
        elif arg == "--debug":
            debug = True
    
    env = os.environ.copy()
    env["PORT"] = str(port)
    if debug:
        env["NODE_OPTIONS"] = env.get("NODE_OPTIONS", "") + " --trace-warnings"
    
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
        subprocess.run(["pnpm", "dev"], cwd=str(ui_dir), env=env, check=False)
    except KeyboardInterrupt:
        print("\n\n👋 BioFlow server stopped.")
    except Exception as e:
        print(f"❌ Error launching BioFlow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
