#!/usr/bin/env python3
"""
CI-style test runner for core BioFlow checks.
"""
import os
import subprocess
import sys


TESTS = [
    ["python", "test_search_api.py"],
    ["python", "test_agent_api.py"],
    ["python", "test_search_filters.py"],
    ["python", "test_phase4_ui.py"],
    ["python", "test_ingestion_api.py"],
]


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    failed = False

    for cmd in TESTS:
        print("=" * 80)
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

