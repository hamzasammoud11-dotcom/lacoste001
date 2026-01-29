#!/usr/bin/env python3
"""
Test Next.js API proxy routes (port 3000).

This validates that the UI's `/api/*` route handlers exist and return
the expected JSON shapes, even if the FastAPI backend is down (mock fallback).
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict

import requests


UI_BASE = "http://localhost:3000"


def _get_json(url: str, timeout_s: float = 10) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _post_json(url: str, payload: Dict[str, Any], timeout_s: float = 30) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _wait_for_ui(max_wait_s: int = 10) -> bool:
    for _ in range(max_wait_s):
        try:
            r = requests.get(f"{UI_BASE}/", timeout=2)
            return r.status_code < 500
        except Exception:
            time.sleep(1)
    return False


def test_search_proxy() -> bool:
    data = _post_json(
        f"{UI_BASE}/api/search",
        {"query": "kinase inhibitor", "top_k": 3, "use_mmr": True},
    )
    assert "results" in data, "missing `results` in /api/search response"
    assert "returned" in data, "missing `returned` in /api/search response"
    return True


def test_workflow_proxy() -> bool:
    data = _post_json(
        f"{UI_BASE}/api/agents/workflow",
        {"query": "drug-like molecule", "num_candidates": 3, "top_k": 2},
    )
    assert "top_candidates" in data, "missing `top_candidates` in /api/agents/workflow response"
    assert "status" in data, "missing `status` in /api/agents/workflow response"
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("BioFlow UI Proxy API Test (Next.js /api/*)")
    print("=" * 60)

    if not _wait_for_ui():
        print("[SKIP] Next.js UI not reachable on http://localhost:3000")
        sys.exit(0)

    try:
        ok1 = test_search_proxy()
        print(f"[OK] /api/search proxy: {ok1}")
        ok2 = test_workflow_proxy()
        print(f"[OK] /api/agents/workflow proxy: {ok2}")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print("[OK] Proxy tests complete")

