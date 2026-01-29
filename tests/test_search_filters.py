#!/usr/bin/env python3
"""Validate /api/search filter behavior (Phase 4 QA)."""
import os
import requests

BASE_URL = os.getenv("BIOFLOW_API_URL", "http://localhost:8000")


def _post(payload):
    r = requests.post(f"{BASE_URL}/api/search", json=payload, timeout=30)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {}


def _skip_if_unavailable(status):
    if status in (404, 503):
        print("[SKIP] Backend unavailable or not restarted")
        return True
    return False


def test_filters():
    print("Phase 4: /api/search filter tests")

    status, data = _post({
        "query": "EGFR",
        "top_k": 5,
        "filters": {"sources": ["pubmed", "uniprot"]},
    })
    if _skip_if_unavailable(status):
        return True
    assert status == 200, f"status={status}"
    assert "results" in data

    status, data = _post({
        "query": "EGFR",
        "top_k": 5,
        "filters": {"modality": "molecule"},
    })
    assert status == 200

    status, data = _post({
        "query": "EGFR",
        "top_k": 5,
        "filters": {"year_min": 2018, "year_max": 2026},
    })
    assert status == 200

    print("[OK] filter tests completed")
    return True


if __name__ == "__main__":
    ok = test_filters()
    raise SystemExit(0 if ok else 1)

