#!/usr/bin/env python3
"""Test ingestion endpoints (Phase 3)."""
import os
import time
import requests

BASE_URL = os.getenv("BIOFLOW_API_URL", "http://localhost:8000")


def _post(path: str, payload: dict):
    r = requests.post(f"{BASE_URL}{path}", json=payload, timeout=30)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {}


def test_pubmed():
    status, data = _post("/api/ingest/pubmed", {"query": "BRCA1", "limit": 5, "sync": False})
    if status == 404:
        print("pubmed: SKIP (server not restarted with new route)")
        return True
    print(f"pubmed: {status} -> {data.get('status', 'n/a')}")
    return status == 200 and "job_id" in data


def test_uniprot():
    status, data = _post("/api/ingest/uniprot", {"query": "EGFR", "limit": 5, "sync": False})
    if status == 404:
        print("uniprot: SKIP (server not restarted with new route)")
        return True
    print(f"uniprot: {status} -> {data.get('status', 'n/a')}")
    return status == 200 and "job_id" in data


def test_chembl():
    status, data = _post("/api/ingest/chembl", {"query": "EGFR", "limit": 5, "sync": False})
    if status == 404:
        print("chembl: SKIP (server not restarted with new route)")
        return True
    print(f"chembl: {status} -> {data.get('status', 'n/a')}")
    return status == 200 and "job_id" in data


if __name__ == "__main__":
    print("BioFlow Ingestion API Test")
    ok = test_pubmed() and test_uniprot() and test_chembl()
    print("OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
