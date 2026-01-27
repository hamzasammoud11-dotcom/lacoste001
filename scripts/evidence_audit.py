#!/usr/bin/env python3
"""
Audit evidence-link coverage in /api/search results.
"""
import os
import requests

BASE_URL = os.getenv("BIOFLOW_API_URL", "http://localhost:8000")

QUERIES = [
    "EGFR inhibitor",
    "BRCA1 breast cancer",
    "kinase inhibitor therapy",
    "TP53 mutation cancer",
]


def main():
    total = 0
    with_evidence = 0

    for q in QUERIES:
        r = requests.post(
            f"{BASE_URL}/api/search",
            json={"query": q, "top_k": 10, "use_mmr": True},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"[SKIP] {q} -> {r.status_code}")
            continue
        data = r.json()
        for item in data.get("results", []):
            total += 1
            if item.get("evidence_links"):
                with_evidence += 1

    if total == 0:
        print("No results returned; evidence audit skipped.")
        return 0

    coverage = (with_evidence / total) * 100.0
    print(f"Evidence coverage: {with_evidence}/{total} ({coverage:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

