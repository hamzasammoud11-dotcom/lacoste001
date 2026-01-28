#!/usr/bin/env python3
"""
Compare /api/search latency + diversity with and without MMR.
"""
import os
import statistics
import time
import requests

BASE_URL = os.getenv("BIOFLOW_API_URL", "http://localhost:8000")

QUERIES = [
    "EGFR inhibitor",
    "BRCA1 breast cancer",
    "kinase inhibitor therapy",
    "TP53 mutation cancer",
    "lung cancer EGFR signaling",
]


def run_batch(use_mmr: bool, runs: int = 10):
    latencies = []
    diversities = []
    for i in range(runs):
        q = QUERIES[i % len(QUERIES)]
        t0 = time.perf_counter()
        r = requests.post(
            f"{BASE_URL}/api/search",
            json={"query": q, "top_k": 20, "use_mmr": use_mmr, "modality": "auto"},
            timeout=30,
        )
        dt = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            continue
        data = r.json()
        latencies.append(dt)
        if data.get("diversity_score") is not None:
            diversities.append(float(data.get("diversity_score") or 0))
    return latencies, diversities


if __name__ == "__main__":
    print("MMR Benchmark")
    lat_no, div_no = run_batch(False, runs=10)
    lat_yes, div_yes = run_batch(True, runs=10)

    def _stats(xs):
        if not xs:
            return "n/a"
        return f"p50={statistics.median(xs):.1f}ms avg={statistics.mean(xs):.1f}ms"

    print(f"MMR OFF: {_stats(lat_no)} | diversity avg={statistics.mean(div_no) if div_no else 'n/a'}")
    print(f"MMR ON : {_stats(lat_yes)} | diversity avg={statistics.mean(div_yes) if div_yes else 'n/a'}")
