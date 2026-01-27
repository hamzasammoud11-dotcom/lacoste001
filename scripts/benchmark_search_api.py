#!/usr/bin/env python3
"""
Benchmark /api/search latency and error rate.

Usage:
  python scripts/benchmark_search_api.py --runs 50 --concurrency 5
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import statistics
import time
from typing import Any, Dict, List, Tuple

import requests


DEFAULT_QUERIES = [
    "EGFR inhibitor",
    "BRCA1 breast cancer",
    "kinase inhibitor therapy",
    "TP53 mutation cancer",
    "lung cancer EGFR signaling",
]


def _one(base_url: str, query: str, top_k: int, use_mmr: bool) -> Tuple[bool, float, str]:
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{base_url}/api/search",
            json={"query": query, "top_k": top_k, "use_mmr": use_mmr, "modality": "auto"},
            timeout=30,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            return False, dt_ms, f"HTTP {r.status_code}"
        return True, dt_ms, ""
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return False, dt_ms, str(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--mmr", action="store_true", help="Enable MMR")
    args = ap.parse_args()

    queries = (DEFAULT_QUERIES * ((args.runs // len(DEFAULT_QUERIES)) + 1))[: args.runs]

    latencies: List[float] = []
    errors: List[str] = []

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(_one, args.base_url, q, args.top_k, bool(args.mmr))
            for q in queries
        ]
        for f in cf.as_completed(futures):
            ok, dt_ms, err = f.result()
            latencies.append(dt_ms)
            if not ok:
                errors.append(err)

    latencies.sort()
    p50 = latencies[int(0.50 * (len(latencies) - 1))]
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    p99 = latencies[int(0.99 * (len(latencies) - 1))]

    print("=" * 60)
    print("BioFlow /api/search Benchmark")
    print("=" * 60)
    print(f"Runs: {args.runs} | Concurrency: {args.concurrency} | top_k: {args.top_k} | mmr: {bool(args.mmr)}")
    print(f"OK: {args.runs - len(errors)} | Errors: {len(errors)}")
    print(f"p50: {p50:.1f}ms | p95: {p95:.1f}ms | p99: {p99:.1f}ms | mean: {statistics.mean(latencies):.1f}ms")
    if errors:
        print("Sample errors:")
        for e in errors[:5]:
            print(f"  - {e}")

    # Non-zero exit on errors to allow CI usage.
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

