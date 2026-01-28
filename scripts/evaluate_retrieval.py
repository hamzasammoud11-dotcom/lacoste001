#!/usr/bin/env python3
"""
Evaluate retrieval quality against a small benchmark file.

Benchmark format (JSON):
{
  "queries": [
    {
      "name": "egfr_lung",
      "query": "EGFR lung cancer",
      "modality": "auto",
      "top_k": 20,
      "relevant_ids": ["..."],
      "relevance_by_id": {"...": 1.0, "...": 2.0}
    }
  ],
  "k": 10
}

Notes:
- `relevant_ids` is used for Recall@k and MRR@k.
- `relevance_by_id` is used for nDCG@k.
- You can provide either or both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import requests

from bioflow.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, help="Path to benchmark JSON")
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()

    bench = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    k = int(bench.get("k", 10))
    queries = bench.get("queries", [])
    if not queries:
        raise SystemExit("Benchmark has no queries")

    recalls: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []

    for q in queries:
        query = q["query"]
        modality = q.get("modality", "auto")
        top_k = int(q.get("top_k", max(k, 20)))

        r = requests.post(
            f"{args.base_url}/api/search",
            json={"query": query, "modality": modality, "top_k": top_k, "use_mmr": False},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()

        ranked_ids = [str(item.get("id")) for item in data.get("results", []) if item.get("id") is not None]

        relevant_ids = set(map(str, q.get("relevant_ids", [])))
        relevance_by_id = {str(k): float(v) for k, v in (q.get("relevance_by_id", {}) or {}).items()}

        if relevant_ids:
            recalls.append(recall_at_k(relevant_ids, ranked_ids, k))
            mrrs.append(mrr_at_k(relevant_ids, ranked_ids, k))

        if relevance_by_id:
            ndcgs.append(ndcg_at_k(relevance_by_id, ranked_ids, k))

        print(f"- {q.get('name', query[:30])}: got={len(ranked_ids)} recall@{k}={recalls[-1] if relevant_ids else 'n/a'} mrr@{k}={mrrs[-1] if relevant_ids else 'n/a'} ndcg@{k}={ndcgs[-1] if relevance_by_id else 'n/a'}")

    def _avg(xs: List[float]) -> float:
        return sum(xs) / float(len(xs)) if xs else 0.0

    print("=" * 60)
    print(f"Aggregate (@{k})")
    if recalls:
        print(f"Recall: {_avg(recalls):.4f}")
        print(f"MRR:    {_avg(mrrs):.4f}")
    if ndcgs:
        print(f"nDCG:   {_avg(ndcgs):.4f}")
    if not (recalls or ndcgs):
        print("No relevance labels provided; nothing to score.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

