from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Set


def recall_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    return len(relevant.intersection(top)) / float(len(relevant))


def mrr_at_k(relevant: Set[str], ranked: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in relevant:
            return 1.0 / float(i + 1)
    return 0.0


def _dcg(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        total += (2.0 ** float(rel) - 1.0) / math.log2(float(i + 2))
    return total


def ndcg_at_k(relevance_by_id: Mapping[str, float], ranked: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    if not relevance_by_id:
        return 0.0

    rels = [float(relevance_by_id.get(doc_id, 0.0)) for doc_id in ranked]
    dcg = _dcg(rels, k)

    ideal_rels = sorted(relevance_by_id.values(), reverse=True)
    idcg = _dcg(ideal_rels, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    denom = math.sqrt(na) * math.sqrt(nb)
    return (dot / denom) if denom else 0.0


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return 1.0 - cosine_similarity(a, b)


def intra_list_diversity_cosine(embeddings: Sequence[Sequence[float]]) -> float:
    """
    Average pairwise cosine distance within a list.
    Higher = more diverse.
    """
    n = len(embeddings)
    if n < 2:
        return 1.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += cosine_distance(embeddings[i], embeddings[j])
            count += 1
    return total / float(count) if count else 1.0

