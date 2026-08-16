"""Standard IR metrics for retrieval evaluation.

Each metric operates on a single query and takes:
    retrieved: ordered list of doc identifiers returned by the retriever
    gold:     set of doc identifiers considered relevant for the query
"""

from __future__ import annotations

import math
from typing import Hashable, Sequence


def recall_at_k(retrieved: Sequence[Hashable], gold: set, k: int) -> float:
    """Fraction of relevant docs that appear in the top-k.

    Recall@k = |relevant ∩ retrieved[:k]| / |relevant|.
    Returns 0.0 when the gold set is empty.
    """
    if not gold:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in gold)
    return hits / len(gold)


def precision_at_k(retrieved: Sequence[Hashable], gold: set, k: int) -> float:
    """Fraction of top-k that are relevant.

    Precision@k = |relevant ∩ retrieved[:k]| / k.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in gold)
    return hits / k


def hit_at_k(retrieved: Sequence[Hashable], gold: set, k: int) -> float:
    """Binary indicator: 1.0 if any relevant doc appears in top-k, else 0.0.

    Often more appropriate than Recall@k when the gold set is small (e.g.
    'cluster gold' where all chunks of one source example count as relevant)
    and we mostly care whether the retriever surfaced *something* useful.
    """
    return 1.0 if any(d in gold for d in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[Hashable], gold: set) -> float:
    """Reciprocal rank of the first relevant doc; 0 if none retrieved.

    MRR over a query set = mean of reciprocal_rank per query.
    """
    for rank, d in enumerate(retrieved, start=1):
        if d in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[Hashable], gold: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain @ k with binary relevance.

    DCG = sum_{i=1..k} rel_i / log2(i + 1)
    IDCG = DCG of an ideal ranking with min(|gold|, k) ones at the top.
    Returns DCG / IDCG, or 0.0 if gold is empty.
    """
    if not gold:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        if d in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: Sequence[Hashable], gold: set) -> float:
    """Average precision: mean of precision@k at each relevant-hit rank.

    AP = (1/|gold|) * sum_k [ rel_k * precision@k ].
    MAP over a query set = mean of AP per query.
    """
    if not gold:
        return 0.0
    hits = 0
    score = 0.0
    for rank, d in enumerate(retrieved, start=1):
        if d in gold:
            hits += 1
            score += hits / rank
    return score / len(gold)
