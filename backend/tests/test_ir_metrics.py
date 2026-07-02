"""IR metric correctness on hand-computed examples.

ir_metrics lives in the evaluation package (no heavy deps — stdlib only), so we
add it to sys.path rather than installing the whole evaluation project in CI.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

from metrics.ir_metrics import (  # noqa: E402
    average_precision,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

GOLD = {"a", "b"}
RETRIEVED = ["x", "a", "y", "b"]  # relevant at ranks 2 and 4


def test_recall_and_precision():
    assert recall_at_k(RETRIEVED, GOLD, 2) == 0.5      # 1 of 2 gold in top-2
    assert recall_at_k(RETRIEVED, GOLD, 4) == 1.0
    assert precision_at_k(RETRIEVED, GOLD, 4) == 0.5   # 2 of 4 retrieved relevant


def test_hit_and_mrr():
    assert hit_at_k(RETRIEVED, GOLD, 1) == 0.0
    assert hit_at_k(RETRIEVED, GOLD, 2) == 1.0
    assert reciprocal_rank(RETRIEVED, GOLD) == 0.5     # first hit at rank 2


def test_empty_gold_is_zero():
    assert recall_at_k(RETRIEVED, set(), 4) == 0.0
    assert ndcg_at_k(RETRIEVED, set(), 4) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x"], GOLD, 3) == pytest.approx(1.0)


def test_ndcg_matches_manual_dcg():
    # hits at ranks 2 and 4: DCG = 1/log2(3) + 1/log2(5)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # ideal: both gold at ranks 1,2
    assert ndcg_at_k(RETRIEVED, GOLD, 4) == pytest.approx(dcg / idcg, rel=1e-6)


def test_average_precision():
    # precision at hit ranks: 1/2 (rank 2) and 2/4 (rank 4); AP = mean over gold
    assert average_precision(RETRIEVED, GOLD) == pytest.approx((0.5 + 0.5) / 2)
