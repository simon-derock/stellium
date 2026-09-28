# Unit checks for corpus-grounded dense retrieval metrics.
from __future__ import annotations

from scripts.evaluate_dense_retrieval import _metrics


def test_dense_retrieval_metrics_count_gold_documents_once() -> None:
    metrics = _metrics(["doc-a", "doc-a", "doc-b"], ["doc-a", "doc-b"], 2)

    assert metrics == {"hit": 1.0, "recall": 0.5, "mrr": 1.0}


def test_dense_retrieval_mrr_respects_cutoff() -> None:
    metrics = _metrics(["doc-x", "doc-y", "doc-gold"], ["doc-gold"], 2)

    assert metrics == {"hit": 0.0, "recall": 0.0, "mrr": 0.0}


def test_dense_retrieval_empty_results_are_zero() -> None:
    metrics = _metrics([], ["doc-gold"], 5)

    assert metrics == {"hit": 0.0, "recall": 0.0, "mrr": 0.0}
