"""Retrieval quality eval: DeepEval Contextual Recall / Precision / Relevancy.

Run from the repo root:  python Evals/run_regression.py --suite retrieval
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os as _os
_os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "1200")  # local judge is slow

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

from golden_dataset.harness import load_goldens, strip_cite_markers, summarize_by_metric
from retrievals.retrieval import hybrid_retrieve

GOLDEN_PATH = ROOT / "golden_dataset" / "gold_retrieval.json"
THRESHOLD = 0.5


def build_metrics(judge, threshold: float = THRESHOLD):
    kw = dict(threshold=threshold, model=judge, include_reason=False, async_mode=False)
    return [
        ContextualRecallMetric(**kw),
        ContextualPrecisionMetric(**kw),
        ContextualRelevancyMetric(**kw),
    ]


def build_test_cases(goldens, rerank_method="cross_encoder", top_k=5):
    """Run the real retriever per golden. Returns (test_cases, retrieved_ids)."""
    cases, retrieved = [], {}
    for g in goldens:
        chunks = hybrid_retrieve(g["question"], final_top_n=top_k, rerank_method=rerank_method)
        retrieved[g["id"]] = [c["id"] for c in chunks]
        cases.append(
            LLMTestCase(
                input=g["question"],
                expected_output=strip_cite_markers(g["ideal_answer"]),
                actual_output="(generator not evaluated in this run)",
                retrieval_context=[c["text"] for c in chunks],
                name=g["id"],
            )
        )
    return cases, retrieved


def run_retrieval_eval(goldens, judge, rerank_method="cross_encoder", limit=None) -> dict:
    """Returns the per-metric summary, with retrieved chunk ids under '_retrieved'."""
    goldens = goldens[:limit] if limit else goldens
    cases, retrieved = build_test_cases(goldens, rerank_method)
    result = evaluate(
        test_cases=cases,
        metrics=build_metrics(judge),
        hyperparameters={
            "embedding_model": "nomic-embed-text",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 5,
            "rrf_weights": "0.7/0.3",
            "rerank": rerank_method,
            "judge_model": judge.get_model_name(),
        },
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=False, show_indicator=False),
    )
    summary = summarize_by_metric(result)
    return {"summary": summary, "retrieved": retrieved}


if __name__ == "__main__":
    from Evals.judge import OllamaJudge
    from golden_dataset.harness import print_summary

    out = run_retrieval_eval(load_goldens(GOLDEN_PATH), OllamaJudge(), limit=5)
    print_summary("Retrieval", out["summary"])
