"""Generation quality eval: DeepEval Faithfulness / Answer Relevancy over rag_answer().

Refusals (status == insufficient_context) are a valid outcome of the confidence
gate; they are counted separately and excluded from the generation metrics.
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
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from citations.generate import rag_answer
from golden_dataset.harness import strip_cite_markers, summarize_by_metric

THRESHOLD = 0.5


def build_metrics(judge, threshold: float = THRESHOLD):
    kw = dict(threshold=threshold, model=judge, include_reason=False, async_mode=False)
    return [FaithfulnessMetric(**kw), AnswerRelevancyMetric(**kw)]


def build_test_cases(goldens):
    """Returns (cases, refused_ids, answered_ids, retrieved_ids)."""
    cases, refused, answered, retrieved = [], [], [], {}
    for g in goldens:
        res = rag_answer(g["question"])
        retrieved[g["id"]] = [c["id"] for c in res.get("chunks_used", [])]
        if res["status"] != "answered":
            refused.append(g["id"])
            continue
        answered.append(g["id"])
        cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=res["answer"],
                expected_output=strip_cite_markers(g["ideal_answer"]),
                retrieval_context=[c["text"] for c in res["chunks_used"]],
                name=g["id"],
            )
        )
    return cases, refused, answered, retrieved


def run_generation_eval(goldens, judge, limit=None) -> dict:
    goldens = goldens[:limit] if limit else goldens
    cases, refused, answered, retrieved = build_test_cases(goldens)
    summary = {}
    if cases:
        result = evaluate(
            test_cases=cases,
            metrics=build_metrics(judge),
            async_config=AsyncConfig(run_async=False),
            display_config=DisplayConfig(print_results=False, show_indicator=False),
        )
        summary = summarize_by_metric(result)
    return {"summary": summary, "refusals": refused, "answered": answered, "retrieved": retrieved}
