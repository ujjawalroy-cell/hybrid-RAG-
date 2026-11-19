"""Pytest/DeepEval smoke subset:  deepeval test run Evals/test_regression.py

Runs the first SMOKE_N goldens (env EVAL_SMOKE_N, default 10) through the real
retriever and asserts the contextual metrics pass. Needs Ollama + a synced index.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
from deepeval import assert_test

from Evals.Eval_retrieval import GOLDEN_PATH, build_metrics, build_test_cases
from Evals.judge import OllamaJudge
from golden_dataset.harness import load_goldens

SMOKE_N = int(os.getenv("EVAL_SMOKE_N", "10"))
_goldens = load_goldens(GOLDEN_PATH)[:SMOKE_N]


@pytest.fixture(scope="module")
def cases():
    import chunking

    if not chunking.verify_sync():
        pytest.fail("Chroma and BM25 are out of sync; run the ingest flow first.")
    cases, _ = build_test_cases(_goldens)
    return {c.name: c for c in cases}


@pytest.mark.parametrize("golden_id", [g["id"] for g in _goldens])
def test_retrieval_quality(cases, golden_id):
    assert_test(cases[golden_id], build_metrics(OllamaJudge()))
