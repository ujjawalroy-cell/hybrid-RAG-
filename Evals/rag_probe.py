"""
Shared instrumentation for the latency / cost / reliability evals.

run_pipeline(question) runs the same path as main.py and Evals_completness.py
(hybrid_retrieve -> generate_answer), but stage by stage so each one can be timed:

  dense    query embedding (Ollama) + Chroma ANN search
  sparse   BM25 scoring (includes unpickling bm25_index.pkl, which load_bm25() does per call)
  fusion   weighted RRF
  hydrate  fetch chunk text + metadata from Chroma
  rerank   cross-encoder over the fused candidates
  generate Ollama answer generation

The stage sequence mirrors retrievals.retrieval.hybrid_retrieve(); keep the two in step.
"""

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)   # CHROMA_DIR and BM25_PATH are relative paths

# goldens contain non-ASCII (e.g. "QKᵀ"); the Windows console / redirected output defaults to cp1252
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

from citations.generate import (
    DEFAULT_MODEL,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    generate_answer_with_usage,
)
from retrievals.retrieval import (
    DENSE_WEIGHT,
    FINAL_TOP_N,
    FUSION_TOP_N,
    SPARSE_WEIGHT,
    dense_search,
    hydrate,
    reciprocal_rank_fusion,
    rerank,
    sparse_search,
)

GOLDEN_PATH = ROOT / "golden_dataset" / "gold_retrieval.json"
RESULTS_DIR = ROOT / "Evals" / "results"
REFUSAL_PHRASE = "I don't have enough information in the course material to answer that."
STAGES = ("dense", "sparse", "fusion", "hydrate", "rerank", "generate")


def run_pipeline(
    question: str,
    top_k: int = FINAL_TOP_N,
    generate: bool = True,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
    dense_k: int = 10,
    sparse_k: int = 10,
) -> dict:
    """
    One instrumented RAG call. Never raises: a failure is recorded with the stage it
    happened in, so the evals can count errors instead of crashing.
    """
    out = {
        "question": question,
        "ok": False,
        "error": None,
        "error_stage": None,
        "stages": {},
        "chunk_ids": [],
        "context_chars": 0,
        "confidence": None,
        "answer": None,
        "usage": None,
    }
    stage = None
    t_start = time.perf_counter()
    try:
        def timed(name, fn, *args, **kwargs):
            nonlocal stage
            stage = name
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            out["stages"][name] = time.perf_counter() - t0
            return result

        dense = timed("dense", dense_search, question, dense_k)
        sparse = timed("sparse", sparse_search, question, sparse_k)
        fused = timed("fusion", reciprocal_rank_fusion, dense, sparse,
                      dense_weight=DENSE_WEIGHT, sparse_weight=SPARSE_WEIGHT)[:FUSION_TOP_N]
        content = timed("hydrate", hydrate, [f["id"] for f in fused])
        candidates = [
            {"id": f["id"], "rrf_score": f["rrf_score"], **content[f["id"]]}
            for f in fused if f["id"] in content
        ]
        chunks = timed("rerank", rerank, question, candidates, top_k) if candidates else []

        out["chunk_ids"] = [c["id"] for c in chunks]
        out["context_chars"] = sum(len(c["text"]) for c in chunks)
        if chunks:
            # same normalisation rag_answer() uses for its confidence gate
            out["confidence"] = 1.0 / (1.0 + math.exp(-chunks[0]["rerank_score"]))

        if generate:
            usage = timed("generate", generate_answer_with_usage, question, chunks, model, None, timeout)
            out["answer"] = usage.pop("answer")
            out["usage"] = usage
            if usage["error"]:
                out["error"], out["error_stage"] = usage["error"], "generate"
                return out

        out["ok"] = True
    except Exception as e:   # noqa: BLE001 -- the reliability eval wants every failure mode
        out["error"] = f"{type(e).__name__}: {e}"
        out["error_stage"] = stage
        out["traceback"] = traceback.format_exc(limit=3)
    finally:
        out["total_s"] = time.perf_counter() - t_start
        out["retrieval_s"] = sum(v for k, v in out["stages"].items() if k != "generate")
    return out


def would_gate(result: dict) -> bool:
    """True if rag_answer()'s confidence gate would refuse this result."""
    return result["confidence"] is not None and result["confidence"] < RETRIEVAL_CONFIDENCE_THRESHOLD


# ------------------------------------------------------------- stats / io
def percentile(values, p):
    """Linear-interpolated percentile, p in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def describe(values) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"),
                "p99": float("nan"), "max": float("nan")}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def load_questions(limit=None) -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        goldens = json.load(f)
    return goldens[:limit] if limit else goldens


def save_results(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def print_checks(checks: list[tuple[str, bool, str]]) -> bool:
    """Prints PASS/FAIL lines for (name, passed, detail) and returns True if all passed."""
    print("\nchecks")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<34} {detail}")
    return all(p for _, p, _ in checks)
