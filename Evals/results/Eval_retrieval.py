"""
Retrieval evaluation with DeepEval: contextual precision and recall for BM25, vector
search and hybrid (weighted RRF, no reranker), on the same golden questions.

Run from the repo root (the Chroma and BM25 paths are CWD-relative):

  python Evals/Eval_retrieval.py [--systems bm25,vector,hybrid] [--k 5] [--limit N]
                                 [--judge MODEL] [--threshold T]
                                 [--dense-weight W] [--rrf-k K]

Each system retrieves its top-k chunks for every question; DeepEval's judge model then scores
them against the golden `ideal_answer`:

  ContextualPrecisionMetric  are the relevant chunks ranked above the irrelevant ones
                             (rank-weighted, so order matters)
  ContextualRecallMetric     is everything in the ideal answer supported by the chunks

Both are LLM-judged, so scores depend on the judge model and can vary slightly between runs.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
os.chdir(ROOT)   # CHROMA_DIR and BM25_PATH are relative paths

# A local judge can take minutes on a long context; DeepEval's default 180s per-attempt limit
# aborted the full run. Must be set BEFORE deepeval is imported.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "1200")

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from golden_dataset.harness import load_goldens, summarize_by_metric
from retrievals.retrieval import (
    DENSE_WEIGHT,
    RRF_K,
    dense_search,
    hydrate,
    reciprocal_rank_fusion,
    sparse_search,
)

load_dotenv()

# ------------------------------------------------------------- config
GOLDEN_PATH = ROOT / "golden_dataset" / "gold_retrieval.json"
RESULTS_DIR = ROOT / "Evals" / "results"
OLLAMA_URL = "http://localhost:11434"
JUDGE_MODEL = "llama3:latest"      # must match `ollama list` exactly
THRESHOLD = 0.7
TOP_K = 5
SYSTEMS = ("bm25", "vector", "hybrid")


# ------------------------------------------------------------- retrieval
def retrieve_ids(system, question, k, dense_weight, rrf_k):
    """Ranked chunk ids (best first) for one system."""
    if system == "bm25":
        return [r["id"] for r in sparse_search(question, k)]
    if system == "vector":
        return [r["id"] for r in dense_search(question, k)]
    # hybrid: fuse the two lists at the same depth, no reranker
    fused = reciprocal_rank_fusion(
        dense_search(question, k),
        sparse_search(question, k),
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        sparse_weight=1.0 - dense_weight,
    )
    return [r["id"] for r in fused[:k]]


def build_test_cases(system, goldens, args):
    cases = []
    for g in goldens:
        ids = retrieve_ids(system, g["question"], args.k, args.dense_weight, args.rrf_k)
        content = hydrate(ids)
        cases.append(
            LLMTestCase(
                input=g["question"],
                expected_output=g["ideal_answer"],
                retrieval_context=[content[cid]["text"] for cid in ids if cid in content],
                actual_output="(generator not evaluated in this run)",
            )
        )
    return cases


# ------------------------------------------------------------- eval
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", default=",".join(SYSTEMS), help="comma-separated: bm25,vector,hybrid")
    p.add_argument("--k", type=int, default=TOP_K, help="chunks retrieved per question")
    p.add_argument("--limit", type=int, default=None, help="only the first N goldens")
    p.add_argument("--judge", default=JUDGE_MODEL)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--dense-weight", type=float, default=DENSE_WEIGHT)
    p.add_argument("--rrf-k", type=int, default=RRF_K)
    args = p.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        sys.exit(f"unknown system(s): {', '.join(sorted(unknown))}")

    goldens = load_goldens(GOLDEN_PATH)
    if args.limit:
        goldens = goldens[: args.limit]

    judge = OllamaModel(model=args.judge, base_url=OLLAMA_URL)

    summaries = {}
    for system in systems:
        print(f"\n=== {system}: retrieving top {args.k} for {len(goldens)} questions ===")
        test_cases = build_test_cases(system, goldens, args)

        # fresh metric objects per system so one run's state never leaks into the next
        metrics = [
            ContextualPrecisionMetric(threshold=args.threshold, model=judge, include_reason=True),
            ContextualRecallMetric(threshold=args.threshold, model=judge, include_reason=True),
        ]
        result = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            async_config=AsyncConfig(run_async=False),   # one request at a time: friendlier to a local Ollama
            error_config=ErrorConfig(ignore_errors=True),   # one bad judge call must not discard an hour of work
            hyperparameters={
                "retriever": system,
                "top_k": args.k,
                "dense_weight": args.dense_weight,
                "sparse_weight": 1.0 - args.dense_weight,
                "rrf_k": args.rrf_k,
                "judge_model": args.judge,
                "golden_set": GOLDEN_PATH.name,
            },
        )
        summaries[system] = summarize_by_metric(result)

    # ------------------------------------------------------------- comparison
    metric_names = sorted({m for s in summaries.values() for m in s})
    print("\n" + "=" * 78)
    print(f"retrieval comparison  (top {args.k}, {len(goldens)} questions, judge {args.judge}, "
          f"threshold {args.threshold})")
    print("=" * 78)
    print(f"{'system':<8}" + "".join(f"{m + ' avg':>26}{'pass%':>8}" for m in metric_names))
    for system, summary in summaries.items():
        row = f"{system:<8}"
        for m in metric_names:
            s = summary.get(m)
            row += f"{s['avg_score']:>26.3f}{s['pass_rate']:>8.0f}" if s else f"{'n/a':>26}{'':>8}"
        print(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"deepeval_retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "systems": systems, "top_k": args.k, "n_questions": len(goldens),
                "judge_model": args.judge, "threshold": args.threshold,
                "dense_weight": args.dense_weight, "rrf_k": args.rrf_k,
                "golden_set": GOLDEN_PATH.name,
            },
            "summary": summaries,
        }, f, indent=2)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
