import os
import sys
import json
from pathlib import Path
from typing import Any, Callable, Optional, Union

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Fix Windows console UTF-8 output encoding for emoji print statements in deepeval
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import OllamaModel

# Project imports
from golden_dataset.harness import load_goldens, summarize_by_metric, print_summary

try:
    from retrievals.retrieval import hybrid_retrieve
except ImportError:
    from retrival import hybrid_retrieve

from citations.generate import generate_answer, rag_answer

load_dotenv()

# ----------------------------------------------------------------- Config
GOLDEN_PATH = ROOT / "golden_dataset" / "gold_retrieval.json"
JUDGE_MODEL_NAME = "llama3:latest"          # Must match `ollama list`
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").strip().strip('"').strip("'").rstrip("/")
THRESHOLD = 0.7
FINAL_TOP_N = 5
RERANK_METHOD = "cross_encoder"           # "cross_encoder" or "llm"
LIMIT = None                               # Default to small sample; set None for all goldens

# DeepEval requires an OllamaModel instance for local Ollama judge models
judge = OllamaModel(model=JUDGE_MODEL_NAME, base_url=OLLAMA_BASE_URL)


# ----------------------------------------------------------------- Default Pipeline
def default_rag_pipeline(
    question: str,
    top_k: int = FINAL_TOP_N,
    rerank_method: str = RERANK_METHOD,
) -> dict:
    """
    Executes hybrid retrieval followed by LLM answer generation.
    Returns a dict with 'answer', 'retrieval_context', and 'chunks'.
    """
    chunks = hybrid_retrieve(
        question,
        final_top_n=top_k,
        rerank_method=rerank_method,
    )
    answer = generate_answer(question, chunks)
    retrieval_context = [c["text"] for c in chunks if isinstance(c, dict) and "text" in c]

    return {
        "answer": answer,
        "retrieval_context": retrieval_context,
        "chunks": chunks,
    }


# ----------------------------------------------------------------- Run Evaluation
def run(rag: Optional[Union[Callable[[str], Any], Any]] = None, limit: Optional[int] = LIMIT):
    """
    Evaluates RAG generation against golden dataset on Correctness and Completeness.
    rag: Optional callable or pipeline object taking question -> output dict or answer string.
         If None, uses default_rag_pipeline.
    limit: Max number of golden test cases to run (None for all).
    """
    # 1. LOAD queries + ideal answers
    goldens = load_goldens(GOLDEN_PATH)
    if limit:
        goldens = goldens[:limit]

    print(f"Loaded {len(goldens)} golden test case(s) from {GOLDEN_PATH.name}...")

    # 2. RUN THE PIPELINE per query and build LLMTestCase
    test_cases = []
    for idx, g in enumerate(goldens, start=1):
        question = g.get("question") or g.get("query") or g.get("input")
        ideal_answer = g.get("ideal_answer") or g.get("expected_output") or g.get("answer")

        if not question or not ideal_answer:
            continue

        print(f"[{idx}/{len(goldens)}] Running RAG pipeline for: {question[:70]}...")

        if rag is not None:
            if callable(rag):
                pipeline_output = rag(question)
            elif hasattr(rag, "ask"):
                pipeline_output = rag.ask(question)
            elif hasattr(rag, "invoke"):
                pipeline_output = rag.invoke(question)
            else:
                raise TypeError(f"Unsupported pipeline object type: {type(rag)}")

            if isinstance(pipeline_output, dict):
                actual_answer = pipeline_output.get("answer", "")
                retrieval_context = pipeline_output.get("retrieval_context") or [
                    c.get("text", "") for c in pipeline_output.get("chunks_used", pipeline_output.get("chunks", []))
                    if isinstance(c, dict) and "text" in c
                ]
            else:
                actual_answer = str(pipeline_output)
                retrieval_context = []
        else:
            pipeline_output = default_rag_pipeline(
                question,
                top_k=FINAL_TOP_N,
                rerank_method=RERANK_METHOD,
            )
            actual_answer = pipeline_output["answer"]
            retrieval_context = pipeline_output["retrieval_context"]

        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=actual_answer,
                expected_output=ideal_answer,
                retrieval_context=retrieval_context if retrieval_context else None,
            )
        )

    # 3. DEFINE QUALITY METRICS

    # 3a. CORRECTNESS --- reference-based, judges TRUTH (not coverage or length)
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
            "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
            "Do NOT deduct for brevity, missing elaboration, or omitted points --- omissions are not errors here.",
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
            Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
            Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )

    # 3b. COMPLETENESS --- reference-based, judges COVERAGE (not correctness)
    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points contained in the expected output.",
            "Check how many of those key points are addressed in the actual output.",
            "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
            "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly --- factual correctness is judged separately.",
            "Do NOT penalize the actual output for adding extra information beyond the expected output.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
            Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
            Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )

    # 3c. STYLE (Optional) --- reference-free, judges TONE only
    # style = GEval(
    #     name="Style",
    #     evaluation_steps=[
    #         "Judge only the teaching style and tone of the actual output, not whether it is factually correct or complete.",
    #         "Reward an intuitive, explanatory tone: plain language, the idea explained before any formula or jargon.",
    #         "Reward a direct, conversational register written in prose.",
    #         "Penalize answers that are stiff, bureaucratic, structured as a bare list with no explanation, or that use unexplained jargon.",
    #     ],
    #     rubric=[
    #         Rubric(score_range=(9, 10), expected_outcome="Intuitive, conversational prose that explains before it formalizes."),
    #         Rubric(score_range=(5, 8),  expected_outcome="Clear and conversational, understandable in places."),
    #         Rubric(score_range=(0, 4),  expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic."),
    #     ],
    #     evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    #     threshold=THRESHOLD,
    #     model=judge,
    #     strict_mode=False,
    # )

    # 4. EVALUATE
    print(f"\nEvaluating {len(test_cases)} test case(s) with judge '{JUDGE_MODEL_NAME}'...")
    metrics = [correctness, completeness]

    try:
        eval_result = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            hyperparameters={
                "retriever": "hybrid_rrf_rerank",
                "rerank_method": RERANK_METHOD,
                "top_k": FINAL_TOP_N,
                "judge_model": JUDGE_MODEL_NAME,
                "golden_set": str(GOLDEN_PATH.name),
            },
        )

        # 5. DETAILED REPORT
        passed_tests = []
        failed_tests = []

        test_results = getattr(eval_result, "test_results", None) or (eval_result if isinstance(eval_result, list) else [])
        for idx, tr in enumerate(test_results, 1):
            if getattr(tr, "success", False):
                passed_tests.append((idx, tr))
            else:
                failed_tests.append((idx, tr))

        total = len(test_results)
        pass_count = len(passed_tests)
        fail_count = len(failed_tests)
        pass_rate = (pass_count / total * 100) if total > 0 else 0

        print("\n" + "=" * 80)
        print(f"📊 EVALUATION SUMMARY: {pass_count}/{total} Passed ({pass_rate:.1f}%) | {fail_count} Failed")
        print("=" * 80)

        if passed_tests:
            print(f"\n✅ PASSED TEST CASES ({pass_count}):")
            print("-" * 80)
            for idx, tr in passed_tests:
                scores = []
                metrics_data = getattr(tr, "metrics_data", None) or getattr(tr, "metrics", [])
                for m in metrics_data:
                    score = f"{m.score:.2f}" if getattr(m, "score", None) is not None else "N/A"
                    scores.append(f"{getattr(m, 'name', 'Metric')}: {score}")
                score_str = " | ".join(scores) if scores else "All metrics passed"
                print(f"  [{idx}/{total}] PASS - Query: {getattr(tr, 'input', '')[:80]}")
                print(f"         Scores: {score_str}")

        if failed_tests:
            print(f"\n❌ FAILED TEST CASES ({fail_count}):")
            print("-" * 80)
            for idx, tr in failed_tests:
                print(f"\n  [{idx}/{total}] FAIL - Query: {getattr(tr, 'input', '')[:80]}")
                print(f"         Expected: {getattr(tr, 'expected_output', '')[:100]}...")
                metrics_data = getattr(tr, "metrics_data", None) or getattr(tr, "metrics", [])
                for m in metrics_data:
                    status = "✅ PASS" if getattr(m, "success", False) else "❌ FAIL"
                    score = f"{m.score:.2f}" if getattr(m, "score", None) is not None else "N/A"
                    thresh = f"{m.threshold:.2f}" if getattr(m, "threshold", None) is not None else "N/A"
                    print(f"         • {getattr(m, 'name', 'Metric')}: {score} (Threshold: {thresh}) -> {status}")
                    if not getattr(m, "success", False) and getattr(m, "reason", None):
                        print(f"           Reason: {m.reason}")
                    if getattr(m, "error", None):
                        print(f"           Error: {m.error}")

        summary = summarize_by_metric(eval_result)
        return summary

    except Exception as e:
        print(f"Error evaluating test cases: {e}")
        return {}


def run_local(limit: Optional[int] = LIMIT):
    """Standalone convenience: run the evaluation on the default RAG pipeline."""
    return run(rag=None, limit=limit)


if __name__ == "__main__":
    summary = run_local()
    if summary:
        print_summary("Application (Correctness & Completeness)", summary)