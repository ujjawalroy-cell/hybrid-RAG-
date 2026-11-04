"""
Latency eval: how long each stage of the RAG pipeline takes, end to end and per stage.

  python Evals/Evals_latency.py [--limit N] [--repeats R] [--warmup W] [--no-generate]
                                [--p95-retrieval S] [--p95-e2e S]

1. Cold start: the very first call, which pays for loading the cross-encoder and the
   Ollama models. Reported separately so it does not skew the steady-state numbers.
2. Warm-up: W more calls, discarded.
3. Measured: every golden question (x R repeats) through rag_probe.run_pipeline().

Reports mean / p50 / p95 / p99 / max per stage, retrieval total and end to end, plus
Ollama's server-side time-to-first-token (load + prompt eval) and generation tokens/s.
Exits with code 1 if a p95 budget is exceeded, so it can gate a regression run.

Numbers are only meaningful on an otherwise idle machine: another eval sharing the same
Ollama server (or GPU) inflates them.
"""

import argparse
import sys

from rag_probe import (
    DEFAULT_MODEL,
    FINAL_TOP_N,
    STAGES,
    describe,
    load_questions,
    print_checks,
    run_pipeline,
    save_results,
)

# ------------------------------------------------------------- config
P95_RETRIEVAL_BUDGET_S = 3.0     # dense + sparse + fusion + hydrate + rerank
P95_E2E_BUDGET_S = 60.0          # retrieval + generation on a local model
WARMUP = 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="only the first N goldens")
    p.add_argument("--repeats", type=int, default=1, help="times each question is run")
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--no-generate", action="store_true", help="time retrieval only")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--top-k", type=int, default=FINAL_TOP_N)
    p.add_argument("--p95-retrieval", type=float, default=P95_RETRIEVAL_BUDGET_S)
    p.add_argument("--p95-e2e", type=float, default=P95_E2E_BUDGET_S)
    args = p.parse_args()

    generate = not args.no_generate
    goldens = load_questions(args.limit)
    questions = [g["question"] for g in goldens]
    run = lambda q: run_pipeline(q, top_k=args.top_k, generate=generate, model=args.model)

    # ------------------------------------------------------------- cold start + warm-up
    print("cold start ...")
    cold = run(questions[0])
    for i in range(args.warmup):
        print(f"warm-up {i + 1}/{args.warmup} ...")
        run(questions[(i + 1) % len(questions)])

    # ------------------------------------------------------------- measured runs
    results = []
    total = len(questions) * args.repeats
    for r in range(args.repeats):
        for i, q in enumerate(questions, start=1):
            res = run(q)
            res["repeat"] = r
            results.append(res)
            status = "ok" if res["ok"] else f"ERROR in {res['error_stage']}: {res['error']}"
            print(f"[{len(results)}/{total}] {res['total_s']:6.2f}s  {q[:60]}  {status}")

    ok = [r for r in results if r["ok"]]
    errors = len(results) - len(ok)

    # ------------------------------------------------------------- stats
    stage_stats = {s: describe([r["stages"].get(s) for r in ok]) for s in STAGES if generate or s != "generate"}
    stage_stats["retrieval"] = describe([r["retrieval_s"] for r in ok])
    stage_stats["end_to_end"] = describe([r["total_s"] for r in ok])

    gen_stats = {}
    if generate:
        usages = [r["usage"] for r in ok if r["usage"]]
        gen_stats = {
            "ttft_server_s": describe([u["load_s"] + u["prompt_eval_s"] for u in usages]),
            "tokens_per_s": describe([u["completion_tokens"] / u["eval_s"] for u in usages if u["eval_s"] > 0]),
            "completion_tokens": describe([u["completion_tokens"] for u in usages]),
        }

    print("\n" + "=" * 78)
    print(f"latency  ({len(ok)} successful calls, {errors} errors excluded, model {args.model})")
    print("=" * 78)
    print(f"cold start: {cold['total_s']:.2f}s  (stages: "
          + ", ".join(f"{k} {v:.2f}" for k, v in cold["stages"].items()) + ")")
    print(f"\n{'stage':<12}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}   (seconds)")
    for name, s in stage_stats.items():
        if name in ("retrieval", "end_to_end"):
            print("-" * 66)
        print(f"{name:<12}" + "".join(f"{s[k]:>9.3f}" for k in ("mean", "p50", "p95", "p99", "max")))

    if gen_stats:
        print(f"\n{'generation':<20}{'mean':>9}{'p50':>9}{'p95':>9}")
        for name, s in gen_stats.items():
            print(f"{name:<20}" + "".join(f"{s[k]:>9.2f}" for k in ("mean", "p50", "p95")))

    if ok:
        slowest = max(ok, key=lambda r: r["total_s"])
        print(f"\nslowest call: {slowest['total_s']:.2f}s  {slowest['question'][:70]}")

    # ------------------------------------------------------------- budgets
    checks = [(
        "retrieval p95 <= budget",
        stage_stats["retrieval"]["p95"] <= args.p95_retrieval,
        f"{stage_stats['retrieval']['p95']:.3f}s vs {args.p95_retrieval}s",
    )]
    if generate:
        checks.append((
            "end-to-end p95 <= budget",
            stage_stats["end_to_end"]["p95"] <= args.p95_e2e,
            f"{stage_stats['end_to_end']['p95']:.3f}s vs {args.p95_e2e}s",
        ))
    checks.append(("at least one successful call", bool(ok), f"{len(ok)}/{len(results)}"))
    passed = print_checks(checks)

    path = save_results("latency", {
        "config": {
            "n_questions": len(questions), "repeats": args.repeats, "warmup": args.warmup,
            "generate": generate, "model": args.model, "top_k": args.top_k,
            "p95_retrieval_budget_s": args.p95_retrieval, "p95_e2e_budget_s": args.p95_e2e,
        },
        "cold_start": cold,
        "stage_stats": stage_stats,
        "generation_stats": gen_stats,
        "errors": errors,
        "checks": [{"name": n, "passed": ok_, "detail": d} for n, ok_, d in checks],
        "per_call": results,
    })
    print(f"\nsaved: {path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
