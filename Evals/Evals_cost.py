"""
Cost eval: tokens and estimated spend per query for the RAG pipeline.

  python Evals/Evals_cost.py [--limit N] [--model M]
                             [--price-in USD] [--price-out USD] [--compute-per-hour USD]
                             [--max-avg-tokens T] [--max-cost-per-query USD]

Everything runs on local Ollama, so there is no API bill. Cost is reported three ways:

  tokens     prompt / completion tokens per query, from Ollama's own counts
  api-equiv  tokens x --price-in / --price-out (USD per 1M tokens) -- set these to a hosted
             model's prices to see what the same traffic would cost there
  compute    wall-clock seconds x --compute-per-hour -- your machine / GPU cost

Prices default to 0; pass the ones you want to compare against. Results are also projected
to 1,000 queries.

It also measures the fixed prompt overhead: the tokens the system prompt and template cost
on every call even with no retrieved context, so you can see how much of each prompt is
retrieved context and how much is boilerplate.

Not counted: query embedding tokens (a few per query) and the cross-encoder (CPU/GPU time
only, which the compute figure covers). If Ollama reuses a cached prompt prefix it may report
fewer prompt tokens than were sent, so treat prompt_tokens as a lower bound.
"""

import argparse
import sys

from rag_probe import (
    DEFAULT_MODEL,
    FINAL_TOP_N,
    describe,
    generate_answer_with_usage,
    load_questions,
    print_checks,
    run_pipeline,
    save_results,
)

# ------------------------------------------------------------- config
MAX_AVG_TOKENS_PER_QUERY = 4000      # prompt + completion; raise if you grow top_k or chunk size


def usd(tokens, price_per_million):
    return tokens * price_per_million / 1_000_000


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="only the first N goldens")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--top-k", type=int, default=FINAL_TOP_N)
    p.add_argument("--price-in", type=float, default=0.0, help="USD per 1M prompt tokens")
    p.add_argument("--price-out", type=float, default=0.0, help="USD per 1M completion tokens")
    p.add_argument("--compute-per-hour", type=float, default=0.0, help="USD per hour of machine time")
    p.add_argument("--max-avg-tokens", type=int, default=MAX_AVG_TOKENS_PER_QUERY)
    p.add_argument("--max-cost-per-query", type=float, default=None, help="USD; checked only if set")
    args = p.parse_args()

    goldens = load_questions(args.limit)

    # ------------------------------------------------------------- warm-up
    # loads the cross-encoder and the Ollama model so the first measured query's
    # wall time (and so its compute cost) is not inflated by model loading
    print("warm-up ...")
    run_pipeline(goldens[0]["question"], top_k=args.top_k, generate=True, model=args.model)

    # ------------------------------------------------------------- fixed overhead
    # no retrieved context: what the system prompt + template cost on every call
    print("measuring fixed prompt overhead ...")
    baseline = generate_answer_with_usage("What is this course about?", [], model=args.model)
    overhead_tokens = None if baseline["error"] else baseline["prompt_tokens"]

    # ------------------------------------------------------------- measured runs
    rows = []
    for i, g in enumerate(goldens, start=1):
        res = run_pipeline(g["question"], top_k=args.top_k, generate=True, model=args.model)
        if not res["ok"]:
            print(f"[{i}/{len(goldens)}] ERROR in {res['error_stage']}: {res['error']}")
            rows.append({"id": g["id"], "ok": False, "error": res["error"]})
            continue
        u = res["usage"]
        row = {
            "id": g["id"],
            "session": g.get("source_session"),
            "ok": True,
            "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"],
            "total_tokens": u["prompt_tokens"] + u["completion_tokens"],
            "context_chars": res["context_chars"],
            "wall_s": res["total_s"],
        }
        row["api_usd"] = usd(row["prompt_tokens"], args.price_in) + usd(row["completion_tokens"], args.price_out)
        row["compute_usd"] = row["wall_s"] * args.compute_per_hour / 3600
        row["total_usd"] = row["api_usd"] + row["compute_usd"]
        rows.append(row)
        print(f"[{i}/{len(goldens)}] {row['prompt_tokens']:>5} in  {row['completion_tokens']:>4} out  "
              f"{row['wall_s']:6.2f}s  {g['question'][:55]}")

    ok = [r for r in rows if r["ok"]]
    if not ok:
        sys.exit("no successful calls; nothing to cost")

    # ------------------------------------------------------------- report
    stats = {k: describe([r[k] for r in ok]) for k in
             ("prompt_tokens", "completion_tokens", "total_tokens", "context_chars", "wall_s",
              "api_usd", "compute_usd", "total_usd")}
    mean_prompt = stats["prompt_tokens"]["mean"]
    per_1k = {k: stats[k]["mean"] * 1000 for k in ("total_tokens", "api_usd", "compute_usd", "total_usd")}

    print("\n" + "=" * 78)
    print(f"cost  ({len(ok)} successful queries, {len(rows) - len(ok)} errors, model {args.model}, top_k {args.top_k})")
    print("=" * 78)
    print(f"{'per query':<20}{'mean':>12}{'p50':>12}{'p95':>12}{'max':>12}")
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "context_chars", "wall_s"):
        s = stats[name]
        print(f"{name:<20}" + "".join(f"{s[k]:>12.1f}" for k in ("mean", "p50", "p95", "max")))
    for name in ("api_usd", "compute_usd", "total_usd"):
        s = stats[name]
        print(f"{name:<20}" + "".join(f"{s[k]:>12.6f}" for k in ("mean", "p50", "p95", "max")))

    if overhead_tokens is None:
        print(f"\nfixed prompt overhead: unavailable (baseline call failed: {baseline['error']})")
    else:
        share = overhead_tokens / mean_prompt if mean_prompt else float("nan")
        print(f"\nfixed prompt overhead: {overhead_tokens} tokens per call "
              f"({share:.0%} of the average prompt; the rest is retrieved context + question)")
    print(f"prices: in ${args.price_in}/1M, out ${args.price_out}/1M, compute ${args.compute_per_hour}/h")
    print(f"per 1,000 queries: {per_1k['total_tokens']:,.0f} tokens, "
          f"api-equiv ${per_1k['api_usd']:.4f}, compute ${per_1k['compute_usd']:.4f}, "
          f"total ${per_1k['total_usd']:.4f}")

    sessions = sorted({r["session"] for r in ok if r["session"]})
    if sessions:
        print("\nby source_session (mean total tokens)")
        for sess in sessions:
            vals = [r["total_tokens"] for r in ok if r["session"] == sess]
            print(f"  {sess}: {sum(vals) / len(vals):,.0f}  (n={len(vals)})")

    # ------------------------------------------------------------- budgets
    checks = [(
        "avg tokens per query <= budget",
        stats["total_tokens"]["mean"] <= args.max_avg_tokens,
        f"{stats['total_tokens']['mean']:.0f} vs {args.max_avg_tokens}",
    )]
    if args.max_cost_per_query is not None:
        checks.append((
            "avg cost per query <= budget",
            stats["total_usd"]["mean"] <= args.max_cost_per_query,
            f"${stats['total_usd']['mean']:.6f} vs ${args.max_cost_per_query}",
        ))
    passed = print_checks(checks)

    path = save_results("cost", {
        "config": {
            "n_questions": len(goldens), "model": args.model, "top_k": args.top_k,
            "price_in_per_1m": args.price_in, "price_out_per_1m": args.price_out,
            "compute_per_hour": args.compute_per_hour,
            "max_avg_tokens": args.max_avg_tokens, "max_cost_per_query": args.max_cost_per_query,
        },
        "fixed_prompt_overhead_tokens": overhead_tokens,
        "stats": stats,
        "per_1k_queries": per_1k,
        "checks": [{"name": n, "passed": ok_, "detail": d} for n, ok_, d in checks],
        "per_query": rows,
    })
    print(f"\nsaved: {path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
