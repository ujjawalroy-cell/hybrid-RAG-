"""Regression runner. Run from the repo root:

    python Evals/run_regression.py --suite all --update-baseline   # record baseline
    python Evals/run_regression.py --suite all                     # compare; exit 1 on regression
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# DeepEval prints symbols (e.g. a warning sign) that crash cp1252 Windows consoles.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests

from golden_dataset.harness import (
    compare_to_baseline,
    file_sha256,
    load_baseline,
    load_goldens,
    print_summary,
    save_baseline,
)

GOLDEN_PATH = ROOT / "golden_dataset" / "gold_retrieval.json"
BASELINE_PATH = ROOT / "Evals" / "baseline.json"
RESULTS_DIR = ROOT / "Evals" / "results"


def preflight() -> None:
    """Fail early: Ollama reachable and the two indexes in sync. Never rebuilds."""
    from Evals.judge import _ollama_host
    import chunking

    host = _ollama_host()
    try:
        requests.get(f"{host}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"Preflight failed: Ollama not reachable at {host} ({e}). Run `ollama serve`.")
    try:
        in_sync = chunking.verify_sync()
    except Exception as e:
        sys.exit(f"Preflight failed: could not read indexes ({e}). Run the ingest flow first.")
    if not in_sync:
        sys.exit("Preflight failed: Chroma and BM25 are out of sync. Run rebuild_bm25() / the ingest flow.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=["retrieval", "generation", "all"], default="all")
    p.add_argument("--limit", type=int, default=None, help="only the first N goldens")
    p.add_argument("--update-baseline", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--rerank", choices=["cross_encoder", "llm"], default="cross_encoder")
    args = p.parse_args()

    os.chdir(ROOT)  # ./chroma_db and ./bm25_index.pkl are CWD-relative
    preflight()

    from Evals.judge import OllamaJudge
    from Evals.Eval_generation import run_generation_eval
    from Evals.Eval_retrieval import run_retrieval_eval

    goldens = load_goldens(GOLDEN_PATH)
    judge = OllamaJudge()
    summary: dict = {}
    record: dict = {}

    if args.suite in ("retrieval", "all"):
        out = run_retrieval_eval(goldens, judge, rerank_method=args.rerank, limit=args.limit)
        summary.update(out["summary"])
        record["retrieved_ids"] = out["retrieved"]
    if args.suite in ("generation", "all"):
        out = run_generation_eval(goldens, judge, limit=args.limit)
        summary.update(out["summary"])
        record["refusals"] = out["refusals"]
        record["answered"] = out["answered"]
        record["generation_retrieved_ids"] = out["retrieved"]
        print(f"\nRefusals (confidence gate): {len(out['refusals'])}  Answered: {len(out['answered'])}")

    print_summary(f"Suite: {args.suite}", summary)

    config = {
        "golden_file_hash": file_sha256(GOLDEN_PATH),
        "n_goldens": len(goldens[: args.limit] if args.limit else goldens),
        "judge_model": judge.get_model_name(),
        "rerank": args.rerank,
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "rrf_weights": "0.7/0.3",
        "embedding_model": "nomic-embed-text",
    }

    rows, exit_code = [], 0
    if args.update_baseline:
        save_baseline(BASELINE_PATH, summary, config)
        print(f"Baseline written to {BASELINE_PATH.relative_to(ROOT)}")
    elif BASELINE_PATH.exists():
        baseline = load_baseline(BASELINE_PATH)
        wanted = set(summary)  # only compare metrics from the suites actually run
        baseline["metrics"] = {k: v for k, v in baseline["metrics"].items() if k in wanted}
        rows = compare_to_baseline(summary, baseline, args.tolerance)
        print("\nMetric vs baseline")
        for r in rows:
            cur = "n/a" if r["current"] is None else f"{r['current']:.2f}"
            d = "n/a" if r["delta"] is None else f"{r['delta']:+.2f}"
            print(f"  {r['metric']:<26} baseline={r['baseline']:.2f} current={cur} delta={d}  "
                  f"{'REGRESSED' if r['regressed'] else 'ok'}")
        bad = [r["metric"] for r in rows if r["regressed"]]
        if bad:
            print(f"\nREGRESSION in: {', '.join(bad)}")
            exit_code = 1
    else:
        print(f"\nNo baseline at {BASELINE_PATH.relative_to(ROOT)}; run with --update-baseline first.")
        exit_code = 1

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{stamp}.json"
    path.write_text(json.dumps(
        {"timestamp": stamp, "suite": args.suite, "config": config, "summary": summary,
         "comparison": rows, "regressed": exit_code == 1, **record},
        indent=2, default=str))
    print(f"Results written to {path.relative_to(ROOT)}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
