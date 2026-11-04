# Spec: RAG Regression Testing With DeepEval

## Overview
Add a repeatable regression suite that scores the Hybrid Search RAG pipeline with the DeepEval framework and fails when quality drops below a stored baseline. Today the only eval is an unfinished, non-runnable script. The suite runs the golden questions through the real pipeline (`hybrid_retrieve` for retrieval, `rag_answer` for generation), scores retrieval with Contextual Recall / Precision / Relevancy and generation with Faithfulness / Answer Relevancy, compares each metric against `Evals/baseline.json`, and exits non-zero on regression. It uses a local Ollama model as the DeepEval judge, so no OpenAI key is needed. This lets any change to chunking, RRF weights, reranking, the confidence gate or the prompt be checked before it ships.

## Current behaviour

**Chunking**
- `splitter/Splitter.py:66` `add_chunk_metadata` splits with `get_text_splitter` (`:36`) and builds `Chunk`s. Chunk id is `source_document::chunking_strategy::chunk_index` (`:19-22`). `main.py:25` uses `recursive_character`, size 1000, overlap 150.
- `ingests/Ingest.py:20-47` `load_documents_from_directory` never returns the `documents` list (no `return` after the loop), so it yields `None`. `DATA_DIR` at `:12` is an absolute Windows path.

**Dense search**
- `chunking/chunking.py:85` `embed_query`, `:91` `get_collection` (Chroma `chunks`, cosine). `upsert_to_chroma` (`:150`) drops near-duplicates above 0.95 (`:28`, `:139`).
- `retrievals/retrieval.py:24` `dense_search` returns `{id, score = 1 - dist, rank}`.

**Sparse search**
- `chunking/chunking.py:201` `rebuild_bm25`, `:212` `load_bm25`, `:218` `verify_sync`.
- `retrievals/retrieval.py:64` `sparse_search` returns `{id, score, rank}`.

**Fusion**
- `retrievals/retrieval.py:103` `reciprocal_rank_fusion`, weights 0.7 / 0.3, k=60 (`:12-14`), top 20 kept (`:15`).

**Reranking**
- `retrievals/retrieval.py:164` `rerank_cross_encoder` (`ms-marco-MiniLM-L-6-v2`), `:189` `rerank_llm` (judge `gemma4:26b`, `:19`), `:224` `hybrid_retrieve` returns `{id, text, metadata, rrf_score, rerank_score}`, final top 5 (`:16`).
- Root `retrival.py` is an older copy; `main.py:4-8` still uses flat imports (`from Ingest`, `from retrival`, `from generate`) that do not resolve as packages.

**Citations and answers**
- `citations/generate.py:56` `generate_answer(query, retrieved_results, model, ...)` returns a plain string (Ollama `/api/generate`, `llama3.2`). There is no `generate_answer_with_usage`.
- `:112` `rag_answer` returns `{query, status, answer, chunks_used, retrieval_confidence}`. Confidence is the sigmoid of the top rerank score; below `RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35` (`:109`, `:167`) it returns `status: "insufficient_context"` without calling the LLM.
- The system prompt (`:69-73`) says to state clearly when context is insufficient. It does not define a fixed refusal phrase.
- Ollama host comes from `OLLAMA_URL` (`:10`); `chunking/chunking.py:22` hardcodes `http://localhost:11434`, and `retrievals/retrieval.py:205` (`chunking.OLLAMA_HOST`) inherits that.

**Evaluation**
- `Evals/Eval_retrieval.py` (62 lines) is broken: `from retrieval import rerank` does not resolve (`:9`), `rerank(query="", candidates=[], ...)` is called as if it returned a retriever with `.invoke` (`:25-30`), it reads the nonexistent `goldens/retriever_goldens.json` (`:13`), it uses keys `query` / `ideal_answer` (`:30`) while the golden file uses `question`, the judge is OpenAI `gpt-4.1-mini` (`:14`), and the hyperparameters claim `text-embedding-3-small` (`:56`).
- `golden_dataset/harness.py:26` `load_goldens`, `:31` `summarize_by_metric` (per-metric pass rate and score stats), `:63` `print_summary`. These are reusable as-is.
- `golden_dataset/gold_retrieval.json`: 50 entries with keys `id` (C001-C050), `question`, `ideal_answer`, `source_session`. All 50 `ideal_answer` strings contain `[cite: n]` markers. There are no labelled relevant chunk ids, no out-of-scope (should-refuse) questions, and no source-document labels.
- `Evals_latency.py`, `Evals_cost.py`, `Evals_reliability.py` and `rag_probe.py` do not exist. `deepeval` is not in `Requirement.txt` or `pyproject.toml`. No baseline file or results directory exists.
- No existing specs in `.claude/specs/`; `CLAUDE.md` marks no step complete.

## Depends on
- `retrievals/retrieval.py` (`hybrid_retrieve`), `citations/generate.py` (`rag_answer`), `chunking/chunking.py` (`verify_sync`, `get_collection`).
- `golden_dataset/harness.py` and `golden_dataset/gold_retrieval.json`.
- A populated index (Chroma `chunks` + `bm25_index.pkl`, in sync) and a running Ollama with `nomic-embed-text`, `llama3.2` and the judge model pulled.

## Modules to change
- `Evals/Eval_retrieval.py`: rewrite to run from the repo root. Remove `from retrieval import rerank` and the OpenAI judge. Import `hybrid_retrieve` from `retrievals.retrieval`, load `golden_dataset/gold_retrieval.json` via `load_goldens`, map `question` / `ideal_answer` to `input` / `expected_output`, build `LLMTestCase` with `retrieval_context = [c["text"] for c in chunks]`, and run `ContextualRecallMetric`, `ContextualPrecisionMetric`, `ContextualRelevancyMetric`, summarising with `summarize_by_metric`. Add a `--rerank` option. Replace the false `hyperparameters` with real values (`nomic-embed-text`, 1000/150, top_k 5, RRF 0.7/0.3).
- `golden_dataset/harness.py`: add `strip_cite_markers(text)` (removes `[cite: ...]`), `load_baseline` / `save_baseline`, and `compare_to_baseline(summary, baseline, tolerance)`. Keep existing functions unchanged.
- `Requirement.txt`: add `deepeval`, `pytest`.
- `CLAUDE.md`: update the "Evals" quirk line and the Commands block to describe the new suite.

## Modules to create
- `Evals/judge.py`: `OllamaJudge(DeepEvalBaseLLM)` that calls `{OLLAMA_URL}/api/generate` (with `format: "json"` support for schema-bound metrics). Host from `OLLAMA_URL` with the same stripping as `citations/generate.py:10`. Model name from env `EVAL_JUDGE_MODEL` (default `llama3.2`).
- `Evals/Eval_generation.py`: runs `rag_answer` per golden, builds `LLMTestCase(actual_output=result["answer"], retrieval_context=[chunk texts])`, scores `FaithfulnessMetric` and `AnswerRelevancyMetric`. Cases with `status == "insufficient_context"` are recorded separately (refusal rate) and excluded from generation metrics.
- `Evals/run_regression.py`: CLI entry point. Flags: `--suite retrieval|generation|all`, `--limit N`, `--update-baseline`, `--tolerance 0.05`, `--rerank cross_encoder|llm`. Pre-flight checks (`verify_sync()` True, Ollama reachable) run before any scoring. Writes `Evals/results/<UTC timestamp>.json`, prints the `print_summary` table plus a per-metric delta against baseline, and exits 1 on any regression.
- `Evals/baseline.json`: committed baseline of per-metric `avg_score` and `pass_rate`, plus the config it was produced with (chunk size, overlap, weights, models, golden file hash).
- `Evals/test_regression.py`: pytest wrapper using DeepEval `assert_test` over a small smoke subset (default 10 goldens) so `deepeval test run Evals/test_regression.py` works in CI.
- `Evals/results/.gitkeep`, plus a `.gitignore` entry for `Evals/results/*.json`.

## Index changes
No index changes. The suite only reads Chroma and BM25. If the indexes are missing or `verify_sync()` is False, the runner stops with a message telling the user to run the ingest flow; it must not rebuild anything itself.

## Interfaces
Added (no changes to existing pipeline signatures):
- `Evals/judge.py`: `class OllamaJudge(DeepEvalBaseLLM)` with `load_model()`, `generate(prompt, schema=None) -> str | BaseModel`, `a_generate(...)`, `get_model_name() -> str`.
- `golden_dataset/harness.py`: `strip_cite_markers(text: str) -> str`; `load_baseline(path) -> dict`; `save_baseline(path, summary, config) -> None`; `compare_to_baseline(summary: dict, baseline: dict, tolerance: float = 0.05) -> list[dict]` returning one entry per metric `{metric, baseline, current, delta, regressed}`.
- `Evals/Eval_retrieval.py`: `run_retrieval_eval(goldens, judge, rerank_method="cross_encoder", limit=None) -> dict` (the per-metric summary).
- `Evals/Eval_generation.py`: `run_generation_eval(goldens, judge, limit=None) -> dict` returning `{summary, refusals, answered}`.

Callers: only `Evals/run_regression.py` and `Evals/test_regression.py` call these. `main.py`, `backend/app.py` and `citations/generate.py` are not touched. The suite calls `hybrid_retrieve(query, rerank_method=...)` and `rag_answer(query)` exactly as `backend/app.py:148` does.

Regression rule: a metric regresses when `current_avg < baseline_avg - tolerance`, or when `pass_rate` drops by more than 10 points. Each metric is judged separately (the point of `summarize_by_metric`), never pooled.

## New dependencies
- `deepeval` (evaluation framework and metrics).
- `pytest` (for `Evals/test_regression.py`).
- No model downloads beyond a judge model in Ollama (default `llama3.2`, already used for generation).

## Rules for implementation
- Run everything from the repo root: `./chroma_db` and `./bm25_index.pkl` are CWD-relative.
- Keep Chroma and BM25 in sync: call `rebuild_bm25()` after any upsert and confirm `verify_sync()` returns True. The eval suite itself must not upsert; it only asserts `verify_sync()`.
- Keep chunk ids in the `file::strategy::index` format. Result files record the chunk ids of every retrieved chunk so runs can be diffed.
- Ollama is reached through `OLLAMA_URL` (default `localhost:11434`). Do not hardcode the host in new code. Do not fix `chunking/chunking.py:22` as part of this spec; note it in the PR if a remote Ollama is needed.
- Changes to retrieval go in `retrievals/retrieval.py`, not `retrival.py`. This spec makes none; eval code imports from `retrievals.retrieval`.
- Keep the citation behaviour in `citations/generate.py` intact, including the confidence gate (0.35) and sigmoid confidence. The generation eval treats `insufficient_context` as a valid outcome, not a failure.
- Strip `[cite: n]` markers from `ideal_answer` before using it as `expected_output`; they refer to source pages outside the corpus and would penalise correct answers.
- The judge must be local (Ollama). Do not add OpenAI or any hosted-judge dependency. Never print or commit `.env` values.
- Judge scores from a small local model are noisy. Keep `temperature` 0 in `OllamaJudge`, and make the tolerance configurable rather than hardcoded.
- Do not claim latency, cost or reliability coverage; this spec covers retrieval and generation quality only.
- Known gap to record, not fix: the golden file has no relevant-chunk-id labels or should-refuse questions, so id-level hit rate / MRR and refusal correctness cannot be measured yet.

## Definition of done
- `pip install -r Requirement.txt` installs `deepeval` and `pytest` without error.
- `python -c "from Evals.judge import OllamaJudge; print(OllamaJudge().get_model_name())"` prints the judge model name.
- `python -c "from chunking import verify_sync; print(verify_sync())"` prints `True` before and after running the suite.
- `python Evals/run_regression.py --suite retrieval --limit 5` completes, prints a per-metric table for Contextual Recall, Precision and Relevancy, and writes a file under `Evals/results/`.
- `python Evals/run_regression.py --suite generation --limit 5` completes, prints Faithfulness and Answer Relevancy plus the refusal count, and writes a file under `Evals/results/`.
- `python Evals/run_regression.py --suite all --update-baseline` writes `Evals/baseline.json` covering all 50 goldens.
- `python Evals/run_regression.py --suite all` exits 0 against the unchanged baseline.
- Regression is caught: temporarily set `DENSE_WEIGHT = 0.0` and `SPARSE_WEIGHT = 1.0` in `retrievals/retrieval.py`, run `python Evals/run_regression.py --suite retrieval`, and confirm it exits 1 and names the regressed metric; revert the change and confirm `git diff retrievals/` is empty.
- `deepeval test run Evals/test_regression.py` runs the smoke subset and reports pass/fail.
- `grep -rn "gpt-4\|openai\|retriever_goldens" Evals/` returns nothing.
- `git diff --stat` shows no changes to `citations/generate.py`, `retrievals/retrieval.py`, `chunking/chunking.py`, `main.py` or `backend/app.py`.
