# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Hybrid-search RAG over PDFs: Chroma (dense) + BM25 (sparse) → Reciprocal Rank Fusion → cross-encoder rerank → cited answer from a local Ollama LLM. Python 3.13 (`.python-version`). `README.md` is empty; there is no test suite, linter config, or build step.

## Commands

Always run from the repo root: `./chroma_db` and `./bm25_index.pkl` are CWD-relative paths.

```bash
pip install -r Requirement.txt        # deps (pyproject.toml has none listed)
ollama serve                          # required; also: ollama pull nomic-embed-text && ollama pull llama3.2
python main.py                        # ingest pdfs/ → chunk → index → sample query → answer
python -m retrievals.retrieval "your query" --rerank cross_encoder   # retrieval CLI (or --rerank llm)
python -m citations.generate          # smoke-test generate_answer with a canned context
python Evals/run_regression.py --suite all --update-baseline   # record eval baseline (needs Ollama + synced index)
python Evals/run_regression.py --suite all    # regression check; exit 1 if a metric drops
deepeval test run Evals/test_regression.py    # pytest smoke subset
uvicorn backend.app:app --port 8000   # FastAPI: /v1/ask, /v1/documents, /v1/strategies, /v1/ingest, /health
streamlit run frontend/web.py         # UI; talks to the backend at http://127.0.0.1:8000
```

Config lives in `.env` (`OLLAMA_URL`, plus LangChain/Groq/HF keys). Never print or commit its values.

## Architecture

Data flow: `ingests/Ingest.py` (PDF → page Documents) → `splitter/Splitter.py` (`add_chunk_metadata` → `Chunk`s) → `chunking/chunking.py` (embed + upsert to Chroma, build BM25 pickle) → `retrievals/retrieval.py` (`hybrid_retrieve`) → `citations/generate.py` (`rag_answer`) → `backend/app.py` → `frontend/web.py`.

Things that need several files to understand:

- **Two indexes must stay in sync.** `upsert_to_chroma` writes the Chroma collection `chunks` (cosine space, `nomic-embed-text` embeddings via Ollama, near-duplicates above 0.95 similarity dropped). BM25 is a separate pickle rebuilt from Chroma by `rebuild_bm25()`. After any upsert call `rebuild_bm25()`, then check `verify_sync()` is True. `backend/app.py` `/v1/ingest` follows this pattern.
- **Chunk ids** are `source_document::chunking_strategy::chunk_index` (property on `Chunk`). Deterministic, so re-ingest upserts instead of duplicating; eval labels and fusion key off these ids.
- **Retrieval** (`retrievals/retrieval.py`): `dense_search` and `sparse_search` → `reciprocal_rank_fusion` (weights 0.7 dense / 0.3 sparse, k=60) → top 20 → `hydrate` (fetch text/metadata from Chroma by id) → `rerank` (cross-encoder `ms-marco-MiniLM-L-6-v2` by default, or an Ollama LLM judge). `filter_document` and `strategy` filter both searches.
- **Answer + refusal gate** (`citations/generate.py`): `rag_answer` takes the top rerank score, sigmoids it into a confidence, and refuses (`status: insufficient_context`) below `RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35` without calling the LLM. `use_sparse=False` bypasses hybrid and uses dense only (confidence is then the raw cosine score).
- **Ollama host** is read from `OLLAMA_URL` in `citations/generate.py`, but `chunking/chunking.py` hardcodes `OLLAMA_HOST = "http://localhost:11434"`. Keep this in mind when pointing at a remote Ollama.

## Repo quirks (check before editing)

- `retrival.py` (root, misspelled) is an older copy of `retrievals/retrieval.py`. Put retrieval changes in `retrievals/retrieval.py`.
- `main.py` uses flat imports (`from Ingest import ...`, `from retrival import ...`) that only work if the subfolders are on `sys.path`; the packages (`chunking/`, `retrievals/`, `citations/` `__init__.py` star-re-export their modules) are imported by package name in `backend/app.py`, which inserts the repo root into `sys.path`. Expect `python main.py` to need import fixes.
- `Evals/` is a DeepEval regression suite using a local Ollama judge (`EVAL_JUDGE_MODEL`, default `llama3.2`): `Eval_retrieval.py` (Contextual Recall/Precision/Relevancy), `Eval_generation.py` (Faithfulness/Answer Relevancy; confidence-gate refusals counted separately), `run_regression.py` (CLI, compares to `Evals/baseline.json`, exit 1 on regression), `test_regression.py` (pytest smoke). Goldens are `golden_dataset/gold_retrieval.json` (no chunk-id labels or should-refuse cases, so no hit-rate/MRR). `golden_dataset/harness.py` holds shared helpers. There are no latency, cost, or reliability evals.
- `Hybrid_search_with_RAG/` (nested folder) only contains a stub README.
- `chroma_db/`, `bm25_index.pkl`, and `.env` are tracked in git.
- No `main` branch exists; the only branch is `master`. Slash commands such as `/Create-spec` expect `main` and `.claude/specs/`.
