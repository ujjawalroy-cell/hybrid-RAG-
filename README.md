# Hybrid-Search RAG: a measured case study

I built a RAG system that combines dense, BM25 and exact-match retrieval, reranks with a cross-encoder, and answers from a local LLM. I evaluated it on a **50-question golden set** with a local LLM judge. This README reports what I measured, including the trade-offs and the things I could not measure.

## Results

All numbers come from 50 golden questions over 9 PDFs (environmental-science lecture notes). The judge is a local `llama3.2` at temperature 0, scored with [DeepEval](https://github.com/confident-ai/deepeval).

| Layer | Metric | Score | Cases passing (≥ 0.5) |
|---|---|---|---|
| Generation | **Faithfulness** | **0.72** | 82% |
| Generation | Answer Relevancy | 0.90 | 92% |
| Retrieval | Contextual Recall | 0.93 | 96% |
| Retrieval | Contextual Precision | 0.98 | 98% |
| Retrieval | Contextual Relevancy | 0.75 | 98% |
| Guardrails | Prompt-injection attacks blocked | 20 / 20 | n/a |
| Guardrails | False positives on benign queries | 0 / 62 | n/a |

Notes on how to read this table:

- **Generation metrics cover 49 of 50 questions.** The retrieval-confidence gate refuses one question without calling the LLM, and refusals are counted separately.
- **Faithfulness is 0.72, not 0.95.** A 3B generator and a 3B judge are both noisy. A single faithfulness run moved between 0.56 and 0.66 on the original prompt, so treat differences under about 0.05 as noise.
- **Citation accuracy is not reported.** The golden set has no labels for which chunk a claim should cite, and the final prompt no longer asks for inline citations (the UI shows the retrieved chunks as sources). Measuring it needs chunk-level labels, listed under future work.
- **The guardrail numbers use a test set I wrote myself.** They show the filters work on the attack styles I thought of, not that the system is secure against adversaries.

## Architecture

```
PDFs -> ingest -> chunk -> embed (nomic-embed-text, Ollama) -> Chroma (dense)
                        \-> BM25 pickle (sparse, also serves exact match)

query -> input guardrail (length, injection regex, PII redaction)
      -> dense_search + sparse_search + exact_search
      -> weighted Reciprocal Rank Fusion (0.7 / 0.3 / 0.5, k=60) -> top 20
      -> cross-encoder rerank (ms-marco-MiniLM-L-6-v2) + exact-match boost
      -> context guardrail (drop chunks that carry injected instructions)
      -> confidence gate (sigmoid of top rerank score < 0.35 -> refuse, no LLM call)
      -> llama3.2 via Ollama (temperature 0, context-only prompt)
      -> output guardrail (system-prompt leak, PII, uncited-answer warning)
      -> FastAPI -> Streamlit
```

Key design points:

- **Two indexes stay in sync.** Chunk ids are deterministic (`source::strategy::index`), so re-ingesting upserts instead of duplicating. BM25 is rebuilt from Chroma after every upsert, and `verify_sync()` checks they match.
- **Near-duplicate chunks above 0.95 cosine similarity are dropped** at ingest.
- **Exact search** pulls identifiers, quoted phrases and acronyms out of the query (`PM10`, `"Project Tiger"`, `AC/DC`). It matches them as whole words over the chunk text and feeds a third ranked list into RRF, plus a small rerank boost.
- **Refusal beats a wrong answer.** If the best rerank score is low, the system says it has insufficient context and never calls the LLM.

## Why hybrid beats dense-only

Dense embeddings capture meaning but blur rare tokens. In technical text the rare token is often the whole question: a standard (`NAAQS`), a chemical (`N2O`), a regulator (`CPCB`), a place or species name. An embedding model may place `PM2.5` and `PM10` near each other, or fail to separate a one-off proper noun from similar sentences. BM25 matches the literal token, so it handles exactly the queries where embeddings are weakest. Dense retrieval handles paraphrases that share no words with the answer.

I tested this on my own corpus in two ways.

**1. Rare-term probe (stress test).** I mined 25 rare ALL-CAPS identifiers (`NAAQS`, `N2O`, `PM10`, `CPCB`, `OSPAR`…) and 25 rare proper nouns (`Fukushima`, `Kaziranga`, `Sellafield`…) from the chunks themselves. Identifiers are queried as `What is {term}?`, proper nouns as `Tell me about {term}.` and again with the term in quotes. The term lists are a seeded sample, and rare terms were screened by hand before any results were seen (acronym and Hindi-word noise removed). A query counts as a hit if one of the top 5 chunks contains the term as a whole word.

| Retriever | Identifiers (25) | Proper nouns (25) | Proper nouns, quoted (25) |
|---|---|---|---|
| Dense only | 80% | 84% | 76% |
| BM25 only | 100% | 100% | 100% |
| **Hybrid** | **100%** | **100%** | **100%** |
| Hybrid + exact | 100% | 100% | 100% |

Dense-only misses 16–24% of rare-term lookups. Hybrid recovers all of them.

Caveats: the pass condition is lexical containment, which favours keyword methods by construction, so read this as a stress test and not a neutral benchmark. The corpus is lecture notes, not API docs or manuals. And the exact-match retriever added nothing here because BM25 already finds these terms. Its value is guaranteeing that a literal identifier or quoted phrase reaches the reranker.

**2. Full 50-question eval, same reranker, only the candidate source changes.**

| Retrieval mode | Recall | Precision | Relevancy | Mean latency |
|---|---|---|---|---|
| Dense only (20 candidates) | 0.889 | 0.972 | 0.707 | 1.40 s |
| Hybrid (10 dense + 10 BM25, RRF) | 0.918 | 0.976 | 0.737 | 1.08 s |
| Hybrid + exact | **0.932** | 0.976 | **0.748** | 1.09 s |

Hybrid changes the top-5 chunks for 23 of 50 questions compared with dense-only. The scores improve at every step, but the gaps are small (at most 0.03 per step) and come from single runs with a 3B judge, so I read this as a consistent trend and not a proven effect. The reranker already hides most weaknesses of dense retrieval on this mostly paraphrase-style question set. Hybrid pays off on identifier and name lookups, as the probe shows.

## Chunking strategy comparison

I rebuilt the whole index for each strategy in a throw-away directory, then ran the same 50 questions through the production retriever (hybrid + exact + rerank, top 5). The chunk size in the first column is characters, except the token splitter, which counts tokens.

| Strategy | Chunk size / overlap | Chunks indexed | Avg chunk chars | Avg context sent to LLM | Recall | Precision | Relevancy |
|---|---|---|---|---|---|---|---|
| Recursive | 500 / 75 | 769 | 416 | 2,220 | 0.914 | 1.000 | 0.730 |
| **Recursive (shipped)** | **1000 / 150** | **426** | **776** | about 4,300 | **0.932** | **0.976** | **0.748** |
| Recursive | 2000 / 300 | 231 | 1,379 | 7,448 | **0.956** | 0.977 | **0.800** |
| Character | 1000 / 150 | 425 | 779 | 4,353 | 0.938 | 0.992 | 0.762 |
| Token | 250 tokens / 40 | 404 | 846 | 4,667 | 0.954 | 0.987 | 0.725 |

What the data shows:

- **The spread is small:** Recall 0.91–0.96, Precision 0.98–1.00, Relevancy 0.73–0.80. With a noisy 3B judge and single runs, most of these differences are not distinguishable from noise.
- **Larger chunks raise recall and relevancy and cost context size.** The 2000-character setting scores best, but it sends about 7,400 characters to the generator, 70% more than the shipped setting, which is slower and gives a small model more text to drift from.
- **Small chunks win on precision but lose recall.** The 500-character index has the highest precision (1.00) and the lowest recall (0.91): each chunk is on topic, but an answer often spans several.
- **I kept 1000 / 150 recursive**, which was my original choice. It sits in the middle of every trade-off. I did not tune it with this data, and I would not claim the experiment proves it is optimal. The 2000-character and token-based variants are worth re-testing with a stronger judge.
- Relevancy for the 500-character variant is averaged over 49 cases, because the judge returned unparseable output for one. All other cells cover all 50.

## Generation: what I changed and what it cost

The first answer prompt asked for citations and a fixed length. Faithfulness stayed about the same and Answer Relevancy dropped, so I replaced it.

| Prompt | Faithfulness | Answer Relevancy |
|---|---|---|
| Original (default temperature) | 0.56 and 0.66 in two runs | 0.96 |
| Strict: 3–5 sentences, per-claim `[chunk id]` citations | 0.64 | 0.87 |
| **Current: context-only, concise, temperature 0** | **0.72** | **0.90** |

The final prompt gains about 0.06–0.16 Faithfulness depending on which original run you compare against, and gives up about 0.06 Answer Relevancy. The answers are shorter and stay closer to the context, but they answer slightly less completely. I did not re-run each prompt multiple times, so the original-to-final gain should be read with the 0.56–0.66 run-to-run spread in mind.

## Guardrails

Implemented in [`guardrails/guardrails.py`](guardrails/guardrails.py) with regex and checksum logic, no extra model calls.

- **Input:** length limits, prompt-injection patterns (instruction override, role hijack, chat-template tokens), PII redaction (email, phone, SSN, API key, credit card with a Luhn check).
- **Context:** drops retrieved chunks that contain instructions aimed at the model (indirect prompt injection).
- **Output:** blocks system-prompt leaks, redacts PII, and flags answers with no citation markers.

[`Evals/Eval_guardrails.py`](Evals/Eval_guardrails.py) results: 20/20 attacks blocked, 0/62 benign queries blocked, 100% PII redaction with 0% over-redaction, 100% poisoned chunks dropped, 100% leaks caught, and the end-to-end flow behaves as expected. I wrote this test set, and my first run missed two attacks, which I fixed by adding patterns. Regex guardrails catch known phrasings. They will not stop a novel attack.

## Limitations

- **Small, single-domain corpus.** 9 PDFs, about 430 chunks, environmental science. Results may not transfer to API docs or manuals.
- **Small local judge.** A 3B model scores everything, and single runs vary by about ±0.05. No confidence intervals.
- **No citation accuracy** (see above), and no latency, cost or reliability evals for the full pipeline.
- **The rare-term probe is biased toward lexical methods** by how success is defined.
- **The 50 goldens have no chunk-level labels**, so hit-rate and MRR are measured only on the probe, not on the golden questions.
- Known bug: `load_documents_from_directory()` in [`ingests/Ingest.py`](ingests/Ingest.py) never returns its list, so `main.py` fails as written. The per-file loader works.

## Run it

Requires Python 3.13 and [Ollama](https://ollama.com).

```bash
pip install -r Requirement.txt
ollama serve
ollama pull nomic-embed-text && ollama pull llama3.2

uvicorn backend.app:app --port 8000      # API: /v1/ask, /v1/ingest, /v1/documents, /health
streamlit run frontend/web.py            # UI

# retrieval from the CLI
python -m retrievals.retrieval "What does CPCB do?" --rerank cross_encoder
```

Reproduce the numbers (run from the repo root; the eval runs take a long time on a local judge):

```bash
python Evals/run_regression.py --suite all          # five DeepEval metrics, compared with Evals/baseline.json
python Evals/Eval_guardrails.py                     # guardrail checks, no LLM needed
python Evals/run_ablation.py probe                  # rare-term probe
python Evals/run_ablation.py modes                  # dense vs hybrid vs hybrid + exact
python Evals/run_ablation.py chunking               # chunking comparison
```

Raw ablation output is in [`Evals/ablations/`](Evals/ablations). Per-run result files are written to `Evals/results/` (gitignored).

## Repo map

| Path | Purpose |
|---|---|
| `ingests/`, `splitter/`, `chunking/` | PDF loading, chunking strategies, Chroma + BM25 indexing |
| `retrievals/retrieval.py` | dense, sparse and exact search, RRF, rerank |
| `citations/generate.py` | confidence gate, prompt, generation, guardrail wiring |
| `guardrails/` | input, context and output guardrails |
| `backend/`, `frontend/` | FastAPI service and Streamlit UI |
| `Evals/`, `golden_dataset/` | DeepEval suite, ablations, 50 golden questions |
