from __future__ import annotations

import os
import requests
from dotenv import load_dotenv
from typing import Any, List, Union,Optional

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_URL", "http://localhost:11434").strip().strip('"').strip("'").rstrip("/")
DEFAULT_MODEL = "llama3.2"


def format_context(context_data: Any) -> str:
    """
    Normalizes various retrieved result formats (list of dicts, search outputs,
    list of strings, or raw string) into a cohesive context string for the prompt.
    """
    if isinstance(context_data, str):
        return context_data.strip()

    passages: List[str] = []

    if isinstance(context_data, dict):
        # Case: output from chunking.search() with keys 'dense' and 'sparse'
        for key in ("dense", "sparse"):
            items = context_data.get(key, [])
            for item in items:
                if isinstance(item, dict) and "text" in item:
                    passages.append(item["text"])
                elif isinstance(item, str):
                    passages.append(item)

    elif isinstance(context_data, list):
        for item in context_data:
            if isinstance(item, dict):
                # Format from hybrid_retrieve(): {"id": ..., "text": ..., "rerank_score": ...}
                text = item.get("text", "")
                heading = item.get("metadata", {}).get("section_heading", "") if isinstance(item.get("metadata"), dict) else ""
                cid = item.get("id", "")
                header = f"[{cid}]" + (f" ({heading})" if heading else "")
                if text:
                    passages.append(f"{header}:\n{text}")
            elif hasattr(item, "page_content"):
                # Langchain Document
                passages.append(item.page_content)
            elif isinstance(item, str):
                passages.append(item)

    if not passages:
        return "No relevant context found."

    return "\n\n---\n\n".join(passages)


def generate_answer(
    query: str,
    retrieved_results: Any,
    model: str = DEFAULT_MODEL,
    system_instruction: str | None = None,
    timeout: int = 120,
) -> str:
    """
    Generates a grounded answer to the user query given the retrieved context passages
    using the local Ollama LLM.
    """
    context_text = format_context(retrieved_results)

    default_system = (
        "You are an accurate, helpful AI assistant. Answer the question using ONLY the provided "
        "context. Do not add outside knowledge, examples or background that the context does not "
        "state. If the context only partly answers the question, answer that part and say what is "
        "missing. If it does not answer at all, say so. Keep the answer concise."
    )
    system = system_instruction or default_system

    prompt = f"""[Context Information]
{context_text}

[Question]
{query}

[Instructions]
Give a direct, concise answer that fully addresses the question, using only facts stated in the context above. Do not add anything the context does not say.

[Answer]
"""

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},  # deterministic, stays closer to the context
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to generate answer from Ollama at {OLLAMA_HOST} (model: {model}): {e}"
        print(f"Error in generate_answer: {error_msg}")
        return f"Error generating answer: {e}"


RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35


def rag_answer(
    query: str,
    top_k: int = 5,
    use_sparse: bool = True,
    retrieval_threshold: float = RETRIEVAL_CONFIDENCE_THRESHOLD,
    model: str = DEFAULT_MODEL,
    filter_document: Optional[str] = None,
    strategy: Optional[str] = None,
    use_guardrails: bool = True,
) -> dict:
    """
    End-to-end RAG execution:
    0. Input guardrail: blocks prompt injection / bad length, redacts PII in the query
    1. Retrieves relevant passages (Hybrid dense + sparse + exact, or dense-only)
    1b. Context guardrail: drops chunks that carry indirect prompt injection
    2. Computes retrieval confidence
    3. Guards against hallucination if confidence < retrieval_threshold
    4. Generates cited answer via local Ollama LLM
    5. Output guardrail: blocks system-prompt leaks, redacts PII, flags missing citations
    """
    import math
    from retrievals.retrieval import hybrid_retrieve, dense_search, hydrate
    import guardrails as gr

    guard_report: dict = {}
    if use_guardrails:
        in_check = gr.check_input(query)
        guard_report["input"] = in_check.to_dict()
        if not in_check.allowed:
            return {
                "query": query,
                "status": "blocked",
                "answer": f"Request blocked by input guardrail: {', '.join(in_check.violations)}.",
                "chunks_used": [],
                "retrieval_confidence": 0.0,
                "guardrails": guard_report,
            }
        query = in_check.text  # PII-redacted query is what reaches retrieval + LLM

    if use_sparse:
        chunks = hybrid_retrieve(
            query,
            final_top_n=top_k,
            filter_document=filter_document,
            strategy=strategy,
        )
    else:
        dense_raw = dense_search(
            query,
            k=top_k,
            filter_document=filter_document,
            strategy=strategy,
        )
        content = hydrate([d["id"] for d in dense_raw])
        chunks = [
            {"id": d["id"], "score": d["score"], **content.get(d["id"], {})}
            for d in dense_raw if d["id"] in content
        ]

    if use_guardrails:
        chunks, ctx_check = gr.filter_context(chunks)
        guard_report["context"] = ctx_check.to_dict()

    if not chunks:
        return {
            "query": query,
            "status": "insufficient_context",
            "answer": "No relevant context found in the knowledge base.",
            "chunks_used": [],
            "retrieval_confidence": 0.0,
            "guardrails": guard_report,
        }

    top_score = chunks[0].get("rerank_score", chunks[0].get("score", 0.0))
    # Normalize cross-encoder logit using sigmoid, or clamp similarity score
    if "rerank_score" in chunks[0]:
        confidence = round(1.0 / (1.0 + math.exp(-top_score)), 3)
    else:
        confidence = round(max(0.0, min(1.0, float(top_score))), 3)

    if confidence < retrieval_threshold:
        return {
            "query": query,
            "status": "insufficient_context",
            "answer": f"Retrieval confidence ({confidence:.2f}) is below threshold ({retrieval_threshold:.2f}). Refusing to answer to avoid hallucination.",
            "chunks_used": chunks,
            "retrieval_confidence": confidence,
            "guardrails": guard_report,
        }

    answer = generate_answer(query, chunks, model=model)
    status = "answered"
    if use_guardrails:
        out_check = gr.check_output(answer, chunks)
        guard_report["output"] = out_check.to_dict()
        answer = out_check.text
        if not out_check.allowed:
            status = "blocked"
    return {
        "query": query,
        "status": status,
        "answer": answer,
        "chunks_used": chunks,
        "retrieval_confidence": confidence,
        "guardrails": guard_report,
    }


if __name__ == "__main__":
    sample_context = [
        {"id": "test::1", "text": "Hybrid search combines dense vector similarity search with sparse lexical BM25 search for optimal retrieval quality.", "metadata": {"section_heading": "Overview"}}
    ]
    ans = generate_answer("What does hybrid search combine?", sample_context)
    print("Generated Answer:\n", ans)
