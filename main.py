from __future__ import annotations

from pathlib import Path
from ingests.Ingest import load_documents_from_directory
from splitter.Splitter import add_chunk_metadata, get_text_splitter
from chunking.chunking import upsert_to_chroma, build, search, rebuild_bm25
from citations.generate import generate_answer
from retrival import hybrid_retrieve

def main():
    # 1. Ingest Documents
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "pdfs"

    print(f"=== Step 1: Loading documents from: {data_dir} ===")
    docs = load_documents_from_directory(data_dir)
    print(f"Loaded {len(docs)} document page(s).\n")

    if not docs:
        print("No documents found in directory. Exiting.")
        return
    # 2. Chunking & Metadata Attachment
    print("=== Step 2: Splitting documents and creating chunks ===")
    chunks = add_chunk_metadata(docs, strategy="recursive_character", chunk_size=1000, chunk_overlap=150)
    print(f"Created {len(chunks)} chunk(s).\n")

    # 3. Dense & Sparse Indexing
    print("=== Step 3: Upserting chunks to Chroma vector database ===")
    upsert_to_chroma(chunks)

    print("=== Step 4: Rebuilding BM25 sparse index ===")
    build()
    rebuild_bm25()

    # 4. Retrieval (Diagnostics side-by-side + Full Hybrid Search)
    query = "how does gogo's infinity works?"
    print(f"\n=== Step 5: Side-by-side Dense vs Sparse Search for: {query!r} ===")
    raw_search_results = search(query, k=4)

    print(f"\n=== Step 6: Hybrid Retrieval (Chroma + BM25 + RRF + Cross-Encoder Reranking) ===")
    hybrid_results = hybrid_retrieve(query, final_top_n=4)
    print(f"Retrieved {len(hybrid_results)} reranked passage(s):")
    for i, r in enumerate(hybrid_results, start=1):
        score = r.get("rerank_score", 0.0)
        print(f"  [{i}] ID: {r['id']} | Rerank Score: {score:.3f}")
        print(f"      Text: {r['text'][:120]}...\n")

    # 5. Answer Generation
    print("=== Step 7: Generating RAG Answer via LLM ===")
    answer = generate_answer(query, hybrid_results)
    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()

