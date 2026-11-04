from pathlib import Path
from deepeval.models import OllamaModel, OllamaEmbeddingModel
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

llm = OllamaModel(model="llama3.2", base_url="http://localhost:11434")
embedder = OllamaEmbeddingModel(model="nomic-embed-text", base_url="http://localhost:11434")

pdf_folder = Path(r"C:\ujjawal\projects\Hybrid_search_with_RAG\pdfs")
document_paths = [str(p) for p in pdf_folder.glob("*.pdf")]
print(f"Found {len(document_paths)} PDFs")

synthesizer = Synthesizer(model=llm)   # object, NOT a string

context_config = ContextConstructionConfig(
    embedder=embedder,
    critic_model=llm,            # <-- the missing piece
)

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=document_paths,
    include_expected_output=True,
    max_goldens_per_context=4,
    context_construction_config=context_config,
)


print(len(goldens))
print(goldens[0].input)
print(goldens[0].expected_output)
print(goldens[0].context)