
"""W7D3: LlamaIndex document indexing, RAG, and ChromaDB comparison."""

import json
import time
from pathlib import Path

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
OUTPUT_DIR = BASE_DIR / "outputs"
COLLECTION_NAME = "w7d3_llamaindex_documents"

QUESTIONS = [
    ("What is machine learning?", "learn patterns from data"),
    ("What are the three common types of machine learning?", "supervised learning"),
    ("How do classification and regression differ?", "continuous numerical values"),
    ("What is unsupervised learning used for?", "hidden patterns"),
    ("What is an embedding?", "numerical vector representation"),
    ("What does retrieval-augmented generation do?", "retrieves relevant information"),
    ("What is the role of LlamaIndex?", "connect data sources"),
    ("What is ChromaDB used for?", "vector database"),
    ("What is overfitting?", "training data"),
    ("What is MLOps?", "lifecycle of machine learning models"),
]


def configure_models():
    """Configure Ollama embedding and language models with a smaller context."""
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        request_timeout=120.0,
    )

    Settings.llm = Ollama(
        model="llama3.2:3b",
        request_timeout=180.0,
        temperature=0.0,
        context_window=2048,
        additional_kwargs={"num_ctx": 2048},
    )


def load_documents():
    """Load TXT documents from the documents directory."""
    if not DOCUMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {DOCUMENTS_DIR}")

    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENTS_DIR),
        required_exts=[".txt"],
    ).load_data()

    if not documents:
        raise ValueError(f"No TXT documents found in {DOCUMENTS_DIR}")

    print(f"Loaded {len(documents)} document(s):")
    for document in documents:
        print(f" - {document.metadata.get('file_name', 'unknown')}")

    return documents


def create_memory_index(documents):
    """Create an in-memory vector index."""
    print("\nCreating in-memory VectorStoreIndex...")
    return VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )


def create_chroma_index(documents):
    """Create a persistent ChromaDB-backed vector index."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset only this project's collection when rerunning the script.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    print("\nCreating ChromaDB-backed VectorStoreIndex...")
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )


def run_queries(index, index_name):
    """Run ten questions and record answers, sources, and latency."""
    query_engine = index.as_query_engine(similarity_top_k=2)
    results = []
    total_latency = 0.0

    print(f"\n{'=' * 60}\n{index_name}\n{'=' * 60}")

    for number, (question, expected_fact) in enumerate(QUESTIONS, start=1):
        start = time.perf_counter()
        response = query_engine.query(question)
        latency = time.perf_counter() - start
        total_latency += latency

        source_nodes = getattr(response, "source_nodes", []) or []
        source_files = sorted({
            item.node.metadata.get("file_name", "unknown")
            for item in source_nodes
        })

        retrieved_text = " ".join(
            item.node.get_content() for item in source_nodes
        ).lower()

        result = {
            "question_number": number,
            "question": question,
            "answer": str(response).strip(),
            "expected_fact": expected_fact,
            "source_files": source_files,
            "sources_returned": bool(source_nodes),
            "expected_fact_in_retrieved_context": (
                expected_fact.lower() in retrieved_text
            ),
            "latency_seconds": round(latency, 3),
        }
        results.append(result)

        print(f"\nQ{number}: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {', '.join(source_files) or 'None returned'}")
        print(
            "Expected fact present in retrieved context:",
            result["expected_fact_in_retrieved_context"],
        )
        print(f"Latency: {latency:.3f} seconds")

    average_latency = total_latency / len(QUESTIONS)
    print(f"\nAverage query latency: {average_latency:.3f} seconds")

    return {
        "index_name": index_name,
        "query_count": len(results),
        "average_latency_seconds": round(average_latency, 3),
        "results": results,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configure_models()
    documents = load_documents()

    # Run queries against the in-memory index.
    memory_index = create_memory_index(documents)
    memory_results = run_queries(
        memory_index,
        "In-memory LlamaIndex",
    )

    # Run the same queries against the persistent ChromaDB index.
    chroma_index = create_chroma_index(documents)
    chroma_results = run_queries(
        chroma_index,
        "LlamaIndex + ChromaDB",
    )

    report = {
        "embedding_model": "nomic-embed-text",
        "llm_model": "llama3.2:3b",
        "context_window": 2048,
        "document_count": len(documents),
        "questions_per_index": len(QUESTIONS),
        "in_memory": memory_results,
        "chroma": chroma_results,
        "latency_comparison_seconds": {
            "in_memory": memory_results["average_latency_seconds"],
            "chroma": chroma_results["average_latency_seconds"],
        },
        "latency_note": (
            "Latency includes retrieval and LLM generation. Model warm-up "
            "and local system load can affect the measurements."
        ),
    }

    report_path = OUTPUT_DIR / "w7d3_query_results.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 40)
    print("LATENCY COMPARISON")
    print("=" * 40)
    print(f"In-memory: {memory_results['average_latency_seconds']:.3f}s")
    print(f"ChromaDB:  {chroma_results['average_latency_seconds']:.3f}s")
    print(f"Detailed results saved to: {report_path}")


if __name__ == "__main__":
    main()
