from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


PDF_DIR = Path(__file__).parent / "pdf_documents"


def load_documents():
    """Convert all PDF files into Haystack Documents."""
    converter = PyPDFToDocument()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    documents = []

    for pdf_file in pdf_files:
        result = converter.run(sources=[str(pdf_file)])
        documents.extend(result["documents"])

    return documents


def create_document_store(documents):
    """Create an in-memory DocumentStore and index documents."""
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    return document_store


def build_bm25_pipeline(document_store):
    """Build a Haystack pipeline using BM25 retrieval."""
    pipeline = Pipeline()

    retriever = InMemoryBM25Retriever(
        document_store=document_store
    )

    pipeline.add_component(
        "retriever",
        retriever,
    )

    return pipeline


def run_bm25_query(pipeline, query):
    """Run a query through the BM25 pipeline."""
    result = pipeline.run(
        {
            "retriever": {
                "query": query,
                "top_k": 3,
            }
        }
    )

    return result["retriever"]["documents"]


def main():
    documents = load_documents()
    document_store = create_document_store(documents)

    print(f"PDF files found: {len(list(PDF_DIR.glob('*.pdf')))}")
    print(f"Documents indexed: {document_store.count_documents()}")

    queries = [
        "What is machine learning?",
        "What is supervised learning?",
        "What is deep learning?",
        "What is a neural network?",
        "What is natural language processing?",
        "What are common NLP tasks?",
        "What is computer vision?",
        "What is object detection?",
        "What is MLOps?",
        "What is model monitoring?",
    ]

    pipeline = build_bm25_pipeline(document_store)

    print("\nBM25 Retrieval Results:")

    for question_number, query in enumerate(queries, start=1):
        retrieved_documents = run_bm25_query(
            pipeline,
            query,
        )

        print(f"\nQuestion {question_number}: {query}")

        for rank, document in enumerate(
            retrieved_documents,
            start=1,
        ):
            file_name = Path(
                document.meta.get("file_path", "unknown")
            ).name

            print(
                f"  {rank}. {file_name} "
                f"(score={document.score:.4f})"
            )


if __name__ == "__main__":
    main()