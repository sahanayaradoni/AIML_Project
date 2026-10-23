from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


PDF_DIR = Path(__file__).parent / "pdf_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
    """Embed and index documents in an in-memory DocumentStore."""
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=EMBEDDING_MODEL
    )
    document_embedder.warm_up()

    embedding_result = document_embedder.run(
        documents=documents
    )

    document_store = InMemoryDocumentStore()
    document_store.write_documents(
        embedding_result["documents"]
    )

    return document_store


def build_dense_pipeline(document_store):
    """Build a Haystack pipeline using dense retrieval."""
    text_embedder = SentenceTransformersTextEmbedder(
        model=EMBEDDING_MODEL
    )
    text_embedder.warm_up()

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "text_embedder",
        text_embedder,
    )

    pipeline.add_component(
        "retriever",
        retriever,
    )

    pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding",
    )

    return pipeline


def run_dense_query(pipeline, query):
    """Run a query through the dense retrieval pipeline."""
    result = pipeline.run(
        {
            "text_embedder": {
                "text": query,
            },
            "retriever": {
                "top_k": 3,
            },
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

    pipeline = build_dense_pipeline(document_store)

    print("\nDense Retrieval Results:")

    for question_number, query in enumerate(queries, start=1):
        retrieved_documents = run_dense_query(
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