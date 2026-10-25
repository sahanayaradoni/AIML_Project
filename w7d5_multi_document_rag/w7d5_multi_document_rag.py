
"""W7D5: Multi-document RAG System using LlamaIndex, Ollama, and ChromaDB."""

from pathlib import Path

import chromadb
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from pypdf import PdfReader


BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "w7d5_documents"

EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen2.5:3b"


def load_pdf_documents(folder_path: str) -> list[dict[str, str]]:
    """Load readable text from PDF files in a folder."""
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"PDF folder does not exist: {folder}")

    documents = []

    for pdf_path in sorted(folder.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()

        if text:
            documents.append({
                "source": pdf_path.name,
                "text": text,
            })

    return documents


def build_vector_index() -> VectorStoreIndex:
    """Index PDF documents in persistent ChromaDB storage."""
    pdf_documents = load_pdf_documents(str(DOCUMENTS_DIR))

    if not pdf_documents:
        raise ValueError(f"No readable PDF text found in {DOCUMENTS_DIR}")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # Rebuild this W7D5 collection to avoid duplicate records.
    if collection.count() > 0:
        chroma_client.delete_collection(COLLECTION_NAME)
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)

    documents = [
        Document(
            text=item["text"],
            metadata={"source": item["source"]},
        )
        for item in pdf_documents
    ]

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(f"PDF documents loaded: {len(pdf_documents)}")
    print(f"ChromaDB collection: {COLLECTION_NAME}")
    print(f"Indexed chunks: {collection.count()}")

    return index


def query_documents(index: VectorStoreIndex, question: str) -> str:
    """Answer a question using the indexed PDF documents."""
    query_engine = index.as_query_engine(
        llm=Settings.llm,
        similarity_top_k=3,
    )
    response = query_engine.query(question)
    return str(response)


def main() -> None:
    """Run the multi-document RAG demonstration."""
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm = Ollama(
        model=LLM_MODEL,
        request_timeout=300.0,
    )

    index = build_vector_index()

    questions = [
        "What is machine learning?",
        "What is deep learning?",
        "What is natural language processing?",
        "What is computer vision?",
        "What is MLOps?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        try:
            answer = query_documents(index, question)
            print(f"Answer: {answer}")
        except Exception as error:
            print(f"Query failed: {error}")


if __name__ == "__main__":
    main()
