from pathlib import Path

from haystack import Document

from w7d1_haystack.w7d1_haystack_pipeline import (
    PDF_DIR,
    create_document_store,
    load_documents,
)
from w7d1_haystack.w7d1_dense_retrieval import (
    EMBEDDING_MODEL,
    create_document_store as create_dense_document_store,
)


def test_pdf_directory_contains_five_documents():
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))

    assert len(pdf_files) == 5


def test_load_documents_returns_five_documents():
    documents = load_documents()

    assert len(documents) == 5
    assert all(isinstance(document, Document) for document in documents)


def test_bm25_document_store_indexes_documents():
    documents = load_documents()
    document_store = create_document_store(documents)

    assert document_store.count_documents() == 5


def test_dense_document_store_indexes_documents():
    documents = load_documents()
    document_store = create_dense_document_store(documents)

    assert document_store.count_documents() == 5


def test_dense_embedding_model_is_configured():
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"