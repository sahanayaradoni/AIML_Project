
"""Tests for the W7D5 multi-document RAG system."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from w7d5_multi_document_rag import (
    load_pdf_documents,
    query_documents,
)


DOCUMENTS_DIR = Path(__file__).parent / "documents"


def test_load_pdf_documents_returns_all_five_pdfs():
    """Verify that all five supplied PDFs are loaded."""
    documents = load_pdf_documents(str(DOCUMENTS_DIR))

    assert len(documents) == 5


def test_loaded_documents_have_source_and_text():
    """Verify that every loaded PDF has a filename and extracted text."""
    documents = load_pdf_documents(str(DOCUMENTS_DIR))

    for document in documents:
        assert document["source"].endswith(".pdf")
        assert document["text"].strip()


def test_expected_pdf_filenames_are_loaded():
    """Verify that the five expected PDF files are present."""
    documents = load_pdf_documents(str(DOCUMENTS_DIR))

    sources = {document["source"] for document in documents}

    assert sources == {
        "computer_vision.pdf",
        "deep_learning.pdf",
        "machine_learning.pdf",
        "mlops.pdf",
        "nlp.pdf",
    }


def test_missing_folder_raises_value_error(tmp_path):
    """Verify that a nonexistent folder raises a clear error."""
    missing_folder = tmp_path / "missing_documents"

    with pytest.raises(ValueError, match="PDF folder does not exist"):
        load_pdf_documents(str(missing_folder))


def test_empty_folder_returns_empty_list(tmp_path):
    """Verify that a valid folder with no PDFs returns an empty list."""
    empty_folder = tmp_path / "empty_documents"
    empty_folder.mkdir()

    documents = load_pdf_documents(str(empty_folder))

    assert documents == []


def test_query_documents_returns_answer():
    """Verify that query_documents returns the generated response."""
    mock_index = Mock()
    mock_query_engine = Mock()
    mock_query_engine.query.return_value = (
        "Machine learning learns from data."
    )
    mock_index.as_query_engine.return_value = mock_query_engine

    mock_settings = Mock()
    mock_settings.llm = Mock()

    with patch(
        "w7d5_multi_document_rag.Settings",
        new=mock_settings,
    ):
        answer = query_documents(
            mock_index,
            "What is machine learning?",
        )

    assert answer == "Machine learning learns from data."
    mock_index.as_query_engine.assert_called_once_with(
        llm=mock_settings.llm,
        similarity_top_k=3,
    )
    mock_query_engine.query.assert_called_once_with(
        "What is machine learning?"
    )
