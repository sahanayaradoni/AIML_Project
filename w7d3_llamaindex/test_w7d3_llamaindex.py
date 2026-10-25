
"""Tests for the W7D3 LlamaIndex RAG project."""

import json
from pathlib import Path

from w7d3_llamaindex_rag import (
    COLLECTION_NAME,
    DOCUMENTS_DIR,
    QUESTIONS,
)

BASE_DIR = Path(__file__).resolve().parent


def test_five_text_documents_exist():
    assert len(list(DOCUMENTS_DIR.glob("*.txt"))) == 5


def test_all_questions_have_expected_facts():
    assert len(QUESTIONS) == 10
    for question, expected_fact in QUESTIONS:
        assert question.strip()
        assert expected_fact.strip()


def test_question_text_is_unique():
    questions = [question for question, _ in QUESTIONS]
    assert len(questions) == len(set(questions))


def test_chroma_collection_name_is_defined():
    assert COLLECTION_NAME == "w7d3_llamaindex_documents"


def test_saved_results_contain_both_indexes():
    report_path = BASE_DIR / "outputs" / "w7d3_query_results.json"
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["document_count"] == 5
    assert report["questions_per_index"] == 10
    assert len(report["in_memory"]["results"]) == 10
    assert len(report["chroma"]["results"]) == 10


def test_all_queries_returned_sources_and_expected_context():
    report_path = BASE_DIR / "outputs" / "w7d3_query_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    for index_name in ("in_memory", "chroma"):
        for result in report[index_name]["results"]:
            assert result["sources_returned"], result["question"]
            assert result["expected_fact_in_retrieved_context"], (
                result["question"]
            )


def test_latency_measurements_are_positive():
    report_path = BASE_DIR / "outputs" / "w7d3_query_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["in_memory"]["average_latency_seconds"] > 0
    assert report["chroma"]["average_latency_seconds"] > 0
