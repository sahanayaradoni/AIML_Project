
from pathlib import Path


PROJECT_DIR = Path(__file__).parent
PDF_DIR = PROJECT_DIR / "pdf_documents"


def test_pdf_documents_directory_exists():
    assert PDF_DIR.is_dir()


def test_five_pdf_documents_exist():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    assert len(pdf_files) == 5


def test_expected_pdf_documents_exist():
    expected_files = {
        "machine_learning.pdf",
        "deep_learning.pdf",
        "nlp.pdf",
        "computer_vision.pdf",
        "mlops.pdf",
    }
    actual_files = {path.name for path in PDF_DIR.glob("*.pdf")}
    assert expected_files.issubset(actual_files)


def test_bm25_evaluation_report_exists():
    report = PROJECT_DIR / "W7D2_BM25_EVALUATION.md"
    assert report.is_file()


def test_dense_evaluation_report_exists():
    report = PROJECT_DIR / "W7D2_DENSE_EVALUATION.md"
    assert report.is_file()


def test_retrieval_comparison_report_exists():
    report = PROJECT_DIR / "W7D2_RETRIEVAL_COMPARISON.md"
    assert report.is_file()


def test_comparison_reports_expected_metrics():
    report = (PROJECT_DIR / "W7D2_RETRIEVAL_COMPARISON.md").read_text(
        encoding="utf-8"
    )
    assert "9/10" in report
    assert "90%" in report
    assert "10/10" in report
    assert "100%" in report
