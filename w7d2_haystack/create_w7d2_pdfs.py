from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


OUTPUT_DIR = Path(__file__).parent / "pdf_documents"


DOCUMENTS = {
    "machine_learning.pdf": (
        "Machine Learning",
        [
            "Machine learning enables computers to learn patterns from data.",
            "Supervised learning uses labelled training data for prediction tasks.",
            "Common supervised algorithms include linear regression, logistic regression, "
            "decision trees, and random forests.",
        ],
    ),
    "deep_learning.pdf": (
        "Deep Learning",
        [
            "Deep learning is a machine learning approach based on neural networks.",
            "Neural networks contain layers of interconnected computational units.",
            "Deep learning is widely used for image recognition, language processing, "
            "speech recognition, and other complex tasks.",
        ],
    ),
    "nlp.pdf": (
        "Natural Language Processing",
        [
            "Natural language processing, or NLP, enables computers to work with human language.",
            "Common NLP tasks include text classification, sentiment analysis, "
            "named entity recognition, and machine translation.",
            "NLP systems can use machine learning and deep learning models to process text.",
        ],
    ),
    "computer_vision.pdf": (
        "Computer Vision",
        [
            "Computer vision enables computers to interpret information from images and videos.",
            "Object detection identifies objects and their locations within images.",
            "Image classification, object detection, and image segmentation are common "
            "computer vision tasks.",
        ],
    ),
    "mlops.pdf": (
        "MLOps",
        [
            "MLOps combines machine learning development with deployment and operations practices.",
            "Model monitoring helps detect changes in model performance after deployment.",
            "MLOps can include model versioning, experiment tracking, deployment automation, "
            "and continuous monitoring.",
        ],
    ),
}


def create_pdf(file_path, title, paragraphs):
    pdf = canvas.Canvas(str(file_path), pagesize=A4)
    width, height = A4

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(72, height - 72, title)

    y_position = height - 110
    pdf.setFont("Helvetica", 11)

    for paragraph in paragraphs:
        pdf.drawString(72, y_position, paragraph)
        y_position -= 35

    pdf.save()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_name, (title, paragraphs) in DOCUMENTS.items():
        create_pdf(
            OUTPUT_DIR / file_name,
            title,
            paragraphs,
        )

    print(f"Created {len(DOCUMENTS)} PDF documents.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()