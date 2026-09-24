from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


OUTPUT_DIR = Path(__file__).parent / "pdf_documents"


DOCUMENTS = {
    "machine_learning.pdf": {
        "title": "Machine Learning",
        "content": (
            "Machine learning is a branch of artificial intelligence that enables "
            "computers to learn patterns from data. Supervised learning uses labeled "
            "examples for tasks such as classification and regression. Unsupervised "
            "learning works with unlabeled data and includes clustering. "
            "A training dataset is used to learn a model, while test data is used "
            "to measure how well the model generalizes to unseen examples. "
            "Common machine learning algorithms include linear regression, logistic "
            "regression, decision trees, random forests, support vector machines, "
            "and k-nearest neighbors."
        ),
    },
    "deep_learning.pdf": {
        "title": "Deep Learning",
        "content": (
            "Deep learning is a subset of machine learning based on neural networks "
            "with multiple layers. A neural network contains interconnected neurons "
            "that transform input data through learned weights and biases. "
            "Activation functions such as ReLU introduce non-linearity into the model. "
            "Deep learning is widely used for image recognition, speech processing, "
            "natural language processing, and other complex pattern-recognition tasks. "
            "Training commonly uses backpropagation and gradient-based optimization."
        ),
    },
    "nlp.pdf": {
        "title": "Natural Language Processing",
        "content": (
            "Natural language processing, or NLP, focuses on enabling computers to "
            "process and understand human language. Important NLP tasks include "
            "text classification, sentiment analysis, named entity recognition, "
            "machine translation, and question answering. Text can be represented "
            "using techniques such as bag-of-words, TF-IDF, word embeddings, and "
            "transformer-based representations. Modern NLP systems often use "
            "transformer architectures because they can capture relationships "
            "between words across a sequence."
        ),
    },
    "computer_vision.pdf": {
        "title": "Computer Vision",
        "content": (
            "Computer vision enables computers to interpret information from images "
            "and videos. Common tasks include image classification, object detection, "
            "image segmentation, and face recognition. Convolutional neural networks "
            "have traditionally been important for extracting spatial features from "
            "images. Computer vision systems can be trained using labeled image "
            "datasets and evaluated using metrics such as accuracy, precision, "
            "recall, and mean average precision."
        ),
    },
    "mlops.pdf": {
        "title": "MLOps",
        "content": (
            "MLOps combines machine learning development with software engineering "
            "and operations practices. It helps teams automate model training, "
            "testing, deployment, monitoring, and versioning. Important MLOps "
            "activities include experiment tracking, data validation, model "
            "versioning, continuous integration, continuous delivery, and model "
            "monitoring. Production machine learning systems should be monitored "
            "for changes in data quality and model performance so that models can "
            "be retrained when necessary."
        ),
    },
}


def create_pdf(filename, title, content):
    output_path = OUTPUT_DIR / filename

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        title=title,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 20),
        Paragraph(content, styles["BodyText"]),
    ]

    document.build(story)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, data in DOCUMENTS.items():
        create_pdf(
            filename,
            data["title"],
            data["content"],
        )

    print(f"Created {len(DOCUMENTS)} PDF documents in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()