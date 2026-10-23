from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
import os


OUTPUT_DIR = "sample_documents"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "aiml_sample.pdf"
)


def create_pdf():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    document = SimpleDocTemplate(
        OUTPUT_FILE,
        pagesize=A4,
        title="AI and Machine Learning Notes"
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    content = []

    content.append(
        Paragraph(
            "AI and Machine Learning Notes",
            title_style
        )
    )

    content.append(Spacer(1, 20))

    sections = [
        (
            "1. Artificial Intelligence",
            "Artificial Intelligence, or AI, is the field of "
            "computer science focused on creating systems that "
            "can perform tasks that normally require human "
            "intelligence. These tasks include reasoning, "
            "learning, problem solving, and understanding language."
        ),
        (
            "2. Machine Learning",
            "Machine Learning is a branch of AI in which computers "
            "learn patterns from data. Instead of explicitly "
            "programming every rule, a machine learning model "
            "uses training data to learn relationships and make "
            "predictions on new data."
        ),
        (
            "3. Deep Learning",
            "Deep Learning is a subfield of machine learning that "
            "uses neural networks with multiple layers. Deep "
            "learning is commonly used for image recognition, "
            "speech recognition, natural language processing, "
            "and other complex tasks."
        ),
        (
            "4. Embeddings",
            "An embedding is a numerical representation of data "
            "such as text. Similar meanings are represented by "
            "vectors that are close to each other in vector space. "
            "Embeddings are widely used for semantic search and "
            "retrieval-augmented generation."
        ),
        (
            "5. Vector Databases",
            "A vector database stores numerical vector "
            "representations and allows efficient similarity "
            "search. ChromaDB is a vector database that can be "
            "used to store embeddings and retrieve documents "
            "that are semantically similar to a query."
        ),
        (
            "6. Cosine Similarity",
            "Cosine similarity measures how similar two vectors "
            "are based on the angle between them. It is commonly "
            "used in semantic search because vectors representing "
            "similar meanings tend to have similar directions."
        ),
        (
            "7. Retrieval-Augmented Generation",
            "Retrieval-Augmented Generation, or RAG, combines "
            "information retrieval with a language model. Relevant "
            "document chunks are retrieved from a vector database "
            "and provided as context to a language model. This "
            "allows the model to generate answers using information "
            "from the retrieved documents."
        ),
        (
            "8. Ollama",
            "Ollama allows developers to run large language models "
            "locally. A local model can receive retrieved document "
            "context and generate an answer without requiring the "
            "document content to be sent to an external service."
        ),
    ]

    for heading, text in sections:
        content.append(
            Paragraph(heading, heading_style)
        )
        content.append(Spacer(1, 8))

        content.append(
            Paragraph(text, body_style)
        )
        content.append(Spacer(1, 15))

    document.build(content)

    print("Sample PDF created successfully!")
    print(f"PDF location: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_pdf()