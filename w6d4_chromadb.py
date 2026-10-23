"""
W6D4: RAG Pipeline — LangChain + ChromaDB

Tasks:
1. Create ChromaDB collection and add 20 documents with embeddings.
2. Perform cosine similarity search and metadata filtering.
3. Create/embed a PDF, retrieve top-3 chunks, and pass context to Ollama.
"""

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "./w6d4_chroma_db"
COLLECTION_NAME = "w6d4_documents"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

PDF_PATH = Path("w6d4_sample.pdf")


# ============================================================
# TASK 1 — CREATE CHROMADB COLLECTION
# ============================================================

def create_collection():
    """Create a persistent ChromaDB collection."""

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    return client, collection


# ============================================================
# TASK 1 — 20 DOCUMENTS
# ============================================================

def get_documents():
    """Return 20 sample documents with metadata."""

    documents = [
        "Python is a popular programming language for data science.",
        "Machine learning allows computers to learn patterns from data.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing works with human language.",
        "LangChain helps developers build applications with language models.",
        "ChromaDB is a vector database designed for embeddings and similarity search.",
        "Embeddings represent text as numerical vectors.",
        "Cosine similarity measures the similarity between two vectors.",
        "RAG combines information retrieval with language generation.",
        "Ollama allows language models to run locally.",
        "FastAPI is a Python framework for building APIs.",
        "Scikit-learn provides machine learning algorithms in Python.",
        "NumPy provides numerical computing capabilities in Python.",
        "Pandas is widely used for data manipulation and analysis.",
        "Matplotlib is a Python library for data visualization.",
        "Git is a distributed version control system.",
        "GitHub hosts Git repositories and supports collaborative development.",
        "Pytest is a Python framework for automated testing.",
        "SQL is used to store and query structured data.",
        "Artificial intelligence enables machines to perform tasks that normally require human intelligence.",
    ]

    topics = [
        "python",
        "machine-learning",
        "deep-learning",
        "nlp",
        "langchain",
        "chromadb",
        "embeddings",
        "similarity",
        "rag",
        "ollama",
        "fastapi",
        "machine-learning",
        "python",
        "data-analysis",
        "visualization",
        "git",
        "github",
        "testing",
        "database",
        "ai",
    ]

    return [
        Document(
            page_content=text,
            metadata={
                "topic": topic,
                "source": "W6D4 sample documents",
            },
        )
        for text, topic in zip(documents, topics)
    ]


def add_documents(collection):
    """Add 20 documents to ChromaDB using Ollama embeddings."""

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        client=chromadb.PersistentClient(path=CHROMA_PATH),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    documents = get_documents()

    ids = [f"doc_{i}" for i in range(1, 21)]

    vectorstore.add_documents(
        documents=documents,
        ids=ids,
    )

    return vectorstore


# ============================================================
# TASK 2 — SIMILARITY SEARCH
# ============================================================

def similarity_search(vectorstore):
    """Perform cosine similarity search."""

    query = "How does artificial intelligence and machine learning work?"

    print("\n" + "=" * 60)
    print("TASK 2A: COSINE SIMILARITY SEARCH")
    print("=" * 60)

    results = vectorstore.similarity_search_with_score(
        query,
        k=5,
    )

    print(f"Query: {query}")
    print("\nTop 5 results:")

    for rank, (document, score) in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Score: {score:.4f}")
        print(f"Topic: {document.metadata.get('topic')}")
        print(f"Text: {document.page_content}")


# ============================================================
# TASK 2 — METADATA FILTERING
# ============================================================

def metadata_filtering(vectorstore):
    """Perform metadata filtering."""

    query = "programming and data"

    print("\n" + "=" * 60)
    print("TASK 2B: METADATA FILTERING")
    print("=" * 60)

    results = vectorstore.similarity_search(
        query,
        k=5,
        filter={"topic": "python"},
    )

    print(f"Query: {query}")
    print("Filter: topic = python")

    print("\nFiltered results:")

    for rank, document in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Topic: {document.metadata.get('topic')}")
        print(f"Text: {document.page_content}")


# ============================================================
# TASK 3 — CREATE SAMPLE PDF
# ============================================================

def create_sample_pdf():
    """Create a small PDF for the RAG demonstration."""

    if PDF_PATH.exists():
        return

    pdf = canvas.Canvas(str(PDF_PATH), pagesize=A4)

    text = pdf.beginText(50, 800)
    text.setFont("Helvetica", 11)

    content = [
        "LangChain and Retrieval Augmented Generation",
        "",
        "LangChain is a framework for building applications",
        "that use large language models.",
        "",
        "Retrieval Augmented Generation, commonly called RAG,",
        "combines information retrieval with language generation.",
        "",
        "A RAG system first retrieves relevant documents or",
        "document chunks from a vector database.",
        "",
        "The retrieved information is then provided to a",
        "language model as context.",
        "",
        "ChromaDB is a vector database that can store embeddings",
        "and perform similarity searches.",
        "",
        "Embeddings convert text into numerical vectors that",
        "capture semantic meaning.",
        "",
        "Ollama can run language models locally.",
        "This makes it possible to build local RAG applications.",
    ]

    for line in content:
        text.textLine(line)

    pdf.drawText(text)
    pdf.save()

    print(f"\nCreated sample PDF: {PDF_PATH}")


# ============================================================
# TASK 3 — READ PDF
# ============================================================

def load_pdf():
    """Read text from the PDF."""

    reader = PdfReader(str(PDF_PATH))

    documents = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": PDF_PATH.name,
                        "page": page_number + 1,
                    },
                )
            )

    return documents


# ============================================================
# TASK 3 — PDF RAG PIPELINE
# ============================================================

def run_rag_pipeline():
    """Embed PDF chunks, retrieve top 3, and ask Ollama."""

    print("\n" + "=" * 60)
    print("TASK 3: CHROMADB + OLLAMA RAG PIPELINE")
    print("=" * 60)

    pdf_documents = load_pdf()

    if not pdf_documents:
        raise ValueError("No text could be extracted from the PDF.")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    pdf_vectorstore = Chroma.from_documents(
        documents=pdf_documents,
        embedding=embeddings,
        collection_name="w6d4_pdf_collection",
        persist_directory="./w6d4_pdf_chroma_db",
    )

    question = "What is RAG and how does ChromaDB help in a RAG system?"

    retrieved_docs = pdf_vectorstore.similarity_search(
        question,
        k=3,
    )

    print(f"Question: {question}")

    print("\nTop-3 retrieved chunks:")

    for rank, document in enumerate(retrieved_docs, start=1):
        print(f"\nChunk {rank}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Page: {document.metadata.get('page')}")
        print(f"Text: {document.page_content[:500]}")


    # --------------------------------------------------------
    # Pass retrieved context to Ollama
    # --------------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in retrieved_docs
    )

    prompt_template = PromptTemplate.from_template(
        """
You are a helpful RAG assistant.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Give a concise and accurate answer.
"""
    )

    prompt = prompt_template.format(
        context=context,
        question=question,
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

    response = llm.invoke(prompt)

    print("\n" + "=" * 60)
    print("OLLAMA RAG ANSWER")
    print("=" * 60)

    print(response.content)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("W6D4: RAG PIPELINE — LANGCHAIN + CHROMADB")
    print("=" * 60)

    # --------------------------------------------------------
    # Task 1
    # --------------------------------------------------------

    client, collection = create_collection()

    print("\nChromaDB collection created:")
    print(f"Collection: {collection.name}")

    vectorstore = add_documents(collection)

    print(f"Documents added: {vectorstore._collection.count()}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Distance metric: cosine")

    # --------------------------------------------------------
    # Task 2
    # --------------------------------------------------------

    similarity_search(vectorstore)
    metadata_filtering(vectorstore)

    # --------------------------------------------------------
    # Task 3
    # --------------------------------------------------------

    create_sample_pdf()
    run_rag_pipeline()

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("W6D4 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("✓ ChromaDB collection created")
    print("✓ 20 documents added with embeddings")
    print("✓ Cosine similarity search performed")
    print("✓ Metadata filtering performed")
    print("✓ PDF embedded")
    print("✓ Top-3 PDF chunks retrieved")
    print("✓ Retrieved context passed to Ollama")
    print("✓ Ollama generated the final answer")

if __name__ == "__main__":
    import io
    import contextlib
    from pathlib import Path

    # Create evidence folder
    evidence_dir = Path("w6d4_outputs")
    evidence_dir.mkdir(exist_ok=True)

    # Capture program output
    output_buffer = io.StringIO()

    with contextlib.redirect_stdout(output_buffer):
        main()

    output = output_buffer.getvalue()

    # Display the captured output in the terminal
    print(output)

    # Save text evidence
    results_file = evidence_dir / "w6d4_results.txt"
    results_file.write_text(output, encoding="utf-8")

    # Create PNG evidence
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 10))
        fig.text(
            0.02,
            0.98,
            output,
            fontsize=9,
            fontfamily="monospace",
            verticalalignment="top",
        )

        plt.axis("off")
        plt.savefig(
            evidence_dir / "w6d4_success.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        print("Evidence PNG saved: w6d4_outputs/w6d4_success.png")

    except Exception as e:
        print(f"PNG evidence could not be created: {e}")

    print("Evidence text saved: w6d4_outputs/w6d4_results.txt")