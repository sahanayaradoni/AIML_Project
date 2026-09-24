import os

import chromadb
import ollama
from pypdf import PdfReader


# ============================================================
# W5D3: PDF + ChromaDB + Ollama RAG
# ============================================================

PDF_PATH = os.path.join(
    "sample_documents",
    "aiml_sample.pdf"
)

OUTPUT_DIR = os.path.join(
    "outputs",
    "w5d3"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "pdf_rag_results.txt"
)

COLLECTION_NAME = "w5d3_pdf_documents"

OLLAMA_MODEL = "llama3.2:3b"


# ============================================================
# Create output directory
# ============================================================

def create_output_directory():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# Extract text from PDF
# ============================================================

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        text = page.extract_text()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip()
                }
            )

    return pages


# ============================================================
# Split text into chunks
# ============================================================

def create_chunks(pages, chunk_size=100):
    chunks = []

    chunk_id = 1

    for page in pages:
        text = page["text"]

        words = text.split()

        for start in range(
            0,
            len(words),
            chunk_size
        ):
            chunk_words = words[
                start:start + chunk_size
            ]

            chunk_text = " ".join(
                chunk_words
            )

            if chunk_text.strip():
                chunks.append(
                    {
                        "id": f"pdf_chunk_{chunk_id}",
                        "text": chunk_text,
                        "page": page["page"]
                    }
                )

                chunk_id += 1

    return chunks


# ============================================================
# Store PDF chunks in ChromaDB
# ============================================================

def create_pdf_collection(chunks):
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )

    # Clear previous chunks when rerunning
    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": "aiml_sample.pdf",
            "page": chunk["page"],
            "chunk_id": chunk["id"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    return collection


# ============================================================
# Retrieve top 3 PDF chunks
# ============================================================

def retrieve_chunks(collection, query):
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    retrieved_chunks = []

    for i, document in enumerate(
        results["documents"][0]
    ):
        distance = results[
            "distances"
        ][0][i]

        metadata = results[
            "metadatas"
        ][0][i]

        retrieved_chunks.append(
            {
                "document": document,
                "distance": distance,
                "metadata": metadata
            }
        )

    return retrieved_chunks


# ============================================================
# Ask Ollama using retrieved context
# ============================================================

def ask_ollama(query, retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):
        context_parts.append(
            f"Chunk {i} "
            f"(Page {chunk['metadata']['page']}):\n"
            f"{chunk['document']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
Use only the information provided in the
context below to answer the question.

If the answer is not present in the context,
say that the information is not available
in the provided document.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# Save results
# ============================================================

def save_results(
    pdf_path,
    pages,
    chunks,
    query,
    retrieved_chunks,
    answer
):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "W5D3 — PDF + ChromaDB + Ollama RAG\n"
        )

        file.write(
            "====================================\n\n"
        )

        file.write(
            f"PDF: {pdf_path}\n"
        )

        file.write(
            f"Pages extracted: {len(pages)}\n"
        )

        file.write(
            f"Chunks created: {len(chunks)}\n\n"
        )

        file.write(
            "QUERY\n"
        )

        file.write(
            "-----\n"
        )

        file.write(
            f"{query}\n\n"
        )

        file.write(
            "TOP 3 RETRIEVED CHUNKS\n"
        )

        file.write(
            "-----------------------\n\n"
        )

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            file.write(
                f"Chunk {i}\n"
            )

            file.write(
                f"Page: "
                f"{chunk['metadata']['page']}\n"
            )

            file.write(
                f"Chunk ID: "
                f"{chunk['metadata']['chunk_id']}\n"
            )

            file.write(
                f"Cosine distance: "
                f"{chunk['distance']:.4f}\n"
            )

            file.write(
                f"Text:\n"
                f"{chunk['document']}\n\n"
            )

        file.write(
            "OLLAMA ANSWER\n"
        )

        file.write(
            "-------------\n\n"
        )

        file.write(
            answer
        )

        file.write("\n")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("W5D3 — PDF + CHROMADB + OLLAMA RAG")
    print("=" * 60)

    create_output_directory()

    # Check PDF
    if not os.path.exists(PDF_PATH):
        print(
            f"ERROR: PDF not found: {PDF_PATH}"
        )
        return

    print(
        f"\nPDF found: {PDF_PATH}"
    )

    # Extract PDF text
    pages = extract_pdf_text(
        PDF_PATH
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    # Create chunks
    chunks = create_chunks(
        pages
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # Store chunks
    collection = create_pdf_collection(
        chunks
    )

    print(
        f"Chunks stored in ChromaDB: "
        f"{collection.count()}"
    )

    # Question
    query = (
        "What is Retrieval-Augmented "
        "Generation and how does it work?"
    )

    print(
        "\nQuery:"
    )

    print(query)

    # Retrieve top 3
    retrieved_chunks = retrieve_chunks(
        collection,
        query
    )

    print(
        "\nTop 3 retrieved chunks:"
    )

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        print(
            f"\nChunk {i}"
        )

        print(
            f"Page: "
            f"{chunk['metadata']['page']}"
        )

        print(
            f"Cosine distance: "
            f"{chunk['distance']:.4f}"
        )

        print(
            f"Text: "
            f"{chunk['document'][:200]}..."
        )

    # Ask Ollama
    print(
        "\nSending retrieved context "
        "to Ollama..."
    )

    answer = ask_ollama(
        query,
        retrieved_chunks
    )

    print(
        "\nOllama answer:"
    )

    print(answer)

    # Save evidence
    save_results(
        PDF_PATH,
        pages,
        chunks,
        query,
        retrieved_chunks,
        answer
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PDF RAG pipeline completed successfully!"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
