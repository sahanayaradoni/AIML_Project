"""
W5D4: PDF Semantic Search with ChromaDB and Ollama

Pipeline:
PDF -> Text Extraction -> Chunks -> ChromaDB
-> Top-3 Retrieval -> Ollama -> Answer

Author: Sahana
"""

import os
import requests
import chromadb
from pypdf import PdfReader


# ============================================================
# 1. CONFIGURATION
# ============================================================

PDF_PATH = "sample_documents/aiml_sample.pdf"

OUTPUT_DIR = "w5d4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "w5d4_pdf_ollama_results.txt"
)

CHROMA_PATH = "./w5d4_pdf_chroma_db"

COLLECTION_NAME = "w5d4_pdf_documents"

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = "llama3.2:3b"


# ============================================================
# 2. CHECK PDF
# ============================================================

print("=" * 70)
print("W5D4 - PDF + CHROMADB + OLLAMA SEMANTIC SEARCH")
print("=" * 70)

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}"
    )

print("\nPDF found:")
print(PDF_PATH)


# ============================================================
# 3. EXTRACT TEXT FROM PDF
# ============================================================

reader = PdfReader(PDF_PATH)

pdf_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        pdf_text += text + "\n"

print("\nPDF pages:", len(reader.pages))
print("PDF text extracted successfully.")
print("Characters extracted:", len(pdf_text))


# ============================================================
# 4. SPLIT TEXT INTO CHUNKS
# ============================================================

chunk_size = 500
overlap = 100

chunks = []

start = 0

while start < len(pdf_text):

    end = start + chunk_size

    chunk = pdf_text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    start = end - overlap


print("\nText chunks created:", len(chunks))


# ============================================================
# 5. CREATE CHROMADB CLIENT
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)

print("\nChromaDB collection created:")
print(collection.name)


# ============================================================
# 6. STORE PDF CHUNKS
# ============================================================

chunk_ids = [
    f"w5d4_pdf_chunk_{i}"
    for i in range(len(chunks))
]

chunk_metadatas = [
    {
        "source": "aiml_sample.pdf",
        "chunk_number": i + 1
    }
    for i in range(len(chunks))
]


collection.upsert(
    ids=chunk_ids,
    documents=chunks,
    metadatas=chunk_metadatas
)

print("PDF chunks stored in ChromaDB.")
print("Documents in collection:", collection.count())


# ============================================================
# 7. SEMANTIC SEARCH
# ============================================================

query = (
    "What is Retrieval-Augmented Generation "
    "and how does it work?"
)

results = collection.query(
    query_texts=[query],
    n_results=3
)

retrieved_chunks = results["documents"][0]
retrieved_ids = results["ids"][0]
retrieved_distances = results["distances"][0]
retrieved_metadatas = results["metadatas"][0]


print("\n" + "=" * 70)
print("TOP-3 SEMANTIC SEARCH RESULTS")
print("=" * 70)

print("\nQuery:")
print(query)

for i in range(len(retrieved_chunks)):

    print(f"\n--- Rank {i + 1} ---")
    print("ID:", retrieved_ids[i])
    print("Chunk number:",
          retrieved_metadatas[i]["chunk_number"])
    print("Cosine distance:",
          round(retrieved_distances[i], 4))
    print("Text:")
    print(retrieved_chunks[i])


# ============================================================
# 8. COMBINE RETRIEVED CHUNKS
# ============================================================

context = "\n\n".join(
    [
        f"Context {i + 1}:\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    ]
)


# ============================================================
# 9. CREATE OLLAMA PROMPT
# ============================================================

prompt = f"""
You are an AI assistant answering questions using retrieved
information from a PDF.

Answer the question using ONLY the provided context.

If the answer is not present in the context, say:
"The information is not available in the provided document."

Question:
{query}

Retrieved Context:
{context}

Give a concise and accurate answer.
"""


# ============================================================
# 10. SEND CONTEXT TO OLLAMA
# ============================================================

print("\n" + "=" * 70)
print("OLLAMA RESPONSE")
print("=" * 70)

payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False
}

try:

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    ollama_result = response.json()

    answer = ollama_result.get(
        "response",
        "No response received from Ollama."
    )

    print("\nModel:", OLLAMA_MODEL)
    print("\nAnswer:")
    print(answer)

except requests.exceptions.RequestException as error:

    print("\nOllama request failed.")
    print("Error:", error)

    answer = f"Ollama request failed: {error}"


# ============================================================
# 11. SAVE OUTPUT EVIDENCE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 70 + "\n")
    file.write("W5D4 - PDF SEMANTIC SEARCH + OLLAMA RESULTS\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"PDF: {PDF_PATH}\n")
    file.write(f"Pages: {len(reader.pages)}\n")
    file.write(f"Chunks: {len(chunks)}\n")
    file.write(f"ChromaDB documents: {collection.count()}\n")
    file.write(f"Ollama model: {OLLAMA_MODEL}\n\n")

    file.write("QUERY\n")
    file.write("-" * 70 + "\n")
    file.write(query + "\n\n")

    file.write("TOP-3 RETRIEVED CHUNKS\n")
    file.write("-" * 70 + "\n")

    for i in range(len(retrieved_chunks)):

        file.write(f"\nRank {i + 1}\n")
        file.write(f"ID: {retrieved_ids[i]}\n")

        file.write(
            f"Chunk number: "
            f"{retrieved_metadatas[i]['chunk_number']}\n"
        )

        file.write(
            f"Cosine distance: "
            f"{retrieved_distances[i]:.4f}\n"
        )

        file.write(
            f"Text:\n{retrieved_chunks[i]}\n"
        )

    file.write("\n\nOLLAMA ANSWER\n")
    file.write("-" * 70 + "\n")
    file.write(answer + "\n")


# ============================================================
# 12. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("W5D4 PDF + OLLAMA TASK COMPLETED")
print("=" * 70)

print("\nEvidence saved to:")
print(OUTPUT_FILE)

print("\nPipeline completed:")
print("PDF -> chunks -> ChromaDB -> top-3 retrieval -> Ollama")