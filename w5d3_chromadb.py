import os
import chromadb


# ============================================================
# W5D3: ChromaDB — Vector Store Setup & Similarity Search
# ============================================================

DOCUMENTS = [
    "Python is a high-level programming language used for data science and machine learning.",
    "Machine learning enables computers to learn patterns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing works with human language and text data.",
    "Computer vision allows machines to interpret images and videos.",
    "Artificial intelligence focuses on creating systems that perform intelligent tasks.",
    "Data preprocessing prepares raw data for machine learning models.",
    "Feature scaling helps machine learning algorithms work effectively with numerical features.",
    "Classification predicts categories or classes from input data.",
    "Regression predicts continuous numerical values.",
    "Clustering groups similar data points without predefined labels.",
    "Cross-validation helps evaluate machine learning models reliably.",
    "Precision measures how many predicted positive cases are actually positive.",
    "Recall measures how many actual positive cases were correctly identified.",
    "F1-score combines precision and recall into a single metric.",
    "Embeddings represent text or other data as numerical vectors.",
    "Vector databases store and search vector representations efficiently.",
    "Semantic search finds information based on meaning rather than exact keywords.",
    "Cosine similarity measures the similarity between two vectors.",
    "Retrieval-augmented generation combines document retrieval with language generation.",
]


# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR = os.path.join("outputs", "w5d3")


def create_output_directory():
    """Create the W5D3 output directory."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Create ChromaDB collection
# ============================================================

def create_collection():
    """Create ChromaDB client and collection."""

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="w5d3_documents",
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )

    return client, collection


# ============================================================
# Add 20 documents
# ============================================================

def add_documents(collection):
    """Add 20 documents with metadata."""

    ids = [
        f"doc_{i + 1}"
        for i in range(len(DOCUMENTS))
    ]

    metadatas = [
        {
            "category": "AI/ML",
            "document_number": i + 1
        }
        for i in range(len(DOCUMENTS))
    ]

    # Clear old data if the program is run again
    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    collection.add(
        ids=ids,
        documents=DOCUMENTS,
        metadatas=metadatas,
    )

    return collection.count()


# ============================================================
# Save collection output
# ============================================================

def save_collection_output(count):
    """Save collection information."""

    file_path = os.path.join(
        OUTPUT_DIR,
        "collection_output.txt"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "W5D3 — ChromaDB Collection Setup\n"
        )

        file.write(
            "================================\n\n"
        )

        file.write(
            "Collection name: w5d3_documents\n"
        )

        file.write(
            "Distance metric: cosine\n"
        )

        file.write(
            f"Number of documents: {count}\n"
        )

        file.write(
            "Embedding generation: ChromaDB\n"
        )

    print(
        f"Collection output saved: {file_path}"
    )


# ============================================================
# Similarity search
# ============================================================

def similarity_search(collection):
    """Perform cosine similarity semantic search."""

    query = (
        "How can computers understand "
        "human language?"
    )

    results = collection.query(
        query_texts=[query],
        n_results=3,
    )

    file_path = os.path.join(
        OUTPUT_DIR,
        "similarity_search.txt"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "W5D3 — Cosine Similarity Search\n"
        )

        file.write(
            "================================\n\n"
        )

        file.write(
            f"Query: {query}\n\n"
        )

        for i, document in enumerate(
            results["documents"][0],
            start=1
        ):

            doc_id = results["ids"][0][i - 1]

            distance = results[
                "distances"
            ][0][i - 1]

            file.write(
                f"Result {i}\n"
            )

            file.write(
                f"Document ID: {doc_id}\n"
            )

            file.write(
                f"Document: {document}\n"
            )

            file.write(
                f"Cosine distance: {distance:.4f}\n\n"
            )

    print(
        f"Similarity results saved: {file_path}"
    )

    return results


# ============================================================
# Metadata filtering
# ============================================================

def metadata_filter_search(collection):
    """Perform similarity search with metadata filtering."""

    query = "machine learning"

    results = collection.query(
        query_texts=[query],
        n_results=5,
        where={
            "category": "AI/ML"
        },
    )

    file_path = os.path.join(
        OUTPUT_DIR,
        "metadata_filtering.txt"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "W5D3 — Metadata Filtering\n"
        )

        file.write(
            "=========================\n\n"
        )

        file.write(
            f"Query: {query}\n"
        )

        file.write(
            'Filter: category == "AI/ML"\n\n'
        )

        for i, document in enumerate(
            results["documents"][0],
            start=1
        ):

            doc_id = results[
                "ids"
            ][0][i - 1]

            file.write(
                f"Result {i}\n"
            )

            file.write(
                f"Document ID: {doc_id}\n"
            )

            file.write(
                f"Document: {document}\n\n"
            )

    print(
        f"Metadata results saved: {file_path}"
    )

    return results


# ============================================================
# Main program
# ============================================================

def main():

    print("=" * 60)
    print("W5D3 — CHROMADB VECTOR STORE")
    print("=" * 60)

    # Create output folder
    create_output_directory()

    # Create ChromaDB collection
    _, collection = create_collection()

    print("\nCollection created successfully!")
    print(
        f"Collection name: {collection.name}"
    )

    # Add 20 documents
    count = add_documents(collection)

    print(
        f"Documents added: {count}"
    )

    # Save collection information
    save_collection_output(count)

    # Similarity search
    print(
        "\nPerforming similarity search..."
    )

    similarity_search(collection)

    # Metadata filtering
    print(
        "\nPerforming metadata filtering..."
    )

    metadata_filter_search(collection)

    # Final status
    print("\n" + "=" * 60)
    print(
        "W5D3 ChromaDB tasks completed successfully!"
    )
    print("=" * 60)

    print("\nOutput files:")
    print(
        "1. outputs/w5d3/collection_output.txt"
    )
    print(
        "2. outputs/w5d3/similarity_search.txt"
    )
    print(
        "3. outputs/w5d3/metadata_filtering.txt"
    )


if __name__ == "__main__":
    main()