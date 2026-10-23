"""
W5D4: Semantic Search with ChromaDB

Tasks:
1. Create a ChromaDB collection.
2. Add 20 documents with embeddings.
3. Perform cosine similarity search.
4. Perform metadata filtering.
5. Save output evidence to files.

Author: Sahana
"""

import os
import chromadb


# ============================================================
# 1. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "w5d4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "w5d4_semantic_search_results.txt"
)

INFO_FILE = os.path.join(
    OUTPUT_DIR,
    "w5d4_collection_info.txt"
)


# ============================================================
# 2. CREATE CHROMADB CLIENT
# ============================================================

print("=" * 60)
print("W5D4 - SEMANTIC SEARCH WITH CHROMADB")
print("=" * 60)

client = chromadb.PersistentClient(
    path="./w5d4_chroma_db"
)

print("\nChromaDB client created successfully.")


# ============================================================
# 3. CREATE COLLECTION
# ============================================================

collection = client.get_or_create_collection(
    name="w5d4_documents",
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)

print("Collection created successfully.")
print("Collection name:", collection.name)


# ============================================================
# 4. CREATE 20 DOCUMENTS
# ============================================================

documents = [
    "Python is a popular programming language used for artificial intelligence.",
    "Machine learning allows computers to learn patterns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables machines to understand images and videos.",
    "Data preprocessing improves the quality of machine learning datasets.",
    "Feature scaling is important for many machine learning algorithms.",
    "Classification predicts categories or labels from input data.",
    "Regression predicts continuous numerical values.",
    "Clustering groups similar data points together.",
    "Semantic search finds documents based on meaning rather than exact keywords.",
    "Embeddings represent text as numerical vectors.",
    "Vector databases store and search high-dimensional embeddings.",
    "ChromaDB is a vector database useful for semantic search applications.",
    "Ollama allows users to run large language models locally.",
    "Large language models can generate and understand natural language.",
    "Prompt engineering improves the quality of responses from language models.",
    "Retrieval augmented generation combines document retrieval with an LLM.",
    "Cosine similarity measures the similarity between two vectors.",
    "Artificial intelligence is used in applications such as recommendation systems."
]


# ============================================================
# 5. CREATE METADATA
# ============================================================

metadatas = [
    {"topic": "python"},
    {"topic": "machine_learning"},
    {"topic": "deep_learning"},
    {"topic": "nlp"},
    {"topic": "computer_vision"},
    {"topic": "preprocessing"},
    {"topic": "preprocessing"},
    {"topic": "classification"},
    {"topic": "regression"},
    {"topic": "clustering"},
    {"topic": "semantic_search"},
    {"topic": "embeddings"},
    {"topic": "vector_database"},
    {"topic": "chromadb"},
    {"topic": "ollama"},
    {"topic": "llm"},
    {"topic": "prompt_engineering"},
    {"topic": "rag"},
    {"topic": "similarity"},
    {"topic": "artificial_intelligence"}
]


# ============================================================
# 6. CREATE DOCUMENT IDS
# ============================================================

ids = [f"w5d4_doc_{i}" for i in range(1, 21)]


# ============================================================
# 7. ADD DOCUMENTS TO CHROMADB
# ============================================================

collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

document_count = collection.count()

print("\nDocuments added successfully.")
print("Number of documents:", document_count)


# ============================================================
# 8. SEMANTIC SIMILARITY SEARCH
# ============================================================

query = "How can computers understand the meaning of human language?"

results = collection.query(
    query_texts=[query],
    n_results=3
)

print("\n" + "=" * 60)
print("COSINE SIMILARITY SEARCH")
print("=" * 60)

print("\nQuery:")
print(query)

print("\nTop 3 Similar Documents:")

similar_documents = results["documents"][0]
similar_ids = results["ids"][0]
similar_distances = results["distances"][0]
similar_metadatas = results["metadatas"][0]

for i in range(len(similar_documents)):
    print(f"\nRank {i + 1}")
    print("ID:", similar_ids[i])
    print("Document:", similar_documents[i])
    print("Topic:", similar_metadatas[i]["topic"])
    print("Cosine distance:", round(similar_distances[i], 4))


# ============================================================
# 9. METADATA FILTERING
# ============================================================

filter_query = "machine learning"

filtered_results = collection.query(
    query_texts=[filter_query],
    n_results=3,
    where={"topic": "machine_learning"}
)

print("\n" + "=" * 60)
print("METADATA FILTERING")
print("=" * 60)

print("\nQuery:")
print(filter_query)

print("\nFilter:")
print('topic = "machine_learning"')

print("\nFiltered Results:")

filtered_documents = filtered_results["documents"][0]
filtered_ids = filtered_results["ids"][0]
filtered_distances = filtered_results["distances"][0]

for i in range(len(filtered_documents)):
    print(f"\nRank {i + 1}")
    print("ID:", filtered_ids[i])
    print("Document:", filtered_documents[i])
    print("Cosine distance:", round(filtered_distances[i], 4))


# ============================================================
# 10. SAVE SEARCH RESULTS TO TXT FILE
# ============================================================

with open(RESULT_FILE, "w", encoding="utf-8") as file:

    file.write("=" * 60 + "\n")
    file.write("W5D4 - CHROMADB SEMANTIC SEARCH RESULTS\n")
    file.write("=" * 60 + "\n\n")

    file.write("Similarity Search\n")
    file.write("-" * 60 + "\n")
    file.write(f"Query: {query}\n\n")

    for i in range(len(similar_documents)):
        file.write(f"Rank {i + 1}\n")
        file.write(f"ID: {similar_ids[i]}\n")
        file.write(f"Document: {similar_documents[i]}\n")
        file.write(
            f"Topic: {similar_metadatas[i]['topic']}\n"
        )
        file.write(
            f"Cosine distance: {similar_distances[i]:.4f}\n\n"
        )

    file.write("\n")
    file.write("Metadata Filtering\n")
    file.write("-" * 60 + "\n")
    file.write(f"Query: {filter_query}\n")
    file.write('Filter: topic = "machine_learning"\n\n')

    for i in range(len(filtered_documents)):
        file.write(f"Rank {i + 1}\n")
        file.write(f"ID: {filtered_ids[i]}\n")
        file.write(f"Document: {filtered_documents[i]}\n")
        file.write(
            f"Cosine distance: {filtered_distances[i]:.4f}\n\n"
        )

print("\nSearch results saved to:")
print(RESULT_FILE)


# ============================================================
# 11. SAVE COLLECTION INFORMATION
# ============================================================

with open(INFO_FILE, "w", encoding="utf-8") as file:

    file.write("W5D4 - CHROMADB COLLECTION INFORMATION\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Collection name: {collection.name}\n")
    file.write(f"Number of documents: {document_count}\n")
    file.write("Distance metric: cosine\n")
    file.write("Database type: ChromaDB\n")
    file.write("Persistent database path: ./w5d4_chroma_db\n")

print("Collection information saved to:")
print(INFO_FILE)


# ============================================================
# 12. FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("W5D4 CHROMADB TASK COMPLETED")
print("=" * 60)

print("\nDeliverables generated:")
print("- 20 documents added")
print("- Cosine similarity search completed")
print("- Top-3 results retrieved")
print("- Metadata filtering completed")
print("- Search results saved")
print("- Collection information saved")

print("\nOutput directory:", OUTPUT_DIR)