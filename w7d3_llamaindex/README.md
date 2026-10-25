# W7D3: LlamaIndex Document Indexing and RAG

## Overview

This project demonstrates document indexing and Retrieval-Augmented Generation (RAG) using LlamaIndex, Ollama, and ChromaDB. It compares an in-memory vector index with a persistent ChromaDB-backed vector index.

## Technologies Used

- Python
- LlamaIndex
- Ollama
- `nomic-embed-text` for embeddings
- `llama3.2:3b` for answer generation
- ChromaDB for persistent vector storage
- Pytest for automated testing

## Project Structure

```text
w7d3_llamaindex/
├── documents/
│   ├── machine_learning.txt
│   ├── rag_and_embeddings.txt
│   ├── vector_databases.txt
│   ├── mlops.txt
│   └── supervised_learning.txt
├── outputs/
│   └── w7d3_query_results.json
├── w7d3_llamaindex_rag.py
├── test_w7d3_llamaindex.py
└── README.md
```

## Implementation

1. Load five TXT documents using `SimpleDirectoryReader`.
2. Configure Ollama embeddings and the local language model.
3. Create an in-memory `VectorStoreIndex`.
4. Create a persistent ChromaDB-backed vector index.
5. Run the same ten questions against both indexes.
6. Record answers, retrieved source filenames, expected facts, and query latency.
7. Save the results as JSON for evaluation.

## Running the Project

Activate the project's virtual environment and ensure Ollama is running with the required models installed.

From the repository root, run:

```bash
python w7d3_llamaindex/w7d3_llamaindex_rag.py
```

The script saves its evaluation results to:

`w7d3_llamaindex/outputs/w7d3_query_results.json`

## Automated Tests

Run the W7D3 tests from the repository root:

```bash
pytest -q w7d3_llamaindex/test_w7d3_llamaindex.py
```

**Latest test result:** 7 passed in 24.75 seconds.

## Evaluation Results

The successful run processed five documents and ten questions per index. All ten questions for each index had their expected facts present in the retrieved context.

| Metric                                      | In-memory index | ChromaDB index |
| ------------------------------------------- | --------------: | -------------: |
| Questions evaluated                         |              10 |             10 |
| Average query latency                       |  23.948 seconds | 12.067 seconds |
| Expected facts present in retrieved context |           10/10 |          10/10 |

These latency values are from one local run. Results may vary with model warm-up, hardware resources, and system load. The measurements include retrieval and LLM answer generation.

## Conclusion

This project demonstrates how LlamaIndex can load documents, build vector indexes, and answer questions using retrieved context. The recorded run showed lower average query latency with ChromaDB, while both indexes retrieved context containing the expected facts for all ten questions.

## Future Improvements

- Evaluate answer correctness with a more comprehensive evaluation dataset.
- Measure indexing time and retrieval latency separately from LLM generation.
- Test larger document collections and different embedding models.
- Explore RAG evaluation with Ragas.
