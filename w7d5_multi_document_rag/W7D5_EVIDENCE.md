# W7D5 — Execution Evidence

## Objective

Demonstrate the multi-document Retrieval-Augmented Generation (RAG) system using LlamaIndex, Ollama, and ChromaDB.

## Automated Test Evidence

Command:

```bash
pytest -q w7d5_multi_document_rag/test_w7d5_multi_document_rag.py
```

Recorded result:

```text
6 passed in 12.86s
```

The tests cover:

- Loading all five PDF documents
- Checking extracted text and source filenames
- Verifying the expected PDF filenames
- Handling a missing directory
- Handling an empty directory
- Testing the query function with a mocked query engine

## Application Execution

The application was run using:

```bash
python w7d5_multi_document_rag\w7d5_multi_document_rag.py
```

The successful run indexed five PDF documents in the `w7d5_documents` ChromaDB collection and answered five example questions about machine learning, deep learning, natural language processing, computer vision, and MLOps.

## Models

- Embedding model: `nomic-embed-text:latest`
- Language model: `qwen2.5:3b`

## Storage

ChromaDB data is stored locally in `chroma_db/` and excluded from Git using `.gitignore`.

## Scope Note

This implementation uses LlamaIndex, Ollama, and ChromaDB. CrewAI and Ragas are not integrated into this W7D5 implementation.
