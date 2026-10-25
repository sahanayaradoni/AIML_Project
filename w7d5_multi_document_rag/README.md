# W7D5 — Multi-Document RAG System

## Objective

Build a Retrieval-Augmented Generation (RAG) system that searches multiple PDF documents and generates answers to questions using LlamaIndex, Ollama, and ChromaDB.

## Technologies

- **Python** — application and testing
- **LlamaIndex** — document indexing and query engine
- **Ollama** — local embedding and language models
- **ChromaDB** — persistent vector storage
- **PyPDF** — PDF text extraction
- **pytest** — automated testing

## Project Structure

```text
w7d5_multi_document_rag/
├── documents/
│   ├── computer_vision.pdf
│   ├── deep_learning.pdf
│   ├── machine_learning.pdf
│   ├── mlops.pdf
│   └── nlp.pdf
├── w7d5_multi_document_rag.py
├── test_w7d5_multi_document_rag.py
└── README.md
```

The `chroma_db/` directory is generated locally when the application runs and should not be committed.

## How It Works

1. Extract text from the PDF files.
2. Attach each PDF filename as document metadata.
3. Generate embeddings using `nomic-embed-text:latest`.
4. Store document vectors in ChromaDB.
5. Retrieve the three most relevant chunks for each question.
6. Generate answers using `qwen2.5:3b` through Ollama.

## Prerequisites

- Python and the project virtual environment
- Ollama installed and running
- Required Python packages installed
- The following Ollama models available:
  - `nomic-embed-text:latest`
  - `qwen2.5:3b`

## Run the Application

From the repository root:

```bash
python w7d5_multi_document_rag\w7d5_multi_document_rag.py
```

The application indexes the PDFs in the W7D5 ChromaDB collection and runs five example questions covering machine learning, deep learning, natural language processing, computer vision, and MLOps.

## Run the Tests

From the repository root:

```bash
pytest -q w7d5_multi_document_rag/test_w7d5_multi_document_rag.py
```

### Current Test Result

**6 tests passed** in 10.65 seconds.

The tests cover PDF loading, expected filenames, extracted text, missing and empty directories, and query-engine response handling.

## Limitations

- Answer quality depends on the extracted PDF text, retrieved context, and local language model.
- The current demonstration uses five example questions.
- CrewAI and Ragas are not integrated into this implementation.
