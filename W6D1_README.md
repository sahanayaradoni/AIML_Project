# W6D1: LangChain Fundamentals — Chains & Prompts

## Objective

Build and test LangChain chains, conversation memory, and a simple agent using Ollama.

## Environment

- Python 3.13
- LangChain 1.4.0
- LangChain Ollama 1.1.0
- Ollama
- Model: `llama3.2:3b`

## Task 1: LangChain Chain

Implemented the following pipeline:

PromptTemplate → Ollama LLM → StrOutputParser

The chain was tested with 5 inputs:

1. What is Python?
2. What is machine learning?
3. What is LangChain?
4. What is an API?
5. What is a database?

The chain successfully generated responses for all 5 inputs.

## Task 2: Conversation Memory

Implemented conversation history that maintains previous user and assistant messages.

Five conversation turns were tested, including questions that required information from earlier turns.

The conversation history was successfully maintained across all 5 turns.

## Task 3: LangChain Agent

Implemented a simple LangChain agent using two tools:

1. `web_search_stub` — simulated web search tool
2. `calculator` — basic arithmetic calculator

The agent was tested with 3 tasks:

1. Calculate `25 * 4 + 10`
2. Search for information about LangChain
3. Calculate `100 / 4` and explain LangChain

The agent successfully completed the three tasks.

## Testing

Automated tests were created in:

`test_w6d1_chain.py`

The tests verify:

- Chain creation
- Chain response generation
- Memory creation
- Conversation history storage
- Memory chain creation

## Files

- `w6d1_chain.py`
- `test_w6d1_chain.py`
- `w6d1_agent.py`
- `W6D1_README.md`

## Git Workflow

Branch:

`feat/aiml-W6-sahana`

CIA Full Stack Mentor Mode review was completed before committing.

## Viva Preparation

### 1. What is a LangChain chain? How does it differ from a single LLM call?

A LangChain chain connects multiple components together in a defined workflow. For example, a prompt can be passed to an LLM and then the result can be processed by an output parser.

A single LLM call directly sends a request to the model and receives a response, while a chain can combine multiple processing steps.

### 2. What problem does LangChain Memory solve?

LangChain Memory allows an application to retain relevant information from previous interactions. This makes it possible to maintain context during a conversation instead of treating every user message as an isolated request.

### 3. What is the ReAct pattern in LangChain agents?

ReAct stands for Reasoning and Acting. An agent reasons about the user's request, decides which tool or action is needed, executes that action, observes the result, and then continues until it can provide a final answer.

# W6D4: RAG Pipeline — LangChain + ChromaDB

## Objective

Implement a Retrieval-Augmented Generation (RAG) pipeline using ChromaDB, LangChain, Ollama embeddings, and the Ollama LLM.

## Technologies Used

- Python 3.13
- ChromaDB
- LangChain
- LangChain Chroma
- Ollama
- nomic-embed-text
- llama3.2:3b
- pypdf
- ReportLab

## Practical Tasks Completed

### 1. ChromaDB Vector Store

- Created a persistent ChromaDB collection named `w6d4_documents`.
- Configured cosine similarity.
- Added 20 documents.
- Generated embeddings using `nomic-embed-text`.

### 2. Similarity Search

- Performed semantic similarity search using cosine similarity.
- Retrieved relevant documents based on query meaning.
- Applied metadata filtering using document topics.
- Manually verified the retrieval results.

### 3. PDF RAG Pipeline

- Created and processed a sample PDF.
- Extracted PDF text using `pypdf`.
- Split the PDF content into chunks.
- Generated embeddings for the PDF chunks.
- Stored the chunks in ChromaDB.
- Retrieved the top 3 relevant chunks.
- Passed the retrieved context to `llama3.2:3b`.
- Generated the final RAG answer.

## Embedding Model

`nomic-embed-text` was used specifically for text embeddings.

The LLM used for answer generation was:

`llama3.2:3b`

## RAG Flow

```text
Documents
    ↓
Text / PDF Extraction
    ↓
Document Chunks
    ↓
nomic-embed-text
    ↓
Vector Embeddings
    ↓
ChromaDB
    ↓
Cosine Similarity Search
    ↓
Top-3 Relevant Chunks
    ↓
Context
    ↓
llama3.2:3b
    ↓
Final Answer
