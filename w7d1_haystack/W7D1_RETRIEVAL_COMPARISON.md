# W7D1 BM25 vs Dense Retrieval Comparison

## Evaluation Setup

- PDF documents: 5
- Questions: 10
- Retrieval results per question: Top 3
- BM25 retriever: `InMemoryBM25Retriever`
- Dense retriever: `InMemoryEmbeddingRetriever`
- Dense embedding model: `sentence-transformers/all-MiniLM-L6-v2`

## Top-1 Retrieval Comparison

| Retriever       | Correct Top-1 Results | Top-1 Retrieval Rate |
| --------------- | --------------------: | -------------------: |
| BM25            |                  9/10 |                  90% |
| Dense Retrieval |                 10/10 |                 100% |

## Question-Level Comparison

| #   | Question                             | BM25 Top-1           | Dense Top-1          |
| --- | ------------------------------------ | -------------------- | -------------------- |
| 1   | What is machine learning?            | machine_learning.pdf | machine_learning.pdf |
| 2   | What is supervised learning?         | machine_learning.pdf | machine_learning.pdf |
| 3   | What is deep learning?               | deep_learning.pdf    | deep_learning.pdf    |
| 4   | What is a neural network?            | deep_learning.pdf    | deep_learning.pdf    |
| 5   | What is natural language processing? | deep_learning.pdf    | nlp.pdf              |
| 6   | What are common NLP tasks?           | nlp.pdf              | nlp.pdf              |
| 7   | What is computer vision?             | computer_vision.pdf  | computer_vision.pdf  |
| 8   | What is object detection?            | computer_vision.pdf  | computer_vision.pdf  |
| 9   | What is MLOps?                       | mlops.pdf            | mlops.pdf            |
| 10  | What is model monitoring?            | mlops.pdf            | mlops.pdf            |

## Observations

### BM25

BM25 performed well on keyword-based queries and correctly ranked the expected document first for 9 out of 10 questions.

For the natural language processing question, `deep_learning.pdf` was ranked first while `nlp.pdf` was ranked second. This shows that lexical overlap can affect keyword-based retrieval when related documents contain similar terminology.

### Dense Retrieval

Dense retrieval ranked the expected document first for all 10 questions.

The NLP-related questions were correctly mapped to `nlp.pdf`, indicating that semantic embeddings helped distinguish the topic from documents containing related terminology.

## Overall Conclusion

On this 10-question manual evaluation set, dense retrieval achieved a higher observed top-1 retrieval rate than BM25.

BM25 remains useful for direct keyword matching, while dense retrieval provides semantic matching based on document and query embeddings.

The evaluation is limited to 5 small documents and 10 manually selected questions, so these results should not be considered a general benchmark.
