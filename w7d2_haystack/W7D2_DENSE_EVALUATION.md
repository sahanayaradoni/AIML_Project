# W7D2 Dense Retrieval Evaluation

## Retrieval Setup

- Documents indexed: 5 PDF documents
- Retrieval method: Dense Retrieval
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Questions tested: 10
- Top-k results: 3 documents per question

## Manual Evaluation

| Question | Expected Document    | Top-1 Retrieved      | Relevant? |
| -------- | -------------------- | -------------------- | --------- |
| 1        | machine_learning.pdf | machine_learning.pdf | Yes       |
| 2        | machine_learning.pdf | machine_learning.pdf | Yes       |
| 3        | deep_learning.pdf    | deep_learning.pdf    | Yes       |
| 4        | deep_learning.pdf    | deep_learning.pdf    | Yes       |
| 5        | nlp.pdf              | nlp.pdf              | Yes       |
| 6        | nlp.pdf              | nlp.pdf              | Yes       |
| 7        | computer_vision.pdf  | computer_vision.pdf  | Yes       |
| 8        | computer_vision.pdf  | computer_vision.pdf  | Yes       |
| 9        | mlops.pdf            | mlops.pdf            | Yes       |
| 10       | mlops.pdf            | mlops.pdf            | Yes       |

## Result

Dense Retrieval returned the expected document as the top-1 result for **10 out of 10 questions**.

Manual top-1 retrieval precision:

**100% (10/10)**

## Observation

Dense Retrieval correctly identified the expected document for all ten questions. The semantic embeddings helped match queries with documents even when the exact wording differed.
