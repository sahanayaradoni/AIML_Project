# W7D1 BM25 Retrieval Evaluation

## Overview

BM25 retrieval was evaluated using 10 questions against 5 indexed PDF documents.

## Manual Evaluation

| Question                             | Expected Document    | BM25 Rank | Result             |
| ------------------------------------ | -------------------- | --------: | ------------------ |
| What is machine learning?            | machine_learning.pdf |         1 | Relevant           |
| What is supervised learning?         | machine_learning.pdf |         1 | Relevant           |
| What is deep learning?               | deep_learning.pdf    |         1 | Relevant           |
| What is a neural network?            | deep_learning.pdf    |         1 | Relevant           |
| What is natural language processing? | nlp.pdf              |         2 | Partially relevant |
| What are common NLP tasks?           | nlp.pdf              |         1 | Relevant           |
| What is computer vision?             | computer_vision.pdf  |         1 | Relevant           |
| What is object detection?            | computer_vision.pdf  |         1 | Relevant           |
| What is MLOps?                       | mlops.pdf            |         1 | Relevant           |
| What is model monitoring?            | mlops.pdf            |         1 | Relevant           |

## Summary

- Total questions: 10
- Directly relevant document ranked #1: 9
- Relevant document not ranked #1: 1
- BM25 top-1 retrieval rate: 90%
- Main observation: BM25 generally retrieved the expected topic document at rank 1.
- Limitation observed: For the NLP question, `deep_learning.pdf` ranked slightly above `nlp.pdf`, showing that keyword-based BM25 retrieval can sometimes favor overlapping terminology.

## Conclusion

BM25 provided strong keyword-based retrieval performance on the evaluation questions. However, the results also show that lexical matching can struggle when documents contain related terminology. Dense retrieval will be evaluated on the same 10 questions for comparison.
