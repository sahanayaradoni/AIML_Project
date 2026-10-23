# W7D1 Dense Retrieval Evaluation

## Overview

Dense retrieval was evaluated using the same 10 questions and 5 indexed PDF documents used for the BM25 evaluation.

The embedding model used was:

`sentence-transformers/all-MiniLM-L6-v2`

## Manual Evaluation

| Question                             | Expected Document    | Dense Rank | Result   |
| ------------------------------------ | -------------------- | ---------: | -------- |
| What is machine learning?            | machine_learning.pdf |          1 | Relevant |
| What is supervised learning?         | machine_learning.pdf |          1 | Relevant |
| What is deep learning?               | deep_learning.pdf    |          1 | Relevant |
| What is a neural network?            | deep_learning.pdf    |          1 | Relevant |
| What is natural language processing? | nlp.pdf              |          1 | Relevant |
| What are common NLP tasks?           | nlp.pdf              |          1 | Relevant |
| What is computer vision?             | computer_vision.pdf  |          1 | Relevant |
| What is object detection?            | computer_vision.pdf  |          1 | Relevant |
| What is MLOps?                       | mlops.pdf            |          1 | Relevant |
| What is model monitoring?            | mlops.pdf            |          1 | Relevant |

## Summary

- Total questions: 10
- Directly relevant document ranked #1: 10
- Relevant document not ranked #1: 0
- Dense top-1 retrieval rate: 100%
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

## Conclusion

Dense retrieval correctly ranked the expected topic document first for all 10 evaluation questions.

Compared with the BM25 evaluation, dense retrieval improved the observed top-1 retrieval rate from 90% to 100% on this manually constructed evaluation set.

The results suggest that semantic embeddings handled the topic-based questions effectively, including the NLP questions where BM25 showed some keyword overlap between related documents.

This is a small manual evaluation set, so the results should not be treated as a general benchmark.
