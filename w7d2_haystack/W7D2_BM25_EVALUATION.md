# W7D2 BM25 Retrieval Evaluation

## Overview

This experiment evaluates BM25 keyword-based retrieval using five PDF documents and ten questions related to machine learning, deep learning, NLP, computer vision, and MLOps.

## Dataset

- PDF documents: 5
- Evaluation questions: 10
- Retrieval method: BM25
- Results per question: Top 3 documents

## Results

| Question                                | Expected Top-1 Document | Actual Top-1 Document | Result    |
| --------------------------------------- | ----------------------- | --------------------- | --------- |
| 1. What is machine learning?            | machine_learning.pdf    | deep_learning.pdf     | Incorrect |
| 2. What is supervised learning?         | machine_learning.pdf    | machine_learning.pdf  | Correct   |
| 3. What is deep learning?               | deep_learning.pdf       | deep_learning.pdf     | Correct   |
| 4. What is a neural network?            | deep_learning.pdf       | deep_learning.pdf     | Correct   |
| 5. What is natural language processing? | nlp.pdf                 | nlp.pdf               | Correct   |
| 6. What are common NLP tasks?           | nlp.pdf                 | nlp.pdf               | Correct   |
| 7. What is computer vision?             | computer_vision.pdf     | computer_vision.pdf   | Correct   |
| 8. What is object detection?            | computer_vision.pdf     | computer_vision.pdf   | Correct   |
| 9. What is MLOps?                       | mlops.pdf               | mlops.pdf             | Correct   |
| 10. What is model monitoring?           | mlops.pdf               | mlops.pdf             | Correct   |

## Evaluation Summary

- Correct top-1 results: 9 out of 10
- Top-1 accuracy: 90%
- Incorrect top-1 result: Question 1
- Observation: BM25 ranked deep_learning.pdf first for the machine learning question, while machine_learning.pdf appeared second.

## Conclusion

BM25 successfully retrieved the expected document for 9 of the 10 evaluation questions. This small experiment suggests that keyword-based retrieval performs well for these documents, although a semantically related document can sometimes rank higher than the intended document.

The results are specific to this dataset and should not be treated as a general benchmark.
