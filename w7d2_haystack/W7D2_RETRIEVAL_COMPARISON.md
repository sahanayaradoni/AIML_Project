# W7D2 BM25 vs Dense Retrieval Comparison

## Overview

This experiment compares BM25 keyword-based retrieval with Dense Retrieval using
sentence-transformers/all-MiniLM-L6-v2 on the same 5 PDF documents and 10 questions.

## Results

| Retrieval Method | Top-1 Correct | Top-1 Precision |
| ---------------- | ------------: | --------------: |
| BM25             |          9/10 |             90% |
| Dense Retrieval  |         10/10 |            100% |

## Question-Level Comparison

| Question | BM25 Top-1           | Dense Top-1          |
| -------- | -------------------- | -------------------- |
| 1        | deep_learning.pdf    | machine_learning.pdf |
| 2        | machine_learning.pdf | machine_learning.pdf |
| 3        | deep_learning.pdf    | deep_learning.pdf    |
| 4        | deep_learning.pdf    | deep_learning.pdf    |
| 5        | nlp.pdf              | nlp.pdf              |
| 6        | nlp.pdf              | nlp.pdf              |
| 7        | computer_vision.pdf  | computer_vision.pdf  |
| 8        | computer_vision.pdf  | computer_vision.pdf  |
| 9        | mlops.pdf            | mlops.pdf            |
| 10       | mlops.pdf            | mlops.pdf            |

## Key Observation

BM25 retrieved the expected document for 9 of 10 questions. The only mismatch
was Question 1, where `deep_learning.pdf` was ranked first and
`machine_learning.pdf` was ranked second.

Dense Retrieval retrieved the expected document for all 10 questions.

In this small manually evaluated dataset, Dense Retrieval achieved a higher
top-1 precision than BM25.

## Important Note

The comparison is based on 5 documents and 10 manually evaluated questions.
Therefore, the results demonstrate the behavior of these retrieval methods
for this specific experiment and should not be treated as a general benchmark.
