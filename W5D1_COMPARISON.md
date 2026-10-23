# W5D1: Local LLM Comparison

## Objective

Compare the response quality of `llama3.2:3b` and `qwen2.5:3b` using the same questions and a common system prompt through Ollama.

## Models

- `llama3.2:3b` — 2.0 GB
- `qwen2.5:3b` — 1.9 GB

Both models were downloaded and verified successfully using Ollama.

## Common System Prompt

> You are an AI/ML learning assistant. Explain concepts clearly for a beginner. Be accurate, concise, and use simple practical examples.

## Questions Tested

1. Explain overfitting in machine learning with a simple example.
2. What is the difference between classification and regression?
3. Why is data preprocessing important in machine learning?

## Comparison

| Question                     | Llama 3.2:3b                                                                           | Qwen 2.5:3b                                                              |
| ---------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Overfitting                  | More detailed and structured explanation with causes and prevention techniques.        | Simpler cookie-based analogy and concise explanation of generalization.  |
| Classification vs Regression | More detailed, including definitions, examples, key differences, and when to use each. | Shorter and direct explanation with clear examples.                      |
| Data preprocessing           | More technically structured coverage of preprocessing techniques.                      | More analogy-oriented explanation covering several important techniques. |

## Overall Observation

For these three test questions, `llama3.2:3b` produced more detailed and structured responses. `qwen2.5:3b` generally produced shorter, more direct responses and used intuitive analogies.

Both models successfully answered all three questions. The comparison is based only on these three test questions and is not intended to represent a general benchmark of the models.

## Result

Both local LLMs were successfully compared using the Ollama Python API. The experiment demonstrated that different local models can produce different response styles even when given the same system prompt and questions.
