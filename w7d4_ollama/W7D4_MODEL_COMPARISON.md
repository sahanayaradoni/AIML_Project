# W7D4 — Ollama Model Comparison

## Objective

Compare `llama3.2:3b` and `qwen2.5:3b` using the same system prompt and three identical questions.

## Setup

- **Models:** `llama3.2:3b` and `qwen2.5:3b`
- **Runtime:** Ollama local API
- **API endpoint:** `/api/chat`
- **System prompt:** Helpful AI/ML learning assistant; clear, concise, beginner-friendly English
- **Temperature:** 0
- **Questions tested:** 3

## Results

### Question 1: Explain machine learning in simple words.

- **Llama 3.2:** Gave a detailed explanation, an example about recognizing animals, and descriptions of supervised, unsupervised, and reinforcement learning.
- **Qwen 2.5:** Gave a shorter explanation with an everyday analogy about recognizing a friend's face.
- **Observation:** Both answers were relevant. Llama provided more detail, while Qwen was more concise.

### Question 2: What is the difference between AI and machine learning?

- **Llama 3.2:** Explained AI as the broader field, ML as a subset, and included examples of AI applications.
- **Qwen 2.5:** Explained the relationship in simpler, shorter language and emphasized that ML learns patterns from data.
- **Observation:** Both answers correctly explained the main distinction. Llama was more detailed, while Qwen was more concise.

### Question 3: Why might someone choose a local LLM?

- **Llama 3.2:** Discussed privacy, reduced network dependence, latency, and local processing.
- **Qwen 2.5:** Misinterpreted “LLM” as a Master of Laws degree and answered about local law programs.
- **Observation:** Llama answered the intended AI/ML question. Qwen's response was irrelevant because it misunderstood the context.

## Conclusion

Both models produced useful answers for the first two questions. Llama 3.2 gave more detailed responses, while Qwen 2.5 was generally more concise. On the third question, Llama interpreted the technical context correctly, but Qwen misunderstood the abbreviation “LLM.” This small comparison shows that response quality depends on context interpretation as well as clarity and detail.

These observations are based on the three questions tested; they do not establish that one model is universally better.

## Evidence

The terminal output from `w7d4_model_comparison.py` records the responses from both models for all three questions.
