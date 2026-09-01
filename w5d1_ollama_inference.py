"""
W5D1: Running LLMs Locally with Ollama

Demonstrates local LLM inference using the Ollama Python API.
"""

import ollama


MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are a helpful AI/ML learning assistant.
Explain concepts clearly and concisely for a beginner.
Use simple language and practical examples when useful.
"""


TEST_PROMPTS = [
    "What is machine learning?",
    "Explain overfitting in simple terms.",
    "What is the difference between classification and regression?",
    "Why is data preprocessing important in machine learning?",
    "What is a neural network?",
]


def run_inference(prompt):
    """Send a prompt to the local Ollama model."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"]


def main():
    """Run all test prompts and display responses."""
    print("=" * 70)
    print("W5D1 - Local LLM Inference with Ollama")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print("System prompt: AI/ML learning assistant for beginners")
    print()

    for index, prompt in enumerate(TEST_PROMPTS, start=1):
        print(f"QUESTION {index}")
        print("-" * 70)
        print(prompt)
        print()
        print("RESPONSE")
        print(run_inference(prompt))
        print()
        print("=" * 70)


if __name__ == "__main__":
    main() 