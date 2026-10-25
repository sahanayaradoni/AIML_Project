
"""W7D4: Compare Llama 3.2 and Qwen 2.5 using identical prompts."""

import requests

OLLAMA_API_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "You are a helpful AI/ML learning assistant. "
    "Answer clearly and concisely using beginner-friendly English."
)

MODELS = ["llama3.2:3b", "qwen2.5:3b"]

COMPARISON_QUESTIONS = [
    "Explain machine learning in simple words.",
    "What is the difference between AI and machine learning?",
    "Why might someone choose a local LLM?",
]


def generate_response(model: str, prompt: str) -> str:
    """Generate a response from the selected local Ollama model."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    response = requests.post(
        OLLAMA_API_URL,
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def main() -> None:
    """Compare both models on the same three questions."""
    for number, question in enumerate(COMPARISON_QUESTIONS, start=1):
        print(f"\n{'=' * 70}")
        print(f"QUESTION {number}: {question}")

        for model in MODELS:
            print(f"\n--- {model} ---")
            try:
                print(generate_response(model, question))
            except requests.RequestException as error:
                print(f"API error for {model}: {error}")


if __name__ == "__main__":
    main()
