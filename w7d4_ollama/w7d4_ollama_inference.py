
"""W7D4: Ollama local LLM inference with a custom system prompt."""

import requests

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a helpful AI/ML learning assistant. "
    "Answer clearly and concisely using beginner-friendly English."
)

TEST_PROMPTS = [
    "What is machine learning?",
    "What is the difference between AI and machine learning?",
    "Explain a Python function in simple words.",
    "Why use a local LLM?",
    "What is prompt engineering?",
]


def generate_response(prompt: str) -> str:
    """Send a prompt to the local Ollama chat API."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    response = requests.post(
        OLLAMA_API_URL,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def main() -> None:
    """Run all five test prompts and print the responses."""
    for number, prompt in enumerate(TEST_PROMPTS, start=1):
        print(f"\n{'=' * 60}")
        print(f"TEST {number}: {prompt}")
        print("-" * 60)

        try:
            print(generate_response(prompt))
        except requests.RequestException as error:
            print(f"Ollama API error: {error}")


if __name__ == "__main__":
    main()
