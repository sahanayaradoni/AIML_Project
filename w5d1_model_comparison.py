"""
W5D1: Compare Local LLMs with Ollama

Compares llama3.2:3b and qwen2.5:3b using the same questions.
"""

import ollama


MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b",
]

QUESTIONS = [
    "Explain overfitting in machine learning with a simple example.",
    "What is the difference between classification and regression?",
    "Why is data preprocessing important in machine learning?",
]


SYSTEM_PROMPT = """
You are an AI/ML learning assistant.
Explain concepts clearly for a beginner.
Be accurate, concise, and use simple practical examples.
"""


def get_response(model, question):
    """Get a response from a local Ollama model."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    return response["message"]["content"]


def main():
    """Compare both models on the same questions."""
    print("=" * 80)
    print("W5D1 - LLAMA 3.2 vs QWEN 2.5")
    print("=" * 80)

    for question_number, question in enumerate(QUESTIONS, start=1):
        print(f"\nQUESTION {question_number}")
        print("-" * 80)
        print(question)

        for model in MODELS:
            print(f"\n[{model}]")
            print("-" * 80)

            try:
                answer = get_response(model, question)
                print(answer)
            except Exception as error:
                print(f"Error: {error}")

        print("=" * 80)


if __name__ == "__main__":
    main()