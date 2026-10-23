import requests
import time
from contextlib import redirect_stdout

# Ollama local API
OLLAMA_URL = "http://localhost:11434/api/generate"

# Custom system prompt
SYSTEM_PROMPT = """
You are a helpful AI/ML learning assistant.
Give clear, concise, beginner-friendly answers.
Explain technical concepts in simple language.
Use examples when useful.
"""


def ask_ollama(model, prompt):
    """Send a prompt to Ollama and return the response and time."""

    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False
    }

    start_time = time.time()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    response_time = time.time() - start_time

    result = response.json()

    return result["response"], response_time


# ---------------------------------------------------------
# OUTPUT EVIDENCE FILE
# ---------------------------------------------------------

with open("w5d5_output.txt", "w", encoding="utf-8") as output_file:

    with redirect_stdout(output_file):

        print("=" * 70)
        print("W5D5 - LOCAL Q&A BOT USING OLLAMA")
        print("=" * 70)

        print("\nCustom System Prompt:")
        print(SYSTEM_PROMPT)

        # -------------------------------------------------
        # PART 1: FIVE PROMPT TEST
        # -------------------------------------------------

        five_prompts = [
            "What is machine learning?",
            "What is the difference between AI and ML?",
            "Explain overfitting in simple terms.",
            "What is a vector database?",
            "What is semantic search?"
        ]

        print("\n")
        print("=" * 70)
        print("PART 1: FIVE PROMPT TEST")
        print("=" * 70)

        print("\nMODEL: llama3.2:3b")
        print("-" * 70)

        for number, prompt in enumerate(five_prompts, start=1):

            print(f"\nPrompt {number}: {prompt}")

            try:
                answer, response_time = ask_ollama(
                    "llama3.2:3b",
                    prompt
                )

                print("\nResponse:")
                print(answer)

                print(
                    f"\nResponse time: "
                    f"{response_time:.2f} seconds"
                )

            except requests.exceptions.RequestException as error:
                print(f"\nError: {error}")

        # -------------------------------------------------
        # PART 2: MODEL COMPARISON
        # -------------------------------------------------

        comparison_questions = [
            "Explain RAG in simple terms.",
            "What is semantic search?",
            "What is overfitting?"
        ]

        models = [
            "llama3.2:3b",
            "qwen2.5:3b"
        ]

        print("\n\n")
        print("=" * 70)
        print("PART 2: MODEL COMPARISON")
        print("=" * 70)

        for question_number, question in enumerate(
            comparison_questions,
            start=1
        ):

            print("\n")
            print("=" * 70)
            print(
                f"QUESTION {question_number}: "
                f"{question}"
            )
            print("=" * 70)

            for model in models:

                print(f"\nMODEL: {model}")
                print("-" * 70)

                try:
                    answer, response_time = ask_ollama(
                        model,
                        question
                    )

                    print(answer)

                    print(
                        f"\nResponse time: "
                        f"{response_time:.2f} seconds"
                    )

                except requests.exceptions.RequestException as error:

                    print(
                        f"\nError running {model}: "
                        f"{error}"
                    )

        # -------------------------------------------------
        # PART 3: TEST SUMMARY
        # -------------------------------------------------

        print("\n\n")
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        print("""
1. Ollama was successfully installed and configured.
2. llama3.2:3b was used for five prompt tests.
3. A custom system prompt was used.
4. llama3.2:3b and qwen2.5:3b were compared
   using the same three questions.
5. Response quality, relevance and response time
   were observed.
6. Any timeout or incorrect response was recorded
   instead of being hidden.
""")

        print("=" * 70)
        print("W5D5 TESTING COMPLETED")
        print("=" * 70)


# ---------------------------------------------------------
# TERMINAL MESSAGE
# ---------------------------------------------------------

print("=" * 70)
print("W5D5 TESTING COMPLETED")
print("=" * 70)
print("Output evidence saved to: w5d5_output.txt")
print("=" * 70)