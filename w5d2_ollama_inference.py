import ollama
from pathlib import Path


# ============================================================
# W5D2 - Prompt Engineering & System Prompts with Ollama
# ============================================================

# Create output folder
OUTPUT_DIR = Path("w5d2_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Custom System Prompt
# ------------------------------------------------------------

SYSTEM_PROMPT = """
You are a helpful AI/ML tutor.
Explain concepts clearly and simply.
Use short examples when useful.
Avoid unnecessary technical jargon.
If the question asks for steps, provide them as a numbered list.
"""


# ------------------------------------------------------------
# Ollama API Function
# ------------------------------------------------------------

def generate_response(model, prompt):
    """Generate a response using the Ollama API."""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ------------------------------------------------------------
# Part 1: Test 5 Prompts with llama3.2:3b
# ------------------------------------------------------------

test_prompts = [
    "What is machine learning?",
    "Explain supervised learning with a simple example.",
    "What is the difference between AI and machine learning?",
    "What is overfitting in machine learning?",
    "Explain the purpose of a train-test split."
]


llama_results = []

llama_results.append("=" * 70)
llama_results.append("W5D2 - LLAMA 3.2:3B PROMPT TESTING")
llama_results.append("=" * 70)
llama_results.append(f"\nSYSTEM PROMPT:\n{SYSTEM_PROMPT.strip()}\n")


for i, prompt in enumerate(test_prompts, start=1):

    print(f"\nPrompt {i}: {prompt}")
    print("-" * 70)

    try:
        answer = generate_response("llama3.2:3b", prompt)

        print(answer)

        llama_results.append(f"\nPrompt {i}: {prompt}")
        llama_results.append("-" * 70)
        llama_results.append(answer)

    except Exception as e:
        error_message = f"Error: {e}"
        print(error_message)
        llama_results.append(error_message)


# Save llama results
llama_file = OUTPUT_DIR / "llama3.2_results.txt"

llama_file.write_text(
    "\n".join(llama_results),
    encoding="utf-8"
)


# ------------------------------------------------------------
# Part 2: Compare Two Models
# ------------------------------------------------------------

comparison_questions = [
    "What is supervised learning?",
    "Explain overfitting in simple terms.",
    "Why is data preprocessing important in machine learning?"
]


comparison_results = []

comparison_results.append("=" * 70)
comparison_results.append("W5D2 - MODEL COMPARISON")
comparison_results.append("llama3.2:3b vs qwen2.5:3b")
comparison_results.append("=" * 70)


for i, question in enumerate(comparison_questions, start=1):

    print(f"\nQuestion {i}: {question}")
    print("=" * 70)

    comparison_results.append(
        f"\nQuestion {i}: {question}"
    )
    comparison_results.append("=" * 70)


    # --------------------------------------------------------
    # llama3.2:3b
    # --------------------------------------------------------

    print("\n[llama3.2:3b]")
    print("-" * 70)

    try:
        llama_answer = generate_response(
            "llama3.2:3b",
            question
        )

        print(llama_answer)

        comparison_results.append("\n[llama3.2:3b]")
        comparison_results.append("-" * 70)
        comparison_results.append(llama_answer)

    except Exception as e:
        llama_error = f"Error: {e}"
        print(llama_error)
        comparison_results.append(llama_error)


    # --------------------------------------------------------
    # qwen2.5:3b
    # --------------------------------------------------------

    print("\n[qwen2.5:3b]")
    print("-" * 70)

    try:
        qwen_answer = generate_response(
            "qwen2.5:3b",
            question
        )

        print(qwen_answer)

        comparison_results.append("\n[qwen2.5:3b]")
        comparison_results.append("-" * 70)
        comparison_results.append(qwen_answer)

    except Exception as e:
        qwen_error = f"Error: {e}"
        print(qwen_error)
        comparison_results.append(qwen_error)


# Save comparison results
comparison_file = OUTPUT_DIR / "model_comparison.txt"

comparison_file.write_text(
    "\n".join(comparison_results),
    encoding="utf-8"
)


# ------------------------------------------------------------
# Completion Message
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("W5D2 TESTING COMPLETED")
print("=" * 70)

print(f"\nResults saved to:")
print(f"- {llama_file}")
print(f"- {comparison_file}")