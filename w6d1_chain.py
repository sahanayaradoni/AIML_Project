from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


# ---------------------------------------------------------
# Task 1: LangChain Chain
# PromptTemplate -> Ollama LLM -> OutputParser
# ---------------------------------------------------------

def create_chain():
    prompt = PromptTemplate(
        input_variables=["topic"],
        template=(
            "You are a helpful AI tutor.\n"
            "Explain the following topic in simple terms:\n"
            "{topic}"
        ),
    )

    llm = OllamaLLM(model="llama3.2:3b")

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain


def test_chain():
    chain = create_chain()

    test_inputs = [
        "What is Python?",
        "What is machine learning?",
        "What is LangChain?",
        "What is an API?",
        "What is a database?",
    ]

    print("=" * 60)
    print("TASK 1: LANGCHAIN CHAIN TEST")
    print("=" * 60)

    for number, topic in enumerate(test_inputs, start=1):
        print(f"\nInput {number}: {topic}")
        response = chain.invoke({"topic": topic})
        print("Response:")
        print(response)
        print("-" * 60)


# ---------------------------------------------------------
# Task 2: Conversation Memory
# Conversation history maintained across 5 turns
# ---------------------------------------------------------

class ConversationBufferMemory:
    def __init__(self):
        self.history = []

    def add_message(self, user_message, assistant_message):
        self.history.append(
            {
                "user": user_message,
                "assistant": assistant_message,
            }
        )

    def get_history(self):
        if not self.history:
            return "No previous conversation."

        formatted_history = []

        for turn_number, turn in enumerate(self.history, start=1):
            formatted_history.append(
                f"Turn {turn_number} - User: {turn['user']}"
            )
            formatted_history.append(
                f"Turn {turn_number} - Assistant: {turn['assistant']}"
            )

        return "\n".join(formatted_history)


def create_memory_chain():
    prompt = PromptTemplate(
        input_variables=["history", "user_input"],
        template=(
            "You are a helpful conversational AI.\n\n"
            "Previous conversation:\n"
            "{history}\n\n"
            "Current user message:\n"
            "{user_input}\n\n"
            "Respond naturally and use the previous conversation "
            "when it is relevant."
        ),
    )

    llm = OllamaLLM(model="llama3.2:3b")

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain


def test_memory():
    chain = create_memory_chain()
    memory = ConversationBufferMemory()

    conversation = [
        "My name is Sahana.",
        "What is my name?",
        "I am learning artificial intelligence.",
        "What am I learning?",
        "What is my name and what am I learning?",
    ]

    print("\n")
    print("=" * 60)
    print("TASK 2: CONVERSATION MEMORY TEST")
    print("=" * 60)

    for turn_number, user_input in enumerate(conversation, start=1):

        history = memory.get_history()

        response = chain.invoke(
            {
                "history": history,
                "user_input": user_input,
            }
        )

        print(f"\nTurn {turn_number}")
        print(f"User: {user_input}")
        print(f"Assistant: {response}")

        memory.add_message(user_input, response)

        print("\nConversation history after this turn:")
        print(memory.get_history())
        print("-" * 60)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    test_chain()
    test_memory()