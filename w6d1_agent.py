from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# ---------------------------------------------------------
# Tool 1: Web Search Stub
# ---------------------------------------------------------

@tool
def web_search_stub(query: str) -> str:
    """Simulate a web search and return a sample result."""
    return (
        f"Web search result for '{query}': "
        "LangChain is a framework for building applications "
        "with language models."
    )


# ---------------------------------------------------------
# Tool 2: Calculator
# ---------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression."""

    try:
        allowed_characters = "0123456789+-*/(). "

        if not all(char in allowed_characters for char in expression):
            return "Error: Only basic arithmetic expressions are allowed."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as error:
        return f"Error: {error}"


# ---------------------------------------------------------
# Create Agent
# ---------------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

tools = [
    web_search_stub,
    calculator
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful AI assistant. "
        "Use the available tools when they are useful. "
        "Use the calculator for arithmetic calculations "
        "and the web search stub for information searches."
    )
)


# ---------------------------------------------------------
# Run Agent Tasks
# ---------------------------------------------------------

def run_agent_task(task_number, user_input):
    print("\n" + "=" * 60)
    print(f"AGENT TASK {task_number}")
    print("=" * 60)

    print(f"User: {user_input}")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
    )

    final_message = result["messages"][-1]

    print(f"Agent: {final_message.content}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("TASK 3: LANGCHAIN AGENT WITH TWO TOOLS")
    print("=" * 60)

    print("\nAvailable tools:")
    print("1. web_search_stub")
    print("2. calculator")

    run_agent_task(
        1,
        "Calculate 25 * 4 + 10."
    )

    run_agent_task(
        2,
        "Search for information about LangChain."
    )

    run_agent_task(
        3,
        "Calculate 100 / 4 and explain what LangChain is."
    )