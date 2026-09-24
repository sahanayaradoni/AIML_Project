from w6d1_chain import create_chain, create_memory_chain, ConversationBufferMemory


def test_chain_creation():
    chain = create_chain()
    assert chain is not None


def test_chain_response():
    chain = create_chain()
    response = chain.invoke({"topic": "Python"})
    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_memory_creation():
    memory = ConversationBufferMemory()
    assert memory.get_history() == "No previous conversation."


def test_memory_stores_conversation():
    memory = ConversationBufferMemory()

    memory.add_message(
        "My name is Sahana.",
        "Nice to meet you, Sahana!"
    )

    history = memory.get_history()

    assert "My name is Sahana." in history
    assert "Nice to meet you, Sahana!" in history


def test_memory_chain_creation():
    chain = create_memory_chain()
    assert chain is not None