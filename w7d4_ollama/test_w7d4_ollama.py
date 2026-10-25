
"""Tests for W7D4 Ollama local inference."""

from unittest.mock import Mock, patch

import requests

from w7d4_ollama_inference import (
    MODEL_NAME,
    OLLAMA_API_URL,
    SYSTEM_PROMPT,
    TEST_PROMPTS,
    generate_response,
)


def test_model_configuration():
    assert MODEL_NAME == "llama3.2:3b"
    assert OLLAMA_API_URL.endswith("/api/chat")
    assert SYSTEM_PROMPT
    assert len(TEST_PROMPTS) == 5


@patch("w7d4_ollama_inference.requests.post")
def test_generate_response_sends_system_and_user_messages(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {
        "message": {"content": "Machine learning helps computers learn."}
    }
    mock_post.return_value = mock_response

    result = generate_response("What is machine learning?")

    assert result == "Machine learning helps computers learn."
    mock_post.assert_called_once()

    args, kwargs = mock_post.call_args
    assert args[0] == OLLAMA_API_URL
    assert kwargs["json"]["model"] == MODEL_NAME
    assert kwargs["json"]["messages"][0]["role"] == "system"
    assert kwargs["json"]["messages"][0]["content"] == SYSTEM_PROMPT
    assert kwargs["json"]["messages"][1] == {
        "role": "user",
        "content": "What is machine learning?",
    }
    assert kwargs["json"]["stream"] is False


@patch("w7d4_ollama_inference.requests.post")
def test_generate_response_raises_for_http_error(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError(
        "Ollama API error"
    )
    mock_post.return_value = mock_response

    try:
        generate_response("Test prompt")
    except requests.HTTPError:
        pass
    else:
        raise AssertionError("Expected requests.HTTPError")
