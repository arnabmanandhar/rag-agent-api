import json
from types import SimpleNamespace

import pytest

from app.llm import generator


def _mock_groq(mocker, content, completion_factory):
    client = mocker.Mock()
    client.chat.completions.create.return_value = completion_factory(content)
    mocker.patch("app.llm.generator.OpenAI", return_value=client)
    return client


def test_valid_groq_json_is_parsed_and_validated(mocker, sample_chunks, completion_factory):
    expected = {
        "answer": "Rotate it in the console.",
        "citations": [{"document": "authentication.md", "section": "Token rotation"}],
        "confidence": 0.8,
        "threshold_used": 0.4,
        "warnings": [],
    }
    client = _mock_groq(mocker, json.dumps(expected), completion_factory)

    result = generator.generate_answer_with_usage("How do I rotate a token?", sample_chunks)

    assert result.response.model_dump() == expected
    client.chat.completions.create.assert_called_once()


def test_citation_not_in_retrieved_chunks_is_rejected(mocker, sample_chunks, completion_factory):
    invalid = {
        "answer": "A claim.",
        "citations": [{"document": "other.md", "section": "Missing section"}],
        "confidence": 0.8,
        "threshold_used": 0.4,
        "warnings": [],
    }
    _mock_groq(mocker, json.dumps(invalid), completion_factory)

    with pytest.raises(ValueError, match="citation absent from retrieved context"):
        generator.generate_answer_with_usage("question", sample_chunks)


def test_malformed_groq_content_is_rejected(mocker, sample_chunks, completion_factory):
    _mock_groq(mocker, "this is not JSON", completion_factory)

    with pytest.raises(ValueError, match="did not match the required answer schema"):
        generator.generate_answer_with_usage("question", sample_chunks)
