"""Shared offline fixtures for the focused RAG/API test suite."""

from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.llm.generator import AnswerResponse, Citation, GenerationResult


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch, tmp_path):
    """Provide harmless test-only configuration and prevent reliance on a real API key."""
    values = {
        "GROQ_API_KEY": "unit-test-key",
        "GROQ_BASE_URL": "https://api.groq.com/openai/v1",
        "EMBEDDING_PROVIDER": "sentence-transformers",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "GENERATION_MODEL": "test-model",
        "SIMILARITY_THRESHOLD": "0.4",
        "TOP_K": "4",
        "MAX_CONTEXT_TOKENS": "5000",
        "CHROMA_PERSIST_DIR": str(tmp_path / "chroma"),
        "MAX_QUESTION_LENGTH": "2000",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_chunks():
    return [("authentication.md", "Token rotation", "Tokens can be rotated in the console.", 0.8)]


@pytest.fixture
def valid_answer():
    return AnswerResponse(
        answer="Rotate the token in the console.",
        citations=[Citation(document="authentication.md", section="Token rotation")],
        confidence=0.8,
        threshold_used=0.4,
        warnings=[],
    )


@pytest.fixture
def generation_result(valid_answer):
    return GenerationResult(response=valid_answer, usage={"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20})


@pytest.fixture
def completion_factory():
    def make(content, usage=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage,
        )
    return make
