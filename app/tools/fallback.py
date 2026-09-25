"""Offline mock search tool for questions unsupported by the local knowledge base."""


def mock_web_search(question: str) -> dict[str, str]:
    """Return a deterministic, explicitly simulated result without network access."""
    return {
        "source": "mock_web_search",
        "result": f"The offline mock search has no verified result for: {question.strip()}",
        "note": "This is a simulated fallback result, not verified knowledge-base evidence.",
    }
