"""Route a question to grounded generation or the offline mock fallback."""

from time import perf_counter

from app.config import get_settings
from app.llm.generator import AnswerResponse, generate_answer_with_usage
from app.retrieval.retriever import retrieve
from app.tools.fallback import mock_web_search


def answer_question(question: str, metrics: dict | None = None) -> AnswerResponse:
    metrics = metrics if metrics is not None else {}
    metrics.update({"retrieval_latency_ms": 0.0, "generation_latency_ms": 0.0, "model": None, "token_usage": None})
    settings = get_settings()
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Question must not be empty")
    if len(normalized_question) > settings.max_question_length:
        raise ValueError(f"Question exceeds the {settings.max_question_length}-character limit")

    started = perf_counter()
    try:
        chunks = retrieve(normalized_question)
    finally:
        metrics["retrieval_latency_ms"] = round((perf_counter() - started) * 1000, 2)
    confidence = max((score for _doc, _section, _text, score in chunks), default=0.0)
    if not chunks or confidence < settings.similarity_threshold:
        fallback_result = mock_web_search(normalized_question)
        metrics["outcome"] = "fallback"
        return AnswerResponse(
            answer=f"{fallback_result['result']} {fallback_result['note']}",
            citations=[],
            confidence=confidence,
            threshold_used=settings.similarity_threshold,
            warnings=["Low knowledge-base similarity; returned the offline mock fallback, not verified evidence."],
        )

    metrics["outcome"] = "grounded"
    started = perf_counter()
    try:
        generated = generate_answer_with_usage(normalized_question, chunks)
        metrics["token_usage"] = generated.usage
        metrics["model"] = settings.generation_model
    finally:
        metrics["generation_latency_ms"] = round((perf_counter() - started) * 1000, 2)
    # Never trust model-reported retrieval values; replace them with the actual retrieval score and threshold.
    return generated.response.model_copy(update={
        "confidence": confidence,
        "threshold_used": settings.similarity_threshold,
    })
