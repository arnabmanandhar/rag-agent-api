"""Route a question to grounded generation or the offline mock fallback."""

from app.config import get_settings
from app.llm.generator import AnswerResponse, generate_answer
from app.retrieval.retriever import retrieve
from app.tools.fallback import mock_web_search


def answer_question(question: str) -> AnswerResponse:
    settings = get_settings()
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Question must not be empty")
    if len(normalized_question) > settings.max_question_length:
        raise ValueError(f"Question exceeds the {settings.max_question_length}-character limit")

    chunks = retrieve(normalized_question)
    confidence = max((score for _doc, _section, _text, score in chunks), default=0.0)
    if not chunks or confidence < settings.similarity_threshold:
        fallback_result = mock_web_search(normalized_question)
        return AnswerResponse(
            answer=f"{fallback_result['result']} {fallback_result['note']}",
            citations=[],
            confidence=confidence,
            threshold_used=settings.similarity_threshold,
            warnings=["Low knowledge-base similarity; returned the offline mock fallback, not verified evidence."],
        )

    generated = generate_answer(normalized_question, chunks)
    # Never trust model-reported retrieval values; replace them with the actual retrieval score and threshold.
    return generated.model_copy(update={
        "confidence": confidence,
        "threshold_used": settings.similarity_threshold,
    })
