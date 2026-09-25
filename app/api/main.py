"""FastAPI endpoints for TaskFlow knowledge-base queries."""

import json
import logging
import time
import uuid
from logging import LogRecord
from typing import Any

import chromadb
from chromadb.errors import ChromaError
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIError
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.inference.orchestrator import answer_question
from app.ingestion.build_index import COLLECTION_NAME
from app.llm.generator import generate_answer_stream
from app.retrieval.retriever import retrieve
from app.tools.fallback import mock_web_search
from app.api.schemas import QueryRequest, QueryResponse


class JsonLogFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        return json.dumps(getattr(record, "structured", {}), separators=(",", ":"), ensure_ascii=False)


logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(JsonLogFormatter())
    logger.addHandler(stream_handler)

app = FastAPI(title="TaskFlow Knowledge Base API", version="1.0.0")


def _log_request(request: Request) -> None:
    state = request.state
    metrics = getattr(state, "metrics", {})
    logger.info("request", extra={"structured": {
        "event": "api_request",
        "request_id": getattr(state, "request_id", ""),
        "retrieval_latency_ms": metrics.get("retrieval_latency_ms", 0.0),
        "generation_latency_ms": metrics.get("generation_latency_ms", 0.0),
        "outcome": metrics.get("outcome", "error"),
        "token_usage": metrics.get("token_usage"),
    }})
    state.logged = True


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied_id = request.headers.get("x-request-id", "").strip()
    request.state.request_id = supplied_id[:128] if supplied_id else str(uuid.uuid4())
    request.state.metrics = {"retrieval_latency_ms": 0.0, "generation_latency_ms": 0.0, "token_usage": None, "outcome": "error"}
    try:
        response = await call_next(request)
    except Exception:
        _log_request(request)
        raise
    response.headers["X-Request-ID"] = request.state.request_id
    if not getattr(request.state, "defer_log", False):
        if response.status_code < 400 and request.url.path == "/health":
            request.state.metrics["outcome"] = "health"
        elif response.status_code >= 400:
            request.state.metrics["outcome"] = "error"
        _log_request(request)
    return response


def _is_unavailable(exc: Exception) -> bool:
    return isinstance(exc, (RuntimeError, OSError, APIError, ChromaError, ConnectionError, TimeoutError))


def _safe_http_exception(exc: Exception) -> HTTPException:
    if _is_unavailable(exc):
        return HTTPException(status_code=503, detail={"code": "service_unavailable", "message": "Retrieval or generation service is unavailable."})
    if isinstance(exc, ValueError):
        return HTTPException(status_code=502, detail={"code": "invalid_model_response", "message": "The answer provider returned an unusable response."})
    return HTTPException(status_code=500, detail={"code": "internal_error", "message": "The query could not be completed."})


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    # Avoid echoing submitted question text or Pydantic's internal error structure.
    return JSONResponse(status_code=422, content={"detail": {"code": "invalid_request", "message": "Request body is invalid."}})


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: Request, payload: QueryRequest):
    try:
        settings = get_settings()
    except Exception as exc:
        request.state.metrics["outcome"] = "error"
        raise _safe_http_exception(exc) from None
    # Enforce the configurable bound before retrieval/model work begins.
    if len(payload.question) > settings.max_question_length:
        raise HTTPException(status_code=422, detail={"code": "question_too_long", "message": "Question exceeds the configured length limit."})

    if not payload.stream:
        try:
            result = await run_in_threadpool(answer_question, payload.question, metrics=request.state.metrics)
            request.state.metrics["outcome"] = request.state.metrics.get("outcome", "grounded")
            return QueryResponse.model_validate(result.model_dump())
        except Exception as exc:
            request.state.metrics["outcome"] = "error"
            raise _safe_http_exception(exc) from None

    request.state.defer_log = True
    return StreamingResponse(
        _stream_query(payload.question, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _stream_query(question: str, request: Request):
    """SSE contract: token data is {text}; done data has citations/confidence/threshold_used/warnings;
    errors use {code, message}. The answer text appears only in token events, never as raw JSON.
    """
    metrics = request.state.metrics
    generation_started = None
    try:
        started = time.perf_counter()
        chunks = retrieve(question)
        metrics["retrieval_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        settings = get_settings()
        confidence = max((score for _doc, _section, _text, score in chunks), default=0.0)

        if not chunks or confidence < settings.similarity_threshold:
            fallback = mock_web_search(question)
            answer = f"{fallback['result']} {fallback['note']}"
            response = QueryResponse(
                answer=answer,
                citations=[],
                confidence=confidence,
                threshold_used=settings.similarity_threshold,
                warnings=["Low knowledge-base similarity; returned the offline mock fallback, not verified evidence."],
            )
            metrics["outcome"] = "fallback"
            yield _sse("token", {"text": response.answer})
        else:
            metrics["outcome"] = "grounded"
            generation_started = time.perf_counter()
            generation = generate_answer_stream(question, chunks)
            response = None
            for event, value in generation:
                if event == "token":
                    yield _sse("token", {"text": value})
                elif event == "result":
                    metrics["token_usage"] = value.usage
                    response = QueryResponse.model_validate(value.response.model_dump()).model_copy(update={
                        "confidence": confidence,
                        "threshold_used": settings.similarity_threshold,
                    })
            metrics["generation_latency_ms"] = round((time.perf_counter() - generation_started) * 1000, 2)
            if response is None:
                raise ValueError("Generation completed without a validated response")

        yield _sse("done", {
            "citations": [citation.model_dump() for citation in response.citations],
            "confidence": response.confidence,
            "threshold_used": response.threshold_used,
            "warnings": response.warnings,
        })
    except Exception as exc:
        metrics["outcome"] = "error"
        if generation_started is not None:
            metrics["generation_latency_ms"] = round((time.perf_counter() - generation_started) * 1000, 2)
        code = "service_unavailable" if _is_unavailable(exc) else "query_failed"
        message = "Retrieval or generation service is unavailable." if code == "service_unavailable" else "The query could not be completed."
        yield _sse("error", {"code": code, "message": message})
    finally:
        _log_request(request)


@app.get("/health")
def health():
    try:
        settings = get_settings()
        db = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        collection = db.get_collection(name=COLLECTION_NAME)
        return {"status": "ok", "collection": COLLECTION_NAME, "count": collection.count()}
    except Exception:
        raise HTTPException(status_code=503, detail={"code": "index_unavailable", "message": "Knowledge-base index is unavailable."}) from None
