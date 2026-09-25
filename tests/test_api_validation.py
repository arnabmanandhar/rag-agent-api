from fastapi.testclient import TestClient

from app.api.main import app


def test_empty_question_returns_422_without_retrieval_or_generation(mocker):
    retrieve = mocker.patch("app.inference.orchestrator.retrieve")
    generate = mocker.patch("app.inference.orchestrator.generate_answer_with_usage")

    response = TestClient(app).post("/api/v1/query", json={"question": "   "})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"
    retrieve.assert_not_called()
    generate.assert_not_called()


def test_oversized_question_returns_422_without_retrieval_or_generation(mocker):
    retrieve = mocker.patch("app.inference.orchestrator.retrieve")
    generate = mocker.patch("app.inference.orchestrator.generate_answer_with_usage")

    response = TestClient(app).post("/api/v1/query", json={"question": "x" * 2001})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "question_too_long"
    retrieve.assert_not_called()
    generate.assert_not_called()


def test_valid_question_passes_validation(mocker, sample_chunks, generation_result):
    retrieve = mocker.patch("app.inference.orchestrator.retrieve", return_value=sample_chunks)
    generate = mocker.patch("app.inference.orchestrator.generate_answer_with_usage", return_value=generation_result)

    response = TestClient(app).post("/api/v1/query", json={"question": "How do I rotate a token?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Rotate the token in the console."
    retrieve.assert_called_once_with("How do I rotate a token?")
    generate.assert_called_once()
