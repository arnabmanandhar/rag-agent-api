from app.inference import orchestrator


def test_high_similarity_uses_generation_not_fallback(mocker, sample_chunks, generation_result):
    retrieve = mocker.patch.object(orchestrator, "retrieve", return_value=sample_chunks)
    generate = mocker.patch.object(orchestrator, "generate_answer_with_usage", return_value=generation_result)
    fallback = mocker.patch.object(orchestrator, "mock_web_search")

    result = orchestrator.answer_question("rotate token")

    assert result.answer == "Rotate the token in the console."
    retrieve.assert_called_once()
    generate.assert_called_once()
    fallback.assert_not_called()


def test_low_similarity_uses_fallback_not_generation(mocker, sample_chunks):
    low_score_chunks = [(doc, section, text, 0.2) for doc, section, text, _score in sample_chunks]
    retrieve = mocker.patch.object(orchestrator, "retrieve", return_value=low_score_chunks)
    generate = mocker.patch.object(orchestrator, "generate_answer_with_usage")
    fallback = mocker.patch.object(orchestrator, "mock_web_search", return_value={
        "source": "mock_web_search",
        "result": "simulated",
        "note": "This is a simulated fallback result, not verified knowledge-base evidence.",
    })

    result = orchestrator.answer_question("unrelated question")

    assert result.citations == []
    assert "simulated" in result.answer
    retrieve.assert_called_once()
    fallback.assert_called_once_with("unrelated question")
    generate.assert_not_called()


def test_empty_retrieval_uses_fallback_not_generation(mocker):
    retrieve = mocker.patch.object(orchestrator, "retrieve", return_value=[])
    generate = mocker.patch.object(orchestrator, "generate_answer_with_usage")
    fallback = mocker.patch.object(orchestrator, "mock_web_search", return_value={
        "source": "mock_web_search",
        "result": "no result",
        "note": "This is a simulated fallback result, not verified knowledge-base evidence.",
    })

    result = orchestrator.answer_question("question with empty index")

    assert result.confidence == 0.0
    assert result.citations == []
    retrieve.assert_called_once()
    fallback.assert_called_once()
    generate.assert_not_called()
