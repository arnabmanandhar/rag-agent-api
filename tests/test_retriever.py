from types import SimpleNamespace

import numpy as np

from app.retrieval import retriever


def test_retriever_embeds_question_and_converts_cosine_distance(mocker):
    model = mocker.Mock()
    model.encode.return_value = np.zeros(384, dtype=np.float32)
    model_factory = mocker.patch("app.retrieval.retriever.SentenceTransformer", return_value=model)
    collection = mocker.Mock()
    collection.query.return_value = {
        "documents": [["near text", "far text"]],
        "metadatas": [[
            {"document": "near.md", "section": "Near"},
            {"document": "far.md", "section": "Far"},
        ]],
        "distances": [[0.2, 0.7]],
    }
    persistent_client = mocker.patch("app.retrieval.retriever.chromadb.PersistentClient")
    persistent_client.return_value.get_collection.return_value = collection
    retriever._embedding_model.cache_clear()

    try:
        results = retriever.retrieve("sample question")
    finally:
        retriever._embedding_model.cache_clear()

    assert results == [
        ("near.md", "Near", "near text", 0.8),
        ("far.md", "Far", "far text", 0.30000000000000004),
    ]
    model_factory.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
    model.encode.assert_called_once_with("sample question", convert_to_numpy=True, show_progress_bar=False)
    collection.query.assert_called_once()
