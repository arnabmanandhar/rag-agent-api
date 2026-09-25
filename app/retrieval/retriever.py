"""Semantic retrieval from the persisted TaskFlow ChromaDB collection."""

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.ingestion.build_index import COLLECTION_NAME


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    """Load the same local embedding model once per process."""
    return SentenceTransformer(get_settings().embedding_model)


def retrieve(question: str) -> list[tuple[str, str, str, float]]:
    """Return (document, section, text, similarity) tuples ordered by relevance."""
    settings = get_settings()
    db = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    try:
        collection = db.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError("TaskFlow vector collection is unavailable; run the ingestion command first.") from exc

    vector = _embedding_model().encode(question, convert_to_numpy=True, show_progress_bar=False).tolist()
    result = collection.query(
        query_embeddings=[vector],
        n_results=settings.top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    matches: list[tuple[str, str, str, float]] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        # The collection uses cosine distance = 1 - cosine similarity; invert it and clamp
        # to [0, 1] so the configured threshold and reported score share one scale.
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        matches.append((metadata["document"], metadata["section"], text, similarity))
    return matches
