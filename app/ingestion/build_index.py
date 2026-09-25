"""Build the local TaskFlow knowledge-base index from Markdown documents."""

import hashlib
import re
from pathlib import Path
from typing import NamedTuple

from openai import OpenAI

from app.config import PROJECT_ROOT, get_settings

DATA_DIR = PROJECT_ROOT / "data" / "taskflow"
COLLECTION_NAME = "taskflow_knowledge"


class Section(NamedTuple):
    title: str
    text: str


def parse_markdown(path: Path) -> list[Section]:
    """Split a Markdown document at level-two headings, preserving section titles."""
    text = path.read_text(encoding="utf-8")
    headings = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", text))
    sections: list[Section] = []
    if not headings:
        body = text.strip()
        if body:
            sections.append(Section(path.stem.replace("-", " ").title(), body))
        return sections

    preamble = text[: headings[0].start()].strip()
    if preamble:
        title_match = re.search(r"(?m)^#\s+(.+?)\s*$", preamble)
        title = title_match.group(1).strip() if title_match else "Overview"
        sections.append(Section(title, preamble))
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[heading.end():end].strip()
        if body:
            sections.append(Section(heading.group(1).strip(), body))
    return sections


def build_index(data_dir: Path = DATA_DIR) -> dict[str, int]:
    """Embed section-sized chunks and upsert them into persisted ChromaDB."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("ChromaDB is not installed. Install dependencies with `pip install -r requirements.txt`.") from exc

    settings = get_settings()
    files = sorted(data_dir.glob("*.md"))
    if not files:
        raise RuntimeError(f"No Markdown documents found in {data_dir}")

    records: list[tuple[str, str, dict[str, str]]] = []
    for file_path in files:
        for section in parse_markdown(file_path):
            chunk_id = hashlib.sha256(f"{file_path.name}\0{section.title}\0{section.text}".encode("utf-8")).hexdigest()
            records.append((chunk_id, section.text, {"document": file_path.name, "section": section.title}))
    if not records:
        raise RuntimeError("No non-empty sections were found in the Markdown dataset")

    client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
    embeddings: list[list[float]] = []
    batch_size = 100
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=[record[1] for record in batch],
        )
        embeddings.extend(item.embedding for item in response.data)

    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    db = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    collection = db.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    # Remove prior entries for this dataset before upsert so deleted/renamed sections do not linger.
    collection.delete(where={"document": {"$in": [path.name for path in files]}})
    collection.upsert(
        ids=[record[0] for record in records],
        documents=[record[1] for record in records],
        metadatas=[record[2] for record in records],
        embeddings=embeddings,
    )
    return {"documents_processed": len(files), "chunks_created": len(records), "collection_size": collection.count()}


def main() -> None:
    try:
        summary = build_index()
    except Exception as exc:
        # Provider exception text can contain request details; avoid printing credentials or raw payloads.
        raise SystemExit(f"Index build failed: {type(exc).__name__}: {exc}") from None
    print("Ingestion completed successfully")
    print(f"Documents processed: {summary['documents_processed']}")
    print(f"Chunks created: {summary['chunks_created']}")
    print(f"Collection size: {summary['collection_size']}")


if __name__ == "__main__":
    main()
