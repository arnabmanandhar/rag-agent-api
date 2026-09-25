"""Application settings loaded from environment and optional project .env."""
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings(BaseModel):
    openai_api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL")
    generation_model: str = Field(default="gpt-4o-mini", validation_alias="GENERATION_MODEL")
    similarity_threshold: float = Field(default=0.75, ge=0, le=1, validation_alias="SIMILARITY_THRESHOLD")
    top_k: int = Field(default=4, ge=1, le=20, validation_alias="TOP_K")
    max_context_tokens: int = Field(default=5000, ge=256, validation_alias="MAX_CONTEXT_TOKENS")
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma", validation_alias="CHROMA_PERSIST_DIR")
    max_question_length: int = Field(default=2000, ge=1, le=100_000, validation_alias="MAX_QUESTION_LENGTH")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load validated settings and fail clearly when the API key is missing."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in the environment or copy .env.example to .env and add your key.")
    try:
        return Settings.model_validate({
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            "GENERATION_MODEL": os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
            "SIMILARITY_THRESHOLD": os.getenv("SIMILARITY_THRESHOLD", "0.75"),
            "TOP_K": os.getenv("TOP_K", "4"),
            "MAX_CONTEXT_TOKENS": os.getenv("MAX_CONTEXT_TOKENS", "5000"),
            "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma")),
            "MAX_QUESTION_LENGTH": os.getenv("MAX_QUESTION_LENGTH", "2000"),
        })
    except ValidationError as exc:
        raise RuntimeError(f"Invalid application configuration: {exc}") from exc
