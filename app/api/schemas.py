"""HTTP request and response models for the query API."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    stream: bool = False

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value


class CitationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: str
    section: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    citations: list[CitationModel]
    confidence: float = Field(ge=0.0, le=1.0)
    threshold_used: float = Field(ge=0.0, le=1.0)
    warnings: list[str]
