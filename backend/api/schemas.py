from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2_000,
        description="Question to ask the local knowledge base.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of retrieved sources.",
    )


class SourceResponse(BaseModel):
    text: str
    score: float
    document: str | None = None
    chunk_id: str | None = None


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    status: str
    service: str
