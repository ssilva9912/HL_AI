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


class SearchMetadata(BaseModel):
    top_k: int
    source_count: int
    elapsed_ms: float


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    metadata: SearchMetadata


class IngestResponse(BaseModel):
    document: str
    size_bytes: int
    document_count: int
    chunk_count: int
    status: str


class HealthResponse(BaseModel):
    status: str
    service: str
