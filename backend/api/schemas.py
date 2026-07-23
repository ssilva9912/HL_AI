from datetime import datetime
from uuid import UUID

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


class EvaluationRequest(BaseModel):
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of documents retrieved for each benchmark question.",
    )


class EvaluationMetricsResponse(BaseModel):
    question_count: int
    hit_at_1: float
    hit_at_5: float
    precision_at_5: float
    recall_at_5: float
    mean_reciprocal_rank: float


class QuestionEvaluationResponse(BaseModel):
    question: str
    relevant_documents: list[str]
    retrieved_documents: list[str]
    hit_at_1: float
    hit_at_5: float
    precision_at_5: float
    recall_at_5: float
    reciprocal_rank: float


class EvaluationResponse(BaseModel):
    metrics: EvaluationMetricsResponse
    questions: list[QuestionEvaluationResponse]
    top_k: int
    elapsed_ms: float


class IngestResponse(BaseModel):
    document: str
    size_bytes: int
    document_count: int
    chunk_count: int
    status: str


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    content_type: str | None
    size_bytes: int
    checksum_sha256: str
    status: str
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class IngestionJobResponse(BaseModel):
    id: UUID
    document_id: UUID
    operation: str
    status: str
    attempt_count: int
    processed_chunks: int
    total_chunks: int | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str
    service: str
