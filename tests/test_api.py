from fastapi.testclient import TestClient

from backend.api.app import app
from backend.api.dependencies import get_rag_service
from backend.api.rag_service import (
    DocumentTooLargeError,
    EmptyDocumentError,
    IngestionResult,
    RAGAnswer,
    RAGSource,
    UnsupportedDocumentTypeError,
)


class FakeRAGService:
    max_upload_bytes = 100

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> RAGAnswer:
        return RAGAnswer(
            answer=f"Test answer for: {question}",
            sources=[
                RAGSource(
                    text="Example source content.",
                    score=0.95,
                    document="sample.txt",
                    chunk_id="chunk-1",
                )
            ][:top_k],
        )

    def ingest_document(
        self,
        filename: str,
        content: bytes,
    ) -> IngestionResult:
        extension = f".{filename.rsplit('.', maxsplit=1)[-1].lower()}" if "." in filename else ""

        if extension not in {".txt", ".md", ".pdf"}:
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type: {extension or '<none>'}."
            )

        if not content:
            raise EmptyDocumentError("Uploaded document cannot be empty.")

        if len(content) > self.max_upload_bytes:
            raise DocumentTooLargeError("Uploaded document exceeds the configured size limit.")

        return IngestionResult(
            document=filename,
            size_bytes=len(content),
            document_count=6,
            chunk_count=8,
        )


fake_rag_service = FakeRAGService()


def override_rag_service() -> FakeRAGService:
    return fake_rag_service


app.dependency_overrides[get_rag_service] = override_rag_service

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "homelab-ai",
    }


def test_search() -> None:
    response = client.post(
        "/search",
        json={
            "question": "What is contained in the knowledge base?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == ("Test answer for: What is contained in the knowledge base?")
    assert len(body["sources"]) == 1
    assert body["sources"][0] == {
        "text": "Example source content.",
        "score": 0.95,
        "document": "sample.txt",
        "chunk_id": "chunk-1",
    }
    assert body["metadata"]["top_k"] == 5
    assert body["metadata"]["source_count"] == 1
    assert body["metadata"]["elapsed_ms"] >= 0


def test_search_uses_requested_top_k() -> None:
    response = client.post(
        "/search",
        json={
            "question": "Test question",
            "top_k": 1,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["metadata"]["top_k"] == 1
    assert body["metadata"]["source_count"] == 1


def test_search_rejects_empty_question() -> None:
    response = client.post(
        "/search",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422


def test_search_rejects_whitespace_only_question() -> None:
    response = client.post(
        "/search",
        json={
            "question": "   ",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Question cannot be empty.",
    }


def test_search_rejects_excessive_top_k() -> None:
    response = client.post(
        "/search",
        json={
            "question": "Test question",
            "top_k": 100,
        },
    )

    assert response.status_code == 422


def test_ingest_text_document() -> None:
    response = client.post(
        "/ingest",
        files={
            "file": (
                "notes.txt",
                b"Homelab AI ingestion test.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201
    assert response.json() == {
        "document": "notes.txt",
        "size_bytes": 26,
        "document_count": 6,
        "chunk_count": 8,
        "status": "indexed",
    }


def test_ingest_markdown_document() -> None:
    response = client.post(
        "/ingest",
        files={
            "file": (
                "guide.md",
                b"# Homelab AI",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["document"] == "guide.md"


def test_ingest_pdf_document() -> None:
    response = client.post(
        "/ingest",
        files={
            "file": (
                "manual.pdf",
                b"%PDF-test-content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["document"] == "manual.pdf"


def test_ingest_rejects_unsupported_extension() -> None:
    response = client.post(
        "/ingest",
        files={
            "file": (
                "data.csv",
                b"name,value",
                "text/csv",
            )
        },
    )

    assert response.status_code == 415


def test_ingest_rejects_empty_document() -> None:
    response = client.post(
        "/ingest",
        files={
            "file": (
                "empty.txt",
                b"",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Uploaded document cannot be empty.",
    }


def test_ingest_rejects_oversized_document() -> None:
    response = client.post(
        "/ingest",
        files={
            "file": (
                "large.txt",
                b"x" * 101,
                "text/plain",
            )
        },
    )

    assert response.status_code == 413
