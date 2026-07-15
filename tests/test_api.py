from fastapi.testclient import TestClient

from backend.api.app import app
from backend.api.dependencies import get_rag_service
from backend.api.rag_service import RAGAnswer, RAGSource


class FakeRAGService:
    def ask(self, question: str, top_k: int = 5) -> RAGAnswer:
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


def override_rag_service() -> FakeRAGService:
    return FakeRAGService()


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
