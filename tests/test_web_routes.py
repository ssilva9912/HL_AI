from typing import Any

from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import get_rag_service
from backend.api.rag_service import RAGAnswer, WebSearchNotConfiguredError


class RecordingService:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._error = error

    def ask(self, question: str, top_k: int = 5, web: bool = False) -> RAGAnswer:
        self.calls.append({"question": question, "top_k": top_k, "web": web})
        if self._error is not None:
            raise self._error
        return RAGAnswer(answer="ok", sources=[])


def client_for(service: RecordingService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: service
    return TestClient(app)


def test_search_does_not_use_the_web_unless_asked() -> None:
    service = RecordingService()

    response = client_for(service).post("/search", json={"question": "hello"})

    assert response.status_code == 200
    assert service.calls == [{"question": "hello", "top_k": 5, "web": False}]


def test_search_passes_the_web_flag_through() -> None:
    service = RecordingService()

    response = client_for(service).post("/search", json={"question": "hello", "web": True})

    assert response.status_code == 200
    assert service.calls[0]["web"] is True


def test_unconfigured_web_search_is_reported_as_service_unavailable() -> None:
    service = RecordingService(WebSearchNotConfiguredError("Web search is not configured."))

    response = client_for(service).post("/search", json={"question": "hello", "web": True})

    assert response.status_code == 503
    assert response.json()["detail"] == "Web search is not configured."
