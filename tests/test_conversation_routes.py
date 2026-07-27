from collections.abc import Iterator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.app import app
from backend.api.dependencies import get_rag_service
from backend.api.rag_service import RAGAnswer, RAGSource
from backend.database import Base, get_database_session


class ConversationRAGService:
    def __init__(self) -> None:
        self.histories: list[list[tuple[str, str]]] = []
        self.should_fail = False

    def ask(
        self,
        question: str,
        top_k: int = 5,
        history: list[tuple[str, str]] | None = None,
        *,
        hybrid: bool = False,
    ) -> RAGAnswer:
        self.histories.append(history or [])
        if self.should_fail:
            raise RuntimeError("Generator unavailable.")
        return RAGAnswer(
            answer=f"Grounded answer: {question}",
            sources=[
                RAGSource(
                    text="Persistent source",
                    score=0.9,
                    document="memory.txt",
                    chunk_id="chunk-1",
                ),
            ][:top_k],
            answer_mode="documents",
        )


def _client() -> Iterator[tuple[TestClient, ConversationRAGService]]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )
    Base.metadata.create_all(engine)
    rag_service = ConversationRAGService()

    def override_database_session() -> Iterator[Session]:
        with factory() as session:
            yield session

    previous_database = app.dependency_overrides.get(get_database_session)
    previous_rag = app.dependency_overrides.get(get_rag_service)
    app.dependency_overrides[get_database_session] = override_database_session
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    try:
        yield TestClient(app), rag_service
    finally:
        if previous_database is None:
            app.dependency_overrides.pop(get_database_session, None)
        else:
            app.dependency_overrides[get_database_session] = previous_database
        if previous_rag is None:
            app.dependency_overrides.pop(get_rag_service, None)
        else:
            app.dependency_overrides[get_rag_service] = previous_rag
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_conversation_lifecycle_and_memory() -> None:
    generator = _client()
    client, rag_service = next(generator)
    try:
        created = client.post(
            "/conversations",
            json={"title": "New conversation"},
        )
        assert created.status_code == 201
        conversation_id = created.json()["id"]

        first = client.post(
            f"/conversations/{conversation_id}/messages",
            json={"content": "What is durable memory?", "top_k": 5},
        )
        assert first.status_code == 200
        assert first.json()["conversation"]["title"] == "What is durable memory?"
        assert first.json()["conversation"]["message_count"] == 2
        assert first.json()["assistant_message"]["sources"][0]["document"] == "memory.txt"
        assert rag_service.histories == [[]]

        second = client.post(
            f"/conversations/{conversation_id}/messages",
            json={"content": "Why does it matter?", "top_k": 5},
        )
        assert second.status_code == 200
        assert rag_service.histories[-1] == [
            ("user", "What is durable memory?"),
            ("assistant", "Grounded answer: What is durable memory?"),
        ]

        renamed = client.patch(
            f"/conversations/{conversation_id}",
            json={"title": "Memory notes"},
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Memory notes"

        listed = client.get("/conversations")
        assert listed.status_code == 200
        assert listed.json()[0]["message_count"] == 4

        deleted = client.delete(f"/conversations/{conversation_id}")
        assert deleted.status_code == 204
        assert client.get(f"/conversations/{conversation_id}").status_code == 404
    finally:
        generator.close()


def test_failed_generation_keeps_only_user_message() -> None:
    generator = _client()
    client, rag_service = next(generator)
    try:
        created = client.post("/conversations", json={})
        conversation_id = created.json()["id"]
        rag_service.should_fail = True

        response = client.post(
            f"/conversations/{conversation_id}/messages",
            json={"content": "This generation will fail."},
        )
        assert response.status_code == 503

        conversation = client.get(f"/conversations/{conversation_id}")
        assert conversation.status_code == 200
        messages = conversation.json()["messages"]
        assert [(message["role"], message["content"]) for message in messages] == [
            ("user", "This generation will fail."),
        ]
    finally:
        generator.close()
