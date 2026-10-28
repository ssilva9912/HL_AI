from unittest.mock import Mock

from fastapi.testclient import TestClient

import backend.api.app as app_module
from backend.config import Settings


def test_app_lifespan_recovers_pending_ingestion(
    monkeypatch,
) -> None:
    ingestion_service = Mock()
    ingestion_service.recover_pending_jobs.return_value = 2

    monkeypatch.setattr(
        app_module,
        "get_queued_ingestion_service",
        lambda: ingestion_service,
    )

    with TestClient(
        app_module.create_app(),
    ):
        pass

    ingestion_service.recover_pending_jobs.assert_called_once_with()


def quiet_ingestion_service() -> Mock:
    service = Mock()
    service.recover_pending_jobs.return_value = 0
    return service


def enable_warm_up(monkeypatch, rag_service: Mock) -> None:
    monkeypatch.setattr(
        app_module,
        "get_settings",
        lambda: Settings(warmup_on_startup=True),
    )
    monkeypatch.setattr(app_module, "get_rag_service", lambda: rag_service)
    monkeypatch.setattr(
        app_module, "get_queued_ingestion_service", lambda: quiet_ingestion_service()
    )


def test_app_warms_up_the_shared_rag_service_at_startup(monkeypatch) -> None:
    rag_service = Mock()
    enable_warm_up(monkeypatch, rag_service)

    with TestClient(app_module.create_app()):
        pass

    rag_service.warm_up.assert_called_once_with()


def test_app_starts_when_warm_up_fails(monkeypatch) -> None:
    rag_service = Mock()
    rag_service.warm_up.side_effect = RuntimeError("Ollama unavailable.")
    enable_warm_up(monkeypatch, rag_service)

    with TestClient(app_module.create_app()) as client:
        assert client.get("/health").status_code == 200


def test_app_starts_when_the_rag_service_cannot_be_created(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "get_settings",
        lambda: Settings(warmup_on_startup=True),
    )
    monkeypatch.setattr(
        app_module,
        "get_rag_service",
        Mock(side_effect=RuntimeError("Misconfigured.")),
    )
    monkeypatch.setattr(
        app_module, "get_queued_ingestion_service", lambda: quiet_ingestion_service()
    )

    with TestClient(app_module.create_app()) as client:
        assert client.get("/health").status_code == 200


def test_app_starts_when_recovery_fails(
    monkeypatch,
) -> None:
    ingestion_service = Mock()
    ingestion_service.recover_pending_jobs.side_effect = RuntimeError(
        "Database unavailable.",
    )

    monkeypatch.setattr(
        app_module,
        "get_queued_ingestion_service",
        lambda: ingestion_service,
    )

    with TestClient(
        app_module.create_app(),
    ):
        pass

    ingestion_service.recover_pending_jobs.assert_called_once_with()
