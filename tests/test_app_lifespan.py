from unittest.mock import Mock

from fastapi.testclient import TestClient

import backend.api.app as app_module


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
