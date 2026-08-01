from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import get_network_share_scanner
from backend.api.ingestion_routes import router
from backend.config import Settings, get_settings
from backend.network_share import NetworkShareScanResult
from backend.network_share.scanner import NetworkShareUnavailableError


class FakeNetworkShareScanner:
    def __init__(self) -> None:
        self.error: NetworkShareUnavailableError | None = None

    def scan(self, *, name: str, root_path: Path) -> NetworkShareScanResult:
        if self.error is not None:
            raise self.error
        return NetworkShareScanResult(
            source_id=uuid4(),
            discovered=2,
            changed=1,
            unchanged=3,
            missing=1,
            unsupported=4,
            unsafe=0,
        )


def _client(
    scanner: FakeNetworkShareScanner,
    *,
    directory: Path | None,
) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_network_share_scanner] = lambda: scanner
    app.dependency_overrides[get_settings] = lambda: Settings(
        network_share_name="Test share",
        network_share_directory=directory,
    )
    return TestClient(app)


def test_manual_sync_returns_inventory_summary(tmp_path: Path) -> None:
    with _client(FakeNetworkShareScanner(), directory=tmp_path) as client:
        response = client.post("/network-share/sync")

    assert response.status_code == 200
    assert response.json() | {"source_id": "ignored"} == {
        "source_id": "ignored",
        "source_name": "Test share",
        "root_path": str(tmp_path),
        "status": "ready",
        "discovered": 2,
        "changed": 1,
        "unchanged": 3,
        "missing": 1,
        "unsupported": 4,
        "unsafe": 0,
    }


def test_manual_sync_requires_configuration() -> None:
    with _client(FakeNetworkShareScanner(), directory=None) as client:
        response = client.post("/network-share/sync")

    assert response.status_code == 409
    assert "not configured" in response.json()["detail"]


def test_manual_sync_reports_unavailable_share(tmp_path: Path) -> None:
    scanner = FakeNetworkShareScanner()
    scanner.error = NetworkShareUnavailableError("share is offline")
    with _client(scanner, directory=tmp_path) as client:
        response = client.post("/network-share/sync")

    assert response.status_code == 503
    assert response.json()["detail"] == "share is offline"
