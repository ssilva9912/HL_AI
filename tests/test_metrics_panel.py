import importlib.util
from pathlib import Path
from types import ModuleType

import httpx
import pytest


def load_panel() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "frontend" / "metrics_panel.py"
    spec = importlib.util.spec_from_file_location("metrics_panel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


panel = load_panel()


def test_dashboard_urls_point_at_the_provisioned_uids() -> None:
    assert panel.DASHBOARDS == {"LLM throughput": "homelab-llm", "Overview": "homelab-ai"}
    assert (
        panel.dashboard_url("http://localhost:3001", "homelab-llm")
        == "http://localhost:3001/d/homelab-llm?theme=dark&refresh=10s"
    )


def test_embedded_urls_hide_grafana_chrome() -> None:
    url = panel.dashboard_url("http://localhost:3001", "homelab-ai", embedded=True)

    assert url.endswith("&kiosk")
    assert url.startswith("http://localhost:3001/d/homelab-ai?")


def test_grafana_url_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOMELAB_GRAFANA_URL", raising=False)
    assert panel.get_grafana_url() == "http://localhost:3001"

    monkeypatch.setenv("HOMELAB_GRAFANA_URL", "https://grafana.lan/")
    assert panel.get_grafana_url() == "https://grafana.lan"

    monkeypatch.setenv("HOMELAB_GRAFANA_URL", "")
    assert panel.get_grafana_url() == "http://localhost:3001"


def test_health_url_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOMELAB_GRAFANA_HEALTH_URL", raising=False)
    assert panel.get_grafana_health_url() is None

    monkeypatch.setenv("HOMELAB_GRAFANA_HEALTH_URL", " http://grafana:3000/api/health ")
    assert panel.get_grafana_health_url() == "http://grafana:3000/api/health"


def client_returning(status: int) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status)))


def test_grafana_is_up_follows_the_health_endpoint() -> None:
    assert panel.grafana_is_up("http://grafana:3000/api/health", client_returning(200))
    assert not panel.grafana_is_up("http://grafana:3000/api/health", client_returning(503))


def test_grafana_is_down_when_the_connection_fails() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(refuse))

    assert not panel.grafana_is_up("http://grafana:3000/api/health", client)


def test_without_a_health_url_grafana_is_assumed_up() -> None:
    assert panel.grafana_is_up(None)
