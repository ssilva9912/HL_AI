import os
from urllib.parse import urlencode

import httpx

DEFAULT_GRAFANA_URL = "http://localhost:3001"
HEALTH_TIMEOUT_SECONDS = 2.0

# Label shown in the app, mapped to the dashboard uid provisioned in docker/observability.
DASHBOARDS = {
    "LLM throughput": "homelab-llm",
    "Overview": "homelab-ai",
}

START_COMMAND = "docker compose --profile observability up -d"


def get_grafana_url() -> str:
    """Address the browser uses to reach Grafana (not necessarily the one the container uses)."""

    return (os.getenv("HOMELAB_GRAFANA_URL") or DEFAULT_GRAFANA_URL).rstrip("/")


def get_grafana_health_url() -> str | None:
    value = os.getenv("HOMELAB_GRAFANA_HEALTH_URL", "").strip()
    return value or None


def dashboard_url(base_url: str, uid: str, *, embedded: bool = False) -> str:
    """Link to a provisioned dashboard; embedded hides Grafana's own menus."""

    query = {"theme": "dark", "refresh": "10s"}
    if embedded:
        return f"{base_url}/d/{uid}?{urlencode(query)}&kiosk"
    return f"{base_url}/d/{uid}?{urlencode(query)}"


def grafana_is_up(health_url: str | None, client: httpx.Client | None = None) -> bool:
    """Whether Grafana answers; without a health URL assume it does and let the iframe show it."""

    if health_url is None:
        return True

    try:
        if client is None:
            with httpx.Client(timeout=HEALTH_TIMEOUT_SECONDS) as owned:
                return owned.get(health_url).is_success
        return client.get(health_url).is_success
    except httpx.HTTPError:
        return False
