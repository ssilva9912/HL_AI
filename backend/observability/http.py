import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.observability.logging_config import request_id_var
from backend.observability.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS,
    HTTP_REQUESTS_IN_PROGRESS,
)

logger = logging.getLogger("backend.http")

REQUEST_ID_HEADER = "x-request-id"
_UNMATCHED_ROUTE = "unmatched"
_MAX_REQUEST_ID_LENGTH = 64

metrics_router = APIRouter()


@metrics_router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class ObservabilityMiddleware:
    """
    Assigns a request id, records HTTP metrics and logs one line per request.

    Implemented as pure ASGI so streaming responses are measured until their
    last body chunk and context variables reach the endpoint unchanged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == "/metrics":
            await self._app(scope, receive, send)
            return

        request_id = self._request_id(scope)
        token = request_id_var.set(request_id)
        started_at = perf_counter()
        status_code = 500
        HTTP_REQUESTS_IN_PROGRESS.inc()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        finally:
            elapsed = perf_counter() - started_at
            route = self._route_template(scope)
            method = scope["method"]
            HTTP_REQUESTS_IN_PROGRESS.dec()
            HTTP_REQUESTS.labels(method=method, route=route, status=str(status_code)).inc()
            HTTP_REQUEST_DURATION.labels(method=method, route=route).observe(elapsed)
            logger.info(
                "%s %s -> %d in %.1f ms",
                method,
                route,
                status_code,
                elapsed * 1_000,
                extra={
                    "http_method": method,
                    "http_route": route,
                    "http_status": status_code,
                    "duration_ms": round(elapsed * 1_000, 2),
                },
            )
            request_id_var.reset(token)

    @staticmethod
    def _request_id(scope: Scope) -> str:
        headers: list[tuple[bytes, bytes]] = scope["headers"]
        for name, value in headers:
            if name.decode("latin-1").lower() == REQUEST_ID_HEADER:
                supplied = value.decode("latin-1").strip()
                if supplied and supplied.isprintable():
                    return supplied[:_MAX_REQUEST_ID_LENGTH]
        return uuid4().hex

    @staticmethod
    def _route_template(scope: Scope) -> str:
        path: object = getattr(scope.get("route"), "path", None)
        return path if isinstance(path, str) else _UNMATCHED_ROUTE
