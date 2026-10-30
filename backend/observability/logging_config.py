import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Literal

LogFormat = Literal["text", "json"]

request_id_var: ContextVar[str] = ContextVar("homelab_request_id", default="-")

_STANDARD_RECORD_ATTRIBUTES = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__,
) | {"message", "asctime", "request_id"}

_HANDLER_MARKER = "_homelab_observability_handler"


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRIBUTES and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(log_format: LogFormat = "text", level: int = logging.INFO) -> None:
    """
    Install one root handler carrying request ids; safe to call repeatedly.

    Replaces our own earlier handler and the bare stream handler that
    ``logging.basicConfig`` installs, so lines are not printed twice. Other
    handlers (for example test capture) are left alone.
    """

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False) or type(handler) is logging.StreamHandler:
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    setattr(handler, _HANDLER_MARKER, True)
    handler.addFilter(RequestIdFilter())
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s [%(request_id)s] - %(message)s"),
        )

    root.addHandler(handler)
    root.setLevel(level)
