from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from time import perf_counter

from backend.observability.metrics import RAG_REQUESTS, RAG_STAGE_DURATION


class RequestTrace:
    """Per-request stage timings in milliseconds. Safe to share across threads."""

    def __init__(self) -> None:
        self._stages: dict[str, float] = {}
        self._lock = Lock()
        self._started_at = perf_counter()

    def record(self, name: str, seconds: float) -> None:
        with self._lock:
            self._stages[name] = self._stages.get(name, 0.0) + seconds * 1_000

    def stages_ms(self) -> dict[str, float]:
        with self._lock:
            return {name: round(value, 2) for name, value in self._stages.items()}

    def elapsed_ms(self) -> float:
        return round((perf_counter() - self._started_at) * 1_000, 2)


_current_trace: ContextVar[RequestTrace | None] = ContextVar(
    "homelab_request_trace",
    default=None,
)


@contextmanager
def use_trace(trace: RequestTrace) -> Iterator[RequestTrace]:
    token = _current_trace.set(trace)
    try:
        yield trace
    finally:
        _current_trace.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """
    Time one leaf stage of the request path.

    The duration always feeds the Prometheus histogram and, when a trace is
    active in this context, that request's own breakdown. Nest stages only if
    double counting in the trace is acceptable; the built-in stages are leaves.
    """

    started_at = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - started_at
        RAG_STAGE_DURATION.labels(stage=name).observe(elapsed)
        trace = _current_trace.get()
        if trace is not None:
            trace.record(name, elapsed)


def observe_stream(
    chunks: Iterable[str],
    trace: RequestTrace,
    *,
    endpoint: str,
    answer_mode: str,
) -> Iterator[str]:
    """
    Wrap a token stream to record time-to-first-token and total generation time.

    The trace is passed explicitly because a stream is consumed lazily, often
    on different threads, where context variables are not reliable.
    """

    started_at = perf_counter()
    first_token_seen = False
    outcome = "cancelled"
    try:
        for chunk in chunks:
            if not first_token_seen:
                first_token_seen = True
                _record(trace, "generation.first_token", perf_counter() - started_at)
            yield chunk
        outcome = "ok"
    except GeneratorExit:
        raise
    except Exception:
        outcome = "error"
        raise
    finally:
        _record(trace, "generation.total", perf_counter() - started_at)
        RAG_REQUESTS.labels(
            endpoint=endpoint,
            answer_mode=answer_mode,
            outcome=outcome,
        ).inc()


def _record(trace: RequestTrace, name: str, seconds: float) -> None:
    RAG_STAGE_DURATION.labels(stage=name).observe(seconds)
    trace.record(name, seconds)
