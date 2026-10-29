import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from backend.observability import RequestTrace, observe_stream, stage, use_trace
from backend.observability.http import ObservabilityMiddleware, metrics_router
from backend.observability.logging_config import (
    JsonFormatter,
    RequestIdFilter,
    configure_logging,
    request_id_var,
)
from backend.observability.metrics import observe_generation


def sample(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_stage_feeds_histogram_and_active_trace() -> None:
    before = sample("homelab_rag_stage_duration_seconds_count", {"stage": "unit.test"})
    trace = RequestTrace()

    with use_trace(trace):
        with stage("unit.test"):
            pass
        with stage("unit.test"):
            pass

    assert sample("homelab_rag_stage_duration_seconds_count", {"stage": "unit.test"}) == before + 2
    assert set(trace.stages_ms()) == {"unit.test"}


def test_stage_without_trace_only_records_metric() -> None:
    before = sample("homelab_rag_stage_duration_seconds_count", {"stage": "unit.untraced"})

    with stage("unit.untraced"):
        pass

    assert (
        sample("homelab_rag_stage_duration_seconds_count", {"stage": "unit.untraced"}) == before + 1
    )


def test_stage_records_even_when_the_body_raises() -> None:
    trace = RequestTrace()

    with pytest.raises(RuntimeError), use_trace(trace), stage("unit.failing"):
        raise RuntimeError("boom")

    assert "unit.failing" in trace.stages_ms()


def test_observe_stream_records_first_token_total_and_outcome() -> None:
    trace = RequestTrace()
    labels = {"endpoint": "unit", "answer_mode": "documents", "outcome": "ok"}
    before = sample("homelab_rag_requests_total", labels)

    chunks = list(
        observe_stream(iter(["a", "b"]), trace, endpoint="unit", answer_mode="documents"),
    )

    assert chunks == ["a", "b"]
    assert {"generation.first_token", "generation.total"} <= set(trace.stages_ms())
    assert sample("homelab_rag_requests_total", labels) == before + 1


def test_observe_stream_marks_abandoned_streams_as_cancelled() -> None:
    trace = RequestTrace()
    labels = {"endpoint": "unit", "answer_mode": "documents", "outcome": "cancelled"}
    before = sample("homelab_rag_requests_total", labels)

    stream = observe_stream(iter(["a", "b"]), trace, endpoint="unit", answer_mode="documents")
    next(stream)
    stream.close()

    assert sample("homelab_rag_requests_total", labels) == before + 1


def test_observe_stream_marks_failures_as_errors() -> None:
    def failing() -> object:
        yield "a"
        raise RuntimeError("model crashed")

    labels = {"endpoint": "unit", "answer_mode": "documents", "outcome": "error"}
    before = sample("homelab_rag_requests_total", labels)

    with pytest.raises(RuntimeError):
        list(
            observe_stream(
                failing(),  # type: ignore[arg-type]
                RequestTrace(),
                endpoint="unit",
                answer_mode="documents",
            )
        )

    assert sample("homelab_rag_requests_total", labels) == before + 1


def test_observe_generation_derives_tokens_per_second() -> None:
    summary = observe_generation(
        {
            "load_duration": 2_000_000_000,
            "prompt_eval_count": 1200,
            "prompt_eval_duration": 500_000_000,
            "eval_count": 100,
            "eval_duration": 2_000_000_000,
        }
    )

    assert summary == {
        "model_load_ms": 2000.0,
        "prompt_tokens": 1200.0,
        "prompt_eval_ms": 500.0,
        "completion_tokens": 100.0,
        "tokens_per_second": 50.0,
    }


def test_observe_generation_accumulates_token_and_time_counters() -> None:
    tokens = {"kind": "completion"}
    seconds = {"phase": "generation"}
    tokens_before = sample("homelab_llm_tokens_total", tokens)
    seconds_before = sample("homelab_llm_processing_seconds_total", seconds)
    prompt_before = sample("homelab_llm_tokens_total", {"kind": "prompt"})

    observe_generation(
        {
            "prompt_eval_count": 300,
            "prompt_eval_duration": 100_000_000,
            "eval_count": 80,
            "eval_duration": 2_000_000_000,
        }
    )

    assert sample("homelab_llm_tokens_total", tokens) == tokens_before + 80
    assert sample("homelab_llm_tokens_total", {"kind": "prompt"}) == prompt_before + 300
    assert sample("homelab_llm_processing_seconds_total", seconds) == seconds_before + 2.0
    assert sample("homelab_llm_last_tokens_per_second") == 40.0


def test_observe_generation_counts_truncated_answers() -> None:
    before = sample("homelab_llm_truncated_total")

    summary = observe_generation({"done_reason": "length"})
    observe_generation({"done_reason": "stop"})

    assert summary == {"truncated": 1.0}
    assert sample("homelab_llm_truncated_total") == before + 1


def test_observe_generation_ignores_missing_and_invalid_fields() -> None:
    assert observe_generation({"eval_count": True, "load_duration": 0, "response": "x"}) == {}


def build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/items/{item_id}")
    def read_item(item_id: int) -> dict[str, int]:
        return {"id": item_id}

    app.include_router(metrics_router)
    app.add_middleware(ObservabilityMiddleware)
    return app


def test_middleware_adds_request_id_and_records_route_template_metrics() -> None:
    labels = {"method": "GET", "route": "/items/{item_id}", "status": "200"}
    before = sample("homelab_http_requests_total", labels)
    client = TestClient(build_app())

    response = client.get("/items/7", headers={"X-Request-ID": "abc-123"})
    client.get("/items/8")

    assert response.headers["x-request-id"] == "abc-123"
    assert sample("homelab_http_requests_total", labels) == before + 2


def test_middleware_generates_request_id_when_absent() -> None:
    response = TestClient(build_app()).get("/items/1")

    assert len(response.headers["x-request-id"]) == 32


def test_unknown_paths_share_one_metric_label() -> None:
    labels = {"method": "GET", "route": "unmatched", "status": "404"}
    before = sample("homelab_http_requests_total", labels)
    client = TestClient(build_app())

    client.get("/nope/1")
    client.get("/nope/2")

    assert sample("homelab_http_requests_total", labels) == before + 2


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    client = TestClient(build_app())
    client.get("/items/1")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "homelab_http_requests_total" in response.text
    assert "homelab_rag_stage_duration_seconds" in response.text


def make_record(message: str) -> logging.LogRecord:
    record = logging.LogRecord("unit", logging.INFO, __file__, 1, message, None, None)
    RequestIdFilter().filter(record)
    return record


def test_configure_logging_replaces_basic_config_handler_instead_of_duplicating() -> None:
    root = logging.getLogger()
    original = list(root.handlers)
    try:
        for handler in original:
            root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO)

        configure_logging("text")
        configure_logging("json")

        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) == 1
        assert isinstance(stream_handlers[0].formatter, JsonFormatter)
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in original:
            root.addHandler(handler)


def test_json_formatter_includes_request_id_and_extra_fields() -> None:
    token = request_id_var.set("req-42")
    try:
        record = make_record("hello")
        record.stages_ms = {"retrieval.rerank": 12.5}
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)

    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "req-42"
    assert payload["stages_ms"] == {"retrieval.rerank": 12.5}
