from collections.abc import Mapping
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

_NANOSECONDS = 1_000_000_000

_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
)

_TOKEN_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

_TOKENS_PER_SECOND_BUCKETS = (1, 2, 5, 10, 20, 30, 40, 60, 80, 120, 200)

HTTP_REQUESTS = Counter(
    "homelab_http_requests_total",
    "HTTP requests handled by the API.",
    ("method", "route", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "homelab_http_request_duration_seconds",
    "Time from request start until the response body is fully sent.",
    ("method", "route"),
    buckets=_LATENCY_BUCKETS,
)
HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "homelab_http_requests_in_progress",
    "HTTP requests currently being handled.",
)

RAG_REQUESTS = Counter(
    "homelab_rag_requests_total",
    "Question-answering requests by endpoint, answer mode and outcome.",
    ("endpoint", "answer_mode", "outcome"),
)
RAG_STAGE_DURATION = Histogram(
    "homelab_rag_stage_duration_seconds",
    "Duration of each leaf stage of the RAG request path.",
    ("stage",),
    buckets=_LATENCY_BUCKETS,
)
RAG_PROMPT_CHARS = Histogram(
    "homelab_rag_prompt_chars",
    "Prompt length in characters sent to the language model.",
    buckets=(500, 1_000, 2_000, 4_000, 6_000, 8_000, 12_000, 16_000),
)
RAG_CORPUS_CHUNKS = Gauge(
    "homelab_rag_corpus_chunks",
    "Chunks held by the currently loaded retrieval corpus.",
)

CACHE_EVENTS = Counter(
    "homelab_cache_events_total",
    "In-process cache lookups by cache and result.",
    ("cache", "result"),
)

WEB_REQUESTS = Counter(
    "homelab_web_requests_total",
    "Web search and page fetch attempts by operation and outcome.",
    ("operation", "outcome"),
)

CRAG_DECISIONS = Counter(
    "homelab_crag_decisions_total",
    "Corrective RAG outcomes by retrieval verdict and where the answer's sources came from.",
    ("verdict", "sources"),
)

OLLAMA_REQUESTS = Counter(
    "homelab_ollama_requests_total",
    "Requests sent to Ollama by operation and outcome.",
    ("operation", "outcome"),
)
OLLAMA_REQUEST_DURATION = Histogram(
    "homelab_ollama_request_duration_seconds",
    "Wall-clock duration of Ollama requests (time to first byte for streams excluded).",
    ("operation",),
    buckets=_LATENCY_BUCKETS,
)
LLM_MODEL_LOAD = Histogram(
    "homelab_llm_model_load_seconds",
    "Time Ollama spent loading the model into memory for a generation.",
    buckets=_LATENCY_BUCKETS,
)
LLM_PROMPT_EVAL = Histogram(
    "homelab_llm_prompt_eval_seconds",
    "Time Ollama spent processing the prompt before the first generated token.",
    buckets=_LATENCY_BUCKETS,
)
LLM_PROMPT_TOKENS = Histogram(
    "homelab_llm_prompt_tokens",
    "Prompt tokens processed per generation.",
    buckets=_TOKEN_BUCKETS,
)
LLM_COMPLETION_TOKENS = Histogram(
    "homelab_llm_completion_tokens",
    "Tokens generated per response.",
    buckets=_TOKEN_BUCKETS,
)
LLM_TRUNCATED = Counter(
    "homelab_llm_truncated_total",
    "Generations cut off because they reached the token limit.",
)
LLM_TOKENS_PER_SECOND = Histogram(
    "homelab_llm_tokens_per_second",
    "Generation throughput reported by Ollama.",
    buckets=_TOKENS_PER_SECOND_BUCKETS,
)


def _positive_number(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def observe_generation(payload: Mapping[str, Any]) -> dict[str, float]:
    """
    Record the timing fields Ollama attaches to a finished generation.

    Returns the values that were present so callers can log them.
    """

    summary: dict[str, float] = {}

    if payload.get("done_reason") == "length":
        LLM_TRUNCATED.inc()
        summary["truncated"] = 1.0

    load_ns = _positive_number(payload, "load_duration")
    if load_ns is not None:
        summary["model_load_ms"] = round(load_ns / _NANOSECONDS * 1_000, 1)
        LLM_MODEL_LOAD.observe(load_ns / _NANOSECONDS)

    prompt_tokens = _positive_number(payload, "prompt_eval_count")
    if prompt_tokens is not None:
        summary["prompt_tokens"] = prompt_tokens
        LLM_PROMPT_TOKENS.observe(prompt_tokens)

    prompt_ns = _positive_number(payload, "prompt_eval_duration")
    if prompt_ns is not None:
        summary["prompt_eval_ms"] = round(prompt_ns / _NANOSECONDS * 1_000, 1)
        LLM_PROMPT_EVAL.observe(prompt_ns / _NANOSECONDS)

    completion_tokens = _positive_number(payload, "eval_count")
    if completion_tokens is not None:
        summary["completion_tokens"] = completion_tokens
        LLM_COMPLETION_TOKENS.observe(completion_tokens)

    eval_ns = _positive_number(payload, "eval_duration")
    if completion_tokens is not None and eval_ns is not None:
        tokens_per_second = completion_tokens / (eval_ns / _NANOSECONDS)
        summary["tokens_per_second"] = round(tokens_per_second, 1)
        LLM_TOKENS_PER_SECOND.observe(tokens_per_second)

    return summary
