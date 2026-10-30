import json
import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from time import perf_counter
from typing import Any

import httpx

from backend.config import get_settings
from backend.observability.metrics import (
    OLLAMA_REQUEST_DURATION,
    OLLAMA_REQUESTS,
    observe_generation,
)

logger = logging.getLogger(__name__)

JsonResponse = dict[str, Any]
PostFunction = Callable[..., httpx.Response]
StreamFunction = Callable[..., AbstractContextManager[httpx.Response]]


class OllamaGenerator:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float = 0.1,
        keep_alive: str | None = None,
        num_predict: int | None = None,
        post_function: PostFunction | None = None,
        stream_function: StreamFunction | None = None,
    ) -> None:
        settings = get_settings()

        resolved_model = settings.llm_model if model is None else model
        resolved_base_url = settings.ollama_url if base_url is None else base_url
        resolved_timeout = settings.request_timeout if timeout_seconds is None else timeout_seconds

        if not resolved_model.strip():
            raise ValueError("model must not be empty")

        if not resolved_base_url.strip():
            raise ValueError("base_url must not be empty")

        if resolved_timeout <= 0:
            raise ValueError("timeout_seconds must be positive")

        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        self._model = resolved_model.strip()
        self._base_url = resolved_base_url.strip().rstrip("/")
        self._timeout_seconds = resolved_timeout
        self._temperature = temperature
        self._keep_alive = settings.ollama_keep_alive if keep_alive is None else keep_alive

        if num_predict is not None and num_predict <= 0:
            raise ValueError("num_predict must be positive")

        self._options: dict[str, Any] = {"temperature": temperature}
        if num_predict is not None:
            self._options["num_predict"] = num_predict
        self._post_function = post_function or httpx.post
        self._stream_function = stream_function or httpx.stream

    def preload(self) -> None:
        """Ask Ollama to load the model now and keep it resident."""

        response = self._post_function(
            f"{self._base_url}/api/generate",
            json={"model": self._model, "keep_alive": self._keep_alive},
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        started_at = perf_counter()
        outcome = "error"
        try:
            response = self._post_function(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt.strip(),
                    "stream": False,
                    "keep_alive": self._keep_alive,
                    "options": self._options,
                },
                timeout=self._timeout_seconds,
            )

            response.raise_for_status()
            outcome = "ok"
        finally:
            OLLAMA_REQUESTS.labels(operation="generate", outcome=outcome).inc()
            OLLAMA_REQUEST_DURATION.labels(operation="generate").observe(
                perf_counter() - started_at
            )

        payload = self._read_payload(response)
        self._log_generation(payload)
        generated_text = payload.get("response")

        if not isinstance(generated_text, str):
            raise RuntimeError("Ollama response did not contain a valid 'response' field")

        generated_text = generated_text.strip()

        if not generated_text:
            raise RuntimeError("Ollama returned an empty response")

        return generated_text

    def stream(self, prompt: str) -> Iterator[str]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        started_at = perf_counter()
        outcome = "error"
        try:
            with self._stream_function(
                "POST",
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt.strip(),
                    "stream": True,
                    "keep_alive": self._keep_alive,
                    "options": self._options,
                },
                timeout=self._timeout_seconds,
            ) as response:
                response.raise_for_status()
                received_text = False
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise RuntimeError(
                            "Ollama returned an invalid streaming response"
                        ) from error
                    if not isinstance(payload, dict):
                        raise RuntimeError("Ollama returned an unexpected streaming response")
                    error_message = payload.get("error")
                    if isinstance(error_message, str) and error_message:
                        raise RuntimeError(error_message)
                    chunk = payload.get("response")
                    if chunk is not None and not isinstance(chunk, str):
                        raise RuntimeError("Ollama returned an invalid streaming token")
                    if chunk:
                        received_text = True
                        yield chunk
                    if payload.get("done") is True:
                        self._log_generation(payload)
                if not received_text:
                    raise RuntimeError("Ollama returned an empty response")
                outcome = "ok"
        except GeneratorExit:
            outcome = "cancelled"
            raise
        finally:
            OLLAMA_REQUESTS.labels(operation="stream", outcome=outcome).inc()
            OLLAMA_REQUEST_DURATION.labels(operation="stream").observe(perf_counter() - started_at)

    @staticmethod
    def _log_generation(payload: JsonResponse) -> None:
        summary = observe_generation(payload)
        if summary:
            logger.info("Ollama generation finished", extra=summary)

    @staticmethod
    def _read_payload(response: httpx.Response) -> JsonResponse:
        try:
            payload = response.json()
        except ValueError as error:
            raise RuntimeError("Ollama returned an invalid JSON response") from error

        if not isinstance(payload, dict):
            raise RuntimeError("Ollama returned an unexpected JSON response")

        return payload
