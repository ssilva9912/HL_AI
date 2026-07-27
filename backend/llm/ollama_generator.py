import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Any

import httpx

from backend.config import get_settings

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
        self._post_function = post_function or httpx.post
        self._stream_function = stream_function or httpx.stream

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = self._post_function(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt.strip(),
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                },
            },
            timeout=self._timeout_seconds,
        )

        response.raise_for_status()

        payload = self._read_payload(response)
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

        with self._stream_function(
            "POST",
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt.strip(),
                "stream": True,
                "options": {
                    "temperature": self._temperature,
                },
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
                    raise RuntimeError("Ollama returned an invalid streaming response") from error
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
            if not received_text:
                raise RuntimeError("Ollama returned an empty response")

    @staticmethod
    def _read_payload(response: httpx.Response) -> JsonResponse:
        try:
            payload = response.json()
        except ValueError as error:
            raise RuntimeError("Ollama returned an invalid JSON response") from error

        if not isinstance(payload, dict):
            raise RuntimeError("Ollama returned an unexpected JSON response")

        return payload
