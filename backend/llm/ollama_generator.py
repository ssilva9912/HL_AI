from collections.abc import Callable
from typing import Any

import httpx

from backend.config import get_settings

JsonResponse = dict[str, Any]
PostFunction = Callable[..., httpx.Response]


class OllamaGenerator:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float = 0.1,
        post_function: PostFunction | None = None,
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

    @staticmethod
    def _read_payload(response: httpx.Response) -> JsonResponse:
        try:
            payload = response.json()
        except ValueError as error:
            raise RuntimeError("Ollama returned an invalid JSON response") from error

        if not isinstance(payload, dict):
            raise RuntimeError("Ollama returned an unexpected JSON response")

        return payload
