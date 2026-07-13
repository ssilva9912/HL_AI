from collections.abc import Callable
from typing import Any

import httpx

JsonResponse = dict[str, Any]
PostFunction = Callable[..., httpx.Response]


class OllamaGenerator:
    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 120.0,
        temperature: float = 0.1,
        post_function: PostFunction | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")

        if not base_url.strip():
            raise ValueError("base_url must not be empty")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        self._model = model.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
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
