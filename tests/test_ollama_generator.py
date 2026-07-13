from typing import Any

import httpx
import pytest

from backend.llm.ollama_generator import OllamaGenerator


def make_response(
    payload: Any,
    status_code: int = 200,
) -> httpx.Response:
    request = httpx.Request(
        method="POST",
        url="http://localhost:11434/api/generate",
    )

    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=request,
    )


def test_generate_returns_model_response() -> None:
    captured_request: dict[str, Any] = {}

    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        captured_request["url"] = url
        captured_request["json"] = json
        captured_request["timeout"] = timeout

        return make_response(
            {
                "response": "Homelab AI produces grounded answers.",
            }
        )

    generator = OllamaGenerator(
        model="test-model",
        base_url="http://localhost:11434/",
        timeout_seconds=30.0,
        temperature=0.2,
        post_function=fake_post,
    )

    result = generator.generate("Explain Homelab AI.")

    assert result == "Homelab AI produces grounded answers."
    assert captured_request["url"] == ("http://localhost:11434/api/generate")
    assert captured_request["timeout"] == 30.0
    assert captured_request["json"] == {
        "model": "test-model",
        "prompt": "Explain Homelab AI.",
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }


def test_generate_strips_prompt_and_response() -> None:
    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        del url
        del timeout

        assert json["prompt"] == "What is RRF?"

        return make_response(
            {
                "response": "  RRF combines ranked results.  ",
            }
        )

    generator = OllamaGenerator(post_function=fake_post)

    result = generator.generate("  What is RRF?  ")

    assert result == "RRF combines ranked results."


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        " ",
        "\n\t",
    ],
)
def test_empty_prompt_raises_value_error(prompt: str) -> None:
    generator = OllamaGenerator(
        post_function=lambda *args, **kwargs: make_response({"response": "unused"})
    )

    with pytest.raises(ValueError, match="prompt must not be empty"):
        generator.generate(prompt)


@pytest.mark.parametrize(
    ("model", "base_url", "timeout_seconds", "temperature"),
    [
        ("", "http://localhost:11434", 10.0, 0.1),
        ("   ", "http://localhost:11434", 10.0, 0.1),
        ("model", "", 10.0, 0.1),
        ("model", "   ", 10.0, 0.1),
        ("model", "http://localhost:11434", 0.0, 0.1),
        ("model", "http://localhost:11434", -1.0, 0.1),
        ("model", "http://localhost:11434", 10.0, -0.1),
    ],
)
def test_invalid_configuration_raises_value_error(
    model: str,
    base_url: str,
    timeout_seconds: float,
    temperature: float,
) -> None:
    with pytest.raises(ValueError):
        OllamaGenerator(
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )


def test_http_error_is_raised() -> None:
    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        del url
        del json
        del timeout

        return make_response(
            {"error": "model not found"},
            status_code=404,
        )

    generator = OllamaGenerator(post_function=fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        generator.generate("Hello")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"response": None},
        {"response": 123},
        {"response": ""},
        {"response": "   "},
    ],
)
def test_invalid_response_field_raises_runtime_error(
    payload: dict[str, Any],
) -> None:
    generator = OllamaGenerator(post_function=lambda *args, **kwargs: make_response(payload))

    with pytest.raises(RuntimeError):
        generator.generate("Hello")


def test_non_dictionary_payload_raises_runtime_error() -> None:
    generator = OllamaGenerator(post_function=lambda *args, **kwargs: make_response(["unexpected"]))

    with pytest.raises(RuntimeError):
        generator.generate("Hello")
