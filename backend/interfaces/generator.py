from typing import Protocol


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...
