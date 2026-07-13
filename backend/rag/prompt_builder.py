from collections.abc import Sequence

from backend.interfaces.retriever import RetrievalResult


class PromptBuilder:
    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are Homelab AI, a local retrieval-augmented assistant. "
        "Answer the question using only the supplied context. "
        "Do not use outside knowledge. "
        "Use reasonable conclusions that follow directly from the context, "
        "but do not introduce unsupported facts. "
        "If the context truly does not contain enough information, say that "
        "you do not know. "
        "Answer directly and concisely. "
        "Cite the source filenames used in the answer."
    )

    def __init__(
        self,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        max_context_chars: int = 6000,
    ) -> None:
        if not system_instruction.strip():
            raise ValueError("system_instruction must not be empty")

        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")

        self._system_instruction = system_instruction.strip()
        self._max_context_chars = max_context_chars

    def build(
        self,
        question: str,
        results: Sequence[RetrievalResult],
    ) -> str:
        if not question.strip():
            raise ValueError("question must not be empty")

        context = self._build_context(results)

        return (
            f"{self._system_instruction}\n\n"
            "CONTEXT\n"
            "=======\n"
            f"{context}\n\n"
            "QUESTION\n"
            "========\n"
            f"{question.strip()}\n\n"
            "ANSWER\n"
            "======\n"
        )

    def _build_context(
        self,
        results: Sequence[RetrievalResult],
    ) -> str:
        if not results:
            return "[No context was retrieved.]"

        sections: list[str] = []
        current_length = 0

        for position, result in enumerate(results, start=1):
            chunk = result.chunk
            source_name = chunk.source_document.metadata.name

            section = (
                f"[Source {position}]\n"
                f"File: {source_name}\n"
                f"Chunk: {chunk.chunk_index}\n"
                f"Content:\n{chunk.content.strip()}"
            )

            separator_length = 2 if sections else 0
            projected_length = current_length + separator_length + len(section)

            if projected_length > self._max_context_chars:
                remaining = self._max_context_chars - current_length - separator_length

                if remaining <= 0:
                    break

                truncated_section = section[:remaining].rstrip()

                if truncated_section:
                    sections.append(truncated_section)

                break

            sections.append(section)
            current_length = projected_length

        if not sections:
            return "[No context fit within the configured context limit.]"

        return "\n\n".join(sections)
