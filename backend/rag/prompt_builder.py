from collections.abc import Sequence

from backend.interfaces.retriever import RetrievalResult


class PromptBuilder:
    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are Homelab AI, a local retrieval-augmented assistant. "
        "Answer the question using only the supplied context. "
        "Do not use outside knowledge. "
        "You may explain implications and make reasonable conclusions when they "
        "follow directly from the supplied context. "
        "For questions asking why, explain how the mechanism described in the "
        "context provides a benefit or solves the stated problem. "
        "Do not require the context to contain the exact wording of the question "
        "when the answer can be logically derived from it. "
        "Do not introduce facts that are not supported by the context. "
        "If the context truly lacks enough information to answer, say that you "
        "do not know. "
        "Answer directly and concisely. "
        "Use inline source citations in the format [filename]. "
        "Do not add a separate source list unless the user asks for one."
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
        history: Sequence[tuple[str, str]] = (),
    ) -> str:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("question must not be empty")

        context = self._build_context(results)
        conversation = self._build_history(history)

        return (
            f"{self._system_instruction}\n\n"
            "CONVERSATION HISTORY\n"
            "====================\n"
            f"{conversation}\n\n"
            "CONTEXT\n"
            "=======\n"
            f"{context}\n\n"
            "QUESTION\n"
            "========\n"
            f"{normalized_question}\n\n"
            "RESPONSE REQUIREMENTS\n"
            "=====================\n"
            "- Answer the question directly.\n"
            "- Base every claim on the supplied context.\n"
            "- For a why question, connect the described mechanism to its benefit.\n"
            "- Cite supporting filenames inline, such as [fusion.txt].\n"
            "- Do not repeat the full context.\n"
            "- Do not include a separate source-filenames section.\n\n"
            "ANSWER\n"
            "======\n"
        )

    @staticmethod
    def _build_history(
        history: Sequence[tuple[str, str]],
    ) -> str:
        if not history:
            return "[No prior conversation.]"

        lines: list[str] = []
        for role, content in history:
            normalized_role = role.strip().lower()
            normalized_content = content.strip()
            if normalized_role not in {"user", "assistant"} or not normalized_content:
                continue
            label = "User" if normalized_role == "user" else "Assistant"
            lines.append(f"{label}: {normalized_content}")

        return "\n".join(lines) or "[No prior conversation.]"

    def build_general(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> str:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        return (
            "You are Homelab AI, a helpful local assistant. "
            "Answer using your general model knowledge and clearly acknowledge "
            "uncertainty when appropriate. Do not invent document citations. "
            "Answer the current question directly. Use prior conversation only "
            "when it materially helps answer the current question. Do not critique, "
            "correct, or revisit an earlier answer unless the user explicitly asks "
            "you to do so. Treat user-provided and document-grounded details in the "
            "conversation as context, including hypothetical or fictional details.\n\n"
            "CONVERSATION HISTORY\n"
            "====================\n"
            f"{self._build_history(history)}\n\n"
            "QUESTION\n"
            "========\n"
            f"{normalized_question}\n\n"
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
                f"Relevance score: {result.score:.4f}\n"
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
