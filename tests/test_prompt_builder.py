from pathlib import Path

import pytest

from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.parser import ParsedDocument
from backend.interfaces.retriever import RetrievalResult
from backend.rag.prompt_builder import PromptBuilder


def make_result(
    content: str,
    filename: str = "source.txt",
    chunk_index: int = 0,
    score: float = 1.0,
) -> RetrievalResult:
    path = Path(filename)

    metadata = FileMetadata(
        name=filename,
        path=path,
        extension=path.suffix.lower(),
        size_bytes=len(content.encode("utf-8")),
    )

    document = ParsedDocument(
        source_path=path,
        file_type=path.suffix.lower(),
        content=content,
        metadata=metadata,
    )

    chunk = DocumentChunk(
        source_document=document,
        content=content,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(content),
    )

    return RetrievalResult(
        chunk=chunk,
        score=score,
        retriever="test",
    )


def test_build_includes_question_and_context() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        question="What does Homelab AI do?",
        results=[
            make_result(
                "Homelab AI retrieves and generates grounded answers.",
            )
        ],
    )

    assert "What does Homelab AI do?" in prompt
    assert "Homelab AI retrieves and generates grounded answers." in prompt
    assert "source.txt" in prompt
    assert "Chunk: 0" in prompt


def test_build_includes_multiple_sources() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        question="How does retrieval work?",
        results=[
            make_result(
                "BM25 performs lexical retrieval.",
                filename="bm25.txt",
                chunk_index=1,
            ),
            make_result(
                "Dense retrieval uses embeddings.",
                filename="dense.txt",
                chunk_index=2,
            ),
        ],
    )

    assert "bm25.txt" in prompt
    assert "dense.txt" in prompt
    assert "BM25 performs lexical retrieval." in prompt
    assert "Dense retrieval uses embeddings." in prompt


def test_empty_results_include_no_context_message() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        question="What is missing?",
        results=[],
    )

    assert "[No context was retrieved.]" in prompt


def test_whitespace_is_removed_from_question() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        question="  What is RRF?  ",
        results=[],
    )

    assert "What is RRF?" in prompt
    assert "  What is RRF?  " not in prompt


def test_context_is_limited() -> None:
    builder = PromptBuilder(max_context_chars=100)

    prompt = builder.build(
        question="What is in the context?",
        results=[
            make_result("A" * 500),
        ],
    )

    context = prompt.split("CONTEXT\n=======\n", maxsplit=1)[1]
    context = context.split("\n\nQUESTION", maxsplit=1)[0]

    assert len(context) <= 100


def test_custom_system_instruction_is_used() -> None:
    builder = PromptBuilder(
        system_instruction="Answer in one sentence.",
    )

    prompt = builder.build(
        question="What is BM25?",
        results=[],
    )

    assert prompt.startswith("Answer in one sentence.")


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "\n\t",
    ],
)
def test_empty_question_raises_value_error(question: str) -> None:
    builder = PromptBuilder()

    with pytest.raises(ValueError):
        builder.build(
            question=question,
            results=[],
        )


@pytest.mark.parametrize(
    ("system_instruction", "max_context_chars"),
    [
        ("", 100),
        ("   ", 100),
        ("Valid instruction", 0),
        ("Valid instruction", -1),
    ],
)
def test_invalid_configuration_raises_value_error(
    system_instruction: str,
    max_context_chars: int,
) -> None:
    with pytest.raises(ValueError):
        PromptBuilder(
            system_instruction=system_instruction,
            max_context_chars=max_context_chars,
        )
