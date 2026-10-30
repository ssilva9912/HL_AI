import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from backend.api.rag_service import HomelabRAGService
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.parser import ParsedDocument
from backend.interfaces.retriever import RetrievalResult
from backend.rag.citations import citation_numbers
from backend.rag.corrective import QueryRewriter
from backend.rag.prompt_builder import PromptBuilder
from tests.test_crag import FakeGenerator


def passage(
    content: str,
    name: str,
    *,
    url: str | None = None,
    chunk_index: int = 0,
) -> RetrievalResult:
    path = Path("web" if url else "docs") / name
    document = ParsedDocument(
        source_path=path,
        file_type="web" if url else ".txt",
        content=content,
        metadata=FileMetadata(name=name, path=path, extension=".txt", size_bytes=len(content)),
        url=url,
    )
    chunk = DocumentChunk(
        source_document=document,
        content=content,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(content),
    )
    return RetrievalResult(chunk=chunk, score=1.0, retriever="test")


RESULTS = [
    passage("first page, part one", "a.example/page", url="https://a.example/page"),
    passage("a file", "notes.txt"),
    passage("first page, part two", "a.example/page", url="https://a.example/page", chunk_index=1),
    passage("second page", "b.example/other", url="https://b.example/other"),
]


def test_passages_from_one_document_share_a_citation_number() -> None:
    assert citation_numbers(RESULTS) == [1, 2, 1, 3]


def test_citation_numbers_for_nothing_is_empty() -> None:
    assert citation_numbers([]) == []


def test_prompt_labels_passages_by_document_number_and_asks_for_numbered_citations() -> None:
    prompt = PromptBuilder().build("What is on the pages?", RESULTS)

    assert prompt.count("[Source 1]") == 2
    assert "[Source 2]" in prompt
    assert "[Source 3]" in prompt
    assert "[Source 4]" not in prompt
    assert "by number" in prompt
    assert "[filename]" not in prompt
    assert "[fusion.txt]" not in prompt


def test_service_sources_carry_their_citation_number_and_web_address() -> None:
    sources = HomelabRAGService._normalize_sources(RESULTS)

    assert [source.citation for source in sources] == [1, 2, 1, 3]
    assert [source.url for source in sources] == [
        "https://a.example/page",
        None,
        "https://a.example/page",
        "https://b.example/other",
    ]
    assert sources[1].document == "notes.txt"


# --- query rewriting -------------------------------------------------------


def test_rewriter_removes_a_year_the_user_never_mentioned() -> None:
    rewriter = QueryRewriter(FakeGenerator('sentiment towards "The Godfather" movie 2023'))

    assert (
        rewriter.rewrite("i meant about the godfather", [("user", "what is the movie about?")])
        == 'sentiment towards "The Godfather" movie'
    )


def test_rewriter_keeps_a_year_that_appears_in_the_conversation() -> None:
    rewriter = QueryRewriter(FakeGenerator("python release schedule 2024"))

    result = rewriter.rewrite("and the next one?", [("user", "python releases in 2024")])

    assert result == "python release schedule 2024"


def test_rewriter_falls_back_when_only_an_invented_year_remains() -> None:
    assert QueryRewriter(FakeGenerator("2023")).rewrite("original question") == "original question"


def test_rewriter_prompt_asks_to_carry_the_request_over_a_correction() -> None:
    generator = FakeGenerator("what people think of the godfather")
    history = [
        ("user", "tell me what the overall sentiment of the public is"),
        ("assistant", "I don't know. " + "plot " * 200),
    ]

    QueryRewriter(generator).rewrite("i meant about the godfather", history)

    prompt = generator.prompts[0]
    assert "carry that request over" in prompt
    assert "Never add a year" in prompt
    assert "User: tell me what the overall sentiment of the public is" in prompt
    assert "LATEST MESSAGE\ni meant about the godfather" in prompt
    assistant_line = next(
        line for line in prompt.splitlines() if line.startswith("Assistant: I don")
    )
    assert len(assistant_line) < 200


# --- frontend --------------------------------------------------------------


def load_frontend_module(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "frontend" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


citations = load_frontend_module("citations")
notes = load_frontend_module("retrieval_notes")


def source(document: str, number: int | None, url: str | None = None, text: str = "t") -> dict:
    return {"document": document, "citation": number, "url": url, "text": text, "score": 1.0}


def test_sources_are_grouped_into_one_entry_per_document() -> None:
    grouped = citations.group_citations(
        [
            source("a.example/page", 1, "https://a.example/page", "one"),
            source("notes.txt", 2),
            source("a.example/page", 1, "https://a.example/page", "two"),
        ]
    )

    assert [(c.number, c.label, c.url) for c in grouped] == [
        (1, "a.example/page", "https://a.example/page"),
        (2, "notes.txt", None),
    ]
    assert [p["text"] for p in grouped[0].passages] == ["one", "two"]


def test_sources_saved_before_numbering_get_numbers_in_order() -> None:
    grouped = citations.group_citations([source("a.txt", None), source("b.txt", None)])

    assert [(c.number, c.label) for c in grouped] == [(1, "a.txt"), (2, "b.txt")]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://a.example/x", "https://a.example/x"),
        ("http://a.example/x_(y)", "http://a.example/x_%28y%29"),
        ("https://a.example/a b", "https://a.example/a%20b"),
        ("javascript:alert(1)", None),
        ("ftp://a.example/file", None),
        ("https://", None),
        (None, None),
        (42, None),
    ],
)
def test_only_plain_web_addresses_become_links(url: object, expected: str | None) -> None:
    assert citations.safe_url(url) == expected


def test_source_lines_link_web_pages_and_leave_files_plain() -> None:
    web, local = citations.group_citations(
        [source("a.example/page", 1, "https://a.example/page"), source("notes [v2].txt", 2)]
    )

    assert citations.format_citation(web) == "**[1]** [a.example/page](https://a.example/page)"
    assert citations.format_citation(local) == "**[2]** notes \\[v2\\].txt"


def test_inline_markers_link_to_their_web_source() -> None:
    grouped = citations.group_citations(
        [source("a.example/page", 1, "https://a.example/page"), source("notes.txt", 2)]
    )

    text = "It is a film [1]. Also see the file [2], and again [1, 2] or [3]."
    linked = citations.link_citations(text, grouped)

    assert linked.startswith("It is a film [[1]](https://a.example/page).")
    assert "the file [2]," in linked
    assert "[3]." in linked
    assert "[1, 2]" in linked


def test_comma_separated_markers_link_each_number() -> None:
    grouped = citations.group_citations(
        [source("a", 1, "https://a.example/"), source("b", 2, "https://b.example/")]
    )

    assert citations.link_citations("fact [1, 2]", grouped) == (
        "fact [[1]](https://a.example/)[[2]](https://b.example/)"
    )


def test_existing_markdown_links_and_plain_answers_are_untouched() -> None:
    grouped = citations.group_citations([source("a", 1, "https://a.example/")])

    assert citations.link_citations("see [1](https://other.example)", grouped) == (
        "see [1](https://other.example)"
    )
    assert citations.link_citations("no markers here", grouped) == "no markers here"
    assert citations.link_citations("fact [1]", []) == "fact [1]"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("I don't know. The context does not say.", True),
        ("I do not know.", True),
        ("  **I don’t know** which model it is", True),
        ("The Godfather is a film. I don't know more.", False),
        ("", False),
    ],
)
def test_abstentions_are_recognised_from_how_the_answer_opens(answer: str, expected: bool) -> None:
    assert notes.looks_like_abstention(answer) is expected


def test_prompt_spells_out_what_a_short_follow_up_means() -> None:
    builder = PromptBuilder()

    restated = builder.build("i meant about the godfather", RESULTS, restated="godfather opinion")
    same = builder.build("godfather opinion", RESULTS, restated="Godfather opinion")
    none = builder.build("i meant about the godfather", RESULTS)

    assert "the user is asking about: godfather opinion" in restated
    assert "the user is asking about" not in same
    assert "the user is asking about" not in none
