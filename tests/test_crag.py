import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import get_rag_service
from backend.api.rag_service import CorrectionSummary, RAGAnswer
from backend.config import Settings
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.parser import ParsedDocument
from backend.interfaces.retriever import RetrievalResult
from backend.rag.corrective import (
    Corrector,
    KnowledgeRefiner,
    QueryRewriter,
    RetrievalEvaluator,
    Verdict,
)
from backend.rag.pipeline import RAGPipeline
from backend.rag.prompt_builder import PromptBuilder
from backend.retrieval.cross_encoder_reranker import CrossEncoderReranker


def make_result(
    content: str,
    score: float,
    *,
    name: str = "doc.txt",
    file_type: str = ".txt",
    retriever: str = "cross_encoder",
) -> RetrievalResult:
    path = Path(name)
    document = ParsedDocument(
        source_path=path,
        file_type=file_type,
        content=content,
        metadata=FileMetadata(
            name=name,
            path=path,
            extension=path.suffix,
            size_bytes=len(content),
        ),
    )
    chunk = DocumentChunk(
        source_document=document,
        content=content,
        chunk_index=0,
        start_char=0,
        end_char=len(content),
    )
    return RetrievalResult(chunk=chunk, score=score, retriever=retriever)


def web_result(content: str, score: float, name: str = "example.com/page") -> RetrievalResult:
    return make_result(content, score, name=name, file_type="web", retriever="web")


class KeywordScorer:
    """Scores a passage 5 when it mentions 'relevant' and -10 otherwise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [5.0 if "relevant" in passage.lower() else -10.0 for passage in passages]


class PassThroughReranker:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        self.queries.append(query)
        return sorted(results, key=lambda result: -result.score)[:top_k]


class FakeWebRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.queries.append(query)
        return self.results[:top_k]


class FakeGenerator:
    def __init__(self, reply: str | Exception = "Generated answer.") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.results[:top_k]


# --- evaluator -------------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([8.5, -3.0], Verdict.CORRECT),
        ([2.0], Verdict.CORRECT),
        ([-1.9, -7.0], Verdict.AMBIGUOUS),
        ([1.99, -5.99], Verdict.AMBIGUOUS),
        ([-6.0, -11.0], Verdict.INCORRECT),
        ([-11.2], Verdict.INCORRECT),
        ([], Verdict.INCORRECT),
    ],
)
def test_evaluator_grades_on_the_best_score(scores: list[float], expected: Verdict) -> None:
    results = [make_result("text", score) for score in scores]

    assert RetrievalEvaluator(2.0, -6.0).evaluate(results) is expected


def test_evaluator_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="smaller"):
        RetrievalEvaluator(correct_threshold=-1.0, incorrect_threshold=1.0)


# --- refiner ---------------------------------------------------------------


def test_refiner_keeps_only_relevant_sentences_in_their_original_order() -> None:
    content = "First relevant fact. Unrelated aside. Second relevant fact. More noise."
    scorer = KeywordScorer()

    refined = KnowledgeRefiner(scorer).refine("q", [make_result(content, 3.0)])

    assert refined[0].chunk.content == "First relevant fact. Second relevant fact."
    assert refined[0].score == 3.0
    assert len(scorer.calls) == 1


def test_refiner_keeps_the_best_sentence_when_none_pass() -> None:
    class FlatScorer:
        def score(self, query: str, passages: Sequence[str]) -> list[float]:
            return [-9.0, -8.0, -9.5][: len(passages)]

    refined = KnowledgeRefiner(FlatScorer()).refine(
        "q", [make_result("Alpha one. Beta two. Gamma three.", 0.0)]
    )

    assert refined[0].chunk.content == "Beta two."


def test_refiner_leaves_short_passages_alone_without_scoring_them() -> None:
    scorer = KeywordScorer()
    results = [make_result("Just one sentence.", 1.0), make_result("Two. Sentences.", 1.0)]

    refined = KnowledgeRefiner(scorer).refine("q", results)

    assert refined == results
    assert scorer.calls == []


def test_refiner_scores_all_passages_in_one_batch_and_keeps_originals_intact() -> None:
    scorer = KeywordScorer()
    original = make_result("Relevant a. Noise b. Relevant c.", 1.0, name="a.txt")
    other = make_result("Noise x. Relevant y. Noise z.", 1.0, name="b.txt")

    refined = KnowledgeRefiner(scorer).refine("q", [original, other])

    assert len(scorer.calls) == 1
    assert [result.chunk.content for result in refined] == [
        "Relevant a. Relevant c.",
        "Relevant y.",
    ]
    assert original.chunk.content == "Relevant a. Noise b. Relevant c."


# --- rewriter --------------------------------------------------------------


def test_rewriter_returns_the_first_non_blank_line_without_quotes() -> None:
    rewriter = QueryRewriter(FakeGenerator('\n  "python latest stable version"  \nextra words'))

    assert rewriter.rewrite("what is the newest one?") == "python latest stable version"


def test_rewriter_includes_recent_conversation_in_the_prompt() -> None:
    generator = FakeGenerator("python 3.15 release date")
    history = [("user", "Tell me about Python"), ("assistant", "It is a language.")]

    QueryRewriter(generator).rewrite("when was the newest released?", history)

    assert "User: Tell me about Python" in generator.prompts[0]
    assert "when was the newest released?" in generator.prompts[0]


@pytest.mark.parametrize("reply", [RuntimeError("model down"), "", "   \n  "])
def test_rewriter_falls_back_to_the_question(reply: str | Exception) -> None:
    assert QueryRewriter(FakeGenerator(reply)).rewrite("original question") == "original question"


def test_rewriter_caps_the_query_length() -> None:
    assert len(QueryRewriter(FakeGenerator("word " * 200)).rewrite("q")) == 200


# --- corrector -------------------------------------------------------------


def make_corrector(
    web: FakeWebRetriever | None,
    *,
    reranker: PassThroughReranker | None = None,
    rewriter: QueryRewriter | None = None,
) -> Corrector:
    return Corrector(
        evaluator=RetrievalEvaluator(2.0, -6.0),
        refiner=KnowledgeRefiner(KeywordScorer()),
        reranker=reranker or PassThroughReranker(),
        web_retriever=web,
        rewriter=rewriter,
    )


def test_correct_documents_never_touch_the_web() -> None:
    web = FakeWebRetriever([web_result("relevant web text", 4.0)])
    local = [make_result("Relevant fact. Noise. Another relevant fact.", 8.0)]

    correction = make_corrector(web).correct("q", local, top_k=3)

    assert correction.verdict is Verdict.CORRECT
    assert not correction.web_used
    assert web.queries == []
    assert correction.results[0].chunk.content == "Relevant fact. Another relevant fact."


def test_incorrect_documents_are_replaced_by_web_results() -> None:
    web = FakeWebRetriever([web_result("relevant web answer", 4.0)])
    local = [make_result("cooking notes", -11.0, name="local.txt")]
    reranker = PassThroughReranker()
    rewriter = QueryRewriter(FakeGenerator("standalone search query"))

    correction = make_corrector(web, reranker=reranker, rewriter=rewriter).correct(
        "what about it?", local, top_k=3
    )

    assert correction.verdict is Verdict.INCORRECT
    assert correction.web_used
    assert correction.search_query == "standalone search query"
    assert [result.chunk.source_document.file_type for result in correction.results] == ["web"]
    assert web.queries == ["standalone search query"]
    assert reranker.queries == ["standalone search query"]


def test_ambiguous_documents_are_combined_with_web_results_by_score() -> None:
    web = FakeWebRetriever([web_result("relevant web text", 4.0), web_result("noise", -9.0)])
    local = [make_result("A relevant note.", -1.9, name="local.txt")]

    correction = make_corrector(web).correct("q", local, top_k=2)

    assert correction.verdict is Verdict.AMBIGUOUS
    assert correction.web_used
    assert [result.chunk.source_document.metadata.name for result in correction.results] == [
        "example.com/page",
        "local.txt",
    ]


def test_search_query_falls_back_to_the_question_without_a_rewriter() -> None:
    web = FakeWebRetriever([web_result("relevant", 4.0)])

    correction = make_corrector(web).correct("plain question", [], top_k=2)

    assert correction.search_query == "plain question"
    assert web.queries == ["plain question"]


def test_without_web_incorrect_documents_are_returned_as_retrieved() -> None:
    local = [make_result("Noise one. Noise two. Noise three.", -11.0)]

    correction = make_corrector(None).correct("q", local, top_k=3)

    assert correction.verdict is Verdict.INCORRECT
    assert not correction.web_used
    assert correction.search_query is None
    assert correction.results == local


def test_without_web_ambiguous_documents_are_still_refined() -> None:
    local = [make_result("Relevant fact. Noise. Other noise.", -1.0)]

    correction = make_corrector(None).correct("q", local, top_k=3)

    assert correction.verdict is Verdict.AMBIGUOUS
    assert correction.results[0].chunk.content == "Relevant fact."


def test_an_empty_web_search_falls_back_to_the_local_results() -> None:
    local = [make_result("Relevant fact. Noise. Other noise.", -1.0)]

    correction = make_corrector(FakeWebRetriever([])).correct("q", local, top_k=3)

    assert not correction.web_used
    assert correction.results[0].chunk.content == "Relevant fact."


def test_incorrect_documents_with_an_empty_web_search_stay_as_retrieved() -> None:
    local = [make_result("Noise one. Noise two. Noise three.", -11.0)]

    correction = make_corrector(FakeWebRetriever([])).correct("q", local, top_k=3)

    assert not correction.web_used
    assert correction.results == local


def test_corrector_rejects_a_non_positive_candidate_multiplier() -> None:
    with pytest.raises(ValueError, match="candidate_multiplier"):
        Corrector(
            evaluator=RetrievalEvaluator(),
            refiner=KnowledgeRefiner(KeywordScorer()),
            reranker=PassThroughReranker(),
            candidate_multiplier=0,
        )


# --- pipeline --------------------------------------------------------------


def make_pipeline(
    local: list[RetrievalResult],
    web: FakeWebRetriever | None,
    generator: FakeGenerator | None = None,
) -> RAGPipeline:
    return RAGPipeline(
        retriever=FakeRetriever(local),
        prompt_builder=PromptBuilder(),
        generator=generator or FakeGenerator("Answer."),
        top_k=3,
        corrector=make_corrector(web),
    )


def test_pipeline_answers_from_the_web_when_documents_do_not_cover_the_question() -> None:
    pipeline = make_pipeline(
        [make_result("cooking notes", -11.0)],
        FakeWebRetriever([web_result("relevant web answer", 4.0)]),
    )

    response = pipeline.ask("latest python?", hybrid=True)

    assert response.answer_mode == "web"
    assert response.correction is not None and response.correction.web_used
    assert [source.chunk.source_document.file_type for source in response.sources] == ["web"]


def test_pipeline_labels_mixed_sources() -> None:
    pipeline = make_pipeline(
        [make_result("A relevant note.", -1.9, name="local.txt")],
        FakeWebRetriever([web_result("relevant web text", 4.0)]),
    )

    prepared = pipeline.prepare("q", hybrid=False)

    assert prepared.answer_mode == "documents_web"
    assert PromptBuilder.WEB_CONTENT_NOTICE in prepared.prompt


def test_pipeline_stays_in_documents_mode_when_they_are_correct() -> None:
    pipeline = make_pipeline([make_result("A relevant note.", 8.0)], FakeWebRetriever([]))

    prepared = pipeline.prepare("q", hybrid=True)

    assert prepared.answer_mode == "documents"
    assert prepared.correction is not None
    assert prepared.correction.verdict is Verdict.CORRECT


def test_pipeline_falls_back_to_general_mode_when_nothing_relevant_is_found() -> None:
    pipeline = make_pipeline([make_result("noise", -11.0)], None)

    prepared = pipeline.prepare("q", hybrid=True)

    assert prepared.answer_mode == "general"
    assert prepared.sources == []


def test_pipeline_without_a_corrector_behaves_as_before() -> None:
    pipeline = RAGPipeline(
        retriever=FakeRetriever([make_result("A note.", 1.0)]),
        prompt_builder=PromptBuilder(),
        generator=FakeGenerator(),
        top_k=3,
    )

    prepared = pipeline.prepare("q")

    assert prepared.correction is None
    assert prepared.answer_mode == "documents"


# --- cross-encoder scoring -------------------------------------------------


class StubModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.pairs: list[tuple[str, str]] = []

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> list[float]:
        self.pairs = sentences
        return self._scores


def test_reranker_scores_arbitrary_passages() -> None:
    model = StubModel([1.5, -2.0])
    reranker = CrossEncoderReranker(model_factory=lambda name: model)

    assert reranker.score("question", ["one", "two"]) == [1.5, -2.0]
    assert model.pairs == [("question", "one"), ("question", "two")]
    assert reranker.score("question", []) == []


def test_reranker_score_rejects_a_blank_query_and_mismatched_output() -> None:
    reranker = CrossEncoderReranker(model_factory=lambda name: StubModel([1.0]))

    with pytest.raises(ValueError, match="query"):
        reranker.score("  ", ["a"])
    with pytest.raises(RuntimeError, match="different number"):
        reranker.score("q", ["a", "b"])


# --- settings --------------------------------------------------------------


def test_settings_reject_inverted_crag_thresholds() -> None:
    with pytest.raises(ValueError, match="crag_incorrect_threshold"):
        Settings(crag_correct_threshold=-1.0, crag_incorrect_threshold=0.0)


def test_crag_is_on_by_default() -> None:
    assert Settings().crag_enabled is True


# --- API payloads ----------------------------------------------------------


class CorrectedService:
    def ask(self, question: str, top_k: int = 5, web: bool = False) -> RAGAnswer:
        return RAGAnswer(
            answer="ok",
            sources=[],
            correction=CorrectionSummary(
                verdict="incorrect",
                web_used=True,
                search_query="python latest version",
                local_score=-9.5,
            ),
        )


def test_search_reports_the_corrective_decision() -> None:
    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: CorrectedService()

    response = TestClient(app).post("/search", json={"question": "hello", "web": True})

    assert response.status_code == 200
    assert response.json()["metadata"]["correction"] == {
        "verdict": "incorrect",
        "web_used": True,
        "search_query": "python latest version",
        "local_score": -9.5,
    }


def test_search_omits_the_decision_when_there_is_none() -> None:
    class PlainService:
        def ask(self, question: str, top_k: int = 5, web: bool = False) -> RAGAnswer:
            return RAGAnswer(answer="ok", sources=[])

    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: PlainService()

    response = TestClient(app).post("/search", json={"question": "hello"})

    assert response.json()["metadata"]["correction"] is None


# --- frontend wording ------------------------------------------------------


def load_frontend_notes() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "frontend" / "retrieval_notes.py"
    spec = importlib.util.spec_from_file_location("retrieval_notes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("correction", "web_enabled", "expected"),
    [
        (None, True, None),
        ({"verdict": "correct", "web_used": False}, True, None),
        (
            {"verdict": "incorrect", "web_used": True, "search_query": "python version"},
            True,
            "Your documents don't cover this, so I searched the web for “python version”.",
        ),
        (
            {"verdict": "incorrect", "web_used": False},
            False,
            "Your documents don't cover this. Tick “Search the web” to look online.",
        ),
        (
            {"verdict": "incorrect", "web_used": False},
            True,
            "Your documents don't cover this, and the web search found nothing usable.",
        ),
        (
            {"verdict": "ambiguous", "web_used": True, "search_query": "q"},
            True,
            "Your documents only partly cover this, so I also searched the web for “q”.",
        ),
        (
            {"verdict": "ambiguous", "web_used": False},
            False,
            "Your documents only partly cover this. Tick “Search the web” to add online results.",
        ),
    ],
)
def test_frontend_explains_the_decision(
    correction: dict[str, Any] | None,
    web_enabled: bool,
    expected: str | None,
) -> None:
    assert load_frontend_notes().describe_correction(correction, web_enabled) == expected


def test_stream_start_event_carries_the_decision() -> None:
    from collections.abc import Iterator

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.pool import StaticPool

    from backend.api.rag_service import RAGSource, RAGStream
    from backend.database import Base, get_database_session

    class StreamService:
        def stream_answer(self, question: str, **kwargs: Any) -> RAGStream:
            return RAGStream(
                chunks=iter(["web ", "answer"]),
                sources=[RAGSource(text="t", score=1.0, document="example.com/p", chunk_id="0")],
                answer_mode="web",
                correction=CorrectionSummary("incorrect", True, "q"),
            )

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    Base.metadata.create_all(engine)

    def session_override() -> Iterator[Session]:
        with factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_database_session] = session_override
    app.dependency_overrides[get_rag_service] = lambda: StreamService()
    client = TestClient(app)

    conversation_id = client.post("/conversations", json={}).json()["id"]
    with client.stream(
        "POST",
        f"/conversations/{conversation_id}/messages/stream",
        json={"content": "latest python?", "web": True},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines()]

    start = events[0]
    assert start["type"] == "start"
    assert start["answer_mode"] == "web"
    assert start["correction"] == {
        "verdict": "incorrect",
        "web_used": True,
        "search_query": "q",
        "local_score": None,
    }

    saved = client.get(f"/conversations/{conversation_id}").json()["messages"][-1]
    assert saved["answer_mode"] == "web"
    Base.metadata.drop_all(engine)
    engine.dispose()
