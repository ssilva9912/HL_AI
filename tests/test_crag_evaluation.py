import json
import math
from pathlib import Path

import pytest

from evaluation.crag import (
    CragCase,
    Observation,
    build_report,
    contains_keywords,
    fit_thresholds,
    grade_by_score,
    load_cases,
    summarize_answers,
    summarize_grader,
    summarize_web,
)
from evaluation.dataset import EvaluationDatasetError

BUNDLED_DATASET = Path(__file__).resolve().parent.parent / "evaluation" / "crag_questions.json"


def observe(
    expected: str,
    verdict: str | None,
    score: float | None = None,
    *,
    answer: str = "",
    web_used: bool = False,
    keywords: tuple[str, ...] = (),
    web_keywords: tuple[str, ...] = (),
    elapsed_ms: float = 100.0,
) -> Observation:
    return Observation(
        case=CragCase(
            question="q",
            expected_verdict=expected,
            answer_keywords=keywords,
            web_answer_keywords=web_keywords,
        ),
        verdict=verdict,
        local_score=score,
        web_used=web_used,
        answer=answer,
        elapsed_ms=elapsed_ms,
    )


def test_bundled_dataset_loads_with_every_verdict_represented() -> None:
    cases = load_cases(BUNDLED_DATASET)

    assert {case.expected_verdict for case in cases} == {"correct", "ambiguous", "incorrect"}
    assert len({case.question for case in cases}) == len(cases)


def test_load_cases_reads_keywords_and_category(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": " Why? ",
                    "expected_verdict": "correct",
                    "answer_keywords": ["a|b"],
                    "web_answer_keywords": ["c"],
                    "category": "documented",
                }
            ]
        ),
        encoding="utf-8",
    )

    assert load_cases(path) == [
        CragCase("Why?", "correct", ("a|b",), ("c",), "documented"),
    ]


@pytest.mark.parametrize(
    "entry",
    [
        "text",
        {"question": "", "expected_verdict": "correct"},
        {"question": "q", "expected_verdict": "maybe"},
        {"question": "q", "expected_verdict": "correct", "answer_keywords": "abc"},
        {"question": "q", "expected_verdict": "correct", "answer_keywords": [""]},
    ],
)
def test_load_cases_rejects_malformed_entries(tmp_path: Path, entry: object) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises(EvaluationDatasetError):
        load_cases(path)


def test_load_cases_rejects_missing_and_empty_datasets(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "missing.json")

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(EvaluationDatasetError):
        load_cases(empty)


def test_keywords_all_required_with_alternatives_and_case_insensitive() -> None:
    answer = "RRF compares rank positions."

    assert contains_keywords(answer, ["rank|score", "rrf"])
    assert not contains_keywords(answer, ["rank", "embedding"])
    assert contains_keywords(answer, [])


def test_grader_summary_counts_confusion_and_critical_failures() -> None:
    summary = summarize_grader(
        [
            observe("correct", "correct"),
            observe("correct", "incorrect"),
            observe("incorrect", "incorrect"),
            observe("incorrect", "correct"),
            observe("incorrect", "incorrect"),
            observe("ambiguous", "ambiguous"),
            observe("ambiguous", None),
        ]
    )

    assert summary.count == 7
    assert summary.accuracy == pytest.approx(4 / 7)
    assert summary.confusion["correct"] == {"correct": 1, "incorrect": 1}
    assert summary.confusion["ambiguous"] == {"ambiguous": 1, "error": 1}
    assert summary.recall == {
        "correct": 0.5,
        "ambiguous": 0.5,
        "incorrect": pytest.approx(2 / 3),
    }
    assert summary.covered_but_discarded == 0.5
    assert summary.uncovered_but_trusted == pytest.approx(1 / 3)
    assert len(summary.mismatches) == 3


def test_failure_rates_are_not_a_number_without_that_class() -> None:
    summary = summarize_grader([observe("ambiguous", "ambiguous")])

    assert math.isnan(summary.covered_but_discarded)
    assert math.isnan(summary.uncovered_but_trusted)


def test_answer_summary_scores_keywords_and_abstention() -> None:
    summary = summarize_answers(
        [
            observe("correct", "correct", answer="It uses rank fusion.", keywords=("rank",)),
            observe("correct", "correct", answer="Not sure.", keywords=("rank",)),
            observe("incorrect", "incorrect", answer="I do not know."),
            observe("incorrect", "incorrect", answer="It is Canberra."),
        ]
    )

    assert summary.keyword_hit_rate == 0.5
    assert summary.keyword_cases == 2
    assert summary.abstain_rate_when_uncovered == 0.5
    assert summary.uncovered_cases == 2
    assert summary.correct_verdicts == 2
    assert summary.abstained_despite_correct_verdict == 0


def test_answer_summary_counts_abstentions_the_grader_did_not_catch() -> None:
    summary = summarize_answers(
        [
            observe("ambiguous", "correct", answer="I don't know which model it uses."),
            observe("correct", "correct", answer="BM25 matches exact terms."),
            observe("incorrect", "incorrect", answer="I do not know."),
        ]
    )

    assert summary.correct_verdicts == 2
    assert summary.abstained_despite_correct_verdict == 1


def test_answer_summary_has_no_rates_without_applicable_questions() -> None:
    summary = summarize_answers([observe("ambiguous", "ambiguous")])

    assert summary.keyword_hit_rate is None
    assert summary.abstain_rate_when_uncovered is None


def test_web_summary_reports_fallback_rate_by_expected_verdict() -> None:
    summary = summarize_web(
        [
            observe("correct", "correct", web_used=False),
            observe(
                "incorrect",
                "incorrect",
                answer="Canberra",
                web_used=True,
                web_keywords=("canberra",),
            ),
            observe("incorrect", "incorrect", answer="I don't know", web_used=False),
        ]
    )

    assert summary.web_used_rate == {"correct": 0.0, "incorrect": 0.5}
    assert summary.keyword_hit_rate == 1.0
    assert summary.still_abstained_when_uncovered == 0.5


@pytest.mark.parametrize(
    ("score", "expected"),
    [(None, "incorrect"), (-6.0, "incorrect"), (-1.0, "ambiguous"), (2.0, "correct")],
)
def test_grade_by_score_matches_the_live_evaluator_boundaries(
    score: float | None, expected: str
) -> None:
    assert grade_by_score(score, correct=2.0, incorrect=-6.0) == expected


def test_threshold_fit_recovers_separating_thresholds() -> None:
    observations = [
        observe("correct", "ambiguous", 5.0),
        observe("correct", "ambiguous", 7.0),
        observe("ambiguous", "ambiguous", -2.0),
        observe("ambiguous", "ambiguous", 0.5),
        observe("incorrect", "ambiguous", -9.0),
        observe("incorrect", "ambiguous", -11.0),
    ]

    best, current = fit_thresholds(observations, current=(6.0, -1.0))

    assert best is not None and current is not None
    assert best.accuracy == 1.0
    assert -9.0 < best.incorrect_threshold < -2.0
    assert 0.5 < best.correct_threshold < 5.0
    assert current.accuracy < 1.0


def test_threshold_fit_needs_scores() -> None:
    assert fit_thresholds([observe("correct", "correct")]) == (None, None)


def test_report_warns_about_small_question_sets() -> None:
    report = build_report(
        [observe("correct", "correct", 4.0), observe("incorrect", "incorrect", -9.0)]
    )

    assert report.notes
    assert report.web is None
    assert report.latency["correct"].count == 1
