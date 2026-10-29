from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.dataset import EvaluationDatasetError

VERDICTS = ("correct", "ambiguous", "incorrect")

ABSTAIN_PATTERN = re.compile(
    r"\b(i do not know|i don't know|i dont know|not enough information|cannot answer)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CragCase:
    question: str
    expected_verdict: str
    answer_keywords: tuple[str, ...] = ()
    web_answer_keywords: tuple[str, ...] = ()
    category: str = ""


@dataclass(frozen=True)
class Observation:
    case: CragCase
    verdict: str | None
    local_score: float | None
    web_used: bool
    answer: str
    elapsed_ms: float

    @property
    def abstained(self) -> bool:
        return bool(ABSTAIN_PATTERN.search(self.answer))


@dataclass(frozen=True)
class ThresholdFit:
    correct_threshold: float
    incorrect_threshold: float
    accuracy: float


@dataclass(frozen=True)
class GraderSummary:
    count: int
    accuracy: float
    recall: dict[str, float]
    confusion: dict[str, dict[str, int]]
    covered_but_discarded: float
    uncovered_but_trusted: float
    mismatches: tuple[Observation, ...]


@dataclass(frozen=True)
class AnswerSummary:
    keyword_hit_rate: float | None
    keyword_cases: int
    abstain_rate_when_uncovered: float | None
    uncovered_cases: int
    abstained_despite_correct_verdict: int = 0
    correct_verdicts: int = 0


@dataclass(frozen=True)
class WebSummary:
    count: int
    web_used_rate: dict[str, float]
    keyword_hit_rate: float | None
    keyword_cases: int
    still_abstained_when_uncovered: float | None


@dataclass(frozen=True)
class LatencySummary:
    count: int
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class CragReport:
    grader: GraderSummary
    answers: AnswerSummary
    latency: dict[str, LatencySummary]
    web: WebSummary | None = None
    thresholds: ThresholdFit | None = None
    current_thresholds: ThresholdFit | None = None
    notes: list[str] = field(default_factory=list)


def load_cases(path: str | Path) -> list[CragCase]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"CRAG evaluation dataset does not exist: {dataset_path}")

    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationDatasetError(f"invalid JSON in {dataset_path}") from exc

    if not isinstance(raw, list) or not raw:
        raise EvaluationDatasetError("CRAG dataset must be a non-empty JSON list")

    return [_parse_case(entry, index) for index, entry in enumerate(raw)]


def _parse_case(entry: Any, index: int) -> CragCase:
    if not isinstance(entry, dict):
        raise EvaluationDatasetError(f"entry {index} must be a JSON object")

    question = entry.get("question")
    if not isinstance(question, str) or not question.strip():
        raise EvaluationDatasetError(f"entry {index} must contain a non-empty question")

    expected = entry.get("expected_verdict")
    if expected not in VERDICTS:
        raise EvaluationDatasetError(f"entry {index} expected_verdict must be one of {VERDICTS}")

    return CragCase(
        question=question.strip(),
        expected_verdict=expected,
        answer_keywords=_parse_keywords(entry, "answer_keywords", index),
        web_answer_keywords=_parse_keywords(entry, "web_answer_keywords", index),
        category=str(entry.get("category", "")),
    )


def _parse_keywords(entry: dict[str, Any], key: str, index: int) -> tuple[str, ...]:
    value = entry.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise EvaluationDatasetError(f"entry {index} {key} must be a list of non-empty strings")
    return tuple(value)


def contains_keywords(answer: str, keywords: Sequence[str]) -> bool:
    """Every keyword must appear; use "a|b" inside one keyword to accept alternatives."""

    lowered = answer.lower()
    return all(
        any(option.strip().lower() in lowered for option in group.split("|")) for group in keywords
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def grade_by_score(score: float | None, correct: float, incorrect: float) -> str:
    if score is None or score <= incorrect:
        return "incorrect"
    return "correct" if score >= correct else "ambiguous"


def summarize_grader(observations: Sequence[Observation]) -> GraderSummary:
    confusion = {expected: Counter[str]() for expected in VERDICTS}
    for observation in observations:
        confusion[observation.case.expected_verdict][observation.verdict or "error"] += 1

    totals = {expected: sum(confusion[expected].values()) for expected in VERDICTS}
    recall = {
        expected: confusion[expected][expected] / totals[expected]
        for expected in VERDICTS
        if totals[expected]
    }
    correct_total = sum(confusion[expected][expected] for expected in VERDICTS)

    return GraderSummary(
        count=len(observations),
        accuracy=correct_total / len(observations) if observations else math.nan,
        recall=recall,
        confusion={expected: dict(counts) for expected, counts in confusion.items()},
        covered_but_discarded=(
            confusion["correct"]["incorrect"] / totals["correct"] if totals["correct"] else math.nan
        ),
        uncovered_but_trusted=(
            confusion["incorrect"]["correct"] / totals["incorrect"]
            if totals["incorrect"]
            else math.nan
        ),
        mismatches=tuple(o for o in observations if o.verdict != o.case.expected_verdict),
    )


def summarize_answers(observations: Sequence[Observation]) -> AnswerSummary:
    keyword_cases = [o for o in observations if o.case.answer_keywords]
    hits = sum(1 for o in keyword_cases if contains_keywords(o.answer, o.case.answer_keywords))
    uncovered = [o for o in observations if o.case.expected_verdict == "incorrect"]
    graded_correct = [o for o in observations if o.verdict == "correct"]

    return AnswerSummary(
        keyword_hit_rate=hits / len(keyword_cases) if keyword_cases else None,
        keyword_cases=len(keyword_cases),
        abstain_rate_when_uncovered=(
            sum(1 for o in uncovered if o.abstained) / len(uncovered) if uncovered else None
        ),
        uncovered_cases=len(uncovered),
        abstained_despite_correct_verdict=sum(1 for o in graded_correct if o.abstained),
        correct_verdicts=len(graded_correct),
    )


def summarize_web(observations: Sequence[Observation]) -> WebSummary:
    by_expected: dict[str, list[Observation]] = {expected: [] for expected in VERDICTS}
    for observation in observations:
        by_expected[observation.case.expected_verdict].append(observation)

    keyword_cases = [o for o in observations if o.case.web_answer_keywords]
    hits = sum(1 for o in keyword_cases if contains_keywords(o.answer, o.case.web_answer_keywords))
    uncovered = by_expected["incorrect"]

    return WebSummary(
        count=len(observations),
        web_used_rate={
            expected: sum(1 for o in group if o.web_used) / len(group)
            for expected, group in by_expected.items()
            if group
        },
        keyword_hit_rate=hits / len(keyword_cases) if keyword_cases else None,
        keyword_cases=len(keyword_cases),
        still_abstained_when_uncovered=(
            sum(1 for o in uncovered if o.abstained) / len(uncovered) if uncovered else None
        ),
    )


def summarize_latency(observations: Sequence[Observation]) -> dict[str, LatencySummary]:
    groups: dict[str, list[float]] = {}
    for observation in observations:
        groups.setdefault(observation.verdict or "error", []).append(observation.elapsed_ms)

    return {
        verdict: LatencySummary(
            count=len(values),
            p50_ms=statistics.median(values),
            p95_ms=percentile(values, 0.95),
        )
        for verdict, values in sorted(groups.items())
    }


def fit_thresholds(
    observations: Sequence[Observation],
    current: tuple[float, float] = (2.0, -6.0),
) -> tuple[ThresholdFit | None, ThresholdFit | None]:
    """
    Replay the stored best-document scores under candidate thresholds.

    Returns (best fit, fit at the current thresholds). It only describes this dataset, so a
    small question set can overfit; treat it as a hint, not a verdict.
    """

    scored = [o for o in observations if o.local_score is not None]
    if len(scored) < 2:
        return None, None

    unique = sorted({o.local_score for o in scored if o.local_score is not None})
    midpoints = [(low + high) / 2 for low, high in zip(unique, unique[1:], strict=False)]
    candidates = [unique[0] - 1, *midpoints, unique[-1] + 1]

    def accuracy(correct: float, incorrect: float) -> float:
        hits = sum(
            1
            for o in observations
            if grade_by_score(o.local_score, correct, incorrect) == o.case.expected_verdict
        )
        return hits / len(observations)

    best: ThresholdFit | None = None
    for incorrect in candidates:
        for correct in candidates:
            if correct <= incorrect:
                continue
            score = accuracy(correct, incorrect)
            if best is None or score > best.accuracy:
                best = ThresholdFit(correct, incorrect, score)

    return best, ThresholdFit(current[0], current[1], accuracy(*current))


def build_report(
    local: Sequence[Observation],
    web: Sequence[Observation] = (),
    current_thresholds: tuple[float, float] = (2.0, -6.0),
) -> CragReport:
    best, current = fit_thresholds(local, current_thresholds)
    notes = []
    if len(local) < 30:
        notes.append("Fewer than 30 questions: the rates above are rough.")
    return CragReport(
        grader=summarize_grader(local),
        answers=summarize_answers(local),
        latency=summarize_latency(local),
        web=summarize_web(web) if web else None,
        thresholds=best,
        current_thresholds=current,
        notes=notes,
    )
