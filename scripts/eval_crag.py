"""
Measure Corrective RAG against a running Homelab AI API.

    uv run python scripts/eval_crag.py
    uv run python scripts/eval_crag.py --web --json after.json

The default pass asks every question with web search off, so nothing leaves your machine. It
checks the grader (did the right questions come back correct, ambiguous or incorrect?) and the
answers. --web adds a second pass with web search allowed to measure the fallback; those
questions are sent to the configured search engine.

Edit evaluation/crag_questions.json to match what your own documents cover.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.crag import (  # noqa: E402
    VERDICTS,
    CragCase,
    CragReport,
    Observation,
    build_report,
    load_cases,
)

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "evaluation" / "crag_questions.json"


def ask(client: httpx.Client, url: str, case: CragCase, top_k: int, web: bool) -> Observation:
    started_at = perf_counter()
    response = client.post(
        f"{url}/search",
        json={"question": case.question, "top_k": top_k, "web": web},
    )
    wall_ms = (perf_counter() - started_at) * 1_000
    response.raise_for_status()
    body = response.json()
    correction = body["metadata"].get("correction") or {}

    return Observation(
        case=case,
        verdict=correction.get("verdict"),
        local_score=correction.get("local_score"),
        web_used=bool(correction.get("web_used")),
        answer=body["answer"],
        elapsed_ms=wall_ms,
    )


def run_pass(
    client: httpx.Client, url: str, cases: list[CragCase], top_k: int, web: bool
) -> list[Observation]:
    label = "web" if web else "local"
    observations = []
    for number, case in enumerate(cases, start=1):
        observation = ask(client, url, case, top_k, web)
        observations.append(observation)
        mark = "ok " if observation.verdict == case.expected_verdict else "MISS"
        score = "   n/a" if observation.local_score is None else f"{observation.local_score:6.2f}"
        print(
            f"[{label} {number:>2}/{len(cases)}] {mark} expected={case.expected_verdict:<9} "
            f"got={observation.verdict or 'error':<9} score={score}  {case.question[:60]}"
        )
    return observations


def percent(value: float | None) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.0%}"


def print_report(report: CragReport) -> None:
    grader = report.grader
    print(f"\n== Grader ({grader.count} questions)")
    print(f"accuracy: {percent(grader.accuracy)}")
    print(
        f"{'expected \\ got':<16}" + "".join(f"{verdict:>11}" for verdict in (*VERDICTS, "error"))
    )
    for expected in VERDICTS:
        row = grader.confusion[expected]
        print(
            f"{expected:<16}"
            + "".join(f"{row.get(verdict, 0):>11}" for verdict in (*VERDICTS, "error"))
        )
    print(
        "recall: "
        + ", ".join(f"{verdict} {percent(value)}" for verdict, value in grader.recall.items())
    )
    print(f"covered but discarded (correct -> incorrect): {percent(grader.covered_but_discarded)}")
    print(f"uncovered but trusted (incorrect -> correct): {percent(grader.uncovered_but_trusted)}")

    if grader.mismatches:
        print("\nmisses (expected -> got, best document score):")
        for item in grader.mismatches:
            score = "n/a" if item.local_score is None else f"{item.local_score:.2f}"
            print(
                f"  {item.case.expected_verdict} -> {item.verdict or 'error'}"
                f"  {score:>6}  {item.case.question}"
            )

    answers = report.answers
    print("\n== Answers (web off)")
    print(
        f"keyword hit rate: {percent(answers.keyword_hit_rate)} of {answers.keyword_cases} "
        f"documented questions"
    )
    print(
        f"abstains when uncovered: {percent(answers.abstain_rate_when_uncovered)} "
        f"of {answers.uncovered_cases}"
    )
    print(
        f'said "I don\'t know" although graded correct: '
        f"{answers.abstained_despite_correct_verdict} of {answers.correct_verdicts} "
        "(the web is never tried for these)"
    )

    print("\n== Latency by verdict (ms)")
    for verdict, latency in report.latency.items():
        print(
            f"{verdict:<10} n={latency.count:<3} p50={latency.p50_ms:>8,.0f}  "
            f"p95={latency.p95_ms:>8,.0f}"
        )

    if report.web is not None:
        web = report.web
        print(f"\n== Web fallback ({web.count} questions, web allowed)")
        for expected, rate in web.web_used_rate.items():
            print(f"web used when expected {expected:<9}: {percent(rate)}")
        print(
            f"keyword hit rate: {percent(web.keyword_hit_rate)} of {web.keyword_cases} "
            f"questions with a reference keyword"
        )
        print(f"still abstains when uncovered: {percent(web.still_abstained_when_uncovered)}")

    if report.thresholds and report.current_thresholds:
        best, current = report.thresholds, report.current_thresholds
        print("\n== Thresholds (replayed from stored scores)")
        print(
            f"current  correct>={current.correct_threshold:.2f} "
            f"incorrect<={current.incorrect_threshold:.2f}: {percent(current.accuracy)}"
        )
        print(
            f"best fit correct>={best.correct_threshold:.2f} "
            f"incorrect<={best.incorrect_threshold:.2f}: {percent(best.accuracy)}"
        )
    for note in report.notes:
        print(f"note: {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--web", action="store_true", help="also run a pass with web search allowed"
    )
    parser.add_argument("--json", help="write the observations and report to this file")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        help="exit with status 1 when grader accuracy falls below this fraction",
    )
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    with httpx.Client(timeout=args.timeout) as client:
        local = run_pass(client, args.url, cases, args.top_k, web=False)
        web = run_pass(client, args.url, cases, args.top_k, web=True) if args.web else []

    report = build_report(local, web)
    print_report(report)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "local": [asdict(item) for item in local],
                    "web": [asdict(item) for item in web],
                    "report": asdict(report),
                },
                handle,
                indent=2,
            )

    if args.min_accuracy is not None and report.grader.accuracy < args.min_accuracy:
        print(f"\naccuracy {percent(report.grader.accuracy)} is below {args.min_accuracy:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
