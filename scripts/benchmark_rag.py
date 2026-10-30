"""
Measure question-answering latency against a running Homelab AI API.

    uv run python scripts/benchmark_rag.py --runs 12
    uv run python scripts/benchmark_rag.py --runs 12 --json before.json

Each question is asked once, then repeated, so the report separates cold starts,
distinct questions, and repeats that should be served from the caches.
"""

import argparse
import json
import statistics
import sys
from time import perf_counter

import httpx

QUESTIONS = (
    "Why does Homelab AI use Reciprocal Rank Fusion instead of averaging scores?",
    "What is BM25 useful for?",
    "How does dense retrieval differ from lexical retrieval?",
    "What does the cross-encoder reranker do after rank fusion?",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def ask(client: httpx.Client, url: str, question: str, top_k: int) -> tuple[float, dict]:
    started_at = perf_counter()
    response = client.post(f"{url}/search", json={"question": question, "top_k": top_k})
    wall_ms = (perf_counter() - started_at) * 1_000
    response.raise_for_status()
    return wall_ms, response.json()["metadata"]


def summarize(label: str, samples: list[float]) -> dict[str, float]:
    summary = {
        "n": len(samples),
        "mean": statistics.fmean(samples),
        "p50": percentile(samples, 0.5),
        "p95": percentile(samples, 0.95),
        "max": max(samples),
    }
    print(
        f"{label:<22} n={summary['n']:<3} mean={summary['mean']:>8,.0f} ms  "
        f"p50={summary['p50']:>8,.0f} ms  p95={summary['p95']:>8,.0f} ms  "
        f"max={summary['max']:>8,.0f} ms"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=12, help="total requests after the first")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json", help="write raw results to this file")
    args = parser.parse_args()

    runs: list[dict] = []
    with httpx.Client(timeout=args.timeout) as client:
        plan = [QUESTIONS[i % len(QUESTIONS)] for i in range(args.runs + 1)]
        seen: set[str] = set()
        print(f"{'#':>3}  {'kind':<9} {'wall ms':>9}  stages (ms)")
        for number, question in enumerate(plan):
            kind = "cold" if number == 0 else ("repeat" if question in seen else "new")
            seen.add(question)
            wall_ms, metadata = ask(client, args.url, question, args.top_k)
            stages = metadata.get("stages_ms", {})
            runs.append({"kind": kind, "wall_ms": wall_ms, "stages_ms": stages})
            breakdown = "  ".join(f"{name}={value:,.0f}" for name, value in sorted(stages.items()))
            print(f"{number:>3}  {kind:<9} {wall_ms:>9,.0f}  {breakdown}")

    print()
    report: dict[str, dict[str, float]] = {}
    for kind in ("cold", "new", "repeat"):
        samples = [run["wall_ms"] for run in runs if run["kind"] == kind]
        if samples:
            report[kind] = summarize(kind, samples)
    warm = [run["wall_ms"] for run in runs if run["kind"] != "cold"]
    if warm:
        report["warm"] = summarize("warm (new + repeat)", warm)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"runs": runs, "summary": report}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
