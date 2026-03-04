"""
Evaluation harness for Horeb.

Runs a hardcoded set of eval cases against the live Claude API and prints
a summary table.

Usage:
    uv run python eval/run_eval.py

Requires ANTHROPIC_API_KEY in the environment (or a .env file).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from horeb.engine import analyze
from horeb.errors import HorebError
from horeb.llm import ClaudeProvider
from horeb.schemas import BookAnalysisResult

from eval.metrics import EvalResult, print_summary_table


EVAL_CASES = [
    # Passages
    {"id": "p1", "ref": "John 3:16-21",     "granularity": "passage"},
    {"id": "p2", "ref": "Romans 8:1-11",    "granularity": "passage"},
    {"id": "p3", "ref": "Psalm 23:1-6",     "granularity": "passage"},
    # Chapters
    {"id": "c1", "ref": "1 Corinthians 13", "granularity": "chapter"},
    {"id": "c2", "ref": "John 3",           "granularity": "chapter"},
    # Books
    {"id": "b1", "ref": "Ruth",             "granularity": "book"},
    {"id": "b2", "ref": "Philemon",         "granularity": "book"},
]


class _CountingProvider:
    """Wraps ClaudeProvider to count total LLM calls made per analyze() invocation."""

    def __init__(self) -> None:
        self._inner = ClaudeProvider()
        self.call_count = 0

    def complete(self, system: str, prompt: str, schema=None, max_tokens=None) -> str:
        self.call_count += 1
        return self._inner.complete(
            system=system,
            prompt=prompt,
            schema=schema,
            max_tokens=max_tokens,
        )


def _run_case(case: dict) -> EvalResult:
    llm = _CountingProvider()
    error: str | None = None
    schema_passed = False
    citation_valid = False
    failed_segments = 0

    try:
        result = analyze(case["ref"], llm=llm)
        schema_passed = True
        citation_valid = True  # analyze() raises CitationOutOfRangeError if invalid
        if isinstance(result, BookAnalysisResult):
            failed_segments = len(result.failed_segments)
    except HorebError as exc:
        error = f"{type(exc).__name__}: {exc}"
        # Distinguish schema vs citation failures
        from horeb.errors import CitationOutOfRangeError, AnalysisFailedError
        if isinstance(exc, CitationOutOfRangeError):
            schema_passed = True  # parse succeeded; citation check failed
        elif isinstance(exc, AnalysisFailedError):
            schema_passed = False
    except Exception as exc:
        error = f"Unexpected: {exc}"

    # retry_count = total calls - 1 base call per segment - 1 synthesis call for books
    # For simplicity, report raw call count minus expected baseline
    retry_count = max(0, llm.call_count - 1)

    return EvalResult(
        case_id=case["id"],
        reference=case["ref"],
        granularity=case["granularity"],
        schema_passed=schema_passed,
        retry_count=retry_count,
        citation_valid=citation_valid,
        failed_segments=failed_segments,
        error=error,
    )


def main() -> None:
    print(f"Running {len(EVAL_CASES)} eval cases...\n")
    results: list[EvalResult] = []
    for case in EVAL_CASES:
        print(f"  [{case['id']}] {case['ref']} ...", end=" ", flush=True)
        result = _run_case(case)
        status = "OK" if result.error is None else "FAIL"
        print(status)
        results.append(result)

    print()
    print_summary_table(results)


if __name__ == "__main__":
    main()
