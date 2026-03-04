"""
Evaluation result types and summary table printer.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalResult:
    case_id: str
    reference: str
    granularity: str        # "passage" | "chapter" | "book"
    schema_passed: bool
    retry_count: int
    citation_valid: bool
    failed_segments: int    # 0 for non-book cases
    error: str | None


def print_summary_table(results: list[EvalResult]) -> None:
    """Print a fixed-width summary table to stdout."""
    header = (
        f"{'ID':<6} {'Reference':<24} {'Gran':<8} "
        f"{'Schema':<8} {'Retries':<8} {'Cites':<7} {'FailSeg':<8} {'Error'}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        schema_str = "OK" if r.schema_passed else "FAIL"
        cites_str = "OK" if r.citation_valid else "FAIL"
        error_str = r.error or ""
        print(
            f"{r.case_id:<6} {r.reference:<24} {r.granularity:<8} "
            f"{schema_str:<8} {r.retry_count:<8} {cites_str:<7} "
            f"{r.failed_segments:<8} {error_str}"
        )
    print(sep)

    total = len(results)
    passed = sum(1 for r in results if r.schema_passed and r.citation_valid and r.error is None)
    total_retries = sum(r.retry_count for r in results)
    total_failures = sum(r.failed_segments for r in results)
    print(
        f"Summary: {passed}/{total} passed  |  "
        f"total retries: {total_retries}  |  "
        f"total failed segments: {total_failures}"
    )
