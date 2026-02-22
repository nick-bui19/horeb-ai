"""
Two-stage JSON repair with a single LLM retry.

Stage 1: json.loads() → Pydantic validation
Stage 2: json_repair.repair() → Pydantic validation
Stage 3: LLM retry with targeted correction message → Pydantic validation
Hard ceiling: 2 total LLM calls. Never more.

[WARN] is always logged to stderr on retry — not gated behind HOREB_DEBUG —
because retry frequency is a correctness signal, not a debug detail.
"""
import json
import sys
from typing import TYPE_CHECKING

import json_repair
from pydantic import ValidationError

from horeb.errors import AnalysisFailedError
from horeb.prompts import SYSTEM_PROMPT
from horeb.schemas import AnalysisResult

if TYPE_CHECKING:
    from horeb.llm import LLMProvider


def repair_and_validate(
    raw: str,
    llm: "LLMProvider",
    prompt: str,
) -> AnalysisResult:
    """
    Attempt to parse and validate a raw LLM response string as AnalysisResult.

    Args:
        raw:    Raw string from LLMProvider.complete().
        llm:    LLMProvider instance for the optional retry call.
        prompt: The original user prompt, used to construct the retry message.

    Returns:
        Validated AnalysisResult.

    Raises:
        AnalysisFailedError: if all stages fail, with raw_response preserved.
    """
    # Stage 1: direct parse + validate
    result = _try_parse_and_validate(raw)
    if result is not None:
        return result

    # Stage 2: structural JSON repair + validate
    try:
        repaired = json_repair.repair(raw)
        result = _try_parse_and_validate(repaired)
        if result is not None:
            return result
    except Exception:
        # json_repair can raise on extreme inputs — treat as repair failure
        pass

    # Stage 3: one LLM retry with a targeted correction prompt
    failure_reason = _get_failure_reason(raw)
    print(
        f"[WARN] LLM retry triggered. Reason: {failure_reason}",
        file=sys.stderr,
    )

    retry_prompt = (
        f"{prompt}\n\n"
        f"Your previous response failed validation. Reason: {failure_reason}\n"
        f"Please call submit_analysis again with a corrected response."
    )
    retry_raw = llm.complete(system=SYSTEM_PROMPT, prompt=retry_prompt)

    result = _try_parse_and_validate(retry_raw)
    if result is not None:
        return result

    raise AnalysisFailedError(
        "Analysis failed after all repair attempts",
        raw_response=retry_raw,
    )


def _try_parse_and_validate(raw: str) -> AnalysisResult | None:
    """
    Attempt json.loads + Pydantic validation. Returns None on any failure.
    Never raises.
    """
    try:
        data = json.loads(raw)
        return AnalysisResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError, Exception):
        return None


def _get_failure_reason(raw: str) -> str:
    """
    Parse the raw string to produce a human-readable failure reason.
    Used to construct a targeted correction prompt for the LLM retry.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"

    try:
        AnalysisResult.model_validate(data)
        return "Unknown validation error"
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = " -> ".join(str(p) for p in first["loc"])
            return f"Validation error on '{loc}': {first['msg']}"
        return "Pydantic validation failed with no details"
