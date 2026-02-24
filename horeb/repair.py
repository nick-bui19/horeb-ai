"""
Two-stage JSON repair with a single LLM retry.

Stage 1: json.loads() → Pydantic validation
Stage 2: json_repair.repair() → Pydantic validation
Stage 3: LLM retry with targeted correction message → Pydantic validation
Hard ceiling: 2 total LLM calls. Never more.

[WARN] is always logged to stderr on retry — not gated behind HOREB_DEBUG —
because retry frequency is a correctness signal, not a debug detail.

repair_and_validate[T] is generic: pass any Pydantic BaseModel subclass as
the schema argument. The caller is responsible for passing the correct
system_prompt for the retry call.
"""
import json
import sys
from typing import TYPE_CHECKING, TypeVar

import json_repair
from pydantic import BaseModel, ValidationError

from horeb.errors import AnalysisFailedError

if TYPE_CHECKING:
    from horeb.llm import LLMProvider

T = TypeVar("T", bound=BaseModel)


def repair_and_validate(
    raw: str,
    schema: type[T],
    llm: "LLMProvider",
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None = None,
) -> T:
    """
    Attempt to parse and validate a raw LLM response string against schema T.

    Args:
        raw:           Raw string from LLMProvider.complete().
        schema:        Pydantic model class to validate against.
        llm:           LLMProvider instance for the optional retry call.
        system_prompt: System prompt for the retry call.
        user_prompt:   Original user prompt; included in the retry message.
        max_tokens:    Optional token limit override forwarded to the retry call.

    Returns:
        Validated instance of T.

    Raises:
        AnalysisFailedError: if all stages fail, with raw_response preserved.
    """
    # Stage 1: direct parse + validate
    result = _try_parse(raw, schema)
    if result is not None:
        return result

    # Stage 2: structural JSON repair + validate
    try:
        repaired = json_repair.repair(raw)
        result = _try_parse(repaired, schema)
        if result is not None:
            return result
    except Exception:
        # json_repair can raise on extreme inputs — treat as repair failure
        pass

    # Stage 3: one LLM retry with a targeted correction prompt
    failure_reason = _get_failure_reason(raw, schema)
    print(
        f"[WARN] LLM retry triggered for {schema.__name__}. Reason: {failure_reason}",
        file=sys.stderr,
    )

    retry_prompt = (
        f"{user_prompt}\n\n"
        f"Your previous response failed validation. Reason: {failure_reason}\n"
        f"Please call the tool again with a corrected response."
    )
    retry_raw = llm.complete(
        system=system_prompt,
        prompt=retry_prompt,
        schema=schema,
        max_tokens=max_tokens,
    )

    result = _try_parse(retry_raw, schema)
    if result is not None:
        return result

    raise AnalysisFailedError(
        f"Analysis failed after all repair attempts for {schema.__name__}",
        raw_response=retry_raw,
    )


def _try_parse(raw: str, schema: type[T]) -> T | None:
    """
    Attempt json.loads + Pydantic validation against schema. Returns None on any failure.
    Never raises.
    """
    try:
        data = json.loads(raw)
        return schema.model_validate(data)
    except Exception:
        return None


def _get_failure_reason(raw: str, schema: type[BaseModel]) -> str:
    """
    Parse the raw string to produce a human-readable failure reason.
    Used to construct a targeted correction prompt for the LLM retry.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"

    try:
        schema.model_validate(data)
        return "Unknown validation error"
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = " -> ".join(str(p) for p in first["loc"])
            return f"Validation error on '{loc}': {first['msg']}"
        return "Pydantic validation failed with no details"
