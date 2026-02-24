"""
Shared test infrastructure.

FixtureLLMProvider         — implements LLMProvider Protocol, returns a fixed string.
SequentialFixtureLLMProvider — returns responses in sequence (for retry path tests).
load_fixture()             — reads a file from tests/fixtures/responses/<subdir>/.
Pytest fixtures            — pre-built FixtureLLMProvider instances for common scenarios.
"""
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pydantic import BaseModel

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESPONSES_DIR = FIXTURES_DIR / "responses"


class FixtureLLMProvider:
    """
    Test double for LLMProvider.

    Returns a fixed response string from complete(), allowing full pipeline
    tests without live API calls. The schema and max_tokens parameters are
    accepted to satisfy the LLMProvider Protocol but are not used.
    """

    def __init__(self, response: str) -> None:
        self._response = response
        self.call_count = 0
        self.last_system: str | None = None
        self.last_prompt: str | None = None
        self.last_schema: type | None = None

    def complete(
        self,
        system: str,
        prompt: str,
        schema: "type[BaseModel] | None" = None,
        max_tokens: int | None = None,
    ) -> str:
        self.call_count += 1
        self.last_system = system
        self.last_prompt = prompt
        self.last_schema = schema
        return self._response


class SequentialFixtureLLMProvider:
    """
    Test double that returns responses in sequence.

    Useful for testing retry paths where the first call fails and the
    second succeeds (or vice versa).
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.call_count = 0
        self.last_prompt: str | None = None

    def complete(
        self,
        system: str,
        prompt: str,
        schema: "type[BaseModel] | None" = None,
        max_tokens: int | None = None,
    ) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        if self._index >= len(self._responses):
            raise RuntimeError(
                f"SequentialFixtureLLMProvider: expected at most "
                f"{len(self._responses)} calls, got {self.call_count}"
            )
        response = self._responses[self._index]
        self._index += 1
        return response


def load_fixture(filename: str, subdir: str = "") -> str:
    """
    Load the content of a fixture file from tests/fixtures/responses/<subdir>/.

    Args:
        filename: The fixture filename (e.g. "john_3_16_valid.json").
        subdir:   Optional subdirectory under responses/ (e.g. "study_guide",
                  "passage", "segment", "book", "similarity").
                  Defaults to the root responses/ directory for Phase 1 compat.
    """
    if subdir:
        path = RESPONSES_DIR / subdir / filename
    else:
        path = RESPONSES_DIR / filename
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 1 / study guide fixtures (responses/study_guide/ after Task 12 migration;
# still at root until fixture reorganization task runs)
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_john_llm() -> FixtureLLMProvider:
    """LLMProvider that returns a valid John 3:16-21 study guide analysis."""
    return FixtureLLMProvider(load_fixture("john_3_16_valid.json"))


@pytest.fixture
def malformed_json_llm() -> FixtureLLMProvider:
    """LLMProvider that returns truncated JSON (json_repair can fix it)."""
    return FixtureLLMProvider(load_fixture("malformed_partial.json"))


@pytest.fixture
def wrong_distribution_llm() -> FixtureLLMProvider:
    """LLMProvider that returns valid JSON with wrong question distribution."""
    return FixtureLLMProvider(load_fixture("wrong_question_distribution.json"))


@pytest.fixture
def wrong_summary_llm() -> FixtureLLMProvider:
    """LLMProvider that returns valid JSON with wrong summary length."""
    return FixtureLLMProvider(load_fixture("wrong_summary_length.json"))


@pytest.fixture
def out_of_range_citation_llm() -> FixtureLLMProvider:
    """LLMProvider that returns an out-of-range citation (John 3:22)."""
    return FixtureLLMProvider(load_fixture("out_of_range_citation.json"))
