"""
Shared test infrastructure.

FixtureLLMProvider — implements LLMProvider Protocol, returns a fixed string.
load_fixture()      — reads a file from tests/fixtures/responses/.
Pytest fixtures     — pre-built FixtureLLMProvider instances for common scenarios.
"""
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESPONSES_DIR = FIXTURES_DIR / "responses"


class FixtureLLMProvider:
    """
    Test double for LLMProvider.

    Returns a fixed response string from complete(), allowing full pipeline
    tests without live API calls.
    """

    def __init__(self, response: str) -> None:
        self._response = response
        self.call_count = 0
        self.last_system: str | None = None
        self.last_prompt: str | None = None

    def complete(self, system: str, prompt: str) -> str:
        self.call_count += 1
        self.last_system = system
        self.last_prompt = prompt
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

    def complete(self, system: str, prompt: str) -> str:
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


def load_fixture(filename: str) -> str:
    """Load the content of a fixture file from tests/fixtures/responses/."""
    path = RESPONSES_DIR / filename
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_john_llm() -> FixtureLLMProvider:
    """LLMProvider that returns a valid John 3:16-21 analysis."""
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
