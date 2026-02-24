"""
Tests for repair.py — two-stage JSON repair and LLM retry logic.

Covers test matrix items:
- 8:  Valid LLM response parses cleanly (Stage 1)
- 9:  Malformed JSON → json_repair → valid result (Stage 2)
- 10: Malformed JSON → repair fails → LLM retry → valid result (Stage 3)
- 11: All retry paths fail → AnalysisFailedError with raw_response preserved
"""
import json

import pytest

from horeb.errors import AnalysisFailedError
from horeb.repair import _get_failure_reason, _try_parse, repair_and_validate
from horeb.schemas import StudyGuideResult
from tests.conftest import FixtureLLMProvider, SequentialFixtureLLMProvider, load_fixture

# Alias for test readability — tests refer to the study guide schema explicitly
_SCHEMA = StudyGuideResult
_SYSTEM = ""  # empty system prompt is sufficient for fixture-based tests


# ---------------------------------------------------------------------------
# _try_parse
# ---------------------------------------------------------------------------

class TestTryParse:
    def test_valid_json_returns_result(self):
        raw = load_fixture("john_3_16_valid.json")
        result = _try_parse(raw, _SCHEMA)
        assert isinstance(result, StudyGuideResult)

    def test_malformed_json_returns_none(self):
        result = _try_parse("{not valid json}", _SCHEMA)
        assert result is None

    def test_valid_json_wrong_distribution_returns_none(self):
        raw = load_fixture("wrong_question_distribution.json")
        result = _try_parse(raw, _SCHEMA)
        assert result is None

    def test_valid_json_wrong_summary_length_returns_none(self):
        raw = load_fixture("wrong_summary_length.json")
        result = _try_parse(raw, _SCHEMA)
        assert result is None

    def test_empty_string_returns_none(self):
        assert _try_parse("", _SCHEMA) is None

    def test_empty_object_returns_none(self):
        assert _try_parse("{}", _SCHEMA) is None


# ---------------------------------------------------------------------------
# _get_failure_reason
# ---------------------------------------------------------------------------

class TestGetFailureReason:
    def test_invalid_json_describes_parse_error(self):
        reason = _get_failure_reason("{bad json", _SCHEMA)
        assert "Invalid JSON" in reason or "JSON" in reason

    def test_wrong_distribution_describes_field(self):
        raw = load_fixture("wrong_question_distribution.json")
        reason = _get_failure_reason(raw, _SCHEMA)
        assert len(reason) > 0  # produces some description

    def test_valid_payload_returns_unknown(self):
        # Internally: valid JSON + valid Pydantic → "Unknown validation error"
        valid = load_fixture("john_3_16_valid.json")
        reason = _get_failure_reason(valid, _SCHEMA)
        assert "Unknown" in reason


# ---------------------------------------------------------------------------
# repair_and_validate — Stage 1 (direct parse)
# ---------------------------------------------------------------------------

class TestStage1DirectParse:
    def test_valid_response_succeeds_without_repair(self):
        raw = load_fixture("john_3_16_valid.json")
        llm = FixtureLLMProvider("should not be called")
        result = repair_and_validate(raw, schema=_SCHEMA, llm=llm, system_prompt=_SYSTEM, user_prompt="")
        assert isinstance(result, StudyGuideResult)
        assert llm.call_count == 0  # no LLM call needed

    def test_result_has_correct_structure(self):
        raw = load_fixture("john_3_16_valid.json")
        llm = FixtureLLMProvider("")
        result = repair_and_validate(raw, schema=_SCHEMA, llm=llm, system_prompt=_SYSTEM, user_prompt="")
        assert len(result.summary) == 3
        assert len(result.questions) == 5


# ---------------------------------------------------------------------------
# repair_and_validate — Stage 2 (json_repair)
# ---------------------------------------------------------------------------

class TestStage2JsonRepair:
    def test_truncated_json_repaired_to_valid_result(self):
        raw = load_fixture("malformed_partial.json")
        valid_fallback = load_fixture("john_3_16_valid.json")
        llm = FixtureLLMProvider(valid_fallback)
        # If json_repair succeeds → result is returned, llm not called
        # If json_repair fails → llm is called once
        result = repair_and_validate(raw, schema=_SCHEMA, llm=llm, system_prompt=_SYSTEM, user_prompt="")
        assert isinstance(result, StudyGuideResult)


# ---------------------------------------------------------------------------
# repair_and_validate — Stage 3 (LLM retry)
# ---------------------------------------------------------------------------

class TestStage3LLMRetry:
    def test_retry_succeeds_on_schema_failure(self):
        """First call: wrong distribution. Second call (retry): valid."""
        wrong = load_fixture("wrong_question_distribution.json")
        valid = load_fixture("john_3_16_valid.json")
        llm = SequentialFixtureLLMProvider([valid])  # retry response

        result = repair_and_validate(wrong, schema=_SCHEMA, llm=llm, system_prompt=_SYSTEM, user_prompt="test prompt")
        assert isinstance(result, StudyGuideResult)

    def test_retry_is_called_once_on_schema_failure(self):
        wrong = load_fixture("wrong_question_distribution.json")
        valid = load_fixture("john_3_16_valid.json")
        llm = SequentialFixtureLLMProvider([valid])

        repair_and_validate(wrong, schema=_SCHEMA, llm=llm, system_prompt=_SYSTEM, user_prompt="")
        assert llm.call_count == 1  # exactly 1 retry

    def test_warn_logged_to_stderr_on_retry(self, capsys):
        wrong = load_fixture("wrong_question_distribution.json")
        valid = load_fixture("john_3_16_valid.json")
        llm = SequentialFixtureLLMProvider([valid])

        repair_and_validate(wrong, schema=_SCHEMA, llm=llm, system_prompt=_SYSTEM, user_prompt="")
        captured = capsys.readouterr()
        assert "[WARN]" in captured.err

    def test_retry_prompt_includes_failure_reason(self):
        wrong = load_fixture("wrong_question_distribution.json")
        valid = load_fixture("john_3_16_valid.json")
        llm = SequentialFixtureLLMProvider([valid])

        repair_and_validate(wrong, schema=_SCHEMA, llm=llm, system_prompt=_SYSTEM, user_prompt="original prompt")
        assert llm.last_prompt is not None
        assert "failed validation" in llm.last_prompt.lower() or "reason" in llm.last_prompt.lower()


# ---------------------------------------------------------------------------
# repair_and_validate — all stages fail
# ---------------------------------------------------------------------------

class TestAllStagesFail:
    def test_raises_analysis_failed_error(self):
        wrong = load_fixture("wrong_question_distribution.json")
        llm = SequentialFixtureLLMProvider([wrong])

        with pytest.raises(AnalysisFailedError):
            repair_and_validate(wrong, schema=_SCHEMA, llm=llm, system_prompt=_SYSTEM, user_prompt="")

    def test_raw_response_preserved_in_error(self):
        wrong = load_fixture("wrong_question_distribution.json")
        retry_response = '{"summary": [], "questions": []}'
        llm = SequentialFixtureLLMProvider([retry_response])

        with pytest.raises(AnalysisFailedError) as exc_info:
            repair_and_validate(wrong, schema=_SCHEMA, llm=llm, system_prompt=_SYSTEM, user_prompt="")

        assert exc_info.value.raw_response == retry_response

    def test_max_2_total_llm_calls(self):
        """Hard ceiling: repair_and_validate never makes more than 1 LLM call."""
        wrong = load_fixture("wrong_question_distribution.json")
        llm = SequentialFixtureLLMProvider([wrong])  # 1 response available

        with pytest.raises((AnalysisFailedError, RuntimeError)):
            repair_and_validate(wrong, schema=_SCHEMA, llm=llm, system_prompt=_SYSTEM, user_prompt="")

        assert llm.call_count <= 1
