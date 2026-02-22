"""
Tests for Pydantic schema validation — AnalysisResult model validators.

Covers test matrix items:
- 12: summary length validator
- 13: question distribution validator
- 16: nullable fields accepted
"""
import pytest
from pydantic import ValidationError

from horeb.schemas import AnalysisResult, QuestionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_questions() -> list[dict]:
    return [
        {"type": "comprehension", "text": "Q1?", "verse_reference": "3:16"},
        {"type": "comprehension", "text": "Q2?", "verse_reference": "3:18"},
        {"type": "reflection", "text": "Q3?", "verse_reference": "3:19"},
        {"type": "reflection", "text": "Q4?", "verse_reference": "3:21"},
        {"type": "application", "text": "Q5?", "verse_reference": "3:17"},
    ]


def _valid_summary() -> list[str]:
    return ["Summary sentence 1.", "Summary sentence 2.", "Summary sentence 3."]


def _valid_payload(**overrides) -> dict:
    base = {
        "summary": _valid_summary(),
        "questions": _valid_questions(),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Summary length
# ---------------------------------------------------------------------------

class TestSummaryLengthValidator:
    def test_exactly_3_accepted(self):
        result = AnalysisResult.model_validate(_valid_payload())
        assert len(result.summary) == 3

    def test_2_items_rejected(self):
        with pytest.raises(ValidationError, match="exactly 3"):
            AnalysisResult.model_validate(_valid_payload(summary=["A.", "B."]))

    def test_4_items_rejected(self):
        with pytest.raises(ValidationError, match="exactly 3"):
            AnalysisResult.model_validate(
                _valid_payload(summary=["A.", "B.", "C.", "D."])
            )

    def test_empty_list_rejected(self):
        with pytest.raises(ValidationError):
            AnalysisResult.model_validate(_valid_payload(summary=[]))


# ---------------------------------------------------------------------------
# Question distribution
# ---------------------------------------------------------------------------

class TestQuestionDistributionValidator:
    def test_correct_2_2_1_accepted(self):
        result = AnalysisResult.model_validate(_valid_payload())
        types = [q.type for q in result.questions]
        assert types.count(QuestionType.COMPREHENSION) == 2
        assert types.count(QuestionType.REFLECTION) == 2
        assert types.count(QuestionType.APPLICATION) == 1

    def test_3_comprehension_rejected(self):
        questions = _valid_questions()
        questions[2]["type"] = "comprehension"  # make 3rd a comprehension
        with pytest.raises(ValidationError, match="comprehension"):
            AnalysisResult.model_validate(_valid_payload(questions=questions))

    def test_0_application_rejected(self):
        questions = _valid_questions()
        questions[4]["type"] = "reflection"  # remove the application question
        # The validator will report whichever distribution mismatch it finds first.
        # The important assertion is that ValidationError is raised at all.
        with pytest.raises(ValidationError):
            AnalysisResult.model_validate(_valid_payload(questions=questions))

    def test_all_same_type_rejected(self):
        questions = [
            {"type": "comprehension", "text": f"Q{i}?", "verse_reference": None}
            for i in range(5)
        ]
        with pytest.raises(ValidationError):
            AnalysisResult.model_validate(_valid_payload(questions=questions))

    def test_invalid_type_string_rejected(self):
        questions = _valid_questions()
        questions[0]["type"] = "summary"  # not a valid QuestionType
        with pytest.raises(ValidationError):
            AnalysisResult.model_validate(_valid_payload(questions=questions))


# ---------------------------------------------------------------------------
# Nullable / optional fields
# ---------------------------------------------------------------------------

class TestNullableFields:
    def test_key_themes_none_accepted(self):
        result = AnalysisResult.model_validate(_valid_payload(key_themes=None))
        assert result.key_themes is None

    def test_named_entities_none_accepted(self):
        result = AnalysisResult.model_validate(_valid_payload(named_entities=None))
        assert result.named_entities is None

    def test_low_confidence_fields_defaults_empty(self):
        result = AnalysisResult.model_validate(_valid_payload())
        assert result.low_confidence_fields == []

    def test_low_confidence_fields_populated(self):
        result = AnalysisResult.model_validate(
            _valid_payload(low_confidence_fields=["key_themes"])
        )
        assert result.low_confidence_fields == ["key_themes"]

    def test_question_verse_reference_none_accepted(self):
        questions = _valid_questions()
        questions[4]["verse_reference"] = None
        result = AnalysisResult.model_validate(_valid_payload(questions=questions))
        assert result.questions[4].verse_reference is None
