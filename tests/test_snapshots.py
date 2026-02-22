"""
Snapshot tests — assert full AnalysisResult structure against known fixtures.

These act as regression tests for schema changes. If a field rename, type
change, or validator change breaks the expected output shape, these tests
catch it before CLI output silently changes.

Covers test matrix item 21.
"""
import json

import pytest

from horeb.engine import analyze
from horeb.repair import repair_and_validate
from horeb.schemas import AnalysisResult, QuestionType
from tests.conftest import FixtureLLMProvider, load_fixture


# ---------------------------------------------------------------------------
# John 3:16-21 snapshot
# ---------------------------------------------------------------------------

class TestJohnSnapshot:
    def test_result_is_analysis_result(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        assert isinstance(result, AnalysisResult)

    def test_summary_has_exactly_3_items(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        assert len(result.summary) == 3

    def test_summary_items_are_nonempty_strings(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        for sentence in result.summary:
            assert isinstance(sentence, str)
            assert len(sentence) > 0

    def test_questions_count_is_5(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        assert len(result.questions) == 5

    def test_question_type_distribution_is_2_2_1(self, valid_john_llm):
        from collections import Counter
        result = analyze("John 3:16-21", llm=valid_john_llm)
        counts = Counter(q.type for q in result.questions)
        assert counts[QuestionType.COMPREHENSION] == 2
        assert counts[QuestionType.REFLECTION] == 2
        assert counts[QuestionType.APPLICATION] == 1

    def test_all_question_texts_nonempty(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        for q in result.questions:
            assert len(q.text) > 0

    def test_named_entities_present(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        # The fixture includes named entities
        assert result.named_entities is not None
        assert len(result.named_entities) > 0

    def test_key_themes_present(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        assert result.key_themes is not None
        assert len(result.key_themes) > 0

    def test_low_confidence_fields_empty(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        assert result.low_confidence_fields == []

    def test_model_dump_round_trips_through_json(self, valid_john_llm):
        """Schema changes that break serialisation are caught here."""
        result = analyze("John 3:16-21", llm=valid_john_llm)
        dumped = result.model_dump()
        restored = AnalysisResult.model_validate(dumped)
        assert restored.model_dump() == dumped

    def test_fixture_matches_expected_summary_content(self):
        """
        Assert the fixture's summary matches expected content.
        This is the strongest regression test — update when fixture changes.
        """
        raw = load_fixture("john_3_16_valid.json")
        llm = FixtureLLMProvider(raw)
        result = analyze("John 3:16-21", llm=llm)

        # Check fixture summary contains key concepts — not exact strings,
        # to allow minor fixture updates without breaking the test.
        full_summary = " ".join(result.summary).lower()
        assert "god" in full_summary or "love" in full_summary
        assert "belief" in full_summary or "believ" in full_summary

    def test_fixture_questions_have_verse_references(self):
        """All questions in the fixture should cite a verse."""
        raw = load_fixture("john_3_16_valid.json")
        llm = FixtureLLMProvider(raw)
        result = analyze("John 3:16-21", llm=llm)
        # At least 4 of 5 questions should have verse_reference
        with_ref = [q for q in result.questions if q.verse_reference is not None]
        assert len(with_ref) >= 4
