"""
Tests for engine.py — full pipeline integration with fixture LLM providers.

Covers test matrix items:
- 8:  Valid response parses cleanly
- 10: Citation in range passes
- 14: Citation out of range → CitationOutOfRangeError
- 15: EmptyPassageError on short passage text
"""
from unittest.mock import patch

import pytest

from horeb.engine import analyze, extract_verse_refs, verify_citations
from horeb.errors import (
    AnalysisFailedError,
    CitationOutOfRangeError,
    EmptyPassageError,
    InvalidReferenceError,
)
from horeb.bible_text import retrieve_passage
from horeb.schemas import AnalysisResult
from tests.conftest import (
    FixtureLLMProvider,
    SequentialFixtureLLMProvider,
    load_fixture,
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestAnalyzeHappyPath:
    def test_valid_fixture_returns_analysis_result(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        assert isinstance(result, AnalysisResult)

    def test_result_has_3_summary_items(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        assert len(result.summary) == 3

    def test_result_has_5_questions(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        assert len(result.questions) == 5

    def test_result_question_distribution_is_2_2_1(self, valid_john_llm):
        from collections import Counter
        from horeb.schemas import QuestionType
        result = analyze("John 3:16-21", llm=valid_john_llm)
        counts = Counter(q.type for q in result.questions)
        assert counts[QuestionType.COMPREHENSION] == 2
        assert counts[QuestionType.REFLECTION] == 2
        assert counts[QuestionType.APPLICATION] == 1


# ---------------------------------------------------------------------------
# Invalid reference propagation
# ---------------------------------------------------------------------------

class TestInvalidReferencePropagation:
    def test_invalid_reference_raises_before_llm_call(self):
        llm = FixtureLLMProvider("should not be called")
        with pytest.raises(InvalidReferenceError):
            analyze("NotARealBook 1:1", llm=llm)
        assert llm.call_count == 0

    def test_chapter_without_verses_raises_invalid_reference(self):
        llm = FixtureLLMProvider("")
        with pytest.raises(InvalidReferenceError):
            analyze("John 3", llm=llm)


# ---------------------------------------------------------------------------
# EmptyPassageError guard
# ---------------------------------------------------------------------------

class TestEmptyPassageGuard:
    def test_empty_passage_text_raises_before_llm_call(self):
        from horeb.schemas import PassageData
        llm = FixtureLLMProvider("should not be called")

        # Patch retrieve_passage to return a passage with very short text
        short_passage = PassageData(
            reference="John 3:16",
            book=43,
            start_chapter=3,
            start_verse=16,
            end_chapter=3,
            end_verse=16,
            text="",
            context_before=None,
            context_after=None,
        )
        with patch("horeb.engine.retrieve_passage", return_value=short_passage):
            with pytest.raises(EmptyPassageError):
                analyze("John 3:16", llm=llm)

        assert llm.call_count == 0


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------

class TestCitationVerification:
    def test_in_range_citation_passes(self, valid_john_llm):
        # john_3_16_valid.json cites verses within 3:16-3:21 — should pass
        result = analyze("John 3:16-21", llm=valid_john_llm)
        assert isinstance(result, AnalysisResult)

    def test_out_of_range_citation_raises(self, out_of_range_citation_llm):
        # out_of_range_citation.json cites John 3:22 which is outside 3:16-21
        with pytest.raises(CitationOutOfRangeError):
            analyze("John 3:16-21", llm=out_of_range_citation_llm)

    def test_extract_verse_refs_from_questions(self, valid_john_llm):
        result = analyze("John 3:16-21", llm=valid_john_llm)
        refs = extract_verse_refs(result)
        assert len(refs) > 0
        # All question refs should be "chapter:verse" format
        for ref in refs:
            assert ":" in ref

    def test_extract_verse_refs_skips_null_references(self):
        from horeb.schemas import Question, QuestionType
        result = AnalysisResult(
            summary=["A.", "B.", "C."],
            questions=[
                Question(type=QuestionType.COMPREHENSION, text="Q1?", verse_reference="3:16"),
                Question(type=QuestionType.COMPREHENSION, text="Q2?", verse_reference=None),
                Question(type=QuestionType.REFLECTION, text="Q3?", verse_reference="3:19"),
                Question(type=QuestionType.REFLECTION, text="Q4?", verse_reference="3:21"),
                Question(type=QuestionType.APPLICATION, text="Q5?", verse_reference=None),
            ],
        )
        refs = extract_verse_refs(result)
        assert refs == ["3:16", "3:19", "3:21"]


# ---------------------------------------------------------------------------
# AnalysisFailedError propagation
# ---------------------------------------------------------------------------

class TestAnalysisFailedPropagation:
    def test_analysis_failed_raises_with_wrong_distribution_and_bad_retry(self):
        wrong = load_fixture("wrong_question_distribution.json")
        # analyze() calls llm.complete() once (initial), then repair_and_validate
        # calls it once more (retry) → 2 total calls, both return wrong distribution
        llm = SequentialFixtureLLMProvider([wrong, wrong])
        with pytest.raises(AnalysisFailedError):
            analyze("John 3:16-21", llm=llm)
