"""
Integration tests — make live Anthropic API calls.

These tests are excluded from the standard pytest run by default.
Run them explicitly with:

    uv run pytest -m integration

Or include them in a CI job that has ANTHROPIC_API_KEY set.

Requires: ANTHROPIC_API_KEY environment variable.
"""
import pytest

from horeb.engine import analyze
from horeb.schemas import AnalysisResult, QuestionType

pytestmark = pytest.mark.integration


class TestLiveJohn316:
    def test_returns_analysis_result(self):
        result = analyze("John 3:16-21")
        assert isinstance(result, AnalysisResult)

    def test_summary_has_3_items(self):
        result = analyze("John 3:16-21")
        assert len(result.summary) == 3

    def test_questions_have_correct_distribution(self):
        from collections import Counter
        result = analyze("John 3:16-21")
        counts = Counter(q.type for q in result.questions)
        assert counts[QuestionType.COMPREHENSION] == 2
        assert counts[QuestionType.REFLECTION] == 2
        assert counts[QuestionType.APPLICATION] == 1

    def test_no_out_of_range_citations(self):
        # If citations were out of range, engine.analyze would raise.
        # Reaching here means verify_citations passed.
        result = analyze("John 3:16-21")
        assert result is not None


class TestLivePsalm23:
    def test_psalm_23_1_6_returns_result(self):
        result = analyze("Psalm 23:1-6")
        assert isinstance(result, AnalysisResult)


class TestLiveEdgeCases:
    def test_single_verse(self):
        result = analyze("John 3:16")
        assert isinstance(result, AnalysisResult)

    def test_single_chapter_book(self):
        # Jude has only one chapter; "Jude 1:1-4" is the correct form
        result = analyze("Jude 1:1-4")
        assert isinstance(result, AnalysisResult)
