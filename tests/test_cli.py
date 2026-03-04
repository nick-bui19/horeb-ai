"""
Tests for cli.py — exit codes, output routing, error messages.

Covers test matrix items:
- 17: Exit code 2 on InvalidReferenceError
- 18: Exit code 3 on EmptyPassageError
- 19: Exit code 4 on CitationOutOfRangeError
- 20: Exit code 5 on AnalysisFailedError
"""
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from horeb.cli import (
    EXIT_ANALYSIS_FAILED,
    EXIT_CITATION_OUT_OF_RANGE,
    EXIT_EMPTY_PASSAGE,
    EXIT_INVALID_REFERENCE,
    app,
    _print_result,
    _print_similar_result,
)
from horeb.errors import (
    AnalysisFailedError,
    CitationOutOfRangeError,
    EmptyPassageError,
    InvalidReferenceError,
)
from horeb.schemas import (
    AnalysisResult,
    BookAnalysisResult,
    OutlineSection,
    PassageAnalysisResult,
    Question,
    QuestionType,
    SimilarityResult,
    SimilarOverlap,
    VerseCitation,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_result() -> AnalysisResult:
    return AnalysisResult(
        summary=["Sentence one.", "Sentence two.", "Sentence three."],
        key_themes=["faith", "love"],
        named_entities=None,
        questions=[
            Question(type=QuestionType.COMPREHENSION, text="Q1?", verse_reference="3:16"),
            Question(type=QuestionType.COMPREHENSION, text="Q2?", verse_reference="3:18"),
            Question(type=QuestionType.REFLECTION, text="Q3?", verse_reference="3:19"),
            Question(type=QuestionType.REFLECTION, text="Q4?", verse_reference="3:21"),
            Question(type=QuestionType.APPLICATION, text="Q5?", verse_reference=None),
        ],
    )


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

class TestExitCodes:
    def test_invalid_reference_exit_code_2(self):
        with patch("horeb.cli.analyze", side_effect=InvalidReferenceError("bad ref")):
            result = runner.invoke(app, ["analyze", "John 3:abc"])
        assert result.exit_code == EXIT_INVALID_REFERENCE

    def test_empty_passage_exit_code_3(self):
        with patch("horeb.cli.analyze", side_effect=EmptyPassageError("empty")):
            result = runner.invoke(app, ["analyze", "John 3:16"])
        assert result.exit_code == EXIT_EMPTY_PASSAGE

    def test_citation_out_of_range_exit_code_4(self):
        with patch("horeb.cli.analyze", side_effect=CitationOutOfRangeError("bad cite")):
            result = runner.invoke(app, ["analyze", "John 3:16-21"])
        assert result.exit_code == EXIT_CITATION_OUT_OF_RANGE

    def test_analysis_failed_exit_code_5(self):
        with patch("horeb.cli.analyze", side_effect=AnalysisFailedError("failed")):
            result = runner.invoke(app, ["analyze", "John 3:16-21"])
        assert result.exit_code == EXIT_ANALYSIS_FAILED

    def test_success_exit_code_0(self):
        with patch("horeb.cli.analyze", return_value=_valid_result()):
            result = runner.invoke(app, ["analyze", "John 3:16-21"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Error messages go to stderr
# ---------------------------------------------------------------------------

class TestErrorOutputRouting:
    def test_error_message_in_output(self):
        # CliRunner mixes stderr into output by default
        with patch("horeb.cli.analyze", side_effect=InvalidReferenceError("bad ref")):
            result = runner.invoke(app, ["analyze", "bad ref"])
        assert "bad ref" in result.output

    def test_analysis_failed_message_in_output(self):
        with patch("horeb.cli.analyze", side_effect=AnalysisFailedError("model failed")):
            result = runner.invoke(app, ["analyze", "John 3:16-21"])
        assert "model failed" in result.output


# ---------------------------------------------------------------------------
# Output format (_print_result)
# ---------------------------------------------------------------------------

class TestPrintResult:
    def test_summary_section_present(self, capsys):
        _print_result(_valid_result())
        captured = capsys.readouterr()
        assert "SUMMARY" in captured.out
        assert "Sentence one." in captured.out

    def test_all_5_questions_printed(self, capsys):
        _print_result(_valid_result())
        captured = capsys.readouterr()
        assert "STUDY QUESTIONS" in captured.out
        for i in range(1, 6):
            assert f"{i}." in captured.out

    def test_question_type_labels_present(self, capsys):
        _print_result(_valid_result())
        captured = capsys.readouterr()
        assert "[comprehension]" in captured.out
        assert "[reflection]" in captured.out
        assert "[application]" in captured.out

    def test_null_key_themes_prints_not_determined(self, capsys):
        result = _valid_result()
        result.key_themes = None
        _print_result(result)
        captured = capsys.readouterr()
        assert "not determined" in captured.out

    def test_low_confidence_fields_noted(self, capsys):
        result = _valid_result()
        result.low_confidence_fields = ["key_themes"]
        _print_result(result)
        captured = capsys.readouterr()
        assert "key_themes" in captured.out
        assert "Low confidence" in captured.out

    def test_output_goes_to_stdout_not_stderr(self, capsys):
        _print_result(_valid_result())
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert captured.err == ""


# ---------------------------------------------------------------------------
# _print_result — PassageAnalysisResult branch
# ---------------------------------------------------------------------------

class TestPrintResultPassage:
    def _passage_result(self) -> PassageAnalysisResult:
        return PassageAnalysisResult(
            summary=["First point.", "Second point.", "Third point."],
            key_themes=["grace", "faith", "love", "hope", "peace"],
            citations=[
                VerseCitation(verse_reference="3:16", quoted_text="For God so loved"),
                VerseCitation(verse_reference="3:17", quoted_text=None),
            ],
        )

    def test_summary_section_present(self, capsys):
        _print_result(self._passage_result())
        out = capsys.readouterr().out
        assert "SUMMARY" in out
        assert "First point." in out

    def test_citations_section_present(self, capsys):
        _print_result(self._passage_result())
        out = capsys.readouterr().out
        assert "CITATIONS" in out
        assert "3:16" in out

    def test_citation_with_quoted_text_prints_snippet(self, capsys):
        _print_result(self._passage_result())
        out = capsys.readouterr().out
        assert "For God so loved" in out

    def test_citation_without_quoted_text_prints_reference_only(self, capsys):
        _print_result(self._passage_result())
        out = capsys.readouterr().out
        assert "3:17" in out

    def test_key_themes_capped_at_5(self, capsys):
        result = self._passage_result()
        result.key_themes = ["a", "b", "c", "d", "e", "f", "g"]
        _print_result(result)
        out = capsys.readouterr().out
        assert "a" in out
        assert "e" in out
        assert "f" not in out  # 6th theme should be truncated

    def test_no_citations_section_when_empty(self, capsys):
        result = PassageAnalysisResult(
            summary=["A.", "B.", "C."],
            citations=[],
        )
        _print_result(result)
        out = capsys.readouterr().out
        assert "CITATIONS" not in out

    def test_no_study_questions_section(self, capsys):
        _print_result(self._passage_result())
        out = capsys.readouterr().out
        assert "STUDY QUESTIONS" not in out


# ---------------------------------------------------------------------------
# _print_result — BookAnalysisResult branch
# ---------------------------------------------------------------------------

class TestPrintResultBook:
    def _book_result(self) -> BookAnalysisResult:
        return BookAnalysisResult(
            summary=["Book point one.", "Book point two.", "Book point three."],
            key_themes=["redemption"],
            outline=[
                OutlineSection(
                    title="Opening narrative",
                    start_verse="1:1",
                    end_verse="1:22",
                    source_segments=[0],
                    summary="Ruth follows Naomi to Bethlehem.",
                ),
                OutlineSection(
                    title="Harvest and provision",
                    start_verse="2:1",
                    end_verse="2:23",
                    source_segments=[1],
                    summary=None,
                ),
            ],
            failed_segments=[],
        )

    def test_summary_section_present(self, capsys):
        _print_result(self._book_result())
        out = capsys.readouterr().out
        assert "SUMMARY" in out
        assert "Book point one." in out

    def test_outline_section_present(self, capsys):
        _print_result(self._book_result())
        out = capsys.readouterr().out
        assert "OUTLINE" in out
        assert "Opening narrative" in out
        assert "1:1" in out
        assert "1:22" in out

    def test_outline_section_summary_printed_when_present(self, capsys):
        _print_result(self._book_result())
        out = capsys.readouterr().out
        assert "Ruth follows Naomi" in out

    def test_no_failed_segments_note_when_empty(self, capsys):
        _print_result(self._book_result())
        out = capsys.readouterr().out
        assert "could not be analyzed" not in out

    def test_failed_segments_note_when_present(self, capsys):
        result = self._book_result()
        result.failed_segments = [2, 3]
        _print_result(result)
        out = capsys.readouterr().out
        assert "2 segment(s) could not be analyzed" in out

    def test_no_study_questions_or_citations_section(self, capsys):
        _print_result(self._book_result())
        out = capsys.readouterr().out
        assert "STUDY QUESTIONS" not in out
        assert "CITATIONS" not in out


# ---------------------------------------------------------------------------
# _print_similar_result
# ---------------------------------------------------------------------------

class TestPrintSimilarResult:
    def _similar_result(self) -> SimilarityResult:
        return SimilarityResult(
            seed_ref="John 3:16",
            candidates=[
                SimilarOverlap(
                    candidate_ref="John 1:12",
                    verbatim_seed_quote="believed on his name",
                    verbatim_candidate_quote="believed on his name",
                    overlap_terms=["believed", "name"],
                    similarity_score=0.75,
                ),
                SimilarOverlap(
                    candidate_ref="Romans 8:1",
                    verbatim_seed_quote="no condemnation",
                    verbatim_candidate_quote="no condemnation",
                    overlap_terms=["condemnation"],
                    similarity_score=0.42,
                ),
            ],
        )

    def test_header_contains_seed_ref(self, capsys):
        _print_similar_result(self._similar_result())
        out = capsys.readouterr().out
        assert "SIMILAR PASSAGES" in out
        assert "John 3:16" in out

    def test_candidate_references_printed(self, capsys):
        _print_similar_result(self._similar_result())
        out = capsys.readouterr().out
        assert "John 1:12" in out
        assert "Romans 8:1" in out

    def test_similarity_scores_printed(self, capsys):
        _print_similar_result(self._similar_result())
        out = capsys.readouterr().out
        assert "0.7500" in out
        assert "0.4200" in out

    def test_overlap_terms_printed(self, capsys):
        _print_similar_result(self._similar_result())
        out = capsys.readouterr().out
        assert "believed" in out
        assert "condemnation" in out

    def test_verbatim_quotes_printed(self, capsys):
        _print_similar_result(self._similar_result())
        out = capsys.readouterr().out
        assert "believed on his name" in out
        assert "no condemnation" in out

    def test_empty_candidates_prints_no_results_message(self, capsys):
        result = SimilarityResult(seed_ref="John 3:16", candidates=[])
        _print_similar_result(result)
        out = capsys.readouterr().out
        assert "No similar passages found" in out

    def test_empty_candidates_prints_nothing_else(self, capsys):
        result = SimilarityResult(seed_ref="John 3:16", candidates=[])
        _print_similar_result(result)
        out = capsys.readouterr().out
        assert "SIMILAR PASSAGES" not in out
