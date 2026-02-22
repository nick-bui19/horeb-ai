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
)
from horeb.errors import (
    AnalysisFailedError,
    CitationOutOfRangeError,
    EmptyPassageError,
    InvalidReferenceError,
)
from horeb.schemas import AnalysisResult, Question, QuestionType

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
            result = runner.invoke(app, ["John 3:abc"])
        assert result.exit_code == EXIT_INVALID_REFERENCE

    def test_empty_passage_exit_code_3(self):
        with patch("horeb.cli.analyze", side_effect=EmptyPassageError("empty")):
            result = runner.invoke(app, ["John 3:16"])
        assert result.exit_code == EXIT_EMPTY_PASSAGE

    def test_citation_out_of_range_exit_code_4(self):
        with patch("horeb.cli.analyze", side_effect=CitationOutOfRangeError("bad cite")):
            result = runner.invoke(app, ["John 3:16-21"])
        assert result.exit_code == EXIT_CITATION_OUT_OF_RANGE

    def test_analysis_failed_exit_code_5(self):
        with patch("horeb.cli.analyze", side_effect=AnalysisFailedError("failed")):
            result = runner.invoke(app, ["John 3:16-21"])
        assert result.exit_code == EXIT_ANALYSIS_FAILED

    def test_success_exit_code_0(self):
        with patch("horeb.cli.analyze", return_value=_valid_result()):
            result = runner.invoke(app, ["John 3:16-21"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Error messages go to stderr
# ---------------------------------------------------------------------------

class TestErrorOutputRouting:
    def test_error_message_in_output(self):
        # CliRunner mixes stderr into output by default
        with patch("horeb.cli.analyze", side_effect=InvalidReferenceError("bad ref")):
            result = runner.invoke(app, ["bad ref"])
        assert "bad ref" in result.output

    def test_analysis_failed_message_in_output(self):
        with patch("horeb.cli.analyze", side_effect=AnalysisFailedError("model failed")):
            result = runner.invoke(app, ["John 3:16-21"])
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
