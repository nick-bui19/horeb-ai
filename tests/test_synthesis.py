"""
Tests for the book pipeline synthesis stage and verify_synthesis_grounding().

Covers:
- verify_synthesis_grounding() all failure modes (synthesis-focused view)
- Partial failure threshold logic in analyze_book()
- failed_segments propagation to BookAnalysisResult
- Synthesis user prompt structure (no raw Bible text beyond verse_texts param)
"""
import re
from unittest.mock import patch

import pytest
import pythonbible as pb

from horeb.bible_text import Segment
from horeb.engine import analyze_book, verify_synthesis_grounding
from horeb.errors import AnalysisFailedError, CitationOutOfRangeError
from horeb.prompts import build_synthesis_user_prompt
from horeb.schemas import (
    BookAnalysisResult,
    OutlineSection,
    SegmentFailure,
    SegmentResult,
)
from tests.conftest import FixtureLLMProvider, load_fixture


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_seg_result(idx: int) -> SegmentResult:
    return SegmentResult(
        segment_index=idx,
        outline_label="Test outline label",
        summary=["A.", "B.", "C."],
    )


def _make_seg_failure(idx: int) -> SegmentFailure:
    return SegmentFailure(
        segment_index=idx,
        chapter_start=idx + 1,
        chapter_end=idx + 1,
        error="Simulated failure",
    )


def _make_fake_segments(n: int) -> list[Segment]:
    """Create n fake Segment objects (chapters 1..n of Ruth)."""
    return [
        Segment(
            book=pb.Book.RUTH,
            segment_index=i,
            start_chapter=i + 1,
            start_verse=1,
            end_chapter=i + 1,
            end_verse=22,
            verse_count=22,
            text="[1:1] In the days when the judges ruled.",
            reference=f"Ruth {i + 1}:1-22",
        )
        for i in range(n)
    ]


def _valid_book_result(source_segments: list[int] | None = None) -> BookAnalysisResult:
    return BookAnalysisResult(
        summary=["A.", "B.", "C."],
        outline=[
            OutlineSection(
                title="Opening narrative",
                start_verse="1:1",
                end_verse="1:22",
                source_segments=source_segments if source_segments is not None else [0],
            )
        ],
        failed_segments=[],
    )


# ---------------------------------------------------------------------------
# verify_synthesis_grounding() — grounding validation
# ---------------------------------------------------------------------------

class TestVerifySynthesisGrounding:
    def _make_segment(self, idx: int) -> Segment:
        return Segment(
            book=pb.Book.RUTH,
            segment_index=idx,
            start_chapter=idx + 1,
            start_verse=1,
            end_chapter=idx + 1,
            end_verse=22,
            verse_count=22,
            text="[1:1] In the days when the judges ruled.",
            reference=f"Ruth {idx + 1}:1-22",
        )

    def test_happy_path_passes(self):
        """Valid source_segments with matching indices — no exception raised."""
        verify_synthesis_grounding(
            _valid_book_result(source_segments=[0]),
            [_make_seg_result(0)],
            [self._make_segment(0)],
        )  # must not raise

    def test_empty_source_segments_raises(self):
        """Outline section with empty source_segments raises CitationOutOfRangeError."""
        book_result = BookAnalysisResult(
            summary=["A.", "B.", "C."],
            outline=[
                OutlineSection(
                    title="Empty section",
                    start_verse="1:1",
                    end_verse="1:5",
                    source_segments=[],
                )
            ],
        )
        with pytest.raises(CitationOutOfRangeError, match="empty source_segments"):
            verify_synthesis_grounding(book_result, [_make_seg_result(0)])

    def test_invalid_segment_index_raises(self):
        """Outline section referencing a non-existent segment index raises."""
        book_result = BookAnalysisResult(
            summary=["A.", "B.", "C."],
            outline=[
                OutlineSection(
                    title="Bad section",
                    start_verse="1:1",
                    end_verse="1:22",
                    source_segments=[99],
                )
            ],
        )
        with pytest.raises(CitationOutOfRangeError, match="does not exist"):
            verify_synthesis_grounding(book_result, [_make_seg_result(0)])


# ---------------------------------------------------------------------------
# analyze_book() partial failure threshold
# ---------------------------------------------------------------------------

class TestAnalyzeBookPartialFailures:
    """Test partial failure threshold and failed_segments propagation.

    segment_book() is patched to return controlled fake segments.
    _run_segment() is patched to return success or failure per segment index.
    verify_synthesis_grounding() is patched to skip citation range checks so
    tests can focus purely on failure-threshold and propagation logic.
    """

    def _run_with_n_segments(
        self,
        n: int,
        failing_indices: set[int],
        synthesis_fixture: str,
    ) -> BookAnalysisResult:
        fake_segs = _make_fake_segments(n)
        llm = FixtureLLMProvider(load_fixture(synthesis_fixture, subdir="book"))

        def fake_run_segment(seg, _llm, _sys):
            if seg.segment_index in failing_indices:
                return _make_seg_failure(seg.segment_index), 0
            return _make_seg_result(seg.segment_index), 1

        with patch("horeb.engine.segment_book", return_value=fake_segs):
            with patch("horeb.engine._run_segment", side_effect=fake_run_segment):
                with patch("horeb.engine.verify_synthesis_grounding"):
                    return analyze_book("Ruth", llm=llm)

    def test_one_failure_in_four_continues(self):
        """1/4 segments fail (25%) < 30% threshold → pipeline continues."""
        result = self._run_with_n_segments(
            n=4,
            failing_indices={2},
            synthesis_fixture="book_synthesis_valid.json",
        )
        assert isinstance(result, BookAnalysisResult)

    def test_three_failures_in_four_raises(self):
        """3/4 segments fail (75%) > 30% threshold → AnalysisFailedError."""
        fake_segs = _make_fake_segments(4)
        llm = FixtureLLMProvider("")

        def fake_run_segment(seg, _llm, _sys):
            if seg.segment_index in {0, 1, 2}:
                return _make_seg_failure(seg.segment_index), 0
            return _make_seg_result(seg.segment_index), 1

        with patch("horeb.engine.segment_book", return_value=fake_segs):
            with patch("horeb.engine._run_segment", side_effect=fake_run_segment):
                with pytest.raises(AnalysisFailedError, match="Too many segment failures"):
                    analyze_book("Ruth", llm=llm)

    def test_failed_segments_propagate_to_output(self):
        """Indices of failed segments appear in BookAnalysisResult.failed_segments."""
        result = self._run_with_n_segments(
            n=4,
            failing_indices={2},
            synthesis_fixture="book_synthesis_valid.json",
        )
        assert 2 in result.failed_segments

    def test_successful_segments_not_in_failed_list(self):
        """Indices that succeeded are not in failed_segments."""
        result = self._run_with_n_segments(
            n=4,
            failing_indices={2},
            synthesis_fixture="book_synthesis_valid.json",
        )
        for idx in (0, 1, 3):
            assert idx not in result.failed_segments


# ---------------------------------------------------------------------------
# Synthesis user prompt structure
# ---------------------------------------------------------------------------

class TestSynthesisPromptStructure:
    def test_no_raw_verse_labels_without_verse_texts(self):
        """Without verse_texts, synthesis prompt contains no [ch:v] raw verse text."""
        segs = [_make_seg_result(0), _make_seg_result(1)]
        prompt = build_synthesis_user_prompt(segs, failed_segments=[])
        verse_label_count = len(re.findall(r"\[\d+:\d+\]", prompt))
        assert verse_label_count == 0

    def test_with_verse_texts_only_cited_verses_appear(self):
        """With verse_texts, only the cited [ref] labels appear — no extra."""
        segs = [_make_seg_result(0)]
        verse_texts = {0: [("1:1", "In the days when the judges ruled.")]}
        prompt = build_synthesis_user_prompt(segs, failed_segments=[], verse_texts=verse_texts)
        labels = re.findall(r"\[(\d+:\d+)\]", prompt)
        assert labels == ["1:1"]

    def test_failed_segment_gap_markers_included(self):
        """Failed segments appear as ANALYSIS FAILED gap markers."""
        segs = [_make_seg_result(0)]
        failures = [SegmentFailure(segment_index=1, chapter_start=2, chapter_end=2, error="err")]
        prompt = build_synthesis_user_prompt(segs, failed_segments=failures)
        assert "ANALYSIS FAILED" in prompt
        assert "Segment 1" in prompt

    def test_segment_summaries_included(self):
        """Segment summaries are included in synthesis prompt."""
        seg = SegmentResult(
            segment_index=0,
            outline_label="Test label",
            summary=["First point.", "Second point.", "Third point."],
        )
        prompt = build_synthesis_user_prompt([seg], failed_segments=[])
        assert "First point." in prompt
        assert "Second point." in prompt
