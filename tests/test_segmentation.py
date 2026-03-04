"""
Tests for bible_text.segment_book() — deterministic book segmentation.

All tests run against the real pythonbible API; no mocks needed since
segment_book() is fully deterministic.
"""
import pytest
import pythonbible as pb

from horeb.bible_text import (
    MAX_BOOK_SEGMENTS,
    _count_chapter_verses,
    segment_book,
)
from horeb.errors import InvalidReferenceError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_verse_coords(book: pb.Book) -> set[tuple[int, int]]:
    """Return set of all (chapter, verse) pairs for a book."""
    coords: set[tuple[int, int]] = set()
    for ch in range(1, pb.get_number_of_chapters(book) + 1):
        for v in range(1, _count_chapter_verses(book, ch) + 1):
            coords.add((ch, v))
    return coords


def _segment_verse_coords(seg) -> set[tuple[int, int]]:
    """Return the set of (chapter, verse) pairs covered by a Segment."""
    coords: set[tuple[int, int]] = set()
    book = seg.book
    for ch in range(seg.start_chapter, seg.end_chapter + 1):
        sv = seg.start_verse if ch == seg.start_chapter else 1
        ev = (
            seg.end_verse
            if ch == seg.end_chapter
            else _count_chapter_verses(book, ch)
        )
        for v in range(sv, ev + 1):
            coords.add((ch, v))
    return coords


# ---------------------------------------------------------------------------
# Ruth — full coverage and structural correctness
# ---------------------------------------------------------------------------

class TestSegmentBookRuth:
    def test_full_verse_coverage(self):
        """Union of all segment verse ranges covers every verse in Ruth."""
        segments = segment_book(pb.Book.RUTH)
        covered: set[tuple[int, int]] = set()
        for seg in segments:
            covered |= _segment_verse_coords(seg)
        assert covered == _all_verse_coords(pb.Book.RUTH)

    def test_no_verse_overlap(self):
        """No verse appears in two different segments."""
        segments = segment_book(pb.Book.RUTH)
        seen: set[tuple[int, int]] = set()
        for seg in segments:
            coords = _segment_verse_coords(seg)
            overlapping = coords & seen
            assert not overlapping, f"Verses {overlapping} appear in multiple segments"
            seen |= coords

    def test_determinism(self):
        """Calling segment_book(Ruth) twice returns identical output."""
        segs1 = segment_book(pb.Book.RUTH)
        segs2 = segment_book(pb.Book.RUTH)
        assert len(segs1) == len(segs2)
        for s1, s2 in zip(segs1, segs2):
            assert s1.segment_index == s2.segment_index
            assert s1.start_chapter == s2.start_chapter
            assert s1.start_verse == s2.start_verse
            assert s1.end_chapter == s2.end_chapter
            assert s1.end_verse == s2.end_verse

    def test_segment_indices_contiguous(self):
        """Segment indices are 0, 1, 2, … N-1 with no gaps."""
        segments = segment_book(pb.Book.RUTH)
        for i, seg in enumerate(segments):
            assert seg.segment_index == i


# ---------------------------------------------------------------------------
# Large-chapter splitting
# ---------------------------------------------------------------------------

class TestLargeChapterSplit:
    def test_large_chapter_produces_multiple_segments(self):
        """A chapter exceeding max_segment_verses is split into verse windows.

        Ruth chapter 1 has 22 verses. With max_segment_verses=10 it splits
        into at least 3 windows.
        """
        segments = segment_book(pb.Book.RUTH, max_segment_verses=10)
        ch1_segs = [
            s for s in segments
            if s.start_chapter == 1 and s.end_chapter == 1
        ]
        assert len(ch1_segs) > 1

    def test_split_segments_cover_full_chapter(self):
        """Split segments together cover all verses in that chapter."""
        segments = segment_book(pb.Book.RUTH, max_segment_verses=10)
        ch1_segs = [
            s for s in segments
            if s.start_chapter == 1 and s.end_chapter == 1
        ]
        covered: set[int] = set()
        for seg in ch1_segs:
            for v in range(seg.start_verse, seg.end_verse + 1):
                covered.add(v)
        expected = set(range(1, _count_chapter_verses(pb.Book.RUTH, 1) + 1))
        assert covered == expected

    def test_each_split_segment_within_verse_budget(self):
        """Every segment respects the max_segment_verses limit."""
        max_v = 10
        segments = segment_book(pb.Book.RUTH, max_segment_verses=max_v)
        for seg in segments:
            assert seg.verse_count <= max_v


# ---------------------------------------------------------------------------
# Single-chapter book
# ---------------------------------------------------------------------------

class TestSingleChapterBook:
    def test_philemon_produces_one_segment(self):
        """Philemon (1 chapter, 25 verses ≤ 30) produces exactly 1 segment."""
        segments = segment_book(pb.Book.PHILEMON)
        assert len(segments) == 1

    def test_philemon_segment_covers_full_book(self):
        """Single segment covers the entire book."""
        seg = segment_book(pb.Book.PHILEMON)[0]
        assert seg.start_chapter == 1
        assert seg.start_verse == 1
        assert seg.end_chapter == 1
        assert seg.end_verse == _count_chapter_verses(pb.Book.PHILEMON, 1)


# ---------------------------------------------------------------------------
# Exceeds MAX_BOOK_SEGMENTS
# ---------------------------------------------------------------------------

class TestExceedsMaxSegments:
    def test_psalms_exceeds_max_segments_raises(self):
        """Psalms has 150 chapters, far exceeding MAX_BOOK_SEGMENTS=60."""
        with pytest.raises(InvalidReferenceError, match="segments"):
            segment_book(pb.Book.PSALMS)
