"""
Tests for bible_text.py — reference parsing, text retrieval, boundary clamping.

Covers test matrix items:
- 1: Valid reference parsing
- 2: Invalid reference → InvalidReferenceError
- 3: Passage too long → InvalidReferenceError
- 4: Empty text (covered via EmptyPassageError in engine tests)
- 5: Context clamped at book start
- 6: Context clamped at book end
- 7: Single-chapter books
"""
import pytest

from horeb.bible_text import (
    MAX_PASSAGE_VERSES,
    retrieve_passage,
)
from horeb.errors import InvalidReferenceError


# ---------------------------------------------------------------------------
# Valid references
# ---------------------------------------------------------------------------

class TestValidReferences:
    def test_john_3_16_21(self):
        p = retrieve_passage("John 3:16-21")
        assert p.start_chapter == 3
        assert p.start_verse == 16
        assert p.end_chapter == 3
        assert p.end_verse == 21
        assert "[3:16]" in p.text
        assert "[3:21]" in p.text

    def test_psalm_23_1_6(self):
        p = retrieve_passage("Psalm 23:1-6")
        assert p.start_verse == 1
        assert p.end_verse == 6
        assert len(p.text) > 0

    def test_single_verse(self):
        p = retrieve_passage("John 3:16")
        assert p.start_verse == 16
        assert p.end_verse == 16
        assert "[3:16]" in p.text

    def test_passage_text_has_verse_labels(self):
        p = retrieve_passage("Genesis 1:1-3")
        for verse in range(1, 4):
            assert f"[1:{verse}]" in p.text

    def test_passage_text_is_nonempty(self):
        p = retrieve_passage("Romans 8:1-4")
        assert len(p.text.strip()) > 50


# ---------------------------------------------------------------------------
# Single-chapter books
# ---------------------------------------------------------------------------

class TestSingleChapterBooks:
    def test_jude_3(self):
        p = retrieve_passage("Jude 3")
        assert "[1:3]" in p.text
        assert p.start_verse == 3

    def test_jude_1_4(self):
        p = retrieve_passage("Jude 1:4")
        assert "[1:4]" in p.text

    def test_philemon_1_10(self):
        p = retrieve_passage("Philemon 1:10")
        assert p.book > 0
        assert len(p.text) > 0

    def test_obadiah_1_3(self):
        p = retrieve_passage("Obadiah 1:3")
        assert len(p.text) > 0


# ---------------------------------------------------------------------------
# Context boundary clamping
# ---------------------------------------------------------------------------

class TestContextBoundaryClamping:
    def test_genesis_1_1_has_no_context_before(self):
        p = retrieve_passage("Genesis 1:1")
        assert p.context_before is None

    def test_genesis_1_1_has_context_after(self):
        p = retrieve_passage("Genesis 1:1")
        assert p.context_after is not None
        assert "[1:2]" in p.context_after

    def test_revelation_22_21_has_no_context_after(self):
        p = retrieve_passage("Revelation 22:21")
        assert p.context_after is None

    def test_revelation_22_21_has_context_before(self):
        p = retrieve_passage("Revelation 22:21")
        assert p.context_before is not None

    def test_context_before_does_not_include_passage_verses(self):
        p = retrieve_passage("John 3:16-21")
        assert "[3:16]" not in (p.context_before or "")

    def test_context_after_does_not_include_passage_verses(self):
        p = retrieve_passage("John 3:16-21")
        assert "[3:21]" not in (p.context_after or "")

    def test_context_before_has_at_most_3_verses(self):
        p = retrieve_passage("John 3:16-21")
        # context_before has at most CONTEXT_VERSES_BEFORE verse label lines
        if p.context_before:
            labels = [line for line in p.context_before.splitlines() if line.startswith("[")]
            assert len(labels) <= 3

    def test_passage_at_chapter_start_crosses_context_to_previous_chapter(self):
        # John 4:1 — context before should come from chapter 3
        p = retrieve_passage("John 4:1")
        assert p.context_before is not None
        assert "[3:" in p.context_before


# ---------------------------------------------------------------------------
# Invalid references
# ---------------------------------------------------------------------------

class TestInvalidReferences:
    def test_empty_string(self):
        with pytest.raises(InvalidReferenceError):
            retrieve_passage("")

    def test_whitespace_only(self):
        with pytest.raises(InvalidReferenceError):
            retrieve_passage("   ")

    def test_chapter_without_verses(self):
        # "John 3" — no verse range → requires verse numbers
        with pytest.raises(InvalidReferenceError, match="verse range"):
            retrieve_passage("John 3")

    def test_malformed_verse_numbers(self):
        with pytest.raises(InvalidReferenceError):
            retrieve_passage("John 3:abc")

    def test_not_a_bible_reference(self):
        with pytest.raises(InvalidReferenceError):
            retrieve_passage("Hello world")


# ---------------------------------------------------------------------------
# Passage length limit
# ---------------------------------------------------------------------------

class TestPassageLengthLimit:
    def test_exceeds_max_raises_invalid_reference(self):
        # Psalm 119 without verse range fails on missing verses,
        # but a large explicit range should trigger MAX_PASSAGE_VERSES
        with pytest.raises(InvalidReferenceError, match="maximum"):
            retrieve_passage(f"Psalm 119:1-{MAX_PASSAGE_VERSES + 1}")

    def test_exactly_max_verses_accepted(self):
        # 30 verses from a long chapter — should not raise
        p = retrieve_passage(f"Psalm 119:1-{MAX_PASSAGE_VERSES}")
        assert len(p.text) > 0

    def test_one_over_max_rejected(self):
        with pytest.raises(InvalidReferenceError, match="maximum"):
            retrieve_passage(f"John 1:1-{MAX_PASSAGE_VERSES + 1}")
