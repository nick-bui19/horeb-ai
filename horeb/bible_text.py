from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum

import pythonbible as pb

from horeb.errors import InvalidReferenceError
from horeb.schemas import PassageData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_VERSES_BEFORE: int = 3
CONTEXT_VERSES_AFTER: int = 3
MAX_PASSAGE_VERSES: int = 30
MAX_SEGMENT_VERSES: int = 30      # max verses per book segment (chapter window)
MAX_BOOK_SEGMENTS: int = 60       # refuse to process books with more segments
MAX_BOOK_LLM_CALLS: int = 130     # hard ceiling: 60 segments × 2 calls + overhead

_VERSION = pb.Version.AMERICAN_STANDARD


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

@dataclass
class _VerseCoord:
    """A single chapter:verse coordinate within a known book."""
    chapter: int
    verse: int


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class Granularity(str, Enum):
    """Detected input granularity for the analyze command."""
    PASSAGE = "passage"   # explicit verse range, e.g. "John 3:16-21"
    CHAPTER = "chapter"   # chapter reference, e.g. "John 3"
    BOOK = "book"         # whole book, e.g. "1 Corinthians"


@dataclass
class Segment:
    """
    One deterministic segment of a book for the two-stage book pipeline.
    Typically one chapter; may be a verse window if the chapter is large.
    """
    book: pb.Book
    segment_index: int       # 0-based index in the segment list
    start_chapter: int
    start_verse: int
    end_chapter: int
    end_verse: int
    verse_count: int
    text: str                # labelled passage text ("[chapter:verse] …")
    reference: str           # human-readable reference, e.g. "Ruth 1:1-22"


# ---------------------------------------------------------------------------
# Verse ID construction
# ---------------------------------------------------------------------------

def _verse_id(book_value: int, chapter: int, verse: int) -> int:
    """Construct pythonbible integer verse ID from components."""
    return book_value * 1_000_000 + chapter * 1_000 + verse


@functools.lru_cache(maxsize=None)
def _get_verse_text(book_value: int, chapter: int, verse: int) -> str | None:
    """
    Return ASV text for the given coordinate, or None if out of range.

    lru_cache eliminates repeated retrieval for overlapping context windows
    and full-corpus scans in find_similar.
    """
    try:
        return pb.get_verse_text(_verse_id(book_value, chapter, verse), version=_VERSION)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reference parsing — strict (passage-only) and permissive (granularity-aware)
# ---------------------------------------------------------------------------

def _parse_reference(reference: str) -> pb.NormalizedReference:
    """
    Parse a Bible reference string into a NormalizedReference.
    Raises InvalidReferenceError on malformed or unrecognised input.
    Requires explicit verse range — chapter/book refs raise.
    """
    if not reference or not reference.strip():
        raise InvalidReferenceError("Reference cannot be empty")

    try:
        results = pb.get_references(reference)
    except Exception as exc:
        raise InvalidReferenceError(f"Could not parse reference: {reference!r}") from exc

    if not results:
        raise InvalidReferenceError(f"No Bible reference found in: {reference!r}")

    ref = results[0]

    # pythonbible returns None verse numbers for chapter-only refs (e.g. "John 3").
    if ref.start_verse is None or ref.end_verse is None:
        raise InvalidReferenceError(
            f"Reference {reference!r} requires a verse range "
            f"(e.g. 'John 3:16-21' or 'Psalm 23:1-6')"
        )

    return ref


def detect_granularity(reference: str) -> tuple[pb.NormalizedReference, Granularity]:
    """
    Permissive reference parse that returns detected granularity without raising
    for chapter-only or book-only references.

    Returns:
        (NormalizedReference, Granularity) — the reference may have None verse
        fields for CHAPTER and BOOK granularities; callers must not use
        start_verse/end_verse without checking.

    Raises:
        InvalidReferenceError: if the reference is empty or not parseable at all.
    """
    if not reference or not reference.strip():
        raise InvalidReferenceError("Reference cannot be empty")

    try:
        results = pb.get_references(reference)
    except Exception as exc:
        raise InvalidReferenceError(f"Could not parse reference: {reference!r}") from exc

    if not results:
        raise InvalidReferenceError(f"No Bible reference found in: {reference!r}")

    ref = results[0]

    if ref.start_verse is not None and ref.end_verse is not None:
        # Explicit verse range → passage
        return ref, Granularity.PASSAGE

    if ref.start_chapter is not None:
        # Chapter without verse range → chapter
        return ref, Granularity.CHAPTER

    # Book-only (start_chapter is None in pythonbible for book-only refs in some versions)
    return ref, Granularity.BOOK


# ---------------------------------------------------------------------------
# Verse counting
# ---------------------------------------------------------------------------

def _count_chapter_verses(book: pb.Book, chapter: int) -> int:
    """Return the number of verses in the given book/chapter."""
    try:
        return pb.get_number_of_verses(book, chapter)
    except Exception:
        return 0


def _count_passage_verses(ref: pb.NormalizedReference) -> int:
    """Count total verses in a potentially multi-chapter reference."""
    book = ref.book
    total = 0
    for chapter in range(ref.start_chapter, ref.end_chapter + 1):
        if chapter == ref.start_chapter == ref.end_chapter:
            total += ref.end_verse - ref.start_verse + 1
        elif chapter == ref.start_chapter:
            total += _count_chapter_verses(book, chapter) - ref.start_verse + 1
        elif chapter == ref.end_chapter:
            total += ref.end_verse
        else:
            total += _count_chapter_verses(book, chapter)
    return total


# ---------------------------------------------------------------------------
# Text building
# ---------------------------------------------------------------------------

def _build_passage_text(ref: pb.NormalizedReference) -> str:
    """Build labelled passage text as '[chapter:verse] text' lines."""
    book_value = ref.book.value
    lines: list[str] = []
    for chapter in range(ref.start_chapter, ref.end_chapter + 1):
        start_v = ref.start_verse if chapter == ref.start_chapter else 1
        end_v = (
            ref.end_verse if chapter == ref.end_chapter
            else _count_chapter_verses(ref.book, chapter)
        )
        for verse in range(start_v, end_v + 1):
            text = _get_verse_text(book_value, chapter, verse)
            if text is not None:
                lines.append(f"[{chapter}:{verse}] {text}")
    return "\n".join(lines)


def _build_text_for_range(
    book: pb.Book, start_chapter: int, start_verse: int, end_chapter: int, end_verse: int
) -> str:
    """Build labelled text for an arbitrary verse range."""
    book_value = book.value
    lines: list[str] = []
    for chapter in range(start_chapter, end_chapter + 1):
        sv = start_verse if chapter == start_chapter else 1
        ev = end_verse if chapter == end_chapter else _count_chapter_verses(book, chapter)
        for verse in range(sv, ev + 1):
            text = _get_verse_text(book_value, chapter, verse)
            if text is not None:
                lines.append(f"[{chapter}:{verse}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context clamping
# ---------------------------------------------------------------------------

def _walk_back(
    book: pb.Book, chapter: int, verse: int, n: int
) -> list[_VerseCoord]:
    """
    Walk backward n verses from (chapter, verse), exclusive of the start.
    Returns coordinates in chronological order. Stops at book start.
    """
    coords: list[_VerseCoord] = []
    ch, v = chapter, verse - 1
    while len(coords) < n:
        if v < 1:
            ch -= 1
            if ch < 1:
                break
            v = _count_chapter_verses(book, ch)
            if v == 0:
                break
        coords.append(_VerseCoord(ch, v))
        v -= 1
    coords.reverse()
    return coords


def _walk_forward(
    book: pb.Book, chapter: int, verse: int, n: int
) -> list[_VerseCoord]:
    """
    Walk forward n verses from (chapter, verse), exclusive of the start.
    Returns coordinates in chronological order. Stops at book end.
    """
    max_chapters = pb.get_number_of_chapters(book)
    coords: list[_VerseCoord] = []
    ch, v = chapter, verse + 1
    while len(coords) < n:
        chapter_max = _count_chapter_verses(book, ch)
        if v > chapter_max:
            ch += 1
            if ch > max_chapters:
                break
            v = 1
        coords.append(_VerseCoord(ch, v))
        v += 1
    return coords


def _coords_to_text(book_value: int, coords: list[_VerseCoord]) -> str | None:
    """Convert a list of verse coordinates to labelled text, or None if empty."""
    lines: list[str] = []
    for c in coords:
        text = _get_verse_text(book_value, c.chapter, c.verse)
        if text is not None:
            lines.append(f"[{c.chapter}:{c.verse}] {text}")
    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Book segmentation
# ---------------------------------------------------------------------------

def segment_book(
    book: pb.Book,
    max_segment_verses: int = MAX_SEGMENT_VERSES,
) -> list[Segment]:
    """
    Deterministically segment a book into a list of Segments.

    Strategy:
    - Prefer chapter boundaries: each chapter is one segment.
    - If a chapter exceeds max_segment_verses, split it into consecutive
      verse windows of max_segment_verses each.
    - Deterministic: same book always produces the same segments.

    Raises:
        InvalidReferenceError: if the resulting segment count exceeds MAX_BOOK_SEGMENTS.
    """
    num_chapters = pb.get_number_of_chapters(book)
    book_name = book.name.replace("_", " ").title()
    segments: list[Segment] = []
    seg_idx = 0

    for chapter in range(1, num_chapters + 1):
        chapter_verses = _count_chapter_verses(book, chapter)
        if chapter_verses == 0:
            continue

        if chapter_verses <= max_segment_verses:
            # Whole chapter fits in one segment
            text = _build_text_for_range(book, chapter, 1, chapter, chapter_verses)
            ref = f"{book_name} {chapter}:1-{chapter_verses}"
            segments.append(Segment(
                book=book,
                segment_index=seg_idx,
                start_chapter=chapter,
                start_verse=1,
                end_chapter=chapter,
                end_verse=chapter_verses,
                verse_count=chapter_verses,
                text=text,
                reference=ref,
            ))
            seg_idx += 1
        else:
            # Split chapter into verse windows
            start_v = 1
            while start_v <= chapter_verses:
                end_v = min(start_v + max_segment_verses - 1, chapter_verses)
                count = end_v - start_v + 1
                text = _build_text_for_range(book, chapter, start_v, chapter, end_v)
                ref = f"{book_name} {chapter}:{start_v}-{end_v}"
                segments.append(Segment(
                    book=book,
                    segment_index=seg_idx,
                    start_chapter=chapter,
                    start_verse=start_v,
                    end_chapter=chapter,
                    end_verse=end_v,
                    verse_count=count,
                    text=text,
                    reference=ref,
                ))
                seg_idx += 1
                start_v = end_v + 1

    if len(segments) > MAX_BOOK_SEGMENTS:
        raise InvalidReferenceError(
            f"{book_name} produces {len(segments)} segments "
            f"(maximum {MAX_BOOK_SEGMENTS}). Use a chapter range instead."
        )

    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_passage(reference: str) -> PassageData:
    """
    Retrieve a Bible passage by reference string.

    Raises:
        InvalidReferenceError: if the reference is malformed, unrecognised,
            or exceeds MAX_PASSAGE_VERSES.
    """
    ref = _parse_reference(reference)

    total_verses = _count_passage_verses(ref)
    if total_verses > MAX_PASSAGE_VERSES:
        raise InvalidReferenceError(
            f"Passage {reference!r} spans {total_verses} verses "
            f"(maximum {MAX_PASSAGE_VERSES}). Use a shorter range."
        )

    book_value = ref.book.value
    text = _build_passage_text(ref)

    before_coords = _walk_back(ref.book, ref.start_chapter, ref.start_verse, CONTEXT_VERSES_BEFORE)
    after_coords = _walk_forward(ref.book, ref.end_chapter, ref.end_verse, CONTEXT_VERSES_AFTER)

    return PassageData(
        reference=reference,
        book=book_value,
        start_chapter=ref.start_chapter,
        start_verse=ref.start_verse,
        end_chapter=ref.end_chapter,
        end_verse=ref.end_verse,
        text=text,
        context_before=_coords_to_text(book_value, before_coords),
        context_after=_coords_to_text(book_value, after_coords),
    )


def retrieve_chapter(book: pb.Book, chapter: int) -> PassageData:
    """
    Retrieve a full chapter as a PassageData.
    Used by the chapter-level analyze pipeline.
    """
    chapter_verses = _count_chapter_verses(book, chapter)
    if chapter_verses == 0:
        raise InvalidReferenceError(f"Chapter {chapter} of {book.name} has no verses.")

    book_value = book.value
    book_name = book.name.replace("_", " ").title()
    reference = f"{book_name} {chapter}:1-{chapter_verses}"
    text = _build_text_for_range(book, chapter, 1, chapter, chapter_verses)

    before_coords = _walk_back(book, chapter, 1, CONTEXT_VERSES_BEFORE)
    after_coords = _walk_forward(book, chapter, chapter_verses, CONTEXT_VERSES_AFTER)

    return PassageData(
        reference=reference,
        book=book_value,
        start_chapter=chapter,
        start_verse=1,
        end_chapter=chapter,
        end_verse=chapter_verses,
        text=text,
        context_before=_coords_to_text(book_value, before_coords),
        context_after=_coords_to_text(book_value, after_coords),
    )
