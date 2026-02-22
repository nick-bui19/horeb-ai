from dataclasses import dataclass

import pythonbible as pb

from horeb.errors import InvalidReferenceError
from horeb.schemas import PassageData

# --- Constants ---

CONTEXT_VERSES_BEFORE: int = 3
CONTEXT_VERSES_AFTER: int = 3
MAX_PASSAGE_VERSES: int = 30

_VERSION = pb.Version.AMERICAN_STANDARD


# --- Internal types ---

@dataclass
class _VerseCoord:
    """A single chapter:verse coordinate within a known book."""
    chapter: int
    verse: int


# --- Verse ID construction ---

def _verse_id(book_value: int, chapter: int, verse: int) -> int:
    """Construct pythonbible integer verse ID from components."""
    return book_value * 1_000_000 + chapter * 1_000 + verse


def _get_verse_text(book_value: int, chapter: int, verse: int) -> str | None:
    """Return ASV text for the given coordinate, or None if out of range."""
    try:
        return pb.get_verse_text(_verse_id(book_value, chapter, verse), version=_VERSION)
    except (pb.VersionMissingVerseError, pb.InvalidVerseError, Exception):
        return None


# --- Reference parsing ---

def _parse_reference(reference: str) -> pb.NormalizedReference:
    """
    Parse a Bible reference string into a NormalizedReference.
    Raises InvalidReferenceError on malformed or unrecognised input.
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

    # pythonbible silently returns None verse numbers for chapter-only references
    # (e.g. "Psalm 23", "John 3") and malformed verse numbers (e.g. "John 3:abc").
    # Phase 1 requires explicit verse ranges.
    if ref.start_verse is None or ref.end_verse is None:
        raise InvalidReferenceError(
            f"Reference {reference!r} requires a verse range "
            f"(e.g. 'John 3:16-21' or 'Psalm 23:1-6')"
        )

    return ref


# --- Verse counting ---

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


# --- Text building ---

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


# --- Context clamping ---

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


# --- Public API ---

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
