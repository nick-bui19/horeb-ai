"""
Tests for app/books.py — no Qt required.
"""
import pythonbible as pb
import pytest

from app.books import NT_BOOKS, OT_BOOKS, validate_navigation


def test_ot_books_count():
    assert len(OT_BOOKS) == 39


def test_nt_books_count():
    assert len(NT_BOOKS) == 27


def test_all_abbreviations_3_to_4_chars():
    all_books = OT_BOOKS + NT_BOOKS
    for book, abbrev in all_books:
        assert 2 <= len(abbrev) <= 4, (
            f"{book.name} abbreviation '{abbrev}' has length {len(abbrev)}, expected 2-4"
        )


def test_validate_navigation_valid():
    # John 3:16 is valid
    assert validate_navigation(pb.Book.JOHN, 3, 16) is None


def test_validate_navigation_chapter_zero():
    result = validate_navigation(pb.Book.JOHN, 0, 1)
    assert isinstance(result, str)
    assert result  # non-empty error message


def test_validate_navigation_chapter_too_large():
    result = validate_navigation(pb.Book.JOHN, 99, 1)
    assert isinstance(result, str)
    assert result


def test_validate_navigation_verse_too_large():
    result = validate_navigation(pb.Book.JOHN, 3, 999)
    assert isinstance(result, str)
    assert result


def test_validate_navigation_philemon_chapter_2():
    # Philemon has only 1 chapter
    result = validate_navigation(pb.Book.PHILEMON, 2, 1)
    assert isinstance(result, str)
    assert result


def test_validate_navigation_psalms_119_verse_176():
    # Psalm 119 has 176 verses — valid
    assert validate_navigation(pb.Book.PSALMS, 119, 176) is None


def test_validate_navigation_psalms_119_verse_177():
    # Psalm 119 has only 176 verses — invalid
    result = validate_navigation(pb.Book.PSALMS, 119, 177)
    assert isinstance(result, str)
    assert result
