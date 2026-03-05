"""
Book abbreviations and navigation validation for the Horeb desktop app.
"""
from __future__ import annotations

import pythonbible as pb

# ---------------------------------------------------------------------------
# OT / NT book lists: (Book enum, abbreviation)
# ---------------------------------------------------------------------------

OT_BOOKS: list[tuple[pb.Book, str]] = [
    (pb.Book.GENESIS, "Gen"),
    (pb.Book.EXODUS, "Exo"),
    (pb.Book.LEVITICUS, "Lev"),
    (pb.Book.NUMBERS, "Num"),
    (pb.Book.DEUTERONOMY, "Deu"),
    (pb.Book.JOSHUA, "Jos"),
    (pb.Book.JUDGES, "Judg"),
    (pb.Book.RUTH, "Rth"),
    (pb.Book.SAMUEL_1, "1Sa"),
    (pb.Book.SAMUEL_2, "2Sa"),
    (pb.Book.KINGS_1, "1Ki"),
    (pb.Book.KINGS_2, "2Ki"),
    (pb.Book.CHRONICLES_1, "1Ch"),
    (pb.Book.CHRONICLES_2, "2Ch"),
    (pb.Book.EZRA, "Eza"),
    (pb.Book.NEHEMIAH, "Neh"),
    (pb.Book.ESTHER, "Est"),
    (pb.Book.JOB, "Job"),
    (pb.Book.PSALMS, "Psa"),
    (pb.Book.PROVERBS, "Pro"),
    (pb.Book.ECCLESIASTES, "Ecc"),
    (pb.Book.SONG_OF_SONGS, "SS"),
    (pb.Book.ISAIAH, "Isa"),
    (pb.Book.JEREMIAH, "Jer"),
    (pb.Book.LAMENTATIONS, "Lam"),
    (pb.Book.EZEKIEL, "Ezk"),
    (pb.Book.DANIEL, "Dan"),
    (pb.Book.HOSEA, "Hos"),
    (pb.Book.JOEL, "Joe"),
    (pb.Book.AMOS, "Amo"),
    (pb.Book.OBADIAH, "Obd"),
    (pb.Book.JONAH, "Jon"),
    (pb.Book.MICAH, "Mic"),
    (pb.Book.NAHUM, "Nah"),
    (pb.Book.HABAKKUK, "Hab"),
    (pb.Book.ZEPHANIAH, "Zep"),
    (pb.Book.HAGGAI, "Hag"),
    (pb.Book.ZECHARIAH, "Zch"),
    (pb.Book.MALACHI, "Mal"),
]

NT_BOOKS: list[tuple[pb.Book, str]] = [
    (pb.Book.MATTHEW, "Mat"),
    (pb.Book.MARK, "Mar"),
    (pb.Book.LUKE, "Luk"),
    (pb.Book.JOHN, "Jn"),
    (pb.Book.ACTS, "Act"),
    (pb.Book.ROMANS, "Rom"),
    (pb.Book.CORINTHIANS_1, "1Co"),
    (pb.Book.CORINTHIANS_2, "2Co"),
    (pb.Book.GALATIANS, "Gal"),
    (pb.Book.EPHESIANS, "Eph"),
    (pb.Book.PHILIPPIANS, "Phi"),
    (pb.Book.COLOSSIANS, "Col"),
    (pb.Book.THESSALONIANS_1, "1Th"),
    (pb.Book.THESSALONIANS_2, "2Th"),
    (pb.Book.TIMOTHY_1, "1Ti"),
    (pb.Book.TIMOTHY_2, "2Ti"),
    (pb.Book.TITUS, "Tit"),
    (pb.Book.PHILEMON, "Phm"),
    (pb.Book.HEBREWS, "Heb"),
    (pb.Book.JAMES, "Jam"),
    (pb.Book.PETER_1, "1Pe"),
    (pb.Book.PETER_2, "2Pe"),
    (pb.Book.JOHN_1, "1Jo"),
    (pb.Book.JOHN_2, "2Jo"),
    (pb.Book.JOHN_3, "3Jo"),
    (pb.Book.JUDE, "Jud"),
    (pb.Book.REVELATION, "Rev"),
]


def validate_navigation(book: pb.Book, chapter: int, verse: int) -> str | None:
    """
    Validate book/chapter/verse for navigation.

    Returns an error message string if invalid, or None if valid.
    """
    if chapter < 1:
        return f"Chapter must be at least 1."

    num_chapters = pb.get_number_of_chapters(book)
    if chapter > num_chapters:
        book_name = book.name.replace("_", " ").title()
        return f"{book_name} has only {num_chapters} chapter(s)."

    if verse < 1:
        return f"Verse must be at least 1."

    num_verses = pb.get_number_of_verses(book, chapter)
    if verse > num_verses:
        book_name = book.name.replace("_", " ").title()
        return f"{book_name} {chapter} has only {num_verses} verse(s)."

    return None
