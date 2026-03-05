"""
BibleReaderWidget — read-only chapter display.
"""
from __future__ import annotations

import pythonbible as pb
from PySide6.QtWidgets import QTextEdit

from app.renderer import chapter_text_to_html
from app.styles import READER_STYLE
from horeb.bible_text import retrieve_chapter


class BibleReaderWidget(QTextEdit):
    """
    Read-only QTextEdit that displays a Bible chapter with HTML formatting.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(READER_STYLE)
        self.setPlaceholderText("Open a chapter using the picker above.")

    def navigate(self, book: pb.Book, chapter: int) -> None:
        """
        Load and display the given chapter. Synchronous — retrieve_chapter is fast.
        """
        data = retrieve_chapter(book, chapter)
        html = chapter_text_to_html(data.text, chapter)
        self.setHtml(html)
        self.moveCursor(self.textCursor().MoveOperation.Start)
