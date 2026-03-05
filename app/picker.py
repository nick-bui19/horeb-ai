"""
PickerModal — book/chapter/verse navigation dialog.
"""
from __future__ import annotations

import pythonbible as pb
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.books import NT_BOOKS, OT_BOOKS, validate_navigation
from app.styles import PICKER_STYLE


class PickerModal(QDialog):
    """
    Dialog for choosing a Bible book, chapter, and verse.

    Emits navigation_confirmed(book, chapter, verse) when the user confirms.
    """

    navigation_confirmed = Signal(object, int, int)  # (pb.Book, chapter, verse)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Navigate")
        self.setStyleSheet(PICKER_STYLE)
        self.setMinimumSize(600, 520)
        self.setModal(True)

        self._selected_book: pb.Book = pb.Book.GENESIS
        self._book_buttons: dict[pb.Book, QPushButton] = {}

        self._build_ui()
        self._update_book_label()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # Top bar: selected book label + close button
        top_row = QHBoxLayout()
        self._book_label = QLabel()
        self._book_label.setObjectName("selected-book")
        top_row.addWidget(self._book_label)
        top_row.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setObjectName("close-btn")
        close_btn.setFixedSize(28, 28)
        close_btn.clicked.connect(self.reject)
        top_row.addWidget(close_btn)
        root.addLayout(top_row)

        # Chapter + Verse inputs
        inputs_row = QHBoxLayout()
        inputs_row.setSpacing(8)

        ch_label = QLabel("Chapter:")
        ch_label.setStyleSheet("color: #8e8e93; font-size: 13px;")
        inputs_row.addWidget(ch_label)
        self._chapter_edit = QLineEdit("1")
        self._chapter_edit.setFixedWidth(70)
        self._chapter_edit.returnPressed.connect(self._on_go)
        inputs_row.addWidget(self._chapter_edit)

        v_label = QLabel("Verse:")
        v_label.setStyleSheet("color: #8e8e93; font-size: 13px;")
        inputs_row.addWidget(v_label)
        self._verse_edit = QLineEdit("1")
        self._verse_edit.setFixedWidth(70)
        self._verse_edit.returnPressed.connect(self._on_go)
        inputs_row.addWidget(self._verse_edit)

        inputs_row.addStretch()

        go_btn = QPushButton("Go")
        go_btn.setObjectName("go-btn")
        go_btn.clicked.connect(self._on_go)
        inputs_row.addWidget(go_btn)
        root.addLayout(inputs_row)

        # Error label (hidden until validation fails)
        self._error_label = QLabel("")
        self._error_label.setObjectName("error-label")
        self._error_label.setVisible(False)
        root.addWidget(self._error_label)

        # Scrollable grid of book buttons
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        grid_container = QWidget()
        grid_container.setStyleSheet("background-color: #1c1c1e;")
        grid_layout = QVBoxLayout(grid_container)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        grid_layout.setSpacing(8)

        # OT section
        ot_header = self._make_section_header("Old Testament")
        grid_layout.addWidget(ot_header)
        ot_grid = self._make_book_grid(OT_BOOKS, cols=11)
        grid_layout.addLayout(ot_grid)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #3a3a3c;")
        grid_layout.addWidget(sep)

        # NT section
        nt_header = self._make_section_header("New Testament")
        grid_layout.addWidget(nt_header)
        nt_grid = self._make_book_grid(NT_BOOKS, cols=11)
        grid_layout.addLayout(nt_grid)

        grid_layout.addStretch()
        scroll.setWidget(grid_container)
        root.addWidget(scroll)

    def _make_section_header(self, text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("section-header")
        return label

    def _make_book_grid(
        self, books: list[tuple[pb.Book, str]], cols: int
    ) -> QGridLayout:
        grid = QGridLayout()
        grid.setSpacing(4)
        for idx, (book, abbrev) in enumerate(books):
            btn = QPushButton(abbrev)
            btn.setObjectName("book-btn")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda checked=False, b=book: self._select_book(b))
            self._book_buttons[book] = btn
            grid.addWidget(btn, idx // cols, idx % cols)
        return grid

    # ------------------------------------------------------------------
    # Interaction logic
    # ------------------------------------------------------------------

    def _select_book(self, book: pb.Book) -> None:
        # Remove selection from previous
        if self._selected_book in self._book_buttons:
            prev_btn = self._book_buttons[self._selected_book]
            prev_btn.setProperty("selected", "false")
            prev_btn.style().unpolish(prev_btn)
            prev_btn.style().polish(prev_btn)
            prev_btn.setStyleSheet("")  # reset inline style

        self._selected_book = book

        # Apply selection style
        if book in self._book_buttons:
            btn = self._book_buttons[book]
            btn.setStyleSheet("background-color: #4caf50; color: #ffffff;")

        self._update_book_label()
        self._error_label.setVisible(False)

    def _update_book_label(self) -> None:
        book_name = self._selected_book.name.replace("_", " ").title()
        self._book_label.setText(book_name)

    def _on_go(self) -> None:
        try:
            chapter = int(self._chapter_edit.text().strip())
        except ValueError:
            self._show_error("Chapter must be a number.")
            return

        try:
            verse = int(self._verse_edit.text().strip())
        except ValueError:
            self._show_error("Verse must be a number.")
            return

        error = validate_navigation(self._selected_book, chapter, verse)
        if error:
            self._show_error(error)
            return

        self._error_label.setVisible(False)
        self.navigation_confirmed.emit(self._selected_book, chapter, verse)
        self.accept()

    def _show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setVisible(True)
