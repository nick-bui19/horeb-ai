"""
SidebarWidget — reference input, action buttons, options, and chat history.
"""
from __future__ import annotations

import pythonbible as pb
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.books import NT_BOOKS, OT_BOOKS
from app.styles import SIDEBAR_STYLE

# "Any book" sentinel for scope combo
_ANY_SENTINEL = "__any__"


class SidebarWidget(QWidget):
    """
    Right-hand panel: inputs at top, collapsible options, chat history below.

    Signals:
        analyze_requested(str): reference string
        find_similar_requested(str, str|None, int, bool):
            (reference, scope_book_name|None, top_n, tags)
    """

    analyze_requested = Signal(str)
    find_similar_requested = Signal(str, object, int, bool)  # (ref, scope, top_n, tags)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setStyleSheet(SIDEBAR_STYLE)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Reference input
        ref_label = QLabel("Reference:")
        ref_label.setStyleSheet("font-size: 13px; color: #636366;")
        root.addWidget(ref_label)

        self._ref_input = QLineEdit()
        self._ref_input.setObjectName("ref-input")
        self._ref_input.setPlaceholderText("e.g. John 3:16-21 or Ruth")
        self._ref_input.returnPressed.connect(self._on_analyze)
        root.addWidget(self._ref_input)

        # Action buttons
        btn_row = QHBoxLayout()
        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.setObjectName("analyze-btn")
        self._analyze_btn.clicked.connect(self._on_analyze)
        btn_row.addWidget(self._analyze_btn)

        self._similar_btn = QPushButton("Find Similar")
        self._similar_btn.setObjectName("similar-btn")
        self._similar_btn.clicked.connect(self._on_find_similar)
        btn_row.addWidget(self._similar_btn)
        root.addLayout(btn_row)

        # Collapsible options group
        self._options_group = QGroupBox("▶  Find Similar Options")
        self._options_group.setObjectName("options-group")
        self._options_group.setCheckable(False)
        self._options_group.setFlat(True)
        self._options_group.setStyleSheet(
            "QGroupBox { font-size:13px; color:#636366; cursor:pointer; }"
            "QGroupBox::title { subcontrol-origin: margin; padding: 0 4px; }"
        )
        self._options_group.mousePressEvent = self._toggle_options  # type: ignore[method-assign]

        opts_layout = QVBoxLayout()
        opts_layout.setContentsMargins(4, 4, 4, 4)
        opts_layout.setSpacing(6)

        # Scope book combo
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self._scope_combo = QComboBox()
        self._scope_combo.addItem("Any book", _ANY_SENTINEL)
        for book, abbrev in OT_BOOKS + NT_BOOKS:
            name = book.name.replace("_", " ").title()
            self._scope_combo.addItem(f"{name} ({abbrev})", book.name)
        scope_row.addWidget(self._scope_combo, stretch=1)
        opts_layout.addLayout(scope_row)

        # Top N spin
        topn_row = QHBoxLayout()
        topn_row.addWidget(QLabel("Top N:"))
        self._topn_spin = QSpinBox()
        self._topn_spin.setRange(1, 50)
        self._topn_spin.setValue(10)
        topn_row.addWidget(self._topn_spin)
        topn_row.addStretch()
        opts_layout.addLayout(topn_row)

        # Tags checkbox
        self._tags_check = QCheckBox("Tags (adds LLM call)")
        opts_layout.addWidget(self._tags_check)

        self._options_widget = QWidget()
        self._options_widget.setLayout(opts_layout)
        self._options_widget.setVisible(False)  # collapsed by default

        group_wrapper = QVBoxLayout()
        group_wrapper.setContentsMargins(0, 0, 0, 0)
        group_wrapper.addWidget(self._options_widget)
        self._options_group.setLayout(group_wrapper)
        root.addWidget(self._options_group)

        # Status bar (hidden when idle)
        self._status_label = QLabel("")
        self._status_label.setObjectName("status-label")
        self._status_label.setVisible(False)
        root.addWidget(self._status_label)

        # Chat history (read-only HTML)
        self._chat = QTextEdit()
        self._chat.setObjectName("chat-history")
        self._chat.setReadOnly(True)
        self._chat.setOpenExternalLinks(False)
        root.addWidget(self._chat, stretch=1)

    # ------------------------------------------------------------------
    # Public API (called by MainWindow)
    # ------------------------------------------------------------------

    def set_running(self, running: bool) -> None:
        self._analyze_btn.setEnabled(not running)
        self._similar_btn.setEnabled(not running)
        self._ref_input.setEnabled(not running)

    def set_status(self, text: str, visible: bool = True) -> None:
        self._status_label.setText(text)
        self._status_label.setVisible(visible)

    def append_user_message(self, text: str) -> None:
        self._chat.append(
            f'<div style="margin: 8px 0; padding: 8px 14px; background: #e0e0e0; '
            f'border-radius: 16px; display:inline-block; max-width:80%; '
            f'color: #1c1c1e; font-size:13px;">{_escape(text)}</div><br>'
        )

    def append_response(self, html: str) -> None:
        self._chat.append(
            f'<div style="margin: 8px 0; background: #ffffff; '
            f'border-radius: 8px; padding: 12px; '
            f'border: 1px solid #e0e0e0;">{html}</div><br>'
        )

    def append_error(self, message: str) -> None:
        self._chat.append(
            f'<div style="margin: 8px 0; color: #ff453a; font-size:13px;">'
            f'<strong>Error:</strong> {_escape(message)}</div><br>'
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_analyze(self) -> None:
        ref = self._ref_input.text().strip()
        if ref:
            self.analyze_requested.emit(ref)

    def _on_find_similar(self) -> None:
        ref = self._ref_input.text().strip()
        if not ref:
            return
        scope_data = self._scope_combo.currentData()
        scope_book = None if scope_data == _ANY_SENTINEL else scope_data
        top_n = self._topn_spin.value()
        tags = self._tags_check.isChecked()
        self.find_similar_requested.emit(ref, scope_book, top_n, tags)

    def _toggle_options(self, _event=None) -> None:
        visible = not self._options_widget.isVisible()
        self._options_widget.setVisible(visible)
        arrow = "▼" if visible else "▶"
        self._options_group.setTitle(f"{arrow}  Find Similar Options")


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
