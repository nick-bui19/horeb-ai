"""
MainWindow — assembles all components and owns the LLM provider + cache.
"""
from __future__ import annotations

from typing import Callable

import pythonbible as pb
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QWidget,
)

from app.picker import PickerModal
from app.reader import BibleReaderWidget
from app.renderer import result_to_html, similar_to_html
from app.sidebar import SidebarWidget
from app.styles import HEADER_STYLE
from app.worker import EngineWorker
from horeb.engine import analyze, find_similar
from horeb.llm import ClaudeProvider


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Horeb")

        self._llm = ClaudeProvider()
        self._cache: dict[tuple, object] = {}
        self._active_thread: QThread | None = None
        self._elapsed_seconds: int = 0

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        self._current_book: pb.Book = pb.Book.GENESIS
        self._current_chapter: int = 1

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header bar
        header = self._build_header()
        main_layout.addWidget(header)

        # Splitter: reader (left) + sidebar (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background: #d1d1d6; }")

        self._reader = BibleReaderWidget()
        splitter.addWidget(self._reader)

        self._sidebar = SidebarWidget()
        splitter.addWidget(self._sidebar)

        splitter.setSizes([820, 460])
        main_layout.addWidget(splitter, stretch=1)

        # Wire sidebar signals
        self._sidebar.analyze_requested.connect(self._on_analyze)
        self._sidebar.find_similar_requested.connect(self._on_find_similar)

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("header-bar")
        bar.setStyleSheet(HEADER_STYLE)
        bar.setFixedHeight(48)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        self._picker_btn = QPushButton("Open Book")
        self._picker_btn.setObjectName("picker-btn")
        self._picker_btn.clicked.connect(self._open_picker)
        layout.addWidget(self._picker_btn)

        self._ref_label = QLabel("–")
        self._ref_label.setObjectName("ref-label")
        layout.addWidget(self._ref_label)

        layout.addStretch()

        version_badge = QLabel("ASV")
        version_badge.setObjectName("version-badge")
        layout.addWidget(version_badge)

        return bar

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _open_picker(self) -> None:
        modal = PickerModal(self)
        modal.navigation_confirmed.connect(self._on_navigation)
        modal.exec()

    def _on_navigation(self, book: pb.Book, chapter: int, verse: int) -> None:
        self._current_book = book
        self._current_chapter = chapter
        book_name = book.name.replace("_", " ").title()
        self._ref_label.setText(f"{book_name} {chapter}  ASV")
        self._reader.navigate(book, chapter)

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    def _cache_key(self, command: str, **kwargs: object) -> tuple:
        return (command, tuple(sorted(kwargs.items())))

    # ------------------------------------------------------------------
    # Engine dispatch
    # ------------------------------------------------------------------

    def _run_engine(self, command: str, fn: Callable, **kwargs: object) -> None:
        key = self._cache_key(command, **kwargs)
        if key in self._cache:
            self._on_result(self._cache[key])
            return

        self._set_running(True, command)

        worker = EngineWorker(fn, **kwargs)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result_ready.connect(lambda r: self._cache.__setitem__(key, r))
        worker.result_ready.connect(self._on_result)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(lambda: self._set_running(False, ""))
        self._active_thread = thread
        thread.start()

    def _on_analyze(self, reference: str) -> None:
        self._sidebar.append_user_message(f'analyze "{reference}"')
        self._run_engine("analyze", analyze, reference=reference, llm=self._llm)

    def _on_find_similar(
        self,
        reference: str,
        scope_book: str | None,
        top_n: int,
        tags: bool,
    ) -> None:
        self._sidebar.append_user_message(f'find-similar "{reference}"')
        llm = self._llm if tags else None
        self._run_engine(
            "find_similar",
            find_similar,
            reference=reference,
            scope_book=scope_book,
            top_n=top_n,
            tags=tags,
            llm=llm,
        )

    # ------------------------------------------------------------------
    # Result / error handlers
    # ------------------------------------------------------------------

    def _on_result(self, result: object) -> None:
        from horeb.schemas import SimilarityResult

        if isinstance(result, SimilarityResult):
            html = similar_to_html(result)
        else:
            # PassageAnalysisResult / BookAnalysisResult / StudyGuideResult
            ref = getattr(result, "reference", "") or ""
            html = result_to_html(result, ref)

        self._sidebar.append_response(html)

    def _on_error(self, message: str) -> None:
        self._sidebar.append_error(message)

    # ------------------------------------------------------------------
    # Running state
    # ------------------------------------------------------------------

    def _set_running(self, running: bool, command: str = "") -> None:
        self._sidebar.set_running(running)
        if running:
            self._elapsed_seconds = 0
            self._elapsed_timer.start()
            label = "Analyzing" if command == "analyze" else "Finding similar passages"
            self._sidebar.set_status(f"⟳ {label}… (0s)", visible=True)
        else:
            self._elapsed_timer.stop()
            self._sidebar.set_status("", visible=False)

    def _tick_elapsed(self) -> None:
        self._elapsed_seconds += 1
        current_text = self._sidebar._status_label.text()
        # Replace the seconds count
        import re
        updated = re.sub(r"\(\d+s\)", f"({self._elapsed_seconds}s)", current_text)
        self._sidebar.set_status(updated, visible=True)

