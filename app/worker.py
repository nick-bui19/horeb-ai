"""
EngineWorker — runs engine calls on a background QThread.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, Signal, Slot

from horeb.errors import (
    AnalysisFailedError,
    CitationOutOfRangeError,
    EmptyPassageError,
    InvalidReferenceError,
)


class EngineWorker(QObject):
    """
    Wraps a callable (engine function) to run on a QThread.

    Signals:
        result_ready(object): Emitted with the return value on success.
        error_occurred(str):  Emitted with an error message on failure.
        finished():           Always emitted last (success or failure).
    """

    result_ready = Signal(object)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, fn: Callable, *args: object, **kwargs: object) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.result_ready.emit(result)
        except (
            InvalidReferenceError,
            EmptyPassageError,
            CitationOutOfRangeError,
            AnalysisFailedError,
        ) as exc:
            self.error_occurred.emit(str(exc))
        except Exception as exc:
            self.error_occurred.emit(f"Unexpected error: {exc}")
        finally:
            self.finished.emit()
