"""
Shared fixtures for desktop app tests.
Sets QT_QPA_PLATFORM=offscreen so tests run headless in CI.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
