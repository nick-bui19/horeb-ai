"""
Entry point for the Horeb desktop app.

Run with:
    uv run python app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `horeb` and `app` are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from PySide6.QtWidgets import QApplication

from app.window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Horeb")

    window = MainWindow()
    window.resize(1280, 800)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
