"""
QSS stylesheet constants for the Horeb desktop app.
"""

PICKER_STYLE = """
QDialog {
    background-color: #1c1c1e;
    color: #ffffff;
}
QLabel {
    color: #ffffff;
}
QPushButton#book-btn {
    background-color: #3a3a3c;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 4px 6px;
    font-size: 12px;
}
QPushButton#book-btn:hover {
    background-color: #48484a;
}
QPushButton#book-btn[selected="true"] {
    background-color: #4caf50;
    color: #ffffff;
}
QLineEdit {
    background-color: #2c2c2e;
    color: #ffffff;
    border: 1px solid #3a3a3c;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 14px;
}
QScrollArea {
    background-color: #1c1c1e;
    border: none;
}
QScrollBar:vertical {
    background: #2c2c2e;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #48484a;
    border-radius: 4px;
}
QPushButton#go-btn {
    background-color: #4caf50;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 8px 24px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#go-btn:hover {
    background-color: #43a047;
}
QPushButton#close-btn {
    background-color: transparent;
    color: #8e8e93;
    border: none;
    font-size: 18px;
}
QPushButton#close-btn:hover {
    color: #ffffff;
}
QLabel#section-header {
    color: #8e8e93;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
    padding: 8px 0 4px 0;
}
QLabel#selected-book {
    background-color: #4caf50;
    color: #ffffff;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 14px;
    font-weight: bold;
}
QLabel#error-label {
    color: #ff453a;
    font-size: 12px;
}
"""

READER_STYLE = """
QTextEdit {
    background-color: #ffffff;
    border: none;
    padding: 40px;
    font-family: Georgia, serif;
    font-size: 18px;
    line-height: 1.8;
    color: #1c1c1e;
}
"""

SIDEBAR_STYLE = """
QWidget#sidebar {
    background-color: #f5f5f5;
}
QLineEdit#ref-input {
    background-color: #ffffff;
    border: 1px solid #d1d1d6;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 14px;
}
QPushButton#analyze-btn, QPushButton#similar-btn {
    background-color: #4caf50;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton#analyze-btn:hover, QPushButton#similar-btn:hover {
    background-color: #43a047;
}
QPushButton#analyze-btn:disabled, QPushButton#similar-btn:disabled {
    background-color: #a5d6a7;
}
QTextEdit#chat-history {
    background-color: #f5f5f5;
    border: none;
    font-size: 14px;
}
QLabel#status-label {
    color: #636366;
    font-size: 13px;
}
QGroupBox#options-group {
    font-size: 13px;
    color: #636366;
    border: none;
    margin-top: 0px;
}
"""

HEADER_STYLE = """
QWidget#header-bar {
    background-color: #1c1c1e;
}
QLabel#ref-label {
    color: #ffffff;
    font-size: 14px;
    font-weight: bold;
}
QLabel#version-badge {
    background-color: #3a3a3c;
    color: #8e8e93;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
}
QPushButton#picker-btn {
    background-color: #3a3a3c;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 13px;
}
QPushButton#picker-btn:hover {
    background-color: #48484a;
}
"""
