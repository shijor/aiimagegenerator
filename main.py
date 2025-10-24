"""
AI Image Generator - Main Application Entry Point
"""
import sys
from PyQt5.QtWidgets import QApplication

from config import APP_NAME, WINDOW_WIDTH, WINDOW_HEIGHT, MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT, STYLESHEET
from ui.main_window import MainWindow


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(STYLESHEET)

    # Create and show main window
    window = MainWindow()
    window.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
    window.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
    window.showMaximized()

    # Start event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
