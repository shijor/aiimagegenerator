"""
AI Image Generator - Main Application Entry Point
"""
import sys
from PyQt5.QtWidgets import QApplication

from config import APP_NAME, WINDOW_WIDTH, WINDOW_HEIGHT, MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT, STYLESHEET
from ui.main_window import MainWindow


def main():
    """Main application entry point."""
    print("🔍 DEBUG: Starting AI Image Generator application")

    # Log system information
    import platform
    print(f"🔍 DEBUG: Platform: {platform.system()} {platform.release()}")
    print(f"🔍 DEBUG: Python version: {sys.version}")

    # Check PyTorch availability
    try:
        import torch
        print(f"🔍 DEBUG: PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"🔍 DEBUG: CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"🔍 DEBUG: CUDA version: {torch.version.cuda}")
        else:
            print("🔍 DEBUG: CUDA not available")
    except ImportError:
        print("🔍 DEBUG: PyTorch not available")

    # Check diffusers availability
    try:
        import diffusers
        print(f"🔍 DEBUG: Diffusers version: {diffusers.__version__}")
    except ImportError:
        print("🔍 DEBUG: Diffusers not available")

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(STYLESHEET)

    print("🔍 DEBUG: Creating main window")
    # Create and show main window
    window = MainWindow()
    window.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
    window.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
    window.showMaximized()

    print("🔍 DEBUG: Application startup complete, entering event loop")
    # Start event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
