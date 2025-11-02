"""
Main application window.
"""
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QSplitter, QStackedWidget, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.model_manager import ModelManager
from services.image_generator import ImageGenerationService
from services.memory_manager import MemoryManager
from services.text_enhancer import TextEnhancerService
from models.settings import SettingsManager, AppSettings
from ui.components.vertical_toolbar import VerticalToolbar
from ui.panels.image_generation_panel import ImageGenerationPanel
from ui.panels.model_management_panel import ModelManagementPanel
from ui.panels.settings_panel import SettingsPanel


class MainWindow(QMainWindow):
    """Main application window with modular panels."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Generator")

        # Initialize services (without loading model)
        self.settings_manager = SettingsManager()
        self.model_manager = ModelManager()
        self.image_service = ImageGenerationService()
        self.text_enhancer = TextEnhancerService(self.settings_manager)
        self.memory_manager = MemoryManager(self.image_service, self.settings_manager)

        # Don't load model on startup - use lazy loading instead
        # Model will be loaded when first generation is requested

        # Start memory monitoring
        self.memory_manager.start_monitoring()

        # Create UI
        self._init_ui()

        # Apply saved theme on startup
        self._apply_saved_theme()

        # Connect signals
        self.toolbar.mode_changed.connect(self._on_mode_changed)
        self.memory_manager.model_auto_unloaded.connect(self._on_model_auto_unloaded)

        # Connect model management to image generation panel
        self.model_mgmt_panel.model_installed.connect(self.image_gen_panel._refresh_model_dropdown)

        # Initialize fullscreen controls
        self._init_fullscreen_controls()

    def _init_ui(self):
        """Initialize the user interface."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        layout = QVBoxLayout(central_widget)

        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Horizontal)

        # Create toolbar
        self.toolbar = VerticalToolbar()

        # Create sidebar stack
        self.sidebar_stack = QStackedWidget()
        self.sidebar_stack.setMaximumWidth(350)

        # Create panels
        self.image_gen_panel = ImageGenerationPanel(self.model_manager, self.image_service, self.text_enhancer)
        self.model_mgmt_panel = ModelManagementPanel(self.model_manager)
        self.settings_panel = SettingsPanel(self.settings_manager, self.model_manager)

        # Add panels to sidebar stack
        self.sidebar_stack.addWidget(self.image_gen_panel.sidebar)
        self.sidebar_stack.addWidget(self.model_mgmt_panel.sidebar)
        self.sidebar_stack.addWidget(self.settings_panel.sidebar)

        # Create main area stack
        self.main_area_stack = QStackedWidget()

        # Add main areas to stack
        self.main_area_stack.addWidget(self.image_gen_panel.main_area)
        self.main_area_stack.addWidget(self.model_mgmt_panel.main_area)
        self.main_area_stack.addWidget(self.settings_panel.main_area)

        # Add widgets to splitter
        self.splitter.addWidget(self.toolbar)
        self.splitter.addWidget(self.sidebar_stack)
        self.splitter.addWidget(self.main_area_stack)

        # Set better proportions for screen space utilization
        # Toolbar: 60px, Sidebar: 350px, Main area: rest of space
        self.splitter.setSizes([60, 350, 1000])
        self.splitter.setStretchFactor(0, 0)  # Toolbar fixed width
        self.splitter.setStretchFactor(1, 0)  # Sidebar fixed width
        self.splitter.setStretchFactor(2, 1)  # Main area expands

        layout.addWidget(self.splitter)

    def _apply_saved_theme(self):
        """Apply the saved theme on application startup."""
        # Load saved settings
        settings = self.settings_manager.load_settings()
        theme = settings.theme

        # Apply the theme to the application
        if theme == "light":
            stylesheet = self._get_light_theme_stylesheet()
        elif theme == "dark":
            stylesheet = self._get_dark_theme_stylesheet()
        else:  # default
            stylesheet = ""

        self.setStyleSheet(stylesheet)

        # Force a repaint of all widgets
        self.update()
        for child in self.findChildren(QWidget):
            child.update()

    def _get_light_theme_stylesheet(self) -> str:
        """Get the light theme stylesheet."""
        return """
            /* Light Theme Styles */
            QWidget {
                background-color: #ffffff;
                color: #333333;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                margin-top: 1ex;
                background-color: #fafafa;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #1976D2;
                font-weight: bold;
            }

            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px 10px;
                color: #333333;
            }

            QPushButton:hover {
                background-color: #e8f4fd;
                border-color: #1976D2;
            }

            QPushButton:pressed {
                background-color: #d1e7f5;
            }

            QLineEdit, QTextEdit, QComboBox {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 3px;
                color: #333333;
            }

            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border-color: #1976D2;
            }

            QListWidget {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                color: #333333;
            }

            QListWidget::item:selected {
                background-color: #e8f4fd;
                color: #1976D2;
            }

            QScrollBar:vertical {
                background-color: #f5f5f5;
                width: 12px;
            }

            QScrollBar::handle:vertical {
                background-color: #cccccc;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #aaaaaa;
            }
        """

    def _get_dark_theme_stylesheet(self) -> str:
        """Get the dark theme stylesheet."""
        return """
            /* Dark Theme Styles */
            QWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                background-color: #3a3a3a;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #64b5f6;
                font-weight: bold;
            }

            QPushButton {
                background-color: #404040;
                border: 1px solid #666666;
                border-radius: 4px;
                padding: 5px 10px;
                color: #e0e0e0;
            }

            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #64b5f6;
            }

            QPushButton:pressed {
                background-color: #333333;
            }

            QLineEdit, QTextEdit, QComboBox {
                background-color: #404040;
                border: 1px solid #666666;
                border-radius: 3px;
                padding: 3px;
                color: #e0e0e0;
            }

            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border-color: #64b5f6;
            }

            QListWidget {
                background-color: #404040;
                border: 1px solid #666666;
                color: #e0e0e0;
            }

            QListWidget::item:selected {
                background-color: #4a4a4a;
                color: #64b5f6;
            }

            QListWidget::item:hover {
                background-color: #454545;
            }

            QScrollBar:vertical {
                background-color: #3a3a3a;
                width: 12px;
            }

            QScrollBar::handle:vertical {
                background-color: #666666;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #888888;
            }

            QLabel {
                color: #e0e0e0;
            }

            QTextEdit {
                background-color: #404040;
                color: #e0e0e0;
            }
        """

    def _on_mode_changed(self, mode: int):
        """Handle mode change from toolbar."""
        self.sidebar_stack.setCurrentIndex(mode)
        self.main_area_stack.setCurrentIndex(mode)

    def _on_model_auto_unloaded(self, model_name: str):
        """Handle automatic model unloading notification."""
        from PyQt5.QtWidgets import QMessageBox

        # Show notification about auto-unload
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Model Auto-Unloaded")
        msg.setText(f"The model '{model_name}' has been automatically unloaded due to 10 minutes of inactivity.")
        msg.setInformativeText("This helps manage memory usage. The model will be reloaded when you generate another image.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

        # Update the image generation panel to reflect no model loaded
        self.image_gen_panel.image_label.setText("Model unloaded due to inactivity - will reload when needed")

    def closeEvent(self, event):
        """Handle application close."""
        # Save settings
        settings = self.settings_manager.load_settings()
        self.settings_manager.save_settings(settings)

        # Stop any running image generation threads
        if (hasattr(self.image_gen_panel, 'current_generator') and
            self.image_gen_panel.current_generator is not None and
            self.image_gen_panel.current_generator.isRunning()):
            self.image_gen_panel.current_generator.quit()
            self.image_gen_panel.current_generator.wait(3000)  # Wait up to 3 seconds

        # Stop any running text enhancement threads
        if (hasattr(self.text_enhancer, 'current_worker') and
            self.text_enhancer.current_worker is not None and
            self.text_enhancer.current_worker.isRunning()):
            print("Stopping text enhancement worker...")
            self.text_enhancer.current_worker.requestInterruption()
            self.text_enhancer.current_worker.wait(2000)  # Wait up to 2 seconds
            if self.text_enhancer.current_worker.isRunning():
                self.text_enhancer.current_worker.terminate()
                self.text_enhancer.current_worker.wait(1000)

        # Clean up
        self.image_service.unload_model()
        self.text_enhancer.unload_model()

        event.accept()

    def _init_fullscreen_controls(self):
        """Initialize fullscreen exit controls."""
        # Create fullscreen overlay widget (compact version)
        self.fullscreen_overlay = QWidget(self)
        self.fullscreen_overlay.setVisible(False)
        self.fullscreen_overlay.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 0.8);
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        """)

        # Create overlay layout
        overlay_layout = QHBoxLayout(self.fullscreen_overlay)
        overlay_layout.setContentsMargins(8, 8, 8, 8)
        overlay_layout.setSpacing(6)

        # Exit fullscreen button (smaller)
        self.exit_fullscreen_btn = QPushButton("⛶")
        self.exit_fullscreen_btn.setFixedSize(32, 32)
        self.exit_fullscreen_btn.setToolTip("Exit Fullscreen (ESC)")
        self.exit_fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.9);
                color: #333;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ffffff;
                border-color: #999;
            }
        """)
        self.exit_fullscreen_btn.clicked.connect(self._exit_fullscreen)
        overlay_layout.addWidget(self.exit_fullscreen_btn)

        # Close application button (smaller)
        self.close_app_btn = QPushButton("✕")
        self.close_app_btn.setFixedSize(32, 32)
        self.close_app_btn.setToolTip("Close Application")
        self.close_app_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(244, 67, 54, 0.9);
                color: white;
                border: 1px solid #d32f2f;
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f44336;
                border-color: #b71c1c;
            }
        """)
        self.close_app_btn.clicked.connect(self.close)
        overlay_layout.addWidget(self.close_app_btn)

        # Position overlay in top-right corner (compact size)
        self.fullscreen_overlay.setFixedSize(80, 48)
        self._update_overlay_position()

    def _update_overlay_position(self):
        """Update the position of the fullscreen overlay."""
        if self.fullscreen_overlay.isVisible():
            # Position in top-right corner with some margin
            margin = 15
            self.fullscreen_overlay.move(
                self.width() - self.fullscreen_overlay.width() - margin,
                margin
            )

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        self._update_overlay_position()

    def keyPressEvent(self, event):
        """Handle keyboard events."""
        # ESC key exits fullscreen
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self._exit_fullscreen()
        else:
            super().keyPressEvent(event)

    def changeEvent(self, event):
        """Handle window state changes."""
        if event.type() == event.WindowStateChange:
            if self.isFullScreen():
                # Entered fullscreen - show overlay
                self.fullscreen_overlay.setVisible(True)
                self._update_overlay_position()
            else:
                # Exited fullscreen - hide overlay
                self.fullscreen_overlay.setVisible(False)
        super().changeEvent(event)

    def _exit_fullscreen(self):
        """Exit fullscreen mode."""
        self.showNormal()
