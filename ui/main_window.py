"""
Main application window.
"""
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QSplitter, QStackedWidget
from PyQt5.QtCore import Qt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.model_manager import ModelManager
from services.image_generator import ImageGenerationService
from services.memory_manager import MemoryManager
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

        # Start maximized
        self.showMaximized()

        # Initialize services (without loading model)
        self.settings_manager = SettingsManager()
        self.model_manager = ModelManager()
        self.image_service = ImageGenerationService()
        self.memory_manager = MemoryManager(self.image_service)

        # Don't load model on startup - use lazy loading instead
        # Model will be loaded when first generation is requested

        # Start memory monitoring
        self.memory_manager.start_monitoring()

        # Create UI
        self._init_ui()

        # Connect signals
        self.toolbar.mode_changed.connect(self._on_mode_changed)
        self.memory_manager.model_auto_unloaded.connect(self._on_model_auto_unloaded)

        # Connect model management to image generation panel
        self.model_mgmt_panel.model_installed.connect(self.image_gen_panel._refresh_model_dropdown)

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
        self.image_gen_panel = ImageGenerationPanel(self.model_manager, self.image_service)
        self.model_mgmt_panel = ModelManagementPanel(self.model_manager)
        self.settings_panel = SettingsPanel(self.settings_manager)

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

        # Clean up
        self.image_service.unload_model()

        event.accept()
