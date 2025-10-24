"""
Settings panel with sidebar and main area.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox
from PyQt5.QtWidgets import QMessageBox

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.settings import SettingsManager, AppSettings


class SettingsPanel:
    """Panel for application settings."""

    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager

        # Create sidebar and main area
        self.sidebar = self._create_sidebar()
        self.main_area = self._create_main_area()

    def _create_sidebar(self) -> QWidget:
        """Create the sidebar widget."""
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Application Settings"))

        theme_btn = QPushButton('🎨 Theme Settings')
        theme_btn.clicked.connect(lambda: QMessageBox.information(widget, "Theme",
                                                                 "Theme customization would open here.\n\n"
                                                                 "Future features:\n"
                                                                 "- Light/Dark themes\n"
                                                                 "- Custom color schemes\n"
                                                                 "- Font settings"))
        layout.addWidget(theme_btn)

        perf_btn = QPushButton('⚡ Performance')
        perf_btn.clicked.connect(lambda: QMessageBox.information(widget, "Performance",
                                                               "Performance settings would open here.\n\n"
                                                               "Future features:\n"
                                                               "- Memory optimization\n"
                                                               "- GPU settings\n"
                                                               "- Cache management"))
        layout.addWidget(perf_btn)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_main_area(self) -> QWidget:
        """Create the main area widget."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Current Settings Overview
        overview_group = QGroupBox("Current Settings")
        overview_layout = QVBoxLayout()

        # Load current settings
        settings = self.settings_manager.load_settings()

        theme_label = QLabel(f"Theme: {settings.theme}")
        overview_layout.addWidget(theme_label)

        perf_label = QLabel(f"Performance Mode: {settings.performance_mode}")
        overview_layout.addWidget(perf_label)

        res_label = QLabel(f"Output Resolution: {settings.output_resolution}")
        overview_layout.addWidget(res_label)

        cache_label = QLabel(f"Cache Location: {settings.cache_location}")
        overview_layout.addWidget(cache_label)

        if settings.default_model_folder:
            folder_label = QLabel(f"Default Model Folder: {settings.default_model_folder}")
            folder_label.setWordWrap(True)
            overview_layout.addWidget(folder_label)
        else:
            folder_label = QLabel("Default Model Folder: Not set")
            overview_layout.addWidget(folder_label)

        overview_group.setLayout(overview_layout)
        layout.addWidget(overview_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget
